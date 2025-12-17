import os
import subprocess
import sys
import traceback
import json
from modules.env_loader import load_env
from modules.input_handler import load_fasta
from modules.ss_generator import pybiomed_ss_candidates
from modules.ark_module import ark_vote_cases, get_default_models, ark_refine_structure, ark_analyze_sequence
from modules.backbone_predictor import run_backbone_fold_multichain
from modules.igpu_predictor import run_backbone_fold_multichain as run_igpu_fold
from modules.igpu_predictor import check_gpu_availability
from modules.visualization import generate_html_view

def run_pipeline(fasta, outdir, env_text, ssn, threshold, use_igpu, use_ext_env, ext_env_name, target_chains=None, log_callback=print):
    """
    Core MiniFold pipeline execution logic.
    
    Args:
        fasta: Path to input FASTA file
        outdir: Output directory
        env_text: Environment description text
        ssn: Number of candidates (SSN)
        threshold: Probability threshold
        use_igpu: Whether to use iGPU acceleration
        use_ext_env: Whether to use external environment
        ext_env_name: Name/path of external environment
        target_chains: Enforce specific number of chains (optional)
        log_callback: Function to handle log messages (default: print)
    """
    try:
        load_env()
        os.makedirs(outdir, exist_ok=True)
        log_callback(f"读取 FASTA: {fasta}")
        sequences = load_fasta(fasta)
        if not sequences:
            log_callback("未发现有效序列。")
            return

        input_base = os.path.basename(fasta)
        prefix = os.path.splitext(input_base)[0]
        workdir = os.path.join(outdir, prefix)
        three_d_dir = os.path.join(workdir, "3d_structures")
        os.makedirs(three_d_dir, exist_ok=True)

        req_path = os.path.join(workdir, "requirements.txt")
        req_lines = [
            f"env={env_text}",
            f"ssn={ssn}",
            f"threshold={threshold}",
        ]
        if target_chains:
            req_lines.append(f"target_chains={target_chains}")
            
        os.makedirs(workdir, exist_ok=True)
        with open(req_path, "w", encoding="utf-8") as f:
            f.write("\n".join(req_lines))
        with open(req_path, "r", encoding="utf-8") as f:
            req_text = f.read()

        gpu_ok, gpu_name, gpu_type = check_gpu_availability()
        auto_igpu = bool(use_igpu or gpu_ok or (use_ext_env and ext_env_name))
        for seq_id, sequence in sequences:
            log_callback(f"处理序列: {seq_id} (长度 {len(sequence)})")
            q_result = pybiomed_ss_candidates(sequence, env_text, num=ssn, target_chains=target_chains)
            cases = q_result.get("cases", [])
            cand_file = os.path.join(workdir, f"{prefix}_ss_candidates.json")
            with open(cand_file, "w", encoding="utf-8") as f:
                json.dump(cases, f, ensure_ascii=False, indent=2)
            raw_file = os.path.join(workdir, "raw_candidates.txt")
            with open(raw_file, "w", encoding="utf-8") as f:
                f.write(q_result.get("raw", ""))
            log_callback(f"候选生成完成，尝试数 {q_result.get('attempts', 0)}，行数 {q_result.get('lines', 0)}。")
            if q_result.get('logs'):
                for l in q_result['logs']:
                    log_callback(f"  [SS-Gen Log] {l}")

            models = get_default_models()
            api_key = os.environ.get("ARK_API_KEY", "")
            if not api_key:
                log_callback("Ark 投票未配置 API Key，继续本地流程并写入空投票结果。")
            votes = ark_vote_cases(models, sequence, env_text, cases, req_text=req_text)
            votes_file = os.path.join(workdir, f"{prefix}_votes.json")
            with open(votes_file, "w", encoding="utf-8") as f:
                json.dump(votes, f, ensure_ascii=False, indent=2)
            best_idx = votes.get("best_idx", -1)
            best_score = votes.get("best_score", 0.0)
            try:
                attempted = len(models)
                succeeded_models = set()
                for it in votes.get("cases", []):
                    for mm in it.get("models", []):
                        succeeded_models.add(mm.get("model"))
                succeeded = len(succeeded_models)
                log_callback(f"Ark 投票完成：共尝试 {attempted} 个模型，成功 {succeeded} 个，最佳分 {best_score:.2f}")
                
                # Detailed voting results
                if votes.get("cases"):
                    for case_it in votes["cases"]:
                        idx = case_it.get("idx") + 1
                        log_callback(f"  Case {idx}: Avg={case_it.get('avg', 0):.2f}, Med={case_it.get('med', 0):.2f}")
                        for m_res in case_it.get("models", []):
                            if m_res.get('p', -1) >= 0:
                                log_callback(f"    - {m_res['model']}: {m_res['p']:.2f}")
                            else:
                                log_callback(f"    - {m_res['model']}: FAILED ({m_res.get('error', 'unknown')})")
            except Exception:
                pass

            kept_cases = []
            
            # Logic Branch: Environment-Aware Refinement vs Standard Selection
            if env_text and cases:
                log_callback(f"检测到环境描述，启动【主考官优化-重审】流程...")
                
                # 1. Identify Top 3 Candidates from Initial Vote
                sorted_cases = sorted(votes.get("cases", []), key=lambda x: (x.get("avg", 0) + x.get("med", 0))/2, reverse=True)
                top_3_indices = [x["idx"] for x in sorted_cases[:3]]
                log_callback(f"  选取初选前三名 (Cases {[i+1 for i in top_3_indices]}) 进行优化...")
                
                chief_model = "doubao-seed-1-6-251015"
                refined_candidates = []
                
                # 2. Chief Examiner Refines Each (Parallel Execution)
                import concurrent.futures
                
                def refine_task(idx, i):
                    if idx >= len(cases): return None
                    case = cases[idx]
                    chains = case.get("chains", [])
                    log_callback(f"  [优化 {i+1}/3] 主考官 ({chief_model}) 正在调整 Case {idx+1} 以适应环境...")
                    r_chains, r_err = ark_refine_structure(chief_model, sequence, env_text, chains, api_key=api_key)
                    if r_chains:
                        return {
                            "origin_idx": idx,
                            "chains": r_chains,
                            "label": f"Refined_from_Case_{idx+1}"
                        }
                    else:
                        log_callback(f"    优化失败 (Case {idx+1}): {r_err}")
                        return None

                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    futures = [executor.submit(refine_task, idx, i) for i, idx in enumerate(top_3_indices)]
                    for f in concurrent.futures.as_completed(futures):
                        try:
                            res = f.result()
                            if res:
                                refined_candidates.append(res)
                        except Exception as e:
                            log_callback(f"    优化任务异常: {e}")
                
                # Sort refined candidates to match input order if needed, but not strictly necessary for re-voting
                
                # 3. Re-voting by Jury (excluding Chief)
                if refined_candidates:
                    jury_models = [m for m in models if m != chief_model]
                    if not jury_models: jury_models = models # Fallback
                    
                    log_callback(f"  [重审] {len(jury_models)} 位评审正在对 {len(refined_candidates)} 个优化后结构进行盲审...")
                    
                    vote_input_cases = [{"chains": rc["chains"]} for rc in refined_candidates]
                    new_votes = ark_vote_cases(jury_models, sequence, env_text, vote_input_cases, req_text=req_text, api_key=api_key)
                    
                    best_new_idx = new_votes.get("best_idx", -1)
                    best_new_score = new_votes.get("best_score", 0.0)
                    
                    log_callback(f"  重审结果: 最高分 {best_new_score:.2f}")
                    
                    if best_new_idx >= 0 and best_new_idx < len(refined_candidates):
                        winner = refined_candidates[best_new_idx]
                        log_callback(f"  >>> 最终优胜: {winner['label']} (Origin Case {winner['origin_idx']+1})")
                        
                        # Save Winner
                        case_dir = os.path.join(workdir, "case_final")
                        os.makedirs(case_dir, exist_ok=True)
                        chain_files = []
                        for m, ss in enumerate(winner["chains"]):
                            path = os.path.join(case_dir, f"{prefix}_final_{m+1}.fasta.txt")
                            with open(path, "w", encoding="utf-8") as f:
                                f.write(ss)
                            chain_files.append(os.path.basename(path))
                        
                        final_meta = {
                            "case": "final",
                            "p": float(best_new_score),
                            "chains": len(winner["chains"]),
                            "files": chain_files,
                            "origin_case": winner["origin_idx"] + 1
                        }
                        with open(os.path.join(case_dir, "case_final_meta.json"), "w", encoding="utf-8") as f:
                            json.dump(final_meta, f, ensure_ascii=False, indent=2)
                        
                        if best_new_score >= threshold:
                            kept_cases.append(final_meta)
                        else:
                            log_callback(f"  警告: 最终优胜得分 {best_new_score:.2f} 仍低于阈值 {threshold}，但将作为最佳结果保留。")
                            kept_cases.append(final_meta)
                    else:
                        log_callback("  重审未产生有效赢家，回退到原始最佳案例。")
                else:
                    log_callback("  所有优化尝试均失败，回退到原始最佳案例。")

            # Fallback / Standard Logic (If no env, or if optimization completely failed/yielded nothing)
            if not kept_cases and isinstance(best_idx, int) and best_idx >= 0 and best_idx < len(cases):
                log_callback(f"  使用原始最佳案例 Case {best_idx+1} (Score: {best_score:.2f})")
                case = cases[best_idx]
                chains = case.get("chains") or []
                case_dir = os.path.join(workdir, f"case_{best_idx+1}")
                os.makedirs(case_dir, exist_ok=True)
                chain_files = []
                for m, ss in enumerate(chains):
                    path = os.path.join(case_dir, f"{prefix}_2_case{best_idx+1}_{m+1}.fasta.txt")
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(ss)
                    chain_files.append(os.path.basename(path))
                meta = {"case": best_idx + 1, "p": float(best_score), "chains": len(chains), "files": chain_files}
                meta_path = os.path.join(case_dir, f"case_{best_idx+1}_meta.json")
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                
                if best_score >= threshold:
                    kept_cases.append(meta)
                else:
                    log_callback(f"  原始最佳案例得分 {best_score:.2f} 低于阈值。")

            kept_file = os.path.join(workdir, f"{prefix}_cases_kept.json")
            with open(kept_file, "w", encoding="utf-8") as f:
                json.dump(kept_cases, f, ensure_ascii=False, indent=2)

            annotation = analyze_sequence(sequence)
            annotation_file = os.path.join(workdir, f"{prefix}_annotation.txt")
            with open(annotation_file, "w", encoding="utf-8") as f:
                f.write(annotation)
            log_callback("功能注释完成。")

            # Extract Constraints if env_text is present
            constraints = []
            if env_text:
                log_callback("正在分析环境描述以提取物理约束...")
                # Use the first available model or chief model
                model_for_constraints = "doubao-seed-1-6-251015"
                constraints, c_err = ark_extract_constraints(model_for_constraints, sequence, env_text, api_key=api_key)
                if c_err:
                    log_callback(f"约束提取警告: {c_err}")
                    constraints = []
                else:
                    log_callback(f"提取到 {len(constraints)} 个物理约束: {[c.get('type') for c in constraints]}")

            generated_pdbs = []
            log_callback("生成 3D 结构...")
            if kept_cases:
                for rank, meta in enumerate(sorted(kept_cases, key=lambda x: x.get("p", 0.0), reverse=True), start=1):
                    prob = meta.get("p", 0.0)
                    case_idx = meta.get("case")
                    case_dir = os.path.join(workdir, f"case_{case_idx}")
                    chains = []
                    for fn in meta.get("files", []):
                        with open(os.path.join(case_dir, fn), "r", encoding="utf-8") as f:
                            chains.append(f.read().strip())
                    suffix = f"case{case_idx}_model_{rank}"
                    pdb_name = f"{prefix}_{suffix}.pdb"
                    pdb_path = os.path.join(three_d_dir, pdb_name)
                    
                    if auto_igpu:
                        if use_ext_env and ext_env_name:
                            log_callback(f"  > Case {case_idx} (p={prob:.2f}): Optimizing backbone (iGPU via external env '{ext_env_name}')...")
                            igpu_input_data = {"sequence": sequence, "chains": chains, "constraints": constraints}
                            tmp_input = os.path.join(workdir, f"igpu_input_case{case_idx}.json")
                            with open(tmp_input, "w", encoding="utf-8") as f:
                                json.dump(igpu_input_data, f)
                            
                            runner_script = os.path.join(os.getcwd(), "modules", "igpu_runner.py")
                            if os.path.sep in ext_env_name or "/" in ext_env_name:
                                # Path to python executable
                                cmd_list = [ext_env_name, runner_script, "--input", tmp_input, "--output", pdb_path]
                                cmd_str = subprocess.list2cmdline(cmd_list)
                            else:
                                # Conda env name
                                cmd_list = ["conda", "run", "-n", ext_env_name, "python", runner_script, "--input", tmp_input, "--output", pdb_path]
                                cmd_str = f'conda run -n {ext_env_name} python "{runner_script}" --input "{tmp_input}" --output "{pdb_path}"'
                            
                            try:
                                # Use shell=True for conda on Windows
                                result = subprocess.run(cmd_str if os.name == "nt" else cmd_list, 
                                                        capture_output=True, 
                                                        text=True, 
                                                        encoding="utf-8", 
                                                        shell=(os.name=="nt"))
                                
                                # Log iGPU Runner output
                                if result.stdout:
                                    log_callback(f"[iGPU Output] {result.stdout.strip()}")
                                if result.stderr:
                                    log_callback(f"[iGPU Error] {result.stderr.strip()}")
                                    
                                if result.returncode == 0:
                                    success = True
                                else:
                                    success = False
                                    log_callback(f"    iGPU External Failed (RC={result.returncode})")
                            except Exception as e:
                                success = False
                                log_callback(f"    Execution Error: {e}")
                            
                            if os.path.exists(tmp_input):
                                try: os.remove(tmp_input) 
                                except: pass
                        else:
                            log_callback(f"  > Case {case_idx} (p={prob:.2f}): Optimizing backbone (iGPU)...")
                            success = run_igpu_fold(sequence, chains, pdb_path)
                    else:
                        log_callback(f"  > Case {case_idx} (p={prob:.2f}): Optimizing backbone (Standard)...")
                        success = run_backbone_fold_multichain(sequence, chains, pdb_path)

                    if success:
                        html_name = f"{prefix}_{suffix}.html"
                        html_path = os.path.join(three_d_dir, html_name)
                        generate_html_view(pdb_path, html_path)
                        generated_pdbs.append({"pdb": pdb_name, "html": html_name, "chains": len(chains), "case": case_idx, "prob": prob})
                        
                        # Assembly step is now integrated into the predictor (Joint Optimization)
                        # So we don't need separate assembly logic here.


            else:
                log_callback("无保留案例，使用回退模型。")
                L = len(sequence)
                pattern = "HHHHH" + "CC" + "EEEEE" + "C"
                s = (pattern * ((L // len(pattern)) + 1))[:L]
                suffix = "fallback_model"
                pdb_name = f"{prefix}_{suffix}.pdb"
                pdb_path = os.path.join(three_d_dir, pdb_name)
                
                success = False
                if auto_igpu:
                    if use_ext_env and ext_env_name:
                        log_callback(f"  > Fallback: Optimizing backbone (iGPU via external env)...")
                        igpu_input_data = {"sequence": sequence, "chains": [s]}
                        tmp_input = os.path.join(workdir, f"igpu_input_fallback.json")
                        with open(tmp_input, "w", encoding="utf-8") as f:
                            json.dump(igpu_input_data, f)
                        
                        runner_script = os.path.join(os.getcwd(), "modules", "igpu_runner.py")
                        if os.path.sep in ext_env_name or "/" in ext_env_name:
                            cmd_list = [ext_env_name, runner_script, "--input", tmp_input, "--output", pdb_path]
                            cmd_str = subprocess.list2cmdline(cmd_list)
                        else:
                            cmd_list = ["conda", "run", "-n", ext_env_name, "python", runner_script, "--input", tmp_input, "--output", pdb_path]
                            cmd_str = f'conda run -n {ext_env_name} python "{runner_script}" --input "{tmp_input}" --output "{pdb_path}"'
                        
                        try:
                            result = subprocess.run(cmd_str if os.name == "nt" else cmd_list, 
                                                    capture_output=True, 
                                                    text=True, 
                                                    encoding="utf-8", 
                                                    shell=(os.name=="nt"))
                            
                            if result.stdout:
                                log_callback(f"[iGPU Output] {result.stdout.strip()}")
                            if result.stderr:
                                log_callback(f"[iGPU Error] {result.stderr.strip()}")
                                
                            if result.returncode == 0:
                                success = True
                            else:
                                success = False
                                log_callback(f"    iGPU External Failed (RC={result.returncode})")
                        except Exception as e:
                            success = False
                            log_callback(f"    Execution Error: {e}")
                        
                        if os.path.exists(tmp_input):
                            try: os.remove(tmp_input) 
                            except: pass
                    else:
                        log_callback(f"  > Fallback: Optimizing backbone (iGPU)...")
                        success = run_igpu_fold(sequence, [s], pdb_path, constraints=constraints)
                    if not success:
                        log_callback("  > Fallback: iGPU path failed, trying Standard optimizer...")
                        success = run_backbone_fold_multichain(sequence, [s], pdb_path)
                else:
                    log_callback(f"  > Fallback: Optimizing backbone (Standard)...")
                    success = run_backbone_fold_multichain(sequence, [s], pdb_path)
                
                if success:
                    html_name = f"{prefix}_{suffix}.html"
                    html_path = os.path.join(three_d_dir, html_name)
                    generate_html_view(pdb_path, html_path)
                    generated_pdbs.append({"pdb": pdb_name, "html": html_name, "chains": 1, "case": "fallback", "prob": 0.0})

            # Report
            log_callback("生成分析报告...")
            report_path = os.path.join(workdir, f"{prefix}_report.md")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"# {prefix} 结构预测报告\n\n")
                f.write(f"## 基本信息\n- 序列长度: {len(sequence)}\n- 候选生成数: {ssn}\n- 筛选阈值: {threshold}\n\n")
                f.write("## 功能注释\n```\n" + annotation + "\n```\n\n")
                f.write("## 投票与聚合\n")
                try:
                    f.write(f"- 最可信案例索引: {votes.get('best_idx', -1)}\n")
                    f.write(f"- 聚合分数: {votes.get('best_score', 0.0):.2f}\n")
                    f.write("- 候选评分概览:\n")
                    for it in votes.get("cases", []):
                        f.write(f"  - Case {it.get('idx')+1}: avg={it.get('avg',0.0):.2f}, med={it.get('med',0.0):.2f}, chains={it.get('chains',0)}\n")
                except Exception:
                    f.write("- 无可用投票数据\n")
                f.write("## 结构模型\n")
                if generated_pdbs:
                    f.write("| 模型 | 类型 | 概率 | 链数 | 文件 |\n|---|---|---|---|---|\n")
                    for g in generated_pdbs:
                        m_type = g.get('type', 'Standard')
                        f.write(f"| {g['pdb']} | {m_type} | {g['prob']:.2f} | {g['chains']} | [View 3D](3d_structures/{g['html']}) |\n")
                else:
                    f.write("未能生成有效模型。\n")
            log_callback(f"报告已生成: {report_path}")

        log_callback("==== 所有任务已完成 ====")

    except Exception as e:
        log_callback(f"发生未捕获异常:\n{traceback.format_exc()}")
