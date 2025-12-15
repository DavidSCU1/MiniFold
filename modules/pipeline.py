import os
import subprocess
import sys
import traceback
import json
from modules.env_loader import load_env
from modules.input_handler import load_fasta
from modules.qwen_module import qwen_ss_candidates
from modules.llm_module import analyze_sequence, deepseek_eval_case
from modules.backbone_predictor import run_backbone_fold_multichain
from modules.igpu_predictor import run_backbone_fold_multichain as run_igpu_fold
from modules.visualization import generate_html_view
from modules.assembler import parse_pdb_chains, assemble_chains, write_complex_pdb

def run_pipeline(fasta, outdir, env_text, ssn, threshold, use_igpu, use_ext_env, ext_env_name, log_callback=print):
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
        os.makedirs(workdir, exist_ok=True)
        with open(req_path, "w", encoding="utf-8") as f:
            f.write("\n".join(req_lines))
        with open(req_path, "r", encoding="utf-8") as f:
            req_text = f.read()

        for seq_id, sequence in sequences:
            log_callback(f"处理序列: {seq_id} (长度 {len(sequence)})")
            q_result = qwen_ss_candidates(sequence, env_text, num=ssn)
            cases = q_result.get("cases", [])
            cand_file = os.path.join(workdir, f"{prefix}_ss_candidates.json")
            with open(cand_file, "w", encoding="utf-8") as f:
                json.dump(cases, f, ensure_ascii=False, indent=2)
            raw_file = os.path.join(workdir, "raw_qwen.txt")
            with open(raw_file, "w", encoding="utf-8") as f:
                f.write(q_result.get("raw", ""))
            log_callback(f"Qwen 候选生成完成，尝试数 {q_result.get('attempts', 0)}，行数 {q_result.get('lines', 0)}。")

            kept_cases = []
            for i, case in enumerate(cases):
                chains = case.get("chains") or []
                if not chains:
                    continue
                case_dir = os.path.join(workdir, f"case_{i+1}")
                os.makedirs(case_dir, exist_ok=True)
                chain_files = []
                for m, ss in enumerate(chains):
                    path = os.path.join(case_dir, f"{prefix}_2_case{i+1}_{m+1}.fasta.txt")
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(ss)
                    chain_files.append(os.path.basename(path))
                p = deepseek_eval_case(sequence, env_text, chains, req_text=req_text)
                meta = {"case": i + 1, "p": p, "chains": len(chains), "files": chain_files}
                meta_path = os.path.join(case_dir, f"case_{i+1}_meta.json")
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                if isinstance(p, float) and p >= threshold:
                    kept_cases.append(meta)
                    log_callback(f"  保留案例 {i+1}，概率 {p:.2f}")
                else:
                    log_callback(f"  丢弃案例 {i+1}，概率 {p}")
                    for fn in chain_files:
                        try:
                            os.remove(os.path.join(case_dir, fn))
                        except Exception:
                            pass
                    try:
                        os.remove(meta_path)
                    except Exception:
                        pass
                    try:
                        os.rmdir(case_dir)
                    except Exception:
                        pass

            kept_file = os.path.join(workdir, f"{prefix}_cases_kept.json")
            with open(kept_file, "w", encoding="utf-8") as f:
                json.dump(kept_cases, f, ensure_ascii=False, indent=2)

            annotation = analyze_sequence(sequence)
            annotation_file = os.path.join(workdir, f"{prefix}_annotation.txt")
            with open(annotation_file, "w", encoding="utf-8") as f:
                f.write(annotation)
            log_callback("功能注释完成。")

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
                    
                    if use_igpu:
                        if use_ext_env and ext_env_name:
                            log_callback(f"  > Case {case_idx} (p={prob:.2f}): Optimizing backbone (iGPU via external env '{ext_env_name}')...")
                            igpu_input_data = {"sequence": sequence, "chains": chains}
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
                if use_igpu:
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
                        success = run_igpu_fold(sequence, [s], pdb_path)
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
