import argparse
import os
import sys
import json
import subprocess

from modules.input_handler import load_fasta
from modules.ss_generator import pybiomed_ss_candidates
from modules.ark_module import ark_vote_cases, ark_analyze_sequence, get_default_models, ark_refine_structure
from modules.backbone_predictor import run_backbone_fold_multichain
from modules.visualization import generate_html_view
from modules.env_loader import load_env
from modules.assembler import parse_pdb_chains, assemble_chains, write_complex_pdb
from modules.refine import run_refinements, analyze_ubiquitin_core
from modules.quality import summarize_structure
from modules.ark_module import ark_audit_structure, get_default_models
from modules.esm_runner import predict_structure_with_esm

def print_progress(percent, step):
    print(f"[PROGRESS] {percent}% - {step}")
    sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser(description="MiniFold: Protein Structure Prediction & Analysis Workflow")
    parser.add_argument("input", help="Path to input FASTA file")
    parser.add_argument("--outdir", default="output", help="Output directory")
    parser.add_argument("--env", default=None, help="Expected protein environment description")
    parser.add_argument("--ssn", type=int, default=5, help="Number of SS candidates from DeepSeek")
    parser.add_argument("--threshold", type=float, default=0.5, help="Likelihood threshold for Qwen filter")
    parser.add_argument("--igpu", action="store_true", help="Enable iGPU acceleration")
    parser.add_argument("--igpu-env", default=None, help="Conda environment for iGPU execution")
    parser.add_argument("--backend", default="auto", choices=["auto", "ipex", "directml", "cuda", "cpu", "oneapi_cpu"], help="Acceleration backend")
    parser.add_argument("--npu", action="store_true", help="Enable NPU head refinement")
    parser.add_argument("--npu-env", default=None, help="Conda environment for NPU execution")
    parser.add_argument("--target-chains", type=int, default=None, help="Enforce specific number of chains (e.g. 1, 2)")
    parser.add_argument("--refine-ramachandran", action="store_true", help="Adjust backbone dihedrals via local torsion smoothing")
    parser.add_argument("--refine-hydrophobic", action="store_true", help="Analyze and nudge hydrophobic core packing")
    parser.add_argument("--repack-long-sides", action="store_true", help="Repack long sidechains (Lys/Gln/Glu)")
    parser.add_argument("--md", default=None, choices=[None, "openmm", "amber", "gromacs"], help="Short MD/minimization engine")
    parser.add_argument("--md-steps", type=int, default=0, help="MD steps for refinement (OpenMM recommended)")
    parser.add_argument("--esm-backbone", action="store_true", help="Use ESM-based backbone model from 3d_moudel/best_model_gpu.pt")
    parser.add_argument("--esm-env", default=None, help="Conda environment or python path for ESM backbone")
    
    args = parser.parse_args()
    
    print_progress(0, "Initializing workflow...")
    load_env()
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
        
    print(f"Reading input from {args.input}...")
    try:
        sequences = load_fasta(args.input)
        print_progress(5, "Loaded FASTA sequences")
    except Exception as e:
        print(f"Error loading FASTA: {e}")
        return

    if not sequences:
        print("No sequences found.")
        return

    # Process each sequence (For now, let's just process the first one or loop)
    # The doc implies a workflow. Let's loop.
    
    # Setup workflow directory per input file prefix
    input_base = os.path.basename(args.input)
    prefix = os.path.splitext(input_base)[0]
    workdir = os.path.join(args.outdir, prefix)
    os.makedirs(workdir, exist_ok=True)
    three_d_dir = os.path.join(workdir, "3d_structures")
    os.makedirs(three_d_dir, exist_ok=True)
    req_path = os.path.join(workdir, "requirements.txt")
    req_lines = [
        f"env={args.env if args.env else ''}",
        f"ssn={args.ssn}",
        f"threshold={args.threshold}"
    ]
    with open(req_path, "w", encoding="utf-8") as f:
        f.write("\n".join(req_lines))
    with open(req_path, "r", encoding="utf-8") as f:
        req_text = f.read()
    
    total_seqs = len(sequences)
    for idx, (seq_id, sequence) in enumerate(sequences):
        print(f"\nProcessing sequence: {seq_id}")
        print_progress(10 + int((idx/total_seqs)*5), f"Processing sequence {seq_id}: Predicting SS (PyBioMed)")
        safe_id = "".join([c if c.isalnum() else "_" for c in seq_id])
        
        # 1. Generate Candidates
        # Pass target_chains to pybiomed_ss_candidates
        if args.target_chains is not None and args.target_chains > 0:
            print(f"Info: Enforcing target chain count = {args.target_chains}")
        q_result = pybiomed_ss_candidates(sequence, args.env, num=args.ssn, target_chains=args.target_chains)
        print_progress(30, "Generated SS candidates")
        cases = q_result.get("cases", [])
        cand_file = os.path.join(workdir, f"{prefix}_ss_candidates.json")
        with open(cand_file, "w", encoding="utf-8") as f:
            json.dump(cases, f, ensure_ascii=False, indent=2)
        raw_file = os.path.join(workdir, "raw_candidates.txt")
        with open(raw_file, "w", encoding="utf-8") as f:
            f.write(q_result.get("raw", ""))
        
        # 2. Verify with Ark (Voting)
        print_progress(35, "Verifying candidates (Ark)")
        
        models = get_default_models()
        # Use simple voting
        votes = ark_vote_cases(models, sequence, args.env, cases, req_text=req_text)
        
        # --- Refinement Loop Start ---
        # User requested: Pick best -> Modify based on Env -> Re-vote -> Check threshold -> Repeat if needed
        best_idx = votes.get("best_idx", -1)
        
        if best_idx != -1 and args.env:
            refine_max_attempts = 3
            refine_cnt = 0
            
            # Loop condition: 
            # 1. First run (refine_cnt == 0) - Always try to incorporate env info
            # 2. Score < Threshold - Keep trying until passed
            # 3. Limit attempts
            
            while refine_cnt < refine_max_attempts:
                best_score = votes.get("best_score", 0.0)
                
                # If we have refined at least once AND score is good enough, stop.
                if refine_cnt > 0 and best_score >= args.threshold:
                    print(f"  > Satisfaction reached ({best_score:.4f} >= {args.threshold}). Stopping refinement.")
                    break
                
                refine_cnt += 1
                best_case = cases[best_idx] # Always refine the current best
                
                print_progress(45, f"Refining best candidate (Attempt {refine_cnt}, Current Best={best_score:.4f})...")
                
                refiner_model = "doubao-seed-1-8-251215"
                if refiner_model not in models:
                    refiner_model = models[0]
                    
                print(f"  > Using {refiner_model} to refine structure based on environment: '{args.env}'")
                
                # We pass the *current best chains*.
                refined_chains, err = ark_refine_structure(refiner_model, sequence, args.env, best_case["chains"], target_chains=args.target_chains)
                
                if refined_chains:
                    # Check if chains changed? If identical, maybe stop?
                    if refined_chains == best_case["chains"]:
                        print("  > Refinement returned identical structure. Stopping.")
                        break

                    print(f"  > Refinement successful. Chains: {len(refined_chains)}")
                    refined_case_idx = len(cases)
                    refined_case = {"chains": refined_chains, "is_refined": True, "parent": best_idx, "round": refine_cnt}
                    cases.append(refined_case)
                    
                    # Correctly set the index for the new vote result BEFORE calling ark_vote_cases
                    # However, ark_vote_cases re-indexes starting from 0.
                    # We need to manually patch the index in the result to match our main 'cases' list.
                    
                    # We pass [refined_case] which has index 0 in the sub-call.
                    new_vote = ark_vote_cases(models, sequence, args.env, [refined_case], req_text=req_text)
                    
                    if new_vote["cases"]:
                        res = new_vote["cases"][0]
                        # Fix the index for display and storage
                        res["idx"] = refined_case_idx
                        
                        # Manually print the correct Case ID log
                        # The sub-call printed "Case 1", but it's actually "Case {refined_case_idx + 1}"
                        print(f"[Ark Vote] (Correction) Above vote was for Refined Candidate {refined_case_idx + 1}")
                        
                        new_score = res["avg"]
                        
                        print(f"  > Refined Score: {new_score:.4f} (Previous Best: {best_score:.4f})")
                        
                        res["idx"] = refined_case_idx
                        votes["cases"].append(res)
                        
                        # UPDATED LOGIC:
                        # If refined score meets threshold, we accept it as the new best,
                        # even if it's slightly lower than original (as long as it satisfies the user's constraints).
                        # Or, strictly speaking, we want the structure that respects the Env.
                        # Since refinement explicitly incorporates Env, we should prefer it if it's "good enough".
                        # But for safety, let's stick to "if it improves OR is above threshold".
                        
                        is_better = new_score > best_score
                        is_good_enough = new_score >= args.threshold
                        
                        if is_better or is_good_enough:
                            if is_better:
                                print(f"  > Refined candidate is BETTER. Replacing original top choice.")
                            else:
                                print(f"  > Refined candidate is GOOD ENOUGH (>{args.threshold}). Replacing original top choice to honor environment constraints.")
                            
                            # OVERWRITE LOGIC:
                            # Instead of just pointing best_idx to the new case, we REPLACE the data in the original best case.
                            # This ensures downstream logic (like 3D gen) uses the refined structure for the original Case ID.
                            
                            cases[best_idx] = refined_case # Overwrite original case data
                            refined_case["idx"] = best_idx # Ensure internal index consistency
                            
                            # We also need to update the vote record for this index
                            # Remove the old vote record for best_idx and append the new one
                            votes["cases"] = [v for v in votes["cases"] if v["idx"] != best_idx]
                            res["idx"] = best_idx
                            votes["cases"].append(res)
                            
                            votes["best_idx"] = best_idx
                            votes["best_score"] = new_score
                            
                            # Clean up the appended temporary case
                            cases.pop() 
                        else:
                            print(f"  > Refined candidate did not improve score and is below threshold.")
                else:
                    print(f"  > Refinement failed: {err}")
                    break # Stop on error
        # --- Refinement Loop End ---

        votes_file = os.path.join(workdir, f"{prefix}_votes.json")
        with open(votes_file, "w", encoding="utf-8") as f:
            json.dump(votes, f, ensure_ascii=False, indent=2)
            
        kept_cases = []
        if votes.get("cases"):
            for case_it in votes["cases"]:
                avg_score = case_it.get("avg", 0.0)
                idx = case_it.get("idx")
                if avg_score >= args.threshold:
                    # Reconstruct meta format expected by downstream
                    # Original meta: {"case": i+1, "p": p, "chains": len(chains), "files": [...]}
                    # We need to ensure chain files exist for visualization/folding
                    
                    case_data = cases[idx]
                    chains = case_data.get("chains", [])
                    case_dir = os.path.join(workdir, f"case_{idx+1}")
                    os.makedirs(case_dir, exist_ok=True)
                    chain_files = []
                    for m, ss in enumerate(chains):
                        path = os.path.join(case_dir, f"{prefix}_2_case{idx+1}_{m+1}.fasta.txt")
                        with open(path, "w", encoding="utf-8") as f:
                            f.write(ss)
                        chain_files.append(os.path.basename(path))
                        
                    meta = {
                        "case": idx+1, 
                        "p": avg_score, 
                        "chains": len(chains), 
                        "files": chain_files
                    }
                    kept_cases.append(meta)

        # STRICT TOP-1 POLICY (Moved Up):
        # Sort and truncate kept_cases IMMEDIATELY after collection.
        # This ensures only the best case is saved to disk and processed further.
        if kept_cases:
            # Sort by probability descending
            kept_cases.sort(key=lambda x: x.get("p", 0.0), reverse=True)
            
            # Keep only the top 1
            if len(kept_cases) > 1:
                print(f"Info: Dropping {len(kept_cases)-1} suboptimal candidates. Keeping only the best (p={kept_cases[0].get('p'):.4f}).")
                kept_cases = kept_cases[:1]

        kept_file = os.path.join(workdir, f"{prefix}_cases_kept.json")
        with open(kept_file, "w", encoding="utf-8") as f:
            json.dump(kept_cases, f, ensure_ascii=False, indent=2)
        
        print_progress(60, "Candidates verified. Analyzing sequence...")
        # 3. LLM Annotation
        annotation = ark_analyze_sequence(sequence)
        annotation_file = os.path.join(workdir, f"{prefix}_annotation.txt")
        with open(annotation_file, "w", encoding='utf-8') as f:
            f.write(annotation)
        print(f"Annotation saved to {annotation_file}")
            
        print("\nGenerating 3D structures via Backbone Predictor...")
        print_progress(65, "Generating 3D Structures")
        generated_pdbs = []
        # Strict Filtering: Ensure we ONLY process cases that are in kept_cases
        # This prevents processing of obsolete or temporary cases that might have been created during refinement
        # but not selected.
        
        # Note: kept_cases is already populated based on the FINAL state of votes['cases']
        # which has been updated by the refinement loop (overwriting original indices).
        # So iterating over kept_cases is correct.
        
        if kept_cases:
            # Sort by probability
            sorted_cases = sorted(kept_cases, key=lambda x: x.get("p", 0.0), reverse=True)
            # Already truncated above, but safe to keep logic general
            
            # Additional safety: Verify that indices are valid and correspond to current 'cases' list
            valid_sorted_cases = []
            for meta in sorted_cases:
                idx = meta.get("case") - 1 # meta['case'] is 1-based index
                if 0 <= idx < len(cases):
                    valid_sorted_cases.append(meta)
            
            # STRICT LIMIT: Only generate 3D structure for the TOP-1 case
            # (Redundant now but harmless double-check)
            if valid_sorted_cases:
                valid_sorted_cases = valid_sorted_cases[:1]
                
            sorted_cases = valid_sorted_cases
            total_kept = len(sorted_cases)
            
            for rank, meta in enumerate(sorted_cases, start=1):
                # Update progress for 3D generation (65% -> 95%)
                p_val = 65 + int(((rank-1)/total_kept) * 30)
                print_progress(p_val, f"Generating 3D Structure {rank}/{total_kept}")
                
                prob = meta.get("p", 0.0)
                case_idx = meta.get("case") # This is the 1-based index from meta
                
                # Double check: does this index point to the correct data in 'cases'?
                # Since we overwrote cases[best_idx] in refinement, cases[idx] holds the REFINED chains.
                
                case_dir = os.path.join(workdir, f"case_{case_idx}")
                chains = []
                
                # Re-read chains from the FILES associated with this case
                # Wait, if we refined, did we update the files on disk?
                # The refinement loop updated 'cases' in memory but NOT the files on disk!
                # We must use the chains from memory (cases[idx]["chains"]) instead of reading old files.
                
                # Fix: Use chains from memory
                real_idx = case_idx - 1
                memory_chains = cases[real_idx].get("chains", [])
                
                if memory_chains:
                    chains = memory_chains
                else:
                    # Fallback to files if memory empty (shouldn't happen)
                    for fn in meta.get("files", []):
                        with open(os.path.join(case_dir, fn), "r", encoding="utf-8") as f:
                            chains.append(f.read().strip())
                
                suffix = f"case{case_idx}_model_{rank}"
                pdb_name = f"{prefix}_{suffix}.pdb"
                pdb_path = os.path.join(three_d_dir, pdb_name)
                
                success = False
                if getattr(args, "esm_backbone", False):
                    print(f"  > Case {case_idx} (p={prob:.2f}): Generating backbone via ESM model (backend={args.backend})...")
                    ss_str_for_case = "".join(chains)
                    ss_path = os.path.join(workdir, f"{prefix}_case{case_idx}_ss.txt")
                    with open(ss_path, "w", encoding="utf-8") as f:
                        f.write(ss_str_for_case)
                    npz_path = os.path.join(workdir, f"{prefix}_case{case_idx}_esm.npz")
                    if getattr(args, "esm_env", None):
                        runner_script = os.path.join(os.getcwd(), "modules", "esm_runner.py")
                        if os.sep in args.esm_env or "/" in args.esm_env:
                            cmd_list = [
                                args.esm_env,
                                runner_script,
                                "--fasta",
                                args.input,
                                "--ss",
                                ss_path,
                                "--output",
                                pdb_path,
                                "--npz",
                                npz_path,
                            ]
                            if args.backend and args.backend != "auto":
                                cmd_list.extend(["--backend", args.backend])
                            cmd_str = subprocess.list2cmdline(cmd_list)
                        else:
                            if sys.platform == "win32":
                                backend_arg = f' --backend {args.backend}' if args.backend and args.backend != "auto" else ""
                                cmd_str = (
                                    f'cmd /c conda run -n {args.esm_env} python "{runner_script}" --fasta "{args.input}"'
                                    f' --ss "{ss_path}" --output "{pdb_path}" --npz "{npz_path}"{backend_arg}'
                                )
                                cmd_list = cmd_str
                            else:
                                cmd_list = [
                                    "conda",
                                    "run",
                                    "-n",
                                    args.esm_env,
                                    "python",
                                    runner_script,
                                    "--fasta",
                                    args.input,
                                    "--ss",
                                    ss_path,
                                    "--output",
                                    pdb_path,
                                    "--npz",
                                    npz_path,
                                ]
                                if args.backend and args.backend != "auto":
                                    cmd_list.extend(["--backend", args.backend])
                                cmd_str = " ".join(cmd_list)
                        try:
                            use_shell = sys.platform == "win32"
                            env_vars = os.environ.copy()
                            env_vars["PYTHONIOENCODING"] = "utf-8"
                            env_vars["PYTHONUTF8"] = "1"
                            result = subprocess.run(
                                cmd_str if sys.platform == "win32" else cmd_list,
                                capture_output=True,
                                text=True,
                                encoding="utf-8",
                                errors="replace",
                                shell=use_shell,
                                env=env_vars,
                            )
                            if result.stdout:
                                print(f"[ESM Output] {result.stdout.strip()}")
                            if result.stderr:
                                print(f"[ESM Error] {result.stderr.strip()}")
                            success = result.returncode == 0 and os.path.exists(pdb_path)
                        except Exception as e:
                            print(f"    ESM backbone external failed: {e}")
                            success = False
                    else:
                        try:
                            info = predict_structure_with_esm(
                                fasta_path=args.input,
                                ss_path=ss_path,
                                output_pdb_path=pdb_path,
                                model_path=None,
                                npz_output_path=npz_path,
                                backend=args.backend,
                            )
                            backend_used = info.get("backend")
                            device_used = info.get("device")
                            if backend_used or device_used:
                                print(f"    ESM backend={backend_used} device={device_used}")
                            success = os.path.exists(pdb_path)
                        except Exception as e:
                            print(f"    ESM backbone generation failed: {e}")
                elif args.igpu:
                    if args.igpu_env:
                        # Use process isolation for iGPU env
                        print(f"  > Case {case_idx} (p={prob:.2f}): Optimizing backbone (iGPU via external env '{args.igpu_env}')...")
                        
                        # Prepare input for isolated runner
                        igpu_input_data = {"sequence": sequence, "chains": chains}
                        tmp_input = os.path.join(workdir, f"igpu_input_case{case_idx}.json")
                        with open(tmp_input, "w", encoding="utf-8") as f:
                            json.dump(igpu_input_data, f)
                        
                        # Construct runner command
                        runner_script = os.path.join(os.getcwd(), "modules", "igpu_runner.py")
                        
                        # Check if env is a path or name
                        if os.sep in args.igpu_env or "/" in args.igpu_env:
                            # Path to python
                            cmd_list = [args.igpu_env, runner_script, "--input", tmp_input, "--output", pdb_path]
                            if args.backend and args.backend != "auto":
                                cmd_list.extend(["--backend", args.backend])
                            cmd_str = subprocess.list2cmdline(cmd_list)
                        else:
                            # Conda env name
                            if sys.platform == "win32":
                                backend_arg = f' --backend {args.backend}' if args.backend and args.backend != "auto" else ''
                                cmd_str = f'cmd /c conda run -n {args.igpu_env} python "{runner_script}" --input "{tmp_input}" --output "{pdb_path}"{backend_arg}'
                                cmd_list = cmd_str # For shell=True/False consideration, win usually needs string for cmd /c
                            else:
                                cmd_list = ["conda", "run", "-n", args.igpu_env, "python", runner_script, "--input", tmp_input, "--output", pdb_path]
                                if args.backend and args.backend != "auto":
                                    cmd_list.extend(["--backend", args.backend])
                                cmd_str = " ".join(cmd_list)

                        try:
                            use_shell = (sys.platform == "win32")
                            env_vars = os.environ.copy()
                            env_vars["PYTHONIOENCODING"] = "utf-8"
                            env_vars["PYTHONUTF8"] = "1"
                            result = subprocess.run(
                                cmd_str if sys.platform == "win32" else cmd_list,
                                capture_output=True,
                                text=True,
                                encoding="utf-8",
                                errors='replace',
                                shell=use_shell,
                                env=env_vars
                            )
                            if result.stdout: print(f"[iGPU Output] {result.stdout.strip()}")
                            if result.stderr: print(f"[iGPU Error] {result.stderr.strip()}")
                            if result.returncode == 0:
                                success = True
                            else:
                                print(f"    iGPU External Failed (RC={result.returncode})")
                        except Exception as e:
                            print(f"    Execution Error: {e}")
                        if os.path.exists(tmp_input):
                            try: os.remove(tmp_input)
                            except: pass
                    else:
                        print(f"  > Case {case_idx} (p={prob:.2f}): Optimizing backbone (iGPU)...")
                        from modules.igpu_predictor import run_backbone_fold_multichain as run_igpu_fold
                        success = run_igpu_fold(sequence, chains, pdb_path, backend=args.backend)
                else:
                    print(f"  > Case {case_idx} (p={prob:.2f}): Optimizing backbone (Standard)...")
                    success = run_backbone_fold_multichain(sequence, chains, pdb_path)
                
                if success:
                    if args.npu:
                        try:
                            runner_script_npu = os.path.join(os.getcwd(), "modules", "npu_runner.py")
                            if args.npu_env:
                                if os.sep in args.npu_env or "/" in args.npu_env:
                                    cmd_list_npu = [args.npu_env, runner_script_npu, "--input", pdb_path, "--output", pdb_path]
                                    cmd_str_npu = subprocess.list2cmdline(cmd_list_npu)
                                else:
                                    if sys.platform == "win32":
                                        cmd_str_npu = f'cmd /c conda run -n {args.npu_env} python "{runner_script_npu}" --input "{pdb_path}" --output "{pdb_path}"'
                                        cmd_list_npu = cmd_str_npu
                                    else:
                                        cmd_list_npu = ["conda", "run", "-n", args.npu_env, "python", runner_script_npu, "--input", pdb_path, "--output", pdb_path]
                                        cmd_str_npu = " ".join(cmd_list_npu)
                                use_shell_npu = (sys.platform == "win32")
                                res_npu = subprocess.run(
                                    cmd_str_npu if sys.platform == "win32" else cmd_list_npu,
                                    capture_output=True,
                                    text=True,
                                    encoding="utf-8",
                                    shell=use_shell_npu
                                )
                                if res_npu.stdout:
                                    print(f"[NPU Output] {res_npu.stdout.strip()}")
                                if res_npu.stderr:
                                    print(f"[NPU Error] {res_npu.stderr.strip()}")
                            else:
                                try:
                                    import modules.npu_runner as _nr
                                    _nr.run_inplace(pdb_path)
                                except Exception as e:
                                    print(f"[NPU Inline Error] {e}")
                        except Exception as e:
                            print(f"[NPU Runner Error] {e}")
                    # Refinement Step (Automatic Assembly & Sidechain Packing)
                    print(f"  > Refinement: Optimizing sidechains and assembly for {pdb_name}...")
                    try:
                        remark_lines = []
                        try:
                            with open(pdb_path, "r", encoding="utf-8", errors="replace") as f:
                                for line in f:
                                    if line.startswith("REMARK"):
                                        remark_lines.append(line.rstrip("\r\n"))
                                    elif line.startswith("ATOM") or line.startswith("HETATM"):
                                        break
                        except Exception:
                            remark_lines = []
                        chains_data = parse_pdb_chains(pdb_path)
                        if chains_data:
                            # 1. Assembly (Docking) if needed
                            if len(chains_data) > 1:
                                assembled_chains = assemble_chains(chains_data)
                            else:
                                assembled_chains = chains_data
                            
                            # 2. Sidechain Packing (Full Atom Refinement)
                            # Update path to reflect refined status
                            refined_pdb_name = f"{prefix}_{suffix}_refined.pdb"
                            refined_pdb_path = os.path.join(three_d_dir, refined_pdb_name)
                            
                            if write_complex_pdb(assembled_chains, sequence, refined_pdb_path, remark_lines=remark_lines):
                                print(f"  > Refinement complete. Saved to {refined_pdb_name}")
                                # Use refined model for final output
                                pdb_path = refined_pdb_path
                                pdb_name = refined_pdb_name
                                try:
                                    run_refinements(
                                        pdb_path,
                                        sequence,
                                        do_ramachandran=args.refine_ramachandran,
                                        do_hydrophobic=args.refine_hydrophobic,
                                        repack_long=args.repack_long_sides,
                                        md_engine=args.md,
                                        md_steps=args.md_steps
                                    )
                                    core_info = analyze_ubiquitin_core(pdb_path, sequence)
                                    if core_info and core_info.get("pair"):
                                        d = core_info["pair"]["distance"]
                                        print(f"  > Ubiquitin Ile44–Val70 distance: {d:.2f} Å")
                                except Exception as _e:
                                    print(f"  > Post-refinement step skipped: {_e}")
                            else:
                                print("  > Refinement failed to write. Using raw backbone.")
                        else:
                            print("  > Refinement skipped (no chains parsed).")
                    except Exception as e:
                        print(f"  > Refinement error: {e}")

                    html_name = os.path.splitext(pdb_name)[0] + ".html"
                    html_path = os.path.join(three_d_dir, html_name)
                    generate_html_view(pdb_path, html_path)
                    generated_pdbs.append({
                        "pdb": pdb_name,
                        "html": html_name,
                        "chains": len(chains),
                        "case": case_idx,
                        "prob": prob
                    })
                    try:
                        ss_str = "".join(chains)
                        summary = summarize_structure(pdb_path, sequence, ss_str)
                        model = get_default_models()[0]
                        verdict, score, err = ark_audit_structure(model, summary)
                        if err:
                            pass
                        else:
                            generated_pdbs[-1]["audit_verdict"] = verdict
                            generated_pdbs[-1]["audit_score"] = score
                            generated_pdbs[-1]["summary"] = summary
                    except Exception:
                        pass
        else:
            print("No valid SS candidates passed Qwen filter. Using fallback SS to build one model.")
            L = len(sequence)
            pattern = ("HHHHH" + "CC" + "EEEEE" + "C")
            s = (pattern * ((L // len(pattern)) + 1))[:L]
            suffix = "fallback_model"
            pdb_name = f"{prefix}_{suffix}.pdb"
            pdb_path = os.path.join(three_d_dir, pdb_name)
            
            success = False
            if getattr(args, "esm_backbone", False):
                print(f"  > Fallback: Generating backbone via ESM model (backend={args.backend})...")
                ss_path = os.path.join(workdir, f"{prefix}_fallback_ss.txt")
                with open(ss_path, "w", encoding="utf-8") as f:
                    f.write(s)
                npz_path = os.path.join(workdir, f"{prefix}_fallback_esm.npz")
                try:
                    info = predict_structure_with_esm(
                        fasta_path=args.input,
                        ss_path=ss_path,
                        output_pdb_path=pdb_path,
                        model_path=None,
                        npz_output_path=npz_path,
                        backend=args.backend,
                    )
                    backend_used = info.get("backend")
                    device_used = info.get("device")
                    if backend_used or device_used:
                        print(f"    ESM backend={backend_used} device={device_used}")
                    success = os.path.exists(pdb_path)
                except Exception as e:
                    print(f"    ESM fallback backbone generation failed: {e}")
            elif args.igpu:
                if args.igpu_env:
                    # Use process isolation for iGPU env
                    print(f"  > Fallback: Optimizing backbone (iGPU via external env '{args.igpu_env}')...")
                    
                    igpu_input_data = {"sequence": sequence, "chains": [s]}
                    tmp_input = os.path.join(workdir, f"igpu_input_fallback.json")
                    with open(tmp_input, "w", encoding="utf-8") as f:
                        json.dump(igpu_input_data, f)
                    
                    runner_script = os.path.join(os.getcwd(), "modules", "igpu_runner.py")
                    
                    if os.sep in args.igpu_env or "/" in args.igpu_env:
                        cmd_list = [args.igpu_env, runner_script, "--input", tmp_input, "--output", pdb_path]
                        if args.backend and args.backend != "auto":
                            cmd_list.extend(["--backend", args.backend])
                        cmd_str = subprocess.list2cmdline(cmd_list)
                    else:
                        if sys.platform == "win32":
                            backend_arg = f' --backend {args.backend}' if args.backend and args.backend != "auto" else ''
                            cmd_str = f'cmd /c conda run -n {args.igpu_env} python "{runner_script}" --input "{tmp_input}" --output "{pdb_path}"{backend_arg}'
                            cmd_list = cmd_str
                        else:
                            cmd_list = ["conda", "run", "-n", args.igpu_env, "python", runner_script, "--input", tmp_input, "--output", pdb_path]
                            if args.backend and args.backend != "auto":
                                cmd_list.extend(["--backend", args.backend])
                            cmd_str = " ".join(cmd_list)

                    try:
                        use_shell = (sys.platform == "win32")
                        result = subprocess.run(
                            cmd_str if sys.platform == "win32" else cmd_list,
                            capture_output=True,
                            text=True,
                            encoding="utf-8",
                            shell=use_shell
                        )
                        
                        if result.stdout: print(f"[iGPU Output] {result.stdout.strip()}")
                        if result.stderr: print(f"[iGPU Error] {result.stderr.strip()}")
                        
                        if result.returncode == 0:
                            success = True
                        else:
                            print(f"    iGPU External Failed (RC={result.returncode})")
                    except Exception as e:
                        print(f"    Execution Error: {e}")
                    
                    if os.path.exists(tmp_input):
                        try: os.remove(tmp_input) 
                        except: pass
                else:
                    from modules.igpu_predictor import run_backbone_fold_multichain as run_igpu_fold
                    success = run_igpu_fold(sequence, [s], pdb_path, backend=args.backend)
            else:
                success = run_backbone_fold_multichain(sequence, [s], pdb_path)
                
                if success:
                    if args.npu:
                        try:
                            runner_script_npu = os.path.join(os.getcwd(), "modules", "npu_runner.py")
                            if args.npu_env:
                                if os.sep in args.npu_env or "/" in args.npu_env:
                                    cmd_list_npu = [args.npu_env, runner_script_npu, "--input", pdb_path, "--output", pdb_path]
                                    cmd_str_npu = subprocess.list2cmdline(cmd_list_npu)
                                else:
                                    if sys.platform == "win32":
                                        cmd_str_npu = f'cmd /c conda run -n {args.npu_env} python "{runner_script_npu}" --input "{pdb_path}" --output "{pdb_path}"'
                                        cmd_list_npu = cmd_str_npu
                                    else:
                                        cmd_list_npu = ["conda", "run", "-n", args.npu_env, "python", runner_script_npu, "--input", pdb_path, "--output", pdb_path]
                                        cmd_str_npu = " ".join(cmd_list_npu)
                                use_shell_npu = (sys.platform == "win32")
                                env_vars_npu = os.environ.copy()
                                env_vars_npu["PYTHONIOENCODING"] = "utf-8"
                                env_vars_npu["PYTHONUTF8"] = "1"
                                res_npu = subprocess.run(
                                    cmd_str_npu if sys.platform == "win32" else cmd_list_npu,
                                    capture_output=True,
                                    text=True,
                                    encoding="utf-8",
                                    shell=use_shell_npu,
                                    env=env_vars_npu
                                )
                                if res_npu.stdout:
                                    print(f"[NPU Output] {res_npu.stdout.strip()}")
                                if res_npu.stderr:
                                    print(f"[NPU Error] {res_npu.stderr.strip()}")
                            else:
                                try:
                                    import modules.npu_runner as _nr
                                    _nr.run_inplace(pdb_path)
                                except Exception as e:
                                    print(f"[NPU Inline Error] {e}")
                        except Exception as e:
                            print(f"[NPU Runner Error] {e}")
                # Refinement Step (Automatic Assembly & Sidechain Packing)
                # For fallback model, it might be single chain, but still useful to refine sidechains.
                print(f"  > Refinement: Optimizing sidechains and assembly for {pdb_name}...")
                try:
                    remark_lines = []
                    try:
                        with open(pdb_path, "r", encoding="utf-8", errors="replace") as f:
                            for line in f:
                                if line.startswith("REMARK"):
                                    remark_lines.append(line.rstrip("\r\n"))
                                elif line.startswith("ATOM") or line.startswith("HETATM"):
                                    break
                    except Exception:
                        remark_lines = []
                    chains_data = parse_pdb_chains(pdb_path)
                    if chains_data:
                        if len(chains_data) > 1:
                            assembled_chains = assemble_chains(chains_data)
                        else:
                            assembled_chains = chains_data
                        
                        refined_pdb_name = f"{prefix}_{suffix}_refined.pdb"
                        refined_pdb_path = os.path.join(three_d_dir, refined_pdb_name)
                        
                        if write_complex_pdb(assembled_chains, sequence, refined_pdb_path, remark_lines=remark_lines):
                            print(f"  > Refinement complete. Saved to {refined_pdb_name}")
                            # Update pdb_path and pdb_name to point to the refined model
                            pdb_path = refined_pdb_path
                            pdb_name = refined_pdb_name
                            try:
                                run_refinements(
                                    pdb_path,
                                    sequence,
                                    do_ramachandran=args.refine_ramachandran,
                                    do_hydrophobic=args.refine_hydrophobic,
                                    repack_long=args.repack_long_sides,
                                    md_engine=args.md,
                                    md_steps=args.md_steps
                                )
                                core_info = analyze_ubiquitin_core(pdb_path, sequence)
                                if core_info and core_info.get("pair"):
                                    d = core_info["pair"]["distance"]
                                    print(f"  > Ubiquitin Ile44–Val70 distance: {d:.2f} Å")
                            except Exception as _e:
                                print(f"  > Post-refinement step skipped: {_e}")
                        else:
                            print("  > Refinement failed to write. Using raw backbone.")
                    else:
                        print("  > Refinement skipped (no chains parsed).")
                except Exception as e:
                    print(f"  > Refinement error: {e}")

                html_name = os.path.splitext(pdb_name)[0] + ".html"
                html_path = os.path.join(three_d_dir, html_name)
                generate_html_view(pdb_path, html_path)
                generated_pdbs.append({
                    "pdb": pdb_name,
                    "html": html_name,
                    "chains": 1,
                    "case": 0,
                    "prob": 0.0
                })
                try:
                    ss_str = s
                    summary = summarize_structure(pdb_path, sequence, ss_str)
                    model = get_default_models()[0]
                    verdict, score, err = ark_audit_structure(model, summary)
                    if not err:
                        generated_pdbs[-1]["audit_verdict"] = verdict
                        generated_pdbs[-1]["audit_score"] = score
                        generated_pdbs[-1]["summary"] = summary
                except Exception:
                    pass
        
        # 5. Generate Report
        report_file = os.path.join(workdir, f"{prefix}_report.md")
        with open(report_file, "w", encoding='utf-8') as f:
            f.write(f"# MiniFold Analysis Report: {seq_id}\n\n")
            f.write("## Sequence\n")
            f.write("```\n" + sequence + "\n```\n\n")
            f.write("## Secondary Structure Candidates (PyBioMed)\n")
            f.write(f"Saved: `{os.path.basename(cand_file)}`\n")
            f.write(f"Raw: `{os.path.basename(raw_file)}`\n\n")
            f.write("## Filtered by Qwen (Cases)\n")
            f.write(f"Saved: `{os.path.basename(kept_file)}`\n\n")
            
            f.write("## Functional Annotation (DeepSeek V3.2)\n")
            f.write(annotation + "\n\n")
            
            f.write("## 3D Models (Backbone Predictor, Multichain)\n")
            if generated_pdbs:
                for m in generated_pdbs:
                    f.write(f"### Case {m['case']} (Likelihood: {m['prob']:.2f})\n")
                    f.write(f"- Chains: {m['chains']}\n")
                    f.write(f"- PDB: `{m['pdb']}`\n")
                    f.write(f"- View: [Interactive 3D]({m['html']})\n\n")
                    if "summary" in m:
                        s = m["summary"]
                        f.write(f"- SS-conf: {s.get('ss_conf', 0.0):.2f}\n")
                        f.write(f"- core-stability: {s.get('core_stability', 0.0):.2f}\n")
                        f.write(f"- loop-uncertainty: {s.get('loop_uncertainty', 0.0):.2f}\n")
                        f.write(f"- helix-lengths: {','.join(str(x) for x in s.get('helix_lengths', []))}\n")
                        f.write(f"- strand-lengths: {','.join(str(x) for x in s.get('strand_lengths', []))}\n")
                        f.write(f"- loop-lengths: {','.join(str(x) for x in s.get('loop_lengths', []))}\n")
                    if "audit_verdict" in m:
                        f.write(f"- Auditor: {m['audit_verdict']} ({m.get('audit_score', 0.0):.2f})\n\n")
            else:
                f.write("No models generated (PyRosetta missing or no candidates).\n")

        
        print(f"Report saved to {report_file}")
        log_path = os.path.join(workdir, "process_report.log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"sequences={len(sequences)}\n")
            f.write(f"cases={len(cases)}\n")
            f.write(f"cases_kept={len(kept_cases)}\n")
            f.write(f"pdb_generated={len(generated_pdbs)}\n")
            f.write(f"qwen_attempts={q_result.get('attempts', 0)}\n")
            f.write(f"qwen_raw_lines={q_result.get('lines', 0)}\n")
        print("-" * 50)

        # Save manifest
        manifest_file = os.path.join(workdir, f"{prefix}_results.json")
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(generated_pdbs, f, ensure_ascii=False, indent=2)
            
        print(f"Workflow completed for {seq_id}. Results saved to {workdir}")
        print_progress(100, f"Completed processing {seq_id}")

    print("\nAll sequences processed.")
    print_progress(100, "All tasks completed")

if __name__ == "__main__":
    main()
