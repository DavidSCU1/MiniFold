import argparse
import os
import sys
import json

from modules.input_handler import load_fasta
from modules.ss_generator import pybiomed_ss_candidates
from modules.ark_module import ark_vote_cases, ark_analyze_sequence, get_default_models
from modules.backbone_predictor import run_backbone_fold_multichain
from modules.igpu_predictor import run_backbone_fold_multichain as run_igpu_fold
from modules.visualization import generate_html_view
from modules.env_loader import load_env
from modules.assembler import parse_pdb_chains, assemble_chains, write_complex_pdb

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
        q_result = pybiomed_ss_candidates(sequence, args.env, num=args.ssn)
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
        if kept_cases:
            sorted_cases = sorted(kept_cases, key=lambda x: x.get("p", 0.0), reverse=True)
            total_kept = len(sorted_cases)
            for rank, meta in enumerate(sorted_cases, start=1):
                # Update progress for 3D generation (65% -> 95%)
                p_val = 65 + int(((rank-1)/total_kept) * 30)
                print_progress(p_val, f"Generating 3D Structure {rank}/{total_kept}")
                
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
                
                success = False
                if args.igpu:
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
                            cmd_str = subprocess.list2cmdline(cmd_list)
                        else:
                            # Conda env name
                            if sys.platform == "win32":
                                cmd_str = f'cmd /c conda run -n {args.igpu_env} python "{runner_script}" --input "{tmp_input}" --output "{pdb_path}"'
                                cmd_list = cmd_str # For shell=True/False consideration, win usually needs string for cmd /c
                            else:
                                cmd_list = ["conda", "run", "-n", args.igpu_env, "python", runner_script, "--input", tmp_input, "--output", pdb_path]
                                cmd_str = " ".join(cmd_list)

                        try:
                            # Execute
                            import subprocess
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
                        print(f"  > Case {case_idx} (p={prob:.2f}): Optimizing backbone (iGPU)...")
                        success = run_igpu_fold(sequence, chains, pdb_path)
                else:
                    print(f"  > Case {case_idx} (p={prob:.2f}): Optimizing backbone (Standard)...")
                    success = run_backbone_fold_multichain(sequence, chains, pdb_path)
                
                if success:
                    # Refinement Step (Automatic Assembly & Sidechain Packing)
                    print(f"  > Refinement: Optimizing sidechains and assembly for {pdb_name}...")
                    try:
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
                            
                            if write_complex_pdb(assembled_chains, sequence, refined_pdb_path):
                                print(f"  > Refinement complete. Saved to {refined_pdb_name}")
                                # Use refined model for final output
                                pdb_path = refined_pdb_path
                                pdb_name = refined_pdb_name
                            else:
                                print("  > Refinement failed to write. Using raw backbone.")
                        else:
                            print("  > Refinement skipped (no chains parsed).")
                    except Exception as e:
                        print(f"  > Refinement error: {e}")

                    html_name = f"{prefix}_{suffix}.html"
                    html_path = os.path.join(three_d_dir, html_name)
                    generate_html_view(pdb_path, html_path)
                    generated_pdbs.append({
                        "pdb": pdb_name,
                        "html": html_name,
                        "chains": len(chains),
                        "case": case_idx,
                        "prob": prob
                    })
        else:
            print("No valid SS candidates passed Qwen filter. Using fallback SS to build one model.")
            L = len(sequence)
            pattern = ("HHHHH" + "CC" + "EEEEE" + "C")
            s = (pattern * ((L // len(pattern)) + 1))[:L]
            suffix = "fallback_model"
            pdb_name = f"{prefix}_{suffix}.pdb"
            # Fallback model often uses default pattern SS, which can be refined.
            # But here suffix is "fallback_model".
            # The logic below uses suffix for refined name: "{prefix}_{suffix}_refined.pdb"
            # Wait, pdb_name definition seems correct.
            pdb_path = os.path.join(three_d_dir, pdb_name)
            
            success = False
            if args.igpu:
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
                        cmd_str = subprocess.list2cmdline(cmd_list)
                    else:
                        if sys.platform == "win32":
                            cmd_str = f'cmd /c conda run -n {args.igpu_env} python "{runner_script}" --input "{tmp_input}" --output "{pdb_path}"'
                            cmd_list = cmd_str
                        else:
                            cmd_list = ["conda", "run", "-n", args.igpu_env, "python", runner_script, "--input", tmp_input, "--output", pdb_path]
                            cmd_str = " ".join(cmd_list)

                    try:
                        import subprocess
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
                    success = run_igpu_fold(sequence, [s], pdb_path)
            else:
                success = run_backbone_fold_multichain(sequence, [s], pdb_path)
                
            if success:
                # Refinement Step (Automatic Assembly & Sidechain Packing)
                # For fallback model, it might be single chain, but still useful to refine sidechains.
                print(f"  > Refinement: Optimizing sidechains and assembly for {pdb_name}...")
                try:
                    chains_data = parse_pdb_chains(pdb_path)
                    if chains_data:
                        if len(chains_data) > 1:
                            assembled_chains = assemble_chains(chains_data)
                        else:
                            assembled_chains = chains_data
                        
                        refined_pdb_name = f"{prefix}_{suffix}_refined.pdb"
                        refined_pdb_path = os.path.join(three_d_dir, refined_pdb_name)
                        
                        if write_complex_pdb(assembled_chains, sequence, refined_pdb_path):
                            print(f"  > Refinement complete. Saved to {refined_pdb_name}")
                            # Update pdb_path and pdb_name to point to the refined model
                            pdb_path = refined_pdb_path
                            pdb_name = refined_pdb_name
                        else:
                            print("  > Refinement failed to write. Using raw backbone.")
                    else:
                        print("  > Refinement skipped (no chains parsed).")
                except Exception as e:
                    print(f"  > Refinement error: {e}")

                html_name = f"{prefix}_{suffix}.html"
                html_path = os.path.join(three_d_dir, html_name)
                generate_html_view(pdb_path, html_path)
                generated_pdbs.append({
                    "pdb": pdb_name,
                    "html": html_name,
                    "chains": 1,
                    "case": 0,
                    "prob": 0.0
                })
        
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
