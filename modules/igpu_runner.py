import argparse
import json
import sys
import os
import traceback
import torch

# Add the project root to sys.path so we can import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging

# Configure logging to stdout so pipeline can capture it
logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)

import subprocess
from modules.env_loader import load_env

def main():
    parser = argparse.ArgumentParser(description="Run iGPU optimization in a separate environment")
    parser.add_argument("--input", required=True, help="Path to input JSON file containing sequence and chains")
    parser.add_argument("--output", required=True, help="Path to output PDB file")
    parser.add_argument("--backend", default="auto", choices=["auto", "ipex", "directml", "cuda", "cpu", "oneapi_cpu"], help="Acceleration backend")
    
    args = parser.parse_args()
    
    try:
        def _bootstrap_oneapi_env():
            try:
                load_env(project_root)
            except Exception:
                pass
            if not os.environ.get("ONEAPI_SETVARS"):
                try:
                    load_env(os.path.join(project_root, ".env.oneapi"))
                except Exception:
                    pass
            vs = os.environ.get("VS2022INSTALLDIR")
            setvars = os.environ.get("ONEAPI_SETVARS")
            if not setvars or not os.path.exists(setvars):
                return
            parts = []
            if vs:
                parts.append(f'set "VS2022INSTALLDIR={vs}"')
            args_val = os.environ.get("ONEAPI_SETVARS_ARGS", "--force")
            print(f"[iGPU Runner] Bootstrapping oneAPI: {setvars} {args_val}")
            parts.append(f'call "{setvars}" {args_val}')
            parts.append('set')
            cmd = 'cmd /c ' + ' && '.join(parts)
            try:
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode == 0 and r.stdout:
                    for line in r.stdout.splitlines():
                        if "=" in line:
                            k, v = line.split("=", 1)
                            os.environ[k] = v
                    return
            except Exception:
                pass
        
        _bootstrap_oneapi_env()
        from modules.igpu_predictor import run_backbone_fold_multichain
        # Read input
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        sequence = data["sequence"]
        chains = data["chains"]
        constraints = data.get("constraints", None)
        
        print(f"[iGPU Runner] Processing sequence length {len(sequence)} with {len(chains)} chains...")
        print(f"[iGPU Runner] Requested backend: {args.backend}")
        if constraints:
            print(f"[iGPU Runner] Applying {len(constraints)} physical constraints.")
        
        # Run optimization
        # The run_backbone_fold_multichain function returns True on success
        success = run_backbone_fold_multichain(sequence, chains, args.output, constraints=constraints, backend=args.backend)
        
        if success:
            print(f"[iGPU Runner] Success. Output written to {args.output}")
            sys.exit(0)
        else:
            print("[iGPU Runner] Optimization returned False.")
            sys.exit(1)
            
    except Exception as e:
        print(f"[iGPU Runner] Error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
