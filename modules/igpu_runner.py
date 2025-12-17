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

from modules.igpu_predictor import run_backbone_fold_multichain

def main():
    parser = argparse.ArgumentParser(description="Run iGPU optimization in a separate environment")
    parser.add_argument("--input", required=True, help="Path to input JSON file containing sequence and chains")
    parser.add_argument("--output", required=True, help="Path to output PDB file")
    
    args = parser.parse_args()
    
    try:
        # Read input
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        sequence = data["sequence"]
        chains = data["chains"]
        constraints = data.get("constraints", None)
        
        print(f"[iGPU Runner] Processing sequence length {len(sequence)} with {len(chains)} chains...")
        if constraints:
            print(f"[iGPU Runner] Applying {len(constraints)} physical constraints.")
        
        # Run optimization
        # The run_backbone_fold_multichain function returns True on success
        success = run_backbone_fold_multichain(sequence, chains, args.output, constraints=constraints)
        
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
