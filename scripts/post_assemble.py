import os
import sys
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.assembler import parse_pdb_chains, assemble_chains, write_complex_pdb
from modules.visualization import generate_html_view

def process_path(target_path):
    print(f"Processing target: {target_path}")
    
    if os.path.isfile(target_path):
        # Case 1: Single PDB file
        if not target_path.endswith(".pdb"):
            print(f"Target is not a PDB file: {target_path}")
            return
        
        # Check if it's already a complex
        if "_complex" in os.path.basename(target_path):
            print(f"Target is already a complex model: {target_path}")
            return
            
        print(f"Processing single file: {target_path}")
        process_single_pdb(target_path)
        
    elif os.path.isdir(target_path):
        # Case 2: Directory
        # Check if it's a '3d_structures' dir or a job dir containing it
        three_d_dir = target_path
        if os.path.exists(os.path.join(target_path, "3d_structures")):
            three_d_dir = os.path.join(target_path, "3d_structures")
            
        print(f"Scanning directory: {three_d_dir}")
        if not os.path.exists(three_d_dir):
            print("No valid directory found.")
            return

        # Find existing PDBs that are NOT complex models
        pdbs = [f for f in os.listdir(three_d_dir) if f.endswith(".pdb") and "_complex" not in f]
        
        count = 0
        for pdb_file in pdbs:
            pdb_path = os.path.join(three_d_dir, pdb_file)
            
            # Check if complex already exists
            base_name = os.path.splitext(pdb_file)[0]
            complex_name = f"{base_name}_complex.pdb"
            complex_path = os.path.join(three_d_dir, complex_name)
            
            if os.path.exists(complex_path):
                print(f"Skipping {pdb_file}, complex already exists.")
                continue
                
            if process_single_pdb(pdb_path):
                count += 1

        print(f"Done. Generated {count} complex models.")

def process_single_pdb(pdb_path):
    try:
        if not os.path.exists(pdb_path):
            print(f"Error: PDB file does not exist: {pdb_path}")
            return False
            
        three_d_dir = os.path.dirname(pdb_path)
        base_name = os.path.splitext(os.path.basename(pdb_path))[0]
        complex_name = f"{base_name}_complex.pdb"
        complex_path = os.path.join(three_d_dir, complex_name)
        
        chains = parse_pdb_chains(pdb_path)
        print(f"Parsed {len(chains)} chains from {os.path.basename(pdb_path)}")
        
        if len(chains) > 1:
            print(f"Assembling {os.path.basename(pdb_path)} ({len(chains)} chains)...")
            assembled = assemble_chains(chains)
            
            # Read sequence from PDB atoms
            full_seq = []
            three_to_one = {
                "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
                "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
                "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
                "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
                "UNK": "X"
            }
            
            with open(pdb_path, 'r') as f:
                for line in f:
                    if line.startswith("ATOM") and line[12:16].strip() == "CA":
                        resn = line[17:20].strip()
                        aa = three_to_one.get(resn, "X")
                        full_seq.append(aa)
                        
            sequence = "".join(full_seq)
            
            # Write
            write_complex_pdb(assembled, sequence, complex_path)
            
            # HTML
            html_name = f"{base_name}_complex.html"
            html_path = os.path.join(three_d_dir, html_name)
            generate_html_view(complex_path, html_path)
            
            print(f"Generated {html_name}")
            return True
        else:
            print(f"Skipping {os.path.basename(pdb_path)}, single chain.")
            return False
            
    except Exception as e:
        print(f"Failed to assemble {pdb_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Path to job directory or specific PDB file")
    args = parser.parse_args()
    
    process_path(args.dir)
