import numpy as np
import math
from scipy.optimize import minimize
import os

def parse_pdb_chains(pdb_path):
    """
    Parses a PDB file and returns a list of chains.
    Each chain is a dict: {'N': np.array, 'CA': np.array, 'C': np.array}
    Assumes standard MiniFold PDB format with N, CA, C atoms.
    """
    chains = []
    current_chain = {'N': [], 'CA': [], 'C': []}
    
    # MiniFold uses TER or chain ID change to separate? 
    # Current write_pdb uses "A{rid}" where rid increments. 
    # But chain breaks are marked by TER.
    
    try:
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith("TER"):
                    if current_chain['CA']:
                        chains.append(_convert_chain_to_numpy(current_chain))
                        current_chain = {'N': [], 'CA': [], 'C': []}
                    continue
                
                if line.startswith("ATOM"):
                    atom_name = line[12:16].strip()
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords = np.array([x, y, z])
                    
                    if atom_name == "N":
                        current_chain['N'].append(coords)
                    elif atom_name == "CA":
                        current_chain['CA'].append(coords)
                    elif atom_name == "C":
                        current_chain['C'].append(coords)
            
            # Add last chain if no trailing TER
            if current_chain['CA']:
                chains.append(_convert_chain_to_numpy(current_chain))
                
    except Exception as e:
        print(f"Error parsing PDB {pdb_path}: {e}")
        return []

    return chains

def _convert_chain_to_numpy(chain_dict):
    return {
        'N': np.array(chain_dict['N']),
        'CA': np.array(chain_dict['CA']),
        'C': np.array(chain_dict['C'])
    }

def _transform_coords(coords, rotation_matrix, translation):
    return (coords @ rotation_matrix.T) + translation

def _euler_to_matrix(alpha, beta, gamma):
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rz = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
    
    return Rz @ Ry @ Rx

def _get_chain_coords_array(chain):
    # Stack N, CA, C for bulk operations
    # Shape: (3 * NumRes, 3)
    return np.vstack([chain['N'], chain['CA'], chain['C']])

def assemble_chains(chains, verbose=False):
    """
    Assembles chains into a compact complex.
    Strategy: Fix first chain, optimize Rigid Body Transform (6 DOF) for others.
    Objective: Minimize Radius of Gyration + Inter-chain Clashes.
    """
    if not chains or len(chains) < 2:
        return chains

    # Center all chains to their own centroids first to make rotation easier
    centered_chains = []
    for c in chains:
        # Calculate centroid of CA atoms
        centroid = np.mean(c['CA'], axis=0)
        new_c = {
            'N': c['N'] - centroid,
            'CA': c['CA'] - centroid,
            'C': c['C'] - centroid
        }
        centered_chains.append(new_c)

    # Optimization variables: 6 per chain (except first)
    # [tx, ty, tz, alpha, beta, gamma]
    num_moving = len(chains) - 1
    x0 = np.zeros(6 * num_moving)
    
    # Randomize initial positions slightly to avoid locking
    x0 = np.random.uniform(-5.0, 5.0, size=x0.shape) 
    # Angles
    x0[3::6] = np.random.uniform(-3.14, 3.14, size=num_moving)
    x0[4::6] = np.random.uniform(-3.14, 3.14, size=num_moving)
    x0[5::6] = np.random.uniform(-3.14, 3.14, size=num_moving)

    def _get_transformed_chains(x):
        transformed = [centered_chains[0]] # First chain fixed at origin
        for i in range(num_moving):
            idx = i * 6
            trans = x[idx:idx+3]
            angs = x[idx+3:idx+6]
            R = _euler_to_matrix(*angs)
            
            orig = centered_chains[i+1]
            new_c = {
                'N': _transform_coords(orig['N'], R, trans),
                'CA': _transform_coords(orig['CA'], R, trans),
                'C': _transform_coords(orig['C'], R, trans)
            }
            transformed.append(new_c)
        return transformed

    def objective(x):
        current_chains = _get_transformed_chains(x)
        
        # 1. Radius of Gyration (Compactness)
        all_ca = np.vstack([c['CA'] for c in current_chains])
        center = np.mean(all_ca, axis=0)
        dists = np.linalg.norm(all_ca - center, axis=1)
        rg_score = np.mean(dists)
        
        # 2. Inter-chain Clash
        # Simple pairwise distance check between CA atoms of different chains
        # This is O(N^2), might be slow for large proteins, but okay for MiniFold
        clash_score = 0.0
        
        # Collect CA arrays
        ca_arrays = [c['CA'] for c in current_chains]
        
        for i in range(len(ca_arrays)):
            for j in range(i + 1, len(ca_arrays)):
                # Pairwise distances
                # Expand dims for broadcasting
                # A: (N, 3) -> (N, 1, 3)
                # B: (M, 3) -> (1, M, 3)
                A = ca_arrays[i]
                B = ca_arrays[j]
                
                # Heuristic: only check if bounding boxes overlap? No, just brute force for now.
                # To speed up, maybe just sample?
                
                diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]
                D = np.linalg.norm(diff, axis=2)
                
                # Soft clash penalty for dist < 4.0 A
                mask = D < 4.0
                if np.any(mask):
                    clash_score += np.sum((4.0 - D[mask]) ** 2)

        # Weights
        # We want compactness, but absolutely NO clashes.
        return rg_score + 10.0 * clash_score

    if verbose: print("Optimizing complex assembly...")
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'disp': verbose})
        final_chains = _get_transformed_chains(res.x)
    except Exception as e:
        print(f"Assembly optimization failed: {e}")
        final_chains = centered_chains # Fallback

    return final_chains

from modules.sidechain_builder import pack_sidechain

def write_complex_pdb(chains, original_sequence, out_path):
    """
    Writes the assembled chains to a PDB file with full atom details.
    """
    # Reconstruct flattened arrays
    all_N = np.concatenate([c['N'] for c in chains])
    all_CA = np.concatenate([c['CA'] for c in chains])
    all_C = np.concatenate([c['C'] for c in chains])
    
    # Environment for sidechain packing (Backbone atoms of the entire complex)
    env_atoms = np.vstack([all_N, all_CA, all_C])
    
    # We need to know where chains break to insert TER
    lengths = [len(c['CA']) for c in chains]
    
    # Three letter codes
    three_letter = {
        "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
        "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
        "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
        "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"
    }
    
    lines = []
    serial = 1
    rid = 1
    
    seq_ptr = 0
    
    # Import math for O placement if needed, or rely on numpy
    import math
    
    for chain_idx, length in enumerate(lengths):
        chain_seq = original_sequence[seq_ptr : seq_ptr + length]
        
        c_N = chains[chain_idx]['N']
        c_CA = chains[chain_idx]['CA']
        c_C = chains[chain_idx]['C']
        
        chain_id = chr(ord('A') + (chain_idx % 26))
        
        for i in range(length):
            aa = chain_seq[i]
            resn = three_letter.get(aa, "UNK")
            
            # Backbone
            lines.append(f"ATOM  {serial:5d}  N   {resn:>3s} {chain_id}{rid:4d}    {c_N[i][0]:8.3f}{c_N[i][1]:8.3f}{c_N[i][2]:8.3f}  1.00  0.00           N")
            serial += 1
            lines.append(f"ATOM  {serial:5d}  CA  {resn:>3s} {chain_id}{rid:4d}    {c_CA[i][0]:8.3f}{c_CA[i][1]:8.3f}{c_CA[i][2]:8.3f}  1.00  0.00           C")
            serial += 1
            lines.append(f"ATOM  {serial:5d}  C   {resn:>3s} {chain_id}{rid:4d}    {c_C[i][0]:8.3f}{c_C[i][1]:8.3f}{c_C[i][2]:8.3f}  1.00  0.00           C")
            serial += 1
            
            # Sidechain + Oxygen
            # Build Sidechain
            if aa != "G":
                sc_atoms = pack_sidechain(aa, c_N[i], c_CA[i], c_C[i], env_atoms)
                
                # Approximate O (Backbone Carbonyl)
                try:
                    # Simple geometric construction for O
                    # O = C + 1.23 * (rotated vector)
                    v = c_C[i] - c_CA[i]
                    v /= np.linalg.norm(v)
                    w = c_N[i] - c_CA[i]
                    w /= np.linalg.norm(w)
                    n = np.cross(v, w)
                    n /= np.linalg.norm(n)
                    
                    # Rotate v around n by 120 degrees (2.1 rad)
                    theta = 2.1
                    v_rot = v * math.cos(theta) + np.cross(n, v) * math.sin(theta) + n * np.dot(n, v) * (1 - math.cos(theta))
                    
                    O_pos = c_C[i] + 1.23 * v_rot
                    lines.append(f"ATOM  {serial:5d}  O   {resn:>3s} {chain_id}{rid:4d}    {O_pos[0]:8.3f}{O_pos[1]:8.3f}{O_pos[2]:8.3f}  1.00  0.00           O")
                    serial += 1
                except:
                    pass

                # Sort and write sidechain atoms
                order = ["CB", "CG", "CG1", "CG2", "OG", "OG1", "SG", 
                         "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD",
                         "CE", "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2",
                         "CZ", "CZ2", "CZ3", "NZ", 
                         "CH2", "NH1", "NH2", "OH"]
                         
                sorted_keys = sorted(sc_atoms.keys(), key=lambda x: order.index(x) if x in order else 99)
                
                for atom_name in sorted_keys:
                    pos = sc_atoms[atom_name]
                    element = atom_name[0]
                    lines.append(f"ATOM  {serial:5d}  {atom_name:<4s}{resn:>3s} {chain_id}{rid:4d}    {pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00  0.00           {element}")
                    serial += 1
            else:
                # Glycine O
                try:
                    v = c_C[i] - c_CA[i]
                    v /= np.linalg.norm(v)
                    w = c_N[i] - c_CA[i]
                    w /= np.linalg.norm(w)
                    n = np.cross(v, w)
                    n /= np.linalg.norm(n)
                    theta = 2.1
                    v_rot = v * math.cos(theta) + np.cross(n, v) * math.sin(theta) + n * np.dot(n, v) * (1 - math.cos(theta))
                    O_pos = c_C[i] + 1.23 * v_rot
                    lines.append(f"ATOM  {serial:5d}  O   {resn:>3s} {chain_id}{rid:4d}    {O_pos[0]:8.3f}{O_pos[1]:8.3f}{O_pos[2]:8.3f}  1.00  0.00           O")
                    serial += 1
                except:
                    pass

            rid += 1
            
        lines.append("TER")
        seq_ptr += length

    lines.append("END")
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        
    return True
