import math
import os

import numpy as np
from scipy.optimize import minimize

three_letter = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"
}

# Precompute bond parameters
BOND_PARAMS = {
    "n_len": 1.329,
    "ca_len": 1.458,
    "c_len": 1.525,
    "ang_C_N_CA": math.radians(121.7),
    "ang_N_CA_C": math.radians(111.2),
    "ang_CA_C_N": math.radians(116.2),
}
BOND_PARAMS["cos_N_CA_C"] = math.cos(BOND_PARAMS["ang_N_CA_C"])
BOND_PARAMS["sin_N_CA_C"] = math.sin(BOND_PARAMS["ang_N_CA_C"])

def ss_to_phi_psi(ss):
    phi = []
    psi = []
    for c in ss:
        if c == "H":
            phi.append(-62.0)
            psi.append(-41.0)
        elif c == "E":
            phi.append(-135.0)
            psi.append(135.0)
        else:
            phi.append(-60.0 + np.random.uniform(-15.0, 15.0))
            psi.append(140.0 + np.random.uniform(-15.0, 15.0))
    return np.deg2rad(np.array(phi)), np.deg2rad(np.array(psi))

def place_atom(a, b, c, bond_len, bond_angle, dihedral):
    ab = b - a
    bc = c - b
    ab_norm = np.linalg.norm(ab)
    bc_norm = np.linalg.norm(bc)
    
    # Robust fallback for zero-length vectors
    if ab_norm < 1e-9:
        ab = np.array([1.0, 0.0, 0.0])
        ab_norm = 1.0
    if bc_norm < 1e-9:
        bc = np.array([0.0, 1.0, 0.0])
        bc_norm = 1.0
        
    n1 = ab / ab_norm
    n2 = bc / bc_norm
    
    # Ensure inputs are numpy arrays for cross product
    n1 = np.asarray(n1)
    n2 = np.asarray(n2)
    
    # Use numpy.cross for efficiency and stability
    n = np.cross(n1, n2)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-9:
        # Handle collinear vectors: pick orthogonal direction
        if abs(n1[0]) > abs(n1[1]):
            n = np.array([-n1[1], n1[0], 0.0])
        else:
            n = np.array([0.0, -n1[2], n1[1]])
        n /= np.linalg.norm(n)
    else:
        n /= n_norm
        
    m = np.cross(n, n2)
    # m is already normalized as n and n2 are orthogonal and unit length
    
    # Use cached trig values if available? (Passed in args?)
    # For now, keep math calls or pass cached sin/cos if optimize further.
    # But since bond_angle/dihedral vary, only bond_angle parts could be cached.
    x = bond_len * math.cos(bond_angle)
    sin_ba = math.sin(bond_angle)
    y = bond_len * sin_ba * math.cos(dihedral)
    z = bond_len * sin_ba * math.sin(dihedral)
    
    return b + (-n2 * x) + (m * y) + (n * z)

def build_backbone(sequence, phi, psi, omega=None):
    if omega is None:
        omega = np.full_like(phi, math.pi)
    
    # Use cached parameters
    p = BOND_PARAMS
    N = [np.array([0.0, 0.0, 0.0])]
    CA = [np.array([p["ca_len"], 0.0, 0.0])]
    C = [np.array([p["ca_len"] + p["c_len"] * p["cos_N_CA_C"], p["c_len"] * p["sin_N_CA_C"], 0.0])]
    
    for i in range(1, len(sequence)):
        Ni = place_atom(C[i-1], N[i-1], CA[i-1], p["n_len"], p["ang_CA_C_N"], omega[i-1])
        CAi = place_atom(Ni, C[i-1], N[i-1], p["ca_len"], p["ang_C_N_CA"], phi[i])
        Ci = place_atom(CAi, Ni, C[i-1], p["c_len"], p["ang_N_CA_C"], psi[i])
        N.append(Ni)
        CA.append(CAi)
        C.append(Ci)
    return np.array(N), np.array(CA), np.array(C)

from modules.sidechain_builder import build_sidechain

def write_pdb(sequence, N, CA, C, out_path, chain_breaks=None):
    three_letter = {
        "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
        "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
        "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
        "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"
    }
    
    with open(out_path, "w", encoding="utf-8") as f:
        atom_idx = 1
        res_idx = 1
        chain_id = "A"
        chain_idx = 0
        
        # Determine chain intervals
        intervals = []
        start = 0
        if chain_breaks:
            for end in chain_breaks:
                intervals.append((start, end))
                start = end
        intervals.append((start, len(sequence)))
        
        current_interval_idx = 0
        next_break = intervals[0][1]
        
        for i, aa in enumerate(sequence):
            # Check for chain break
            if i >= next_break:
                f.write("TER\n")
                current_interval_idx += 1
                if current_interval_idx < len(intervals):
                    next_break = intervals[current_interval_idx][1]
                chain_idx += 1
                chain_id = chr(ord('A') + (chain_idx % 26))
                # Reset res_idx? Usually yes for new chain
                res_idx = 1
                
            resn = three_letter.get(aa, "UNK")
            
            # Backbone
            # N
            f.write(f"ATOM  {atom_idx:5d}  N   {resn:>3s} {chain_id}{res_idx:4d}    {N[i][0]:8.3f}{N[i][1]:8.3f}{N[i][2]:8.3f}  1.00  0.00           N\n")
            atom_idx += 1
            # CA
            f.write(f"ATOM  {atom_idx:5d}  CA  {resn:>3s} {chain_id}{res_idx:4d}    {CA[i][0]:8.3f}{CA[i][1]:8.3f}{CA[i][2]:8.3f}  1.00  0.00           C\n")
            atom_idx += 1
            # C
            f.write(f"ATOM  {atom_idx:5d}  C   {resn:>3s} {chain_id}{res_idx:4d}    {C[i][0]:8.3f}{C[i][1]:8.3f}{C[i][2]:8.3f}  1.00  0.00           C\n")
            atom_idx += 1
            
            # Sidechain
            # Only if not GLY
            if aa != "G":
                # Build sidechain atoms
                sc_atoms = build_sidechain(aa, N[i], CA[i], C[i])
                
                # Order matters for PDB but visualizing tools are flexible.
                # Standard order: N, CA, C, O, CB...
                
                # We missed Oxygen (O) in backbone!
                # O is bonded to C.
                # Standard geometry: C=O bond length 1.23, bisects N-C-CA?
                # Actually, in planar peptide bond, O is in plane defined by CA(i), C(i), N(i+1).
                # But we don't have N(i+1) easily here if it's the last residue or chain break.
                # We can approximate O position: 
                # Vector C-O is opposite to C-CA roughly? No.
                # Angle CA-C-O ~ 120.8, N(+1)-C-O ~ 123.
                # It lies in the plane CA-C-N(+1).
                # Since psi determines C rotation, and we built C based on psi,
                # we can place O relative to CA-C frame.
                
                # Let's add O first (backbone carbonyl)
                # Need N, CA, C coordinates.
                # Use NeRF-like placement or geometric relative to C.
                # Vector u_c_ca = (CA - C).norm
                # Vector u_c_n_next?
                # Without next residue, assume trans planar.
                # Place O in the plane of CA-C-N(previous) but rotated?
                # Actually, O is usually placed such that C=O and N-H are anti-parallel in sheets/helices.
                
                # Simple approximation:
                # O is in the plane of N-CA-C? No.
                # O is in the peptide plane.
                # We can construct O using CA, C and a virtual atom?
                
                # Let's compute O coordinate:
                # O = place_atom(CA[i], C[i], N[i], 1.23, 120 deg, 180 deg) ? 
                # This would place it trans to N relative to C-CA bond.
                try:
                    O_pos = place_atom(CA[i], C[i], N[i], 1.23, math.radians(120.8), math.pi)
                    f.write(f"ATOM  {atom_idx:5d}  O   {resn:>3s} {chain_id}{res_idx:4d}    {O_pos[0]:8.3f}{O_pos[1]:8.3f}{O_pos[2]:8.3f}  1.00  0.00           O\n")
                    atom_idx += 1
                except:
                    pass

                # Write Sidechain atoms
                # Sort keys to maintain some standard order (CB, CG, CD...)
                # Specific ordering: CB, CG, OD1, ND2...
                # Simple sort by name length then alphabetical?
                # PDB order is specific.
                order = ["CB", "CG", "CG1", "CG2", "OG", "OG1", "SG", 
                         "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD",
                         "CE", "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2",
                         "CZ", "CZ2", "CZ3", "NZ", 
                         "CH2", "NH1", "NH2", "OH"]
                         
                sorted_keys = sorted(sc_atoms.keys(), key=lambda x: order.index(x) if x in order else 99)
                
                for atom_name in sorted_keys:
                    pos = sc_atoms[atom_name]
                    # Element symbol
                    element = atom_name[0]
                    f.write(f"ATOM  {atom_idx:5d}  {atom_name:<4s}{resn:>3s} {chain_id}{res_idx:4d}    {pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00  0.00           {element}\n")
                    atom_idx += 1
            else:
                # GLY still has O
                try:
                    O_pos = place_atom(CA[i], C[i], N[i], 1.23, math.radians(120.8), math.pi)
                    f.write(f"ATOM  {atom_idx:5d}  O   {resn:>3s} {chain_id}{res_idx:4d}    {O_pos[0]:8.3f}{O_pos[1]:8.3f}{O_pos[2]:8.3f}  1.00  0.00           O\n")
                    atom_idx += 1
                except:
                    pass

            res_idx += 1
            
        f.write("TER\n")
        f.write("END\n")
    return True

def _objective(x, sequence, phi_ref, psi_ref):
    n = len(sequence)
    phi = x[:n]
    psi = x[n:]
    try:
        N, CA, C = build_backbone(sequence, phi, psi)
    except Exception:
        return 1e9
    adj = CA[1:] - CA[:-1]
    adj_len = np.linalg.norm(adj, axis=1)
    t1 = np.sum((adj_len - 3.80) ** 2)
    dmat = CA[:, None, :] - CA[None, :, :]
    dd = np.linalg.norm(dmat, axis=2)
    mask = (dd < 2.5) & (dd > 0)
    if np.any(mask):
        t2 = np.sum((2.5 - dd[mask]) ** 2)
    else:
        t2 = 0.0
    t3 = np.sum((phi - phi_ref) ** 2 + (psi - psi_ref) ** 2)
    # Adjusted weights: reduce collision penalty (t2: 10->5), increase angle penalty (t3: 0.01->0.1)
    return t1 + 5.0 * t2 + 0.1 * t3

def _gradient(x, sequence, phi_ref, psi_ref):
    # Numerical gradient approximation (centered difference) for L-BFGS-B
    # This is faster than 2-sided approximation inside minimize if we control epsilon
    eps = 1e-5
    n = len(x)
    grad = np.zeros(n)
    # Baseline
    f0 = _objective(x, sequence, phi_ref, psi_ref)
    
    # We can't easily vectorize _objective due to build_backbone loop
    # But for 122 dims, this loop is the bottleneck.
    # To speed up, we only compute grad for phi/psi that actually changed?
    # No, all coords change if one angle changes.
    # So we must loop.
    
    # NOTE: Since calculating full gradient numerically is expensive (2N calls),
    # and we don't have analytic gradient yet, we rely on L-BFGS-B's internal approximation.
    # However, to avoid "hanging", we explicitly use a larger epsilon if we were to implement this.
    # But actually, providing NO jacobian to minimize() causes it to use 2-point approximation.
    # The bottleneck is build_backbone speed.
    # So we skip implementing _gradient here and rely on the speedups in build_backbone
    # and reduced iterations.
    return None

def _optimize_single_chain_coords(sequence, ss_str, iters=2):
    seq_len = len(sequence)
    if not ss_str or len(ss_str) != seq_len:
        return None
        
    # Generate initial reference angles from SS
    phi0, psi0 = ss_to_phi_psi(ss_str)
    
    n = len(sequence)
    best = None
    best_val = None
    
    # Try multiple initializations
    for _ in range(iters):
        # Add random noise to initial guess
        x0 = np.concatenate([
            phi0 + np.random.uniform(-0.3, 0.3, n), 
            psi0 + np.random.uniform(-0.3, 0.3, n)
        ])
        
        # Optimize
        try:
            res = minimize(_objective, x0, args=(sequence, phi0, psi0), 
                          method="L-BFGS-B", 
                          options={'maxiter': 30, 'ftol': 1e-3, 'disp': False, 'maxls': 20},
                          bounds=[(-np.pi, np.pi)] * (2 * n))
            
            val = res.fun
            if best is None or val < best_val:
                best = res.x
                best_val = val
        except Exception as e:
            if best is None:
                best = x0
                best_val = 1e9
            
    # Rebuild final structure
    phi_final = best[:n]
    psi_final = best[n:]
    N, CA, C = build_backbone(sequence, phi_final, psi_final)
    return N, CA, C

def optimize_from_ss(sequence, chain_ss_list, output_pdb, iters=2):
    seq_len = len(sequence)
    if not chain_ss_list:
        return False
    lengths = [len(s) for s in chain_ss_list]
    if sum(lengths) != seq_len:
        return False
    
    all_N = []
    all_CA = []
    all_C = []
    
    start_idx = 0
    for i, ss in enumerate(chain_ss_list):
        L = len(ss)
        sub_seq = sequence[start_idx : start_idx + L]
        start_idx += L
        
        coords = _optimize_single_chain_coords(sub_seq, ss, iters=iters)
        if coords is None:
            return False
            
        N, CA, C = coords
        
        # Offset subsequent chains to avoid visual overlap (simulate separate molecules)
        # Shift along X axis by 30A * chain_index
        offset = np.array([30.0 * i, 0.0, 0.0])
        all_N.append(N + offset)
        all_CA.append(CA + offset)
        all_C.append(C + offset)
    
    # Concatenate coordinates
    final_N = np.concatenate(all_N)
    final_CA = np.concatenate(all_CA)
    final_C = np.concatenate(all_C)
        
    # Calculate chain breaks (cumulative indices)
    breaks = []
    acc = 0
    for L in lengths[:-1]:
        acc += L
        breaks.append(acc)
        
    return write_pdb(sequence, final_N, final_CA, final_C, output_pdb, chain_breaks=breaks)

def run_backbone_fold_multichain(sequence, chain_ss_list, output_pdb, chain_slices=None):
    # Backward compatibility wrapper: now calls optimization
    return optimize_from_ss(sequence, chain_ss_list, output_pdb)

