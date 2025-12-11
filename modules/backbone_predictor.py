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

def write_pdb(sequence, N, CA, C, out_path, chain_breaks=None):
    lines = []
    resn = [three_letter.get(a, "UNK") for a in sequence]
    serial = 1
    rid = 1
    for i in range(len(sequence)):
        lines.append(
            f"ATOM  {serial:5d}  N   {resn[i]:>3s} A{rid:4d}    {N[i][0]:8.3f}{N[i][1]:8.3f}{N[i][2]:8.3f}  1.00  0.00           N"
        )
        serial += 1
        lines.append(
            f"ATOM  {serial:5d}  CA  {resn[i]:>3s} A{rid:4d}    {CA[i][0]:8.3f}{CA[i][1]:8.3f}{CA[i][2]:8.3f}  1.00  0.00           C"
        )
        serial += 1
        lines.append(
            f"ATOM  {serial:5d}  C   {resn[i]:>3s} A{rid:4d}    {C[i][0]:8.3f}{C[i][1]:8.3f}{C[i][2]:8.3f}  1.00  0.00           C"
        )
        serial += 1
        rid += 1
        if chain_breaks and (i+1) in chain_breaks:
            lines.append("TER")
    lines.append("END")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
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

def optimize_from_ss(sequence, chain_ss_list, output_pdb, iters=2):
    seq_len = len(sequence)
    if not chain_ss_list:
        return False
    lengths = [len(s) for s in chain_ss_list]
    if sum(lengths) != seq_len:
        return False
        
    # Generate initial reference angles from SS
    phi_list = []
    psi_list = []
    for s in chain_ss_list:
        p, q = ss_to_phi_psi(s)
        phi_list.append(p)
        psi_list.append(q)
    phi0 = np.concatenate(phi_list)
    psi0 = np.concatenate(psi_list)
    
    n = len(sequence)
    best = None
    best_val = None
    
    # Try multiple initializations
    for _ in range(iters):
        # Add random noise to initial guess (increased from 0.1 to 0.3)
        x0 = np.concatenate([
            phi0 + np.random.uniform(-0.3, 0.3, n), 
            psi0 + np.random.uniform(-0.3, 0.3, n)
        ])
        
        # Optimize
        # Use numerical approximation for gradient (jac=None)
        # Reduced iterations/precision for speed
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
            print(f"Optimization warning: {e}")
            if best is None:
                best = x0
                best_val = 1e9
            
    # Rebuild final structure
    phi_final = best[:n]
    psi_final = best[n:]
    N, CA, C = build_backbone(sequence, phi_final, psi_final)
    
    # Calculate chain breaks
    breaks = []
    acc = 0
    for L in lengths[:-1]:
        acc += L
        breaks.append(acc)
        
    return write_pdb(sequence, N, CA, C, output_pdb, chain_breaks=breaks)

def run_backbone_fold_multichain(sequence, chain_ss_list, output_pdb, chain_slices=None):
    # Backward compatibility wrapper: now calls optimization
    return optimize_from_ss(sequence, chain_ss_list, output_pdb)

