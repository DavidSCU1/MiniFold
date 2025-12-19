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

# Refined Bond Parameters (Engh & Huber, 1991)
BOND_PARAMS = {
    "n_len": 1.329,
    "ca_len": 1.46,
    "c_len": 1.52,
    "ang_C_N_CA": math.radians(121.7),
    "ang_N_CA_C": math.radians(111.0), # Tetrahedral sp3 is 109.5, but backbone is slightly strained
    "ang_CA_C_N": math.radians(116.2), # Planar sp2 at C is 120, but N-C-O > 120 usually
}
BOND_PARAMS["cos_N_CA_C"] = math.cos(BOND_PARAMS["ang_N_CA_C"])
BOND_PARAMS["sin_N_CA_C"] = math.sin(BOND_PARAMS["ang_N_CA_C"])

_BOND_CO_LEN = 1.229

def ss_to_phi_psi(ss, sequence=None):
    phi = []
    psi = []
    seq = sequence if sequence is not None else ("A" * len(ss))
    for i, c in enumerate(ss):
        aa = seq[i] if i < len(seq) else "A"
        if c == "H":
            p = -57.0
            s = -47.0
        elif c == "E":
            p = -119.0
            s = 113.0
        else:
            if aa == "G":
                p = -120.0 + np.random.uniform(-30.0, 30.0)
                s = 130.0 + np.random.uniform(-30.0, 30.0)
            else:
                r = np.random.rand()
                if r < 0.6 and aa != "P":
                    p = -60.0 + np.random.uniform(-20.0, 20.0)
                    s = -45.0 + np.random.uniform(-20.0, 20.0)
                else:
                    p = -120.0 + np.random.uniform(-30.0, 30.0)
                    s = 130.0 + np.random.uniform(-30.0, 30.0)

        if aa == "P":
            p = -65.0 + np.random.uniform(-10.0, 10.0)

        phi.append(math.radians(p))
        psi.append(math.radians(s))
    return np.array(phi), np.array(psi)

def _wrap_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

_RAMA_PREF = {
    "General": np.array([[-60.0, -45.0], [-120.0, 120.0], [-75.0, 145.0]], dtype=float) * (np.pi / 180.0),
    "GLY": np.array([[-60.0, -45.0], [-120.0, 120.0], [-75.0, 145.0], [60.0, 30.0]], dtype=float) * (np.pi / 180.0),
    "PRO": np.array([[-65.0, 150.0], [-75.0, 145.0]], dtype=float) * (np.pi / 180.0),
}
_RAMA_SIGMA = {
    "General": np.array([25.0, 25.0], dtype=float) * (np.pi / 180.0),
    "GLY": np.array([35.0, 35.0], dtype=float) * (np.pi / 180.0),
    "PRO": np.array([20.0, 20.0], dtype=float) * (np.pi / 180.0),
}

def _rama_loss(phi, psi, sequence):
    if len(phi) == 0:
        return 0.0
    loss = 0.0
    for i in range(len(phi)):
        aa = sequence[i] if i < len(sequence) else "A"
        grp = "General"
        if aa == "G":
            grp = "GLY"
        elif aa == "P":
            grp = "PRO"
        centers = _RAMA_PREF[grp]
        sigma = _RAMA_SIGMA[grp]
        dp = _wrap_pi(phi[i] - centers[:, 0]) / sigma[0]
        dq = _wrap_pi(psi[i] - centers[:, 1]) / sigma[1]
        loss += float(np.min(dp * dp + dq * dq))
    return loss

def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-9:
        return None
    return v / n

def carbonyl_oxygen_position(Ci, CAi, Nnext, bond_len=_BOND_CO_LEN):
    u1 = _normalize(CAi - Ci)
    u2 = _normalize(Nnext - Ci)
    if u1 is None or u2 is None:
        return None
    d = -(u1 + u2)
    d = _normalize(d)
    if d is None:
        return None
    return Ci + d * bond_len

def place_atom(a, b, c, bond_len, bond_angle, torsion):
    """
    NeRF: Places atom D such that:
    Distance C-D = bond_len
    Angle B-C-D = bond_angle
    Dihedral A-B-C-D = torsion
    """
    bc = c - b
    ab = b - a
    
    # Normalize
    bc_l = np.linalg.norm(bc)
    ab_l = np.linalg.norm(ab)
    
    if bc_l < 1e-6 or ab_l < 1e-6:
        # Fallback for degenerate cases (start of chain)
        return c + np.array([bond_len, 0, 0])

    bc_u = bc / bc_l
    
    # Normal to plane ABC
    n = np.cross(ab, bc_u)
    n_l = np.linalg.norm(n)
    if n_l < 1e-6:
        # Collinear case: arbitrary normal
        n = np.array([1.0, 0.0, 0.0]) if abs(bc_u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    else:
        n = n / n_l
        
    # Binormal vector (in plane ABC, perp to BC)
    nb = np.cross(n, bc_u)
    
    # D relative to C in local frame (bc, nb, n)
    # x component along extension of BC (which is -bc_u if bond angle 180)
    # Standard NeRF derivation:
    # D = C + R * D_local
    # D_local = [ -len * cos(angle), len * sin(angle) * cos(torsion), len * sin(angle) * sin(torsion) ]
    # But basis vectors: 
    # v1 = bc_u
    # v2 = nb (in plane)
    # v3 = n (perp)
    
    # Careful with angle definition. bond_angle is B-C-D.
    # If B-C-D is 180, D is along BC extension.
    # Formula usually assumes exterior angle or interior.
    # Let's use standard NeRF formula:
    # D = C + len * ( -bc_u * cos(angle) + nb * sin(angle) * cos(torsion) + n * sin(angle) * sin(torsion) )
    # This assumes torsion=0 is cis (planar with A-B-C)?
    # Standard: torsion is angle between plane ABC and BCD.
    
    x = -bond_len * math.cos(bond_angle)
    y = bond_len * math.sin(bond_angle) * math.cos(torsion)
    z = bond_len * math.sin(bond_angle) * math.sin(torsion)
    
    d = c + (bc_u * x) + (nb * y) + (n * z)
    return d

def build_backbone(sequence, phi, psi, omega=None):
    if omega is None:
        omega = np.full(len(phi), math.pi) # Trans peptide bond
    
    # Initial atoms: N at origin, CA on x-axis, C in xy-plane
    p = BOND_PARAMS
    
    # Initialize list of coordinates
    # N(0)
    n0 = np.array([0.0, 0.0, 0.0])
    
    # CA(0): dist(N-CA), arbitrary angle
    ca0 = np.array([p["ca_len"], 0.0, 0.0])
    
    # C(0): dist(CA-C), angle(N-CA-C)
    # torsion? Arbitrary, put in xy plane.
    # Use place_atom with virtual previous atom (-1,0,0)
    c0 = place_atom(np.array([-1.0, 0.0, 0.0]), n0, ca0, p["c_len"], p["ang_N_CA_C"], 0.0)
    
    N = [n0]
    CA = [ca0]
    C = [c0]
    
    for i in range(1, len(sequence)):
        # 1. Place N(i)
        # Bond: C(i-1)-N(i)
        # Angle: CA(i-1)-C(i-1)-N(i)
        # Torsion: N(i-1)-CA(i-1)-C(i-1)-N(i) = psi(i-1)
        prev_psi = psi[i-1]
        ni = place_atom(N[i-1], CA[i-1], C[i-1], p["n_len"], p["ang_CA_C_N"], prev_psi)
        
        # 2. Place CA(i)
        # Bond: N(i)-CA(i)
        # Angle: C(i-1)-N(i)-CA(i)
        # Torsion: CA(i-1)-C(i-1)-N(i)-CA(i) = omega(i-1) (peptide bond, ~180)
        prev_omega = omega[i-1]
        cai = place_atom(CA[i-1], C[i-1], ni, p["ca_len"], p["ang_C_N_CA"], prev_omega)
        
        # 3. Place C(i)
        # Bond: CA(i)-C(i)
        # Angle: N(i)-CA(i)-C(i)
        # Torsion: C(i-1)-N(i)-CA(i)-C(i) = phi(i)
        curr_phi = phi[i]
        ci = place_atom(C[i-1], ni, cai, p["c_len"], p["ang_N_CA_C"], curr_phi)
        
        N.append(ni)
        CA.append(cai)
        C.append(ci)
        
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
        env_atoms = []
        
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
            
            next_N = None
            if i + 1 < len(sequence) and (i + 1) < next_break:
                next_N = N[i + 1]
            else:
                next_N = N[i]
            O_pos = carbonyl_oxygen_position(C[i], CA[i], next_N, bond_len=_BOND_CO_LEN)
            if O_pos is not None:
                v1 = CA[i] - N[i]
                v2 = C[i] - CA[i]
                nrm = np.cross(v1, v2)
                nl = np.linalg.norm(nrm)
                if nl > 1e-6:
                    nrm = nrm / nl
                    dproj = np.dot(O_pos - N[i], nrm)
                    O_pos = O_pos - dproj * nrm
            if O_pos is not None:
                f.write(f"ATOM  {atom_idx:5d}  O   {resn:>3s} {chain_id}{res_idx:4d}    {O_pos[0]:8.3f}{O_pos[1]:8.3f}{O_pos[2]:8.3f}  1.00  0.00           O\n")
                atom_idx += 1
                env_atoms.append(O_pos)
            env_atoms.append(N[i])
            env_atoms.append(CA[i])
            env_atoms.append(C[i])

            if aa != "G":
                from modules.sidechain_builder import pack_sidechain
                sc_atoms = pack_sidechain(aa, N[i], CA[i], C[i], local_environment_atoms=np.array(env_atoms) if len(env_atoms) > 0 else None)

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
                    env_atoms.append(pos)

            res_idx += 1
            
        f.write("TER\n")
        f.write("END\n")
    return True

def _rotation_matrix(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return Rz @ Ry @ Rx

def _apply_transform(coords, trans, rot_angles):
    R = _rotation_matrix(*rot_angles)
    return (coords @ R.T) + trans

def _objective_joint(x, chain_data, total_residues):
    phi_all = x[:total_residues]
    psi_all = x[total_residues : 2*total_residues]
    rb_params = x[2*total_residues:]
    
    all_CA = []
    idx = 0
    rb_idx = 0
    t_ang = 0.0
    t_rama = 0.0
    
    for i, (seq, ss, phi_ref, psi_ref) in enumerate(chain_data):
        n = len(seq)
        phi = phi_all[idx : idx+n]
        psi = psi_all[idx : idx+n]
        idx += n
        
        # Weighted angle constraint
        # SS 'C' -> weight 0.1, others 1.0
        # Can be precomputed, but here SS is string.
        weights = np.array([0.5 if c == 'C' else 1.0 for c in ss])
        
        d_phi = (phi - phi_ref + np.pi) % (2 * np.pi) - np.pi
        d_psi = (psi - psi_ref + np.pi) % (2 * np.pi) - np.pi
        t_ang += np.sum(weights * (d_phi**2 + d_psi**2))

        try:
            t_rama += _rama_loss(phi, psi, seq)
        except Exception:
            pass
        
        try:
            N_c, CA_c, C_c = build_backbone(seq, phi, psi)
        except:
            return 1e9
            
        if i > 0:
            params = rb_params[rb_idx : rb_idx+6]
            rb_idx += 6
            trans = params[:3]
            rot = params[3:]
            N_c = _apply_transform(N_c, trans, rot)
            CA_c = _apply_transform(CA_c, trans, rot)
            C_c = _apply_transform(C_c, trans, rot)
            
        all_CA.append(CA_c)
        
    full_CA = np.concatenate(all_CA)
    centroid = np.mean(full_CA, axis=0)
    rg2 = np.sum(np.sum((full_CA - centroid)**2, axis=1)) / len(full_CA)
    
    dmat = full_CA[:, None, :] - full_CA[None, :, :]
    dd = np.linalg.norm(dmat, axis=2)
    mask = np.triu(np.ones(dd.shape, dtype=bool), k=2)
    dists = dd[mask]
    
    clash_mask = dists < 3.6
    if np.any(clash_mask):
        t_clash = np.sum((3.5 - dists[clash_mask]) ** 2)
    else:
        t_clash = 0.0
        
    return 0.5 * rg2 + 10.0 * t_clash + 5.0 * t_ang + 3.0 * t_rama

def optimize_from_ss(sequence, chain_ss_list, output_pdb, iters=2):
    if not chain_ss_list: return False
    lengths = [len(s) for s in chain_ss_list]
    if sum(lengths) != len(sequence): return False
    
    chain_data = []
    start_idx = 0
    total_residues = 0
    
    # Prepare chain data
    for ss in chain_ss_list:
        L = len(ss)
        sub_seq = sequence[start_idx : start_idx + L]
        start_idx += L
        phi0, psi0 = ss_to_phi_psi(ss, sub_seq)
        chain_data.append((sub_seq, ss, phi0, psi0))
        total_residues += L
        
    # Initialize variables
    # [phi_1..N, psi_1..N, (tx,ty,tz,rx,ry,rz)_2..M]
    num_chains = len(chain_ss_list)
    num_rb = (num_chains - 1) * 6
    
    best_x = None
    best_val = 1e9
    
    for _ in range(iters):
        # Initial guess
        phi_init = []
        psi_init = []
        for _, _, p, q in chain_data:
            phi_init.append(p + np.random.uniform(-0.1, 0.1, len(p)))
            psi_init.append(q + np.random.uniform(-0.1, 0.1, len(q)))
        
        rb_init = []
        if num_chains > 1:
            # Randomize initial positions for chains > 0
            for k in range(num_chains - 1):
                # Random translation +/- 20A
                t = np.random.uniform(-20, 20, 3)
                # Random rotation
                r = np.random.uniform(-np.pi, np.pi, 3)
                rb_init.extend(t)
                rb_init.extend(r)
                
        x0 = np.concatenate([np.concatenate(phi_init), np.concatenate(psi_init), np.array(rb_init)])
        
        # Bounds
        bounds = [(-np.pi, np.pi)] * (2 * total_residues)
        if num_rb > 0:
            # Unbounded translation, but let's loose bound it
            bounds.extend([(-100, 100)] * 3 + [(-np.pi, np.pi)] * 3)
            # Repeat for each chain
            bounds = bounds[:2*total_residues] + (bounds[2*total_residues:] * (num_chains - 1))

        try:
            res = minimize(_objective_joint, x0, args=(chain_data, total_residues),
                          method="L-BFGS-B",
                          options={'maxiter': 300, 'ftol': 1e-4, 'disp': False},
                          bounds=bounds)
            
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
            
    if best_x is None:
        best_x = x0 # Fallback
        
    # Reconstruct final structure
    phi_all = best_x[:total_residues]
    psi_all = best_x[total_residues : 2*total_residues]
    rb_params = best_x[2*total_residues:]
    
    final_N, final_CA, final_C = [], [], []
    idx = 0
    rb_idx = 0
    
    for i, (seq, _, _, _) in enumerate(chain_data):
        n = len(seq)
        phi = phi_all[idx : idx+n]
        psi = psi_all[idx : idx+n]
        idx += n
        
        N_c, CA_c, C_c = build_backbone(seq, phi, psi)
        
        if i > 0:
            params = rb_params[rb_idx : rb_idx+6]
            rb_idx += 6
            N_c = _apply_transform(N_c, params[:3], params[3:])
            CA_c = _apply_transform(CA_c, params[:3], params[3:])
            C_c = _apply_transform(C_c, params[:3], params[3:])
            
        final_N.append(N_c)
        final_CA.append(CA_c)
        final_C.append(C_c)
        
    final_N = np.concatenate(final_N)
    final_CA = np.concatenate(final_CA)
    final_C = np.concatenate(final_C)
    
    breaks = []
    acc = 0
    for L in lengths[:-1]:
        acc += L
        breaks.append(acc)
        
    return write_pdb(sequence, final_N, final_CA, final_C, output_pdb, chain_breaks=breaks)

def run_backbone_fold_multichain(sequence, chain_ss_list, output_pdb, chain_slices=None):
    # Backward compatibility wrapper: now calls optimization
    return optimize_from_ss(sequence, chain_ss_list, output_pdb)
