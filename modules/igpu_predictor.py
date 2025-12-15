import math
import logging
import platform
import time
import numpy as np
import torch
import cpuinfo
import os
try:
    import torch_directml
    _HAS_DML = True
except Exception:
    _HAS_DML = False

# Setup logging
logger = logging.getLogger(__name__)
ALGO_VERSION = "igpu_dml_v2.0_fullatom"

# Precompute bond parameters (PyTorch tensors)
# Engh & Huber 1991
BOND_PARAMS = {
    "n_len": 1.33,
    "ca_len": 1.46,
    "c_len": 1.52,
    "ang_C_N_CA": math.radians(121.7),
    "ang_N_CA_C": math.radians(111.0),
    "ang_CA_C_N": math.radians(116.2),
}

three_letter = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"
}

# --- Advanced Physics Parameters ---

# 1. Hydrogen Bonding Constants
HB_OPT_DIST = 2.9  # N-O distance
HB_OPT_ANGLE = math.radians(160) # N-H...O angle
HB_MAX_DIST = 3.5
HB_MIN_ANGLE = math.radians(120)

# 2. Ramachandran Preferred Regions (Simplified Gaussian Mixtures)
# We define "centers" for General, Glycine, Proline
RAMA_PREF = {
    "General": [
        (-1.2, 2.4), # Beta
        (-1.0, -0.8) # Alpha
    ],
    "GLY": [
        (1.4, -2.8),
        (-1.4, 2.8),
        (1.4, 2.8),
        (-1.4, -2.8)
    ],
    "PRO": [
        (-1.0, 2.6), # PolyPro II
        (-1.0, -0.5) # Alpha
    ]
}

# Advanced Rotamer Library (Approximation of Dunbrack 2010 top probabilities)
# {AA: [(chi1, chi2, ...), ...]}
# Angles in degrees
# AlphaFold/Rosetta style: Extensive libraries preferred.
# Here we keep it concise but representative of the "Golden" states.
ROTAMER_LIB = {
    'ALA': [[]],
    'ARG': [[180, 180, 180, 180], [-60, 180, 180, 180], [-60, -60, 180, 180], [180, 60, 180, 180], [-60, 180, -60, 180]],
    'ASN': [[-60, -60], [-60, 90], [180, -60], [-60, 0]],
    'ASP': [[-60, -60], [-180, -60], [-60, 90], [180, 90]],
    'CYS': [[-60], [180], [60]],
    'GLN': [[-60, -60, 0], [-180, 60, 0], [-60, 180, 0], [-60, 60, 0]],
    'GLU': [[-60, -60, 0], [-180, 60, 0], [-60, 180, 0], [-60, 60, 0]],
    'GLY': [[]],
    'HIS': [[-60, 180], [-60, -60], [180, 60], [60, -60]],
    'ILE': [[-60, 180], [-60, -60], [180, 60]],
    'LEU': [[-60, 180], [180, 60], [-60, 60]],
    'LYS': [[-60, 180, 180, 180], [-60, -60, 180, 180], [180, 180, 180, 180]],
    'MET': [[-60, 180, 180], [-60, 180, 60], [-60, -60, 180]],
    'PHE': [[-60, 90], [180, 90], [-60, 0]],
    'PRO': [[0, 0], [10, 0]], 
    'SER': [[-60], [60], [180]],
    'THR': [[-60], [60], [180]],
    'TRP': [[-60, 90], [180, 90], [-60, 0]],
    'TYR': [[-60, 90], [180, 90], [-60, 0]],
    'VAL': [[180], [-60], [60]]
}

def check_gpu_availability():
    """
    Checks for available hardware acceleration devices in order of preference.
    Support:
    1. DirectML (Windows/Linux) -> AMD, Intel, NVIDIA, Qualcomm (Broadest iGPU support)
    2. CUDA (NVIDIA) -> NVIDIA dGPU
    3. ROCm (AMD) -> AMD dGPU/APU (Linux mostly)
    4. MPS (macOS) -> Apple Silicon (M1/M2/M3)
    5. Intel XPU (IPEX) -> Intel Arc/Iris (Future support)
    """
    # 1. DirectML (Best for Windows iGPU: Intel Iris/Arc, AMD Radeon, Qualcomm Adreno)
    if _HAS_DML:
        try:
            dml_dev = torch_directml.device()
            # Verify basic tensor op
            _ = torch.tensor([1.0], device=dml_dev)
            return True, "DirectML (Intel/AMD/NVIDIA/Qualcomm)", "DirectML"
        except Exception as e:
            logger.warning(f"DirectML available but failed init: {e}")

    # 2. CUDA (NVIDIA Standard)
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "NVIDIA CUDA"
        return True, name, "NVIDIA Tensor Core"
        
    # 3. Apple MPS (Mac)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return True, "Apple Neural Engine (MPS)", "Apple NPU"
        
    # 4. Intel XPU (Linux/WSL specific, usually requires intel_extension_for_pytorch)
    try:
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
             return True, "Intel XPU (Arc/Iris)", "Intel XPU"
    except:
        pass

    return False, "No dedicated GPU detected", "CPU"

def _normalize_chains_to_seq_len(chain_ss_list, seq_len):
    if not chain_ss_list:
        return ["C" * seq_len]
    total = sum(len(s) for s in chain_ss_list)
    parts = chain_ss_list[:]
    if total == seq_len:
        return parts
    if total > seq_len:
        over = total - seq_len
        for i in range(len(parts)-1, -1, -1):
            if over <= 0:
                break
            cut = min(over, len(parts[i]))
            parts[i] = parts[i][:-cut]
            over -= cut
        parts = [p for p in parts if p]
        if not parts:
            return ["C" * seq_len]
        return parts
    add = seq_len - total
    parts[-1] = parts[-1] + ("C" * add)
    return parts

def ss_to_phi_psi_tensor(ss):
    phi = []
    psi = []
    for c in ss:
        if c == "H":
            phi.append(-math.radians(57.0))
            psi.append(-math.radians(47.0))
        elif c == "E":
            phi.append(-math.radians(119.0))
            psi.append(math.radians(113.0))
        else:
            phi.append(math.radians(-60.0 + np.random.uniform(-10.0, 10.0)))
            psi.append(math.radians(140.0 + np.random.uniform(-10.0, 10.0)))
    return torch.tensor(phi, dtype=torch.float32), torch.tensor(psi, dtype=torch.float32)

def place_atom_tensor(a, b, c, bond_len, bond_angle, dihedral):
    bc = c - b
    ab = b - a
    bc_l = torch.norm(bc) + 1e-9
    bc_u = bc / bc_l
    n = torch.cross(ab, bc_u, dim=0)
    n_l = torch.norm(n)
    
    # Robust normal calculation for collinear case
    # If n_l is too small, ab and bc are collinear.
    # We construct an arbitrary normal perpendicular to bc_u.
    
    # We use a mask-free approach for differentiability where possible, 
    # but here a conditional is safer for stability.
    # Since this function is usually called in a loop (not batched), we can check n_l value if scalar?
    # But n_l is a tensor (scalar tensor).
    
    # If we are in a batch, we need torch.where. 
    # Assuming single-instance calls for now as per loop in build_full_structure_tensor.
    
    if n_l < 1e-6:
        # Create arbitrary vector not parallel to bc_u
        # If bc_u is near x-axis, use y-axis
        if torch.abs(bc_u[0]) < 0.9:
            arb = torch.tensor([1.0, 0.0, 0.0], device=a.device, dtype=a.dtype)
        else:
            arb = torch.tensor([0.0, 1.0, 0.0], device=a.device, dtype=a.dtype)
            
        n = torch.cross(arb, bc_u, dim=0)
        n_l = torch.norm(n)
        
    n = n / (n_l + 1e-9)
    nb = torch.cross(n, bc_u, dim=0)
    nb = nb / (torch.norm(nb) + 1e-9)
    
    x = -bond_len * torch.cos(bond_angle)
    y = bond_len * torch.sin(bond_angle) * torch.cos(dihedral)
    z = bond_len * torch.sin(bond_angle) * torch.sin(dihedral)
    
    return c + (bc_u * x) + (nb * y) + (n * z)

def _rotation_matrix_tensor(rx, ry, rz, device):
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)
    
    zero = torch.tensor(0.0, device=device)
    one = torch.tensor(1.0, device=device)
    
    Rz = torch.stack([
        torch.stack([cz, -sz, zero]),
        torch.stack([sz, cz, zero]),
        torch.stack([zero, zero, one])
    ])
    Ry = torch.stack([
        torch.stack([cy, zero, sy]),
        torch.stack([zero, one, zero]),
        torch.stack([-sy, zero, cy])
    ])
    Rx = torch.stack([
        torch.stack([one, zero, zero]),
        torch.stack([zero, cx, -sx]),
        torch.stack([zero, sx, cx])
    ])
    return torch.mm(Rz, torch.mm(Ry, Rx))

def _apply_transform_tensor(coords, trans, rot_angles, device):
    R = _rotation_matrix_tensor(rot_angles[0], rot_angles[1], rot_angles[2], device)
    # coords: (N, 3). R: (3, 3). R.t(): (3, 3).
    # Matmul: (N,3) x (3,3) -> (N,3)
    return torch.matmul(coords, R.t()) + trans

def build_full_structure_tensor(sequence, phi, psi, device):
    """
    Builds Backbone + CB (Centroid) + H (Amide) + O (Carbonyl).
    Note: Full sidechain rotamer optimization is expensive in pure python loop.
    We implement "Backbone + Centroid" model for speed, 
    but include H and O for H-bond calculation.
    """
    p = BOND_PARAMS
    n_len = torch.tensor(p["n_len"], device=device)
    ca_len = torch.tensor(p["ca_len"], device=device)
    c_len = torch.tensor(p["c_len"], device=device)
    ang_C_N_CA = torch.tensor(p["ang_C_N_CA"], device=device)
    ang_N_CA_C = torch.tensor(p["ang_N_CA_C"], device=device)
    ang_CA_C_N = torch.tensor(p["ang_CA_C_N"], device=device)
    
    # Init
    N = torch.tensor([0.0, 0.0, 0.0], device=device)
    CA = torch.tensor([ca_len.item(), 0.0, 0.0], device=device)
    
    # Virtual previous atom C(-1)
    # Must NOT be collinear with N-CA (X-axis)
    # Place it at angle ~120 deg to X-axis in XY plane
    # x = -cos(120)*len, y = sin(120)*len
    prev_virtual = torch.tensor([-0.5 * c_len.item(), 0.866 * c_len.item(), 0.0], device=device)
    
    C = place_atom_tensor(prev_virtual, N, CA, c_len, ang_N_CA_C, torch.tensor(0.0, device=device))
    
    coords_N = [N]
    coords_CA = [CA]
    coords_C = [C]
    coords_O = [] # Carbonyl Oxygen
    coords_H = [] # Amide Hydrogen
    coords_CB = [] # Sidechain Centroid/CB
    
    omega = torch.tensor(math.pi, device=device) # Trans
    
    # H-bond parameters
    bond_NH = torch.tensor(1.01, device=device) # N-H length
    bond_CO = torch.tensor(1.23, device=device) # C=O length
    
    # Place first H? No H on N-term usually (NH3+)
    coords_H.append(torch.tensor([0.0,0.0,0.0], device=device)) # Dummy
    
    # Place first O
    # O is in the plane of CA-C-N(next) usually, but here we place relative to N-CA-C frame
    # Vector C=O bisects N-C-CA exterior angle? 
    # Roughly: place O opposite to bisector of N-C-CA
    # Or simply: place_atom(CA, C, N, bond_CO, 120deg, pi)
    O0 = place_atom_tensor(CA, C, N, bond_CO, torch.tensor(math.radians(120.8), device=device), torch.tensor(math.pi, device=device))
    coords_O.append(O0)
    
    # Place first CB
    # Standard geometry from N, C, CA
    # place_atom(N, CA, C, 1.53, 110deg, -120deg)?
    CB0 = place_atom_tensor(N, CA, C, torch.tensor(1.53, device=device), torch.tensor(math.radians(110.5), device=device), torch.tensor(math.radians(-122.5), device=device))
    coords_CB.append(CB0)

    for i in range(1, len(sequence)):
        Ni = place_atom_tensor(coords_N[i-1], coords_CA[i-1], coords_C[i-1], n_len, ang_CA_C_N, psi[i-1])
        CAi = place_atom_tensor(coords_CA[i-1], coords_C[i-1], Ni, ca_len, ang_C_N_CA, omega)
        Ci = place_atom_tensor(coords_C[i-1], Ni, CAi, c_len, ang_N_CA_C, phi[i])
        
        coords_N.append(Ni)
        coords_CA.append(CAi)
        coords_C.append(Ci)
        
        # Place H on N(i)
        # H is on bisector of C(i-1)-N(i)-CA(i)
        # place_atom(C(i-1), N(i), CA(i), 1.01, 120, pi)
        Hi = place_atom_tensor(coords_C[i-1], Ni, CAi, bond_NH, torch.tensor(math.radians(119.5), device=device), torch.tensor(math.pi, device=device))
        coords_H.append(Hi)
        
        # Place O on C(i)
        # place_atom(CA(i), C(i), N(i), 1.23, 120.8, pi)
        # Note: N(i) is 'prev' for C(i), but for place_atom(a,b,c), 'a' is prev-prev.
        # Here frame is CA, C, N(prev)? No.
        # place_atom inputs: (prev_prev, prev, current, len, angle, torsion)
        # To place O at C, we need atoms defining C's frame. That is Ni, CAi.
        # So place_atom(Ni, CAi, Ci, ...)
        Oi = place_atom_tensor(Ni, CAi, Ci, bond_CO, torch.tensor(math.radians(120.8), device=device), torch.tensor(math.pi, device=device))
        coords_O.append(Oi)
        
        # Place CB on CA(i)
        # Frame: Ni, Ci, CAi
        # place_atom(Ni, CAi, Ci, ...)
        CBi = place_atom_tensor(Ni, CAi, Ci, torch.tensor(1.53, device=device), torch.tensor(math.radians(110.5), device=device), torch.tensor(math.radians(122.5), device=device))
        coords_CB.append(CBi)
        
    return (torch.stack(coords_N), torch.stack(coords_CA), torch.stack(coords_C), 
            torch.stack(coords_O), torch.stack(coords_H), torch.stack(coords_CB))

def optimize_from_ss_gpu(sequence, chain_ss_list, output_pdb):
    """
    Advanced Full-Atom Joint Optimization Pipeline.
    """
    is_gpu, device_name, device_type = check_gpu_availability()
    logger.info(f"iGPU Optimization Enabled [{ALGO_VERSION}]. Device: {device_name} ({device_type})")
    
    device = torch.device('cpu')
    if is_gpu:
        if device_type == "DirectML":
            device = torch_directml.device()
        elif device_type == "NVIDIA Tensor Core" or device_type == "AMD ROCm":
            device = torch.device('cuda')
        elif device_type == "Apple NPU":
            device = torch.device('mps')
            
    try:
        torch.set_default_device(device)
    except Exception:
        pass
        
    start_time = time.time()
    
    # Prepare chain data
    chain_data = []
    start_idx = 0
    
    for ss in chain_ss_list:
        L = len(ss)
        sub_seq = sequence[start_idx : start_idx + L]
        start_idx += L
        p, q = ss_to_phi_psi_tensor(ss)
        
        # Create weights: 0.1 for 'C', 1.0 for others
        w_list = [0.1 if c == 'C' else 1.0 for c in ss]
        w_tensor = torch.tensor(w_list, dtype=torch.float32, device=device)
        
        # Prepare Residue Properties
        hydro_indices = []
        charge_vals = []
        
        # Refined Hydrophobic Mapping (Kyte-Doolittle Scale)
        # Normalized (0.0 to 1.0), only positive hydrophobicity considered for core.
        # AlphaFold uses "Residue Interaction Networks" implicitly via Evoformer.
        # We explicitly model the physical property.
        # I: 1.0, V: 0.9, L: 0.8, F: 0.6, M: 0.4, A: 0.4
        HP_MAP = {
            'I': 1.0, 'V': 0.9, 'L': 0.8, 'F': 0.6, 'M': 0.4, 'A': 0.4
        }
        # C, Y, W treated as weak or polar for this purpose.
        
        # Charge: D=-1, E=-1, K=1, R=1, H=0.5 (at pH 7)
        CHARGE_MAP = {'D': -1.0, 'E': -1.0, 'K': 1.0, 'R': 1.0, 'H': 0.5}
        
        h_idx_list = []
        c_val_list = []
        
        for aa in sub_seq:
            h_idx_list.append(HP_MAP.get(aa, 0.0))
            c_val_list.append(CHARGE_MAP.get(aa, 0.0))
            
        hydro_mask = torch.tensor(h_idx_list, device=device)
        charge = torch.tensor(c_val_list, device=device)
        
        chain_data.append({
            "seq": sub_seq,
            "ss": ss,
            "phi_ref": p.to(device),
            "psi_ref": q.to(device),
            "ss_weights": w_tensor,
            "hydro_mask": hydro_mask,
            "charge": charge
        })
        
    # Initialize Parameters
    phi_params = []
    psi_params = []
    
    for d in chain_data:
        p0 = d["phi_ref"] + (torch.rand_like(d["phi_ref"]) * 0.1)
        q0 = d["psi_ref"] + (torch.rand_like(d["psi_ref"]) * 0.1)
        phi_params.append(p0.requires_grad_(True))
        psi_params.append(q0.requires_grad_(True))
        
    rb_params = None
    if len(chain_data) > 1:
        init_rb = []
        for _ in range(len(chain_data) - 1):
            t = (torch.rand(3, device=device) * 40.0) - 20.0
            r = (torch.rand(3, device=device) * 2 * math.pi) - math.pi
            init_rb.append(torch.cat([t, r]))
        rb_params = torch.stack(init_rb).requires_grad_(True)
        
    all_params = phi_params + psi_params
    if rb_params is not None:
        all_params.append(rb_params)
        
    # LBFGS is powerful but can be slow per step. 
    # For initial convergence, we can use Adam, then refine with LBFGS.
    # Or just optimize LBFGS settings.
    # Reducing history size and relaxing tolerance slightly can help speed without hurting final structure much.
    
    # Strategy: Two-stage optimization
    # Stage 1: Adam (Fast coarse folding)
    # Stage 2: LBFGS (Fine refinement)
    
    # Stage 1: Adam
    logger.info("  > Stage 1: Coarse folding (Adam)...")
    opt_adam = torch.optim.Adam(all_params, lr=0.05)
    for _ in range(200): # 200 steps of Adam
        opt_adam.zero_grad()
        
        # Inlined closure logic for speed (avoid function call overhead? Not significant but cleaner)
        # Just call closure function but adapted for simple forward/backward
        # Actually, LBFGS closure requires re-evaluation. Adam just needs forward.
        # Let's extract loss calculation to a function to reuse.
        
        # Reuse closure logic via a helper? 
        # For simplicity in this script, we can just define a `calc_loss()` function inside `optimize_from_ss_gpu`.
        pass # Will implemented below
        
    # We need to restructure the code to share the loss function.
    
    def calc_loss():
        all_N, all_CA, all_C, all_O, all_H, all_CB = [], [], [], [], [], []
        loss_ss = torch.tensor(0.0, device=device)
        loss_rama = torch.tensor(0.0, device=device)
        
        # Build
        for i, d in enumerate(chain_data):
            phi = phi_params[i]
            psi = psi_params[i]
            
            # SS Restraint (Adaptive)
            ss_weights = torch.ones_like(phi, device=device)
            if "ss_weights" in d:
                ss_weights = d["ss_weights"]
            
            d_phi = torch.remainder(phi - d["phi_ref"] + math.pi, 2*math.pi) - math.pi
            d_psi = torch.remainder(psi - d["psi_ref"] + math.pi, 2*math.pi) - math.pi
            loss_ss = loss_ss + torch.sum(ss_weights * (d_phi**2 + d_psi**2))
            
            # Ramachandran Potential
            d_alpha = (phi + 1.0)**2 + (psi + 0.8)**2
            d_beta = (phi + 1.2)**2 + (psi - 2.4)**2
            loss_rama = loss_rama + torch.sum(torch.min(d_alpha, d_beta))
            
            # Build Structure
            N, CA, C, O, H, CB = build_full_structure_tensor(d["seq"], phi, psi, device)
            
            if i > 0 and rb_params is not None:
                params = rb_params[i-1]
                t, r = params[:3], params[3:]
                N = _apply_transform_tensor(N, t, r, device)
                CA = _apply_transform_tensor(CA, t, r, device)
                C = _apply_transform_tensor(C, t, r, device)
                O = _apply_transform_tensor(O, t, r, device)
                H = _apply_transform_tensor(H, t, r, device)
                CB = _apply_transform_tensor(CB, t, r, device)
                
            all_N.append(N)
            all_CA.append(CA)
            all_C.append(C)
            all_O.append(O)
            all_H.append(H)
            all_CB.append(CB)
            
        full_CA = torch.cat(all_CA)
        full_CB = torch.cat(all_CB)
        
        # Collect property tensors
        all_hydro = []
        all_charge = []
        for d in chain_data:
            all_hydro.append(d["hydro_mask"])
            all_charge.append(d["charge"])
        full_hydro = torch.cat(all_hydro)
        full_charge = torch.cat(all_charge)
        
        # Rg (Compactness)
        centroid = torch.mean(full_CA, dim=0)
        rg2 = torch.sum(torch.sum((full_CA - centroid)**2, dim=1)) / len(full_CA)
        
        # Clash (CA-CA + CB-CB)
        full_atoms = torch.cat([full_CA, full_CB])
        diff_mat = full_atoms.unsqueeze(0) - full_atoms.unsqueeze(1)
        dist_mat = torch.norm(diff_mat, dim=2)
        mask = torch.triu(torch.ones_like(dist_mat), diagonal=2) > 0
        clash_dists = dist_mat[mask]
        clash_loss = torch.sum(torch.relu(3.5 - clash_dists)**2)
        
        # Hydrophobic Effect
        diff_cb = full_CB.unsqueeze(0) - full_CB.unsqueeze(1)
        dist_cb = torch.norm(diff_cb, dim=2) + 1e-6
        hp_mat = full_hydro.unsqueeze(0) * full_hydro.unsqueeze(1)
        mask_nonlocal = (torch.triu(torch.ones_like(dist_cb), diagonal=3) > 0).float()
        
        hp_dist_sum = torch.sum(dist_cb * hp_mat * mask_nonlocal)
        n_hp_pairs = torch.sum(hp_mat * mask_nonlocal) + 1.0
        loss_hydro = hp_dist_sum / n_hp_pairs
        
        # Electrostatics
        q_mat = full_charge.unsqueeze(0) * full_charge.unsqueeze(1)
        elec_pot = q_mat / torch.sqrt(dist_cb**2 + 1.0)
        loss_elec = torch.sum(elec_pot * mask_nonlocal) * 10.0
        
        # Hydrogen Bonds
        full_O = torch.cat(all_O)
        full_H = torch.cat(all_H)
        diff_ho = full_H.unsqueeze(0) - full_O.unsqueeze(1)
        dist_ho = torch.norm(diff_ho, dim=2)
        hb_score = torch.relu(3.5 - dist_ho)
        
        n_res = len(full_CA)
        idx = torch.arange(n_res, device=device)
        idx_diff = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
        hb_mask = (idx_diff > 3).float()
        
        hb_energy = -torch.sum(hb_score * hb_mask) * 0.1
        
        # Weights
        w_rg = 2.0
        w_clash = 5.0
        w_ss = 2.0
        w_rama = 1.0
        w_hb = 3.0
        w_hydro = 5.0
        w_elec = 2.0
        
        total_loss = (w_rg * rg2 + 
                      w_clash * clash_loss + 
                      w_ss * loss_ss + 
                      w_rama * loss_rama + 
                      w_hb * hb_energy + 
                      w_hydro * loss_hydro + 
                      w_elec * loss_elec)
                      
        return total_loss
    
    frames_dir = os.path.join(os.path.dirname(output_pdb), "_frames")
    try:
        os.makedirs(frames_dir, exist_ok=True)
    except Exception:
        pass
    frame_counter = 0
    # Helper to save snapshot
    def save_snapshot(step_name):
        try:
            with torch.no_grad():
                final_N_list, final_CA_list, final_C_list = [], [], []
                for i, d in enumerate(chain_data):
                    phi, psi = phi_params[i], psi_params[i]
                    N, CA, C, _, _, _ = build_full_structure_tensor(d["seq"], phi, psi, device)
                    if i > 0 and rb_params is not None:
                        params = rb_params[i-1]
                        t, r = params[:3], params[3:]
                        N = _apply_transform_tensor(N, t, r, device)
                        CA = _apply_transform_tensor(CA, t, r, device)
                        C = _apply_transform_tensor(C, t, r, device)
                    final_N_list.append(N)
                    final_CA_list.append(CA)
                    final_C_list.append(C)
                
                final_N = torch.cat(final_N_list).cpu().numpy()
                final_CA = torch.cat(final_CA_list).cpu().numpy()
                final_C = torch.cat(final_C_list).cpu().numpy()
                
                lengths = [len(s) for s in chain_ss_list]
                breaks = []
                acc = 0
                for L in lengths[:-1]:
                    acc += L
                    breaks.append(acc)
                
                tmp_path = output_pdb + ".tmp"
                # Use global write_pdb
                write_pdb(sequence, final_N, final_CA, final_C, tmp_path, chain_breaks=breaks)
                
                # Atomic replace (simulated for Windows) with retry
                max_retries = 5
                for _ in range(max_retries):
                    try:
                        if os.path.exists(output_pdb):
                            os.remove(output_pdb)
                        os.rename(tmp_path, output_pdb)
                        break
                    except PermissionError:
                        time.sleep(0.01)
                    except Exception:
                        break
                
                nonlocal frame_counter
                frame_counter += 1
                frame_name = f"frame_{frame_counter:05d}.pdb"
                frame_tmp = os.path.join(frames_dir, frame_name + ".tmp")
                write_pdb(sequence, final_N, final_CA, final_C, frame_tmp, chain_breaks=breaks)
                for _ in range(max_retries):
                    try:
                        frame_out = os.path.join(frames_dir, frame_name)
                        if os.path.exists(frame_out):
                            os.remove(frame_out)
                        os.rename(frame_tmp, frame_out)
                        break
                    except PermissionError:
                        time.sleep(0.01)
                    except Exception:
                        break
                    
        except Exception as e:
            logger.warning(f"Snapshot failed: {e}")

    # Save initial state
    save_snapshot("init")

    # Stage 1: Adam
    logger.info("  > Stage 1: Coarse folding (Adam)...")
    opt_adam = torch.optim.Adam(all_params, lr=0.05)
    for step in range(100): # 100 steps is usually enough for coarse structure
        opt_adam.zero_grad()
        loss = calc_loss()
        loss.backward()
        opt_adam.step()
        
        # Save every step for maximum smoothness
        save_snapshot(f"adam_{step+1}")
        
    # Stage 2: LBFGS (Refinement)
    logger.info("  > Stage 2: Refinement (LBFGS)...")
    opt_lbfgs = torch.optim.LBFGS(all_params,
                                  max_iter=100, # Reduced max_iter because Adam did the heavy lifting
                                  tolerance_grad=1e-5, # Relaxed slightly for speed
                                  tolerance_change=1e-5,
                                  history_size=20, # Reduced history
                                  line_search_fn="strong_wolfe")
                                  
    lbfgs_step_count = 0
    def closure():
        nonlocal lbfgs_step_count
        opt_lbfgs.zero_grad()
        loss = calc_loss()
        loss.backward()
        
        lbfgs_step_count += 1
        # Save every step
        save_snapshot(f"lbfgs_{lbfgs_step_count}")
            
        return loss
        
    try:
        opt_lbfgs.step(closure)
    except Exception as e:
        logger.warning(f"Optimization step failed: {e}")
        
    # Reconstruct Final
    final_N_list = []
    final_CA_list = []
    final_C_list = []
    
    with torch.no_grad():
        for i, d in enumerate(chain_data):
            phi = phi_params[i]
            psi = psi_params[i]
            N, CA, C, _, _, _ = build_full_structure_tensor(d["seq"], phi, psi, device)
            
            if i > 0 and rb_params is not None:
                params = rb_params[i-1]
                t, r = params[:3], params[3:]
                N = _apply_transform_tensor(N, t, r, device)
                CA = _apply_transform_tensor(CA, t, r, device)
                C = _apply_transform_tensor(C, t, r, device)
                
            final_N_list.append(N)
            final_CA_list.append(CA)
            final_C_list.append(C)
            
    final_N = torch.cat(final_N_list).cpu().numpy()
    final_CA = torch.cat(final_CA_list).cpu().numpy()
    final_C = torch.cat(final_C_list).cpu().numpy()
    
    elapsed = time.time() - start_time
    logger.info(f"iGPU Optimization completed in {elapsed:.4f}s [{ALGO_VERSION}]")
    
    # Breaks
    lengths = [len(s) for s in chain_ss_list]
    breaks = []
    acc = 0
    for L in lengths[:-1]:
        acc += L
        breaks.append(acc)
        
    return write_pdb(sequence, final_N, final_CA, final_C, output_pdb, chain_breaks=breaks)

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
        
        intervals = []
        start = 0
        if chain_breaks:
            for end in chain_breaks:
                intervals.append((start, end))
                start = end
        intervals.append((start, len(sequence)))
        
        next_break = intervals[0][1]
        current_interval_idx = 0
        
        for i, aa in enumerate(sequence):
            if i >= next_break:
                f.write("TER\n")
                current_interval_idx += 1
                if current_interval_idx < len(intervals):
                    next_break = intervals[current_interval_idx][1]
                chain_idx += 1
                chain_id = chr(ord('A') + (chain_idx % 26))
                res_idx = 1
                
            resn = three_letter.get(aa, "UNK")
            
            f.write(f"ATOM  {atom_idx:5d}  N   {resn:>3s} {chain_id}{res_idx:4d}    {N[i][0]:8.3f}{N[i][1]:8.3f}{N[i][2]:8.3f}  1.00  0.00           N\n")
            atom_idx += 1
            f.write(f"ATOM  {atom_idx:5d}  CA  {resn:>3s} {chain_id}{res_idx:4d}    {CA[i][0]:8.3f}{CA[i][1]:8.3f}{CA[i][2]:8.3f}  1.00  0.00           C\n")
            atom_idx += 1
            f.write(f"ATOM  {atom_idx:5d}  C   {resn:>3s} {chain_id}{res_idx:4d}    {C[i][0]:8.3f}{C[i][1]:8.3f}{C[i][2]:8.3f}  1.00  0.00           C\n")
            atom_idx += 1
            
            # Re-calculate O for PDB (based on final coords)
            # Use place_atom logic locally or simplified
            try:
                # O vector: bisect N-CA-C angle approx?
                # or place relative to C using CA, N
                v_ca_c = C[i] - CA[i]
                v_ca_c /= np.linalg.norm(v_ca_c)
                v_n_ca = CA[i] - N[i]
                v_n_ca /= np.linalg.norm(v_n_ca)
                
                # In plane defined by N, CA, C
                # O is ~opposite to bisector
                # Simple: rotate v_ca_c by 120 deg in plane
                n_plane = np.cross(v_n_ca, v_ca_c)
                n_plane /= np.linalg.norm(n_plane)
                
                # Rodrigues
                theta = math.radians(120.0)
                v_o = v_ca_c * math.cos(theta) + np.cross(n_plane, v_ca_c) * math.sin(theta) + n_plane * np.dot(n_plane, v_ca_c) * (1 - math.cos(theta))
                O_pos = C[i] + 1.23 * v_o
                
                f.write(f"ATOM  {atom_idx:5d}  O   {resn:>3s} {chain_id}{res_idx:4d}    {O_pos[0]:8.3f}{O_pos[1]:8.3f}{O_pos[2]:8.3f}  1.00  0.00           O\n")
                atom_idx += 1
            except:
                pass
            
            # Sidechain
            if aa != "G":
                # Use sidechain_builder's pack_sidechain to pick best rotamer if possible
                # Or just build with default (first) one for now
                from modules.sidechain_builder import pack_sidechain
                sc_atoms = pack_sidechain(aa, N[i], CA[i], C[i])
                
                order = ["CB", "CG", "CG1", "CG2", "OG", "OG1", "SG", 
                         "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD",
                         "CE", "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2",
                         "CZ", "CZ2", "CZ3", "NZ", 
                         "CH2", "NH1", "NH2", "OH"]
                sorted_keys = sorted(sc_atoms.keys(), key=lambda x: order.index(x) if x in order else 99)
                
                for atom_name in sorted_keys:
                    pos = sc_atoms[atom_name]
                    element = atom_name[0]
                    f.write(f"ATOM  {atom_idx:5d}  {atom_name:<4s}{resn:>3s} {chain_id}{res_idx:4d}    {pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00  0.00           {element}\n")
                    atom_idx += 1
            
            res_idx += 1
            
        f.write("TER\n")
        f.write("END\n")
    return True

# Interface compatibility wrapper
def run_backbone_fold_multichain(sequence, chain_ss_list, output_pdb, chain_slices=None):
    try:
        return optimize_from_ss_gpu(sequence, chain_ss_list, output_pdb)
    except Exception as e:
        logger.error(f"iGPU Optimization failed: {e}. Falling back to standard predictor.")
        return False
