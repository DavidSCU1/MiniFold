import math
import logging
import platform
import time
import numpy as np
import torch
import cpuinfo
import os
import shutil
from modules.quality import rama_pass_rate, contact_energy, tm_score_proxy, write_results_json
try:
    import torch_directml
    _HAS_DML = True
except Exception:
    _HAS_DML = False

# Setup logging
logger = logging.getLogger(__name__)
ALGO_VERSION = "igpu_dml_v2.4_phys_mj_pi"

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
RAMA_SIGMA = {
    "General": (0.4, 0.4),
    "GLY": (0.6, 0.6),
    "PRO": (0.5, 0.5)
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

MJ_ORDER = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
MJ_INDEX = {aa:i for i,aa in enumerate(MJ_ORDER)}
MJ_VALUES = [
[-0.59, 0.01, 0.12, 0.06,-0.34,-0.10,-0.24,-0.13, 0.15,-0.72,-0.89, 0.10,-0.47,-0.70, 0.11,-0.18,-0.14,-0.66,-0.46,-0.61],
[ 0.01,-0.68, 0.09,-0.10, 0.37,-0.34,-0.24, 0.19,-0.20, 0.22, 0.38,-0.63, 0.29, 0.36, 0.04, 0.14, 0.19,-0.18,-0.10, 0.23],
[ 0.12, 0.09,-0.47,-0.28, 0.07,-0.27,-0.17, 0.14,-0.28,-0.02,-0.07, 0.08, 0.02, 0.02, 0.06, 0.05, 0.02,-0.13,-0.10, 0.02],
[ 0.06,-0.10,-0.28,-0.53, 0.09,-0.25,-0.28, 0.16,-0.19, 0.06, 0.08,-0.07, 0.06, 0.07, 0.07, 0.04, 0.05,-0.07,-0.04, 0.07],
[-0.34, 0.37, 0.07, 0.09,-0.89, 0.14, 0.11,-0.08, 0.20,-0.57,-0.73, 0.34,-0.46,-0.66, 0.12,-0.12,-0.10,-0.60,-0.44,-0.50],
[-0.10,-0.34,-0.27,-0.25, 0.14,-0.54,-0.46, 0.06,-0.24,-0.09, 0.00,-0.31,-0.09, 0.01, 0.03,-0.04,-0.01,-0.21,-0.17,-0.05],
[-0.24,-0.24,-0.17,-0.28, 0.11,-0.46,-0.62, 0.07,-0.20, 0.02, 0.05,-0.25,-0.03, 0.03, 0.05,-0.01, 0.01,-0.16,-0.12, 0.02],
[-0.13, 0.19, 0.14, 0.16,-0.08, 0.06, 0.07,-0.26, 0.09,-0.20,-0.30, 0.16,-0.16,-0.24, 0.11,-0.08,-0.06,-0.28,-0.20,-0.18],
[ 0.15,-0.20,-0.28,-0.19, 0.20,-0.24,-0.20, 0.09,-0.56, 0.14, 0.18,-0.25, 0.12, 0.12, 0.07, 0.03, 0.06,-0.10,-0.12, 0.11],
[-0.72, 0.22,-0.02, 0.06,-0.57,-0.09, 0.02,-0.20, 0.14,-0.88,-0.98, 0.25,-0.59,-0.83, 0.12,-0.16,-0.12,-0.71,-0.52,-0.78],
[-0.89, 0.38,-0.07, 0.08,-0.73, 0.00, 0.05,-0.30, 0.18,-0.98,-1.08, 0.41,-0.68,-0.92, 0.14,-0.21,-0.16,-0.82,-0.62,-0.88],
[ 0.10,-0.63, 0.08,-0.07, 0.34,-0.31,-0.25, 0.16,-0.25, 0.25, 0.41,-0.80, 0.32, 0.37, 0.03, 0.15, 0.20,-0.16,-0.10, 0.26],
[-0.47, 0.29, 0.02, 0.06,-0.46,-0.09,-0.03,-0.16, 0.12,-0.59,-0.68, 0.32,-0.71,-0.79, 0.13,-0.15,-0.10,-0.68,-0.49,-0.64],
[-0.70, 0.36, 0.02, 0.07,-0.66, 0.01, 0.03,-0.24, 0.12,-0.83,-0.92, 0.37,-0.79,-0.96, 0.14,-0.19,-0.14,-0.86,-0.66,-0.84],
[ 0.11, 0.04, 0.06, 0.07, 0.12, 0.03, 0.05, 0.11, 0.07, 0.12, 0.14, 0.03, 0.13, 0.14,-0.26, 0.09, 0.12, 0.08, 0.09, 0.12],
[-0.18, 0.14, 0.05, 0.04,-0.12,-0.04,-0.01,-0.08, 0.03,-0.16,-0.21, 0.15,-0.15,-0.19, 0.09,-0.22,-0.18,-0.23,-0.17,-0.20],
[-0.14, 0.19, 0.02, 0.05,-0.10,-0.01, 0.01,-0.06, 0.06,-0.12,-0.16, 0.20,-0.10,-0.14, 0.12,-0.18,-0.14,-0.19,-0.14,-0.16],
[-0.66,-0.18,-0.13,-0.07,-0.60,-0.21,-0.16,-0.28,-0.10,-0.71,-0.82,-0.16,-0.68,-0.86, 0.08,-0.23,-0.19,-1.00,-0.75,-0.76],
[-0.46,-0.10,-0.10,-0.04,-0.44,-0.17,-0.12,-0.20,-0.12,-0.52,-0.62,-0.10,-0.49,-0.66, 0.09,-0.17,-0.14,-0.75,-0.58,-0.60],
[-0.61, 0.23, 0.02, 0.07,-0.50,-0.05, 0.02,-0.18, 0.11,-0.78,-0.88, 0.26,-0.64,-0.84, 0.12,-0.20,-0.16,-0.76,-0.60,-0.84]
]

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
    def _cross3(u, v):
        return torch.stack([
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0]
        ])
    n = _cross3(ab, bc_u)
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
            
        n = _cross3(arb, bc_u)
        n_l = torch.norm(n)
        
    n = n / (n_l + 1e-9)
    nb = _cross3(n, bc_u)
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

def optimize_from_ss_gpu(sequence, chain_ss_list, output_pdb, constraints=None):
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
        POLAR_SET = set(['D','E','K','R','H','N','Q','S','T','Y','C','W'])
        AROMATIC_SET = set(['F','Y','W'])
        
        h_idx_list = []
        c_val_list = []
        polar_list = []
        arom_list = []
        
        for aa in sub_seq:
            h_idx_list.append(HP_MAP.get(aa, 0.0))
            c_val_list.append(CHARGE_MAP.get(aa, 0.0))
            polar_list.append(1.0 if aa in POLAR_SET else 0.0)
            arom_list.append(1.0 if aa in AROMATIC_SET else 0.0)
            
        hydro_mask = torch.tensor(h_idx_list, device=device)
        charge = torch.tensor(c_val_list, device=device)
        polar_mask = torch.tensor(polar_list, device=device)
        aromatic_mask = torch.tensor(arom_list, device=device)
        
        chain_data.append({
            "seq": sub_seq,
            "ss": ss,
            "phi_ref": p.to(device),
            "psi_ref": q.to(device),
            "ss_weights": w_tensor,
            "hydro_mask": hydro_mask,
            "charge": charge,
            "polar_mask": polar_mask
            ,"aromatic_mask": aromatic_mask
        })
        
    num_chains = len(chain_data)
    
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
    
    # unified loss function and optimization below
    
    mj_tensor = torch.tensor(MJ_VALUES, dtype=torch.float32, device=device)
    def calc_loss():
        all_N, all_CA, all_C, all_O, all_H, all_CB = [], [], [], [], [], []
        loss_ss = torch.tensor(0.0, device=device)
        loss_rama = torch.tensor(0.0, device=device)
        loss_smooth = torch.tensor(0.0, device=device)
        chain_ids_collect = []
        
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
            
            group = "General"
            if len(d["seq"]) > 0:
                aa = d["seq"][0]
                if aa == "G":
                    group = "GLY"
                elif aa == "P":
                    group = "PRO"
            centers = RAMA_PREF.get(group, RAMA_PREF["General"])
            sigma = RAMA_SIGMA.get(group, RAMA_SIGMA["General"])
            
            # Vectorized Rama Loss
            c_t = torch.tensor(centers, device=device, dtype=torch.float32) # (K, 2)
            s_t = torch.tensor(sigma, device=device, dtype=torch.float32)   # (2,)
            
            p_ex = phi.unsqueeze(1) # (L, 1)
            q_ex = psi.unsqueeze(1) # (L, 1)
            
            cp = c_t[:, 0].unsqueeze(0) # (1, K)
            cq = c_t[:, 1].unsqueeze(0) # (1, K)
            
            term1 = ((p_ex - cp) / s_t[0])**2
            term2 = ((q_ex - cq) / s_t[1])**2
            
            # Min over K, Sum over L
            rama_sum = torch.sum(torch.min(term1 + term2, dim=1)[0])
            loss_rama = loss_rama + rama_sum
            
            diff_phi = phi[1:] - phi[:-1]
            diff_psi = psi[1:] - psi[:-1]
            loss_smooth = loss_smooth + torch.sum(diff_phi**2 + diff_psi**2)
            
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
            chain_ids_collect.append(torch.full((len(d["seq"]),), i, device=device, dtype=torch.long))
            
        full_CA = torch.cat(all_CA)
        full_CB = torch.cat(all_CB)
        full_N = torch.cat(all_N)
        full_C = torch.cat(all_C)
        chain_ids = torch.cat(chain_ids_collect)
        aa_idx_collect = []
        for d in chain_data:
            aa_idx_collect.append(torch.tensor([MJ_INDEX.get(a, MJ_INDEX['A']) for a in d["seq"]], device=device))
        aa_indices = torch.cat(aa_idx_collect)
        
        # Collect property tensors
        all_hydro = []
        all_charge = []
        all_polar = []
        all_aromatic = []
        for d in chain_data:
            all_hydro.append(d["hydro_mask"])
            all_charge.append(d["charge"])
            all_polar.append(d["polar_mask"])
            all_aromatic.append(d["aromatic_mask"])
        full_hydro = torch.cat(all_hydro)
        full_charge = torch.cat(all_charge)
        full_polar = torch.cat(all_polar)
        full_aromatic = torch.cat(all_aromatic)
        
        # Rg (Compactness)
        centroid = torch.mean(full_CA, dim=0)
        rg2 = torch.sum(torch.sum((full_CA - centroid)**2, dim=1)) / len(full_CA)
        rg = torch.sqrt(rg2 + 1e-9)
        rg_target = 2.2 * (len(full_CA) ** 0.38)
        loss_rg_target = (rg - rg_target) ** 2
        
        # Clash (CA-CA + CB-CB)
        full_atoms = torch.cat([full_CA, full_CB, full_N, full_C])
        diff_mat = full_atoms.unsqueeze(0) - full_atoms.unsqueeze(1)
        dist_mat = torch.norm(diff_mat, dim=2)
        mask = torch.triu(torch.ones_like(dist_mat), diagonal=2) > 0
        clash_dists = dist_mat[mask]
        clash_loss = torch.sum(torch.relu(3.5 - clash_dists)**2)
        full_O = torch.cat(all_O)
        heavy_atoms = torch.cat([full_CA, full_CB, full_O, full_N, full_C])
        hdiff = heavy_atoms.unsqueeze(0) - heavy_atoms.unsqueeze(1)
        hdist = torch.norm(hdiff, dim=2)
        hmask = torch.triu(torch.ones_like(hdist), diagonal=2) > 0
        heavy_loss = torch.sum(torch.relu(2.6 - hdist[hmask])**2)
        
        # Hydrophobic Effect
        diff_cb = full_CB.unsqueeze(0) - full_CB.unsqueeze(1)
        dist_cb = torch.norm(diff_cb, dim=2) + 1e-6
        u_vec = full_CB - full_CA
        u_vec = u_vec / (torch.norm(u_vec, dim=1, keepdim=True) + 1e-6)
        v_unit = (-diff_cb) / (dist_cb.unsqueeze(2))
        dot_i = torch.sum(u_vec.unsqueeze(1) * v_unit, dim=2).clamp(min=0.0)
        dot_j = torch.sum(u_vec.unsqueeze(0) * (-v_unit), dim=2).clamp(min=0.0)
        ori_gate = dot_i * dot_j
        hp_mat = full_hydro.unsqueeze(0) * full_hydro.unsqueeze(1)
        mask_nonlocal = (torch.triu(torch.ones_like(dist_cb), diagonal=3) > 0).float()
        
        hp_dist_sum = torch.sum(dist_cb * hp_mat * mask_nonlocal)
        n_hp_pairs = torch.sum(hp_mat * mask_nonlocal) + 1.0
        loss_hydro = hp_dist_sum / n_hp_pairs
        
        cross_mask = (chain_ids.unsqueeze(0) != chain_ids.unsqueeze(1)).float()
        iface_sum = torch.sum(dist_cb * hp_mat * mask_nonlocal * cross_mask * ori_gate)
        iface_pairs = torch.sum(hp_mat * mask_nonlocal * cross_mask) + 1.0
        loss_iface_hydro = iface_sum / iface_pairs
        
        mj_selected = mj_tensor.index_select(0, aa_indices).index_select(1, aa_indices)
        contact_w = torch.exp(-((dist_cb - 6.0)**2) / (2.0 * (2.0**2)))
        mj_mask = ((dist_cb < 10.0) & (dist_cb > 3.0)).float()
        mj_pairs = torch.sum(mj_selected * contact_w * mask_nonlocal * cross_mask * mj_mask * ori_gate)
        mj_count = torch.sum(mask_nonlocal * cross_mask * mj_mask) + 1.0
        loss_mj = - mj_pairs / mj_count
        
        q_mat = full_charge.unsqueeze(0) * full_charge.unsqueeze(1)
        eps = 10.0 + 70.0 * torch.clamp(dist_cb / 12.0, min=0.0, max=1.0)
        elec_pot = q_mat / (eps * (dist_cb + 1e-6))
        elec_pot = elec_pot * torch.exp(-(dist_cb) / 8.0)
        sr_elec_mask = ((dist_cb < 12.0) & (dist_cb > 3.0)).float()
        salt_mask = ((q_mat < 0.0).float() * ((dist_cb < 6.0) & (dist_cb > 3.0)).float())
        elec_pot = elec_pot * (1.0 + 0.5 * salt_mask)
        elec_mask = mask_nonlocal * sr_elec_mask * ((1.0 - cross_mask) + cross_mask * ori_gate)
        loss_elec = torch.sum(elec_pot * elec_mask) * 10.0
        
        cat = (full_charge > 0.4).float()
        arom = full_aromatic
        catpi_bin = torch.clamp(cat.unsqueeze(0) * arom.unsqueeze(1) + cat.unsqueeze(1) * arom.unsqueeze(0), max=1.0)
        pi_mask = ((dist_cb < 6.0) & (dist_cb > 3.0)).float()
        contact_pi = torch.exp(-((dist_cb - 5.0)**2) / (2.0 * (1.5**2)))
        catpi_score = torch.sum(catpi_bin * contact_pi * mask_nonlocal * pi_mask * ((1.0 - cross_mask) + cross_mask * ori_gate))
        catpi_count = torch.sum(catpi_bin * mask_nonlocal * pi_mask) + 1.0
        loss_catpi = - catpi_score / catpi_count
        
        arom_mat = full_aromatic.unsqueeze(0) * full_aromatic.unsqueeze(1)
        dot_norm = torch.sum(u_vec.unsqueeze(1) * u_vec.unsqueeze(0), dim=2)
        parallel_w = torch.exp(-((torch.abs(dot_norm) - 1.0)**2) / 0.2)
        tshape_w = torch.exp(-(dot_norm**2) / 0.2)
        pi_stack_mask = ((dist_cb < 7.0) & (dist_cb > 3.0)).float()
        contact_par = torch.exp(-((dist_cb - 4.5)**2) / (2.0 * (0.8**2)))
        contact_t = torch.exp(-((dist_cb - 5.5)**2) / (2.0 * (1.0**2)))
        pi_par_score = torch.sum(arom_mat * parallel_w * contact_par * mask_nonlocal * pi_stack_mask * ((1.0 - cross_mask) + cross_mask * ori_gate))
        pi_t_score = torch.sum(arom_mat * tshape_w * contact_t * mask_nonlocal * pi_stack_mask * ((1.0 - cross_mask) + cross_mask * ori_gate))
        pi_count = torch.sum(arom_mat * mask_nonlocal * pi_stack_mask) + 1.0
        loss_pipi = - (pi_par_score + 0.7 * pi_t_score) / pi_count
        
        sr_hydro_mask = ((dist_cb < 9.0) & (dist_cb > 3.0)).float()
        r = torch.clamp(dist_cb, min=3.0) + 1e-6
        sigma = 4.0
        epsilon = 0.05
        lj_term = epsilon * ((sigma / r)**12 - (sigma / r)**6)
        loss_lj = torch.sum(lj_term * hp_mat * mask_nonlocal * sr_hydro_mask * ((1.0 - cross_mask) + cross_mask * ori_gate))
        
        diff_ca = full_CA.unsqueeze(0) - full_CA.unsqueeze(1)
        dist_ca = torch.norm(diff_ca, dim=2) + 1e-6
        n_res = len(full_CA)
        idx = torch.arange(n_res, device=device)
        idx_diff = torch.abs(idx.unsqueeze(1) - idx.unsqueeze(0))
        near_mask = ((idx_diff > 2) & (dist_ca < 8.0)).float()
        neighbor_counts = torch.sum(near_mask, dim=1)
        exposure = 1.0 / (1.0 + neighbor_counts)
        loss_burial = torch.sum(full_hydro * exposure)
        loss_polar = torch.sum(full_polar * (1.0 - exposure))
        
        full_H = torch.cat(all_H)
        diff_ho = full_H.unsqueeze(1) - full_O.unsqueeze(0)
        dist_ho = torch.norm(diff_ho, dim=2) + 1e-6
        hb_mask = ((idx_diff > 3) & (dist_ho < HB_MAX_DIST)).float()
        hn_vec = full_H - torch.cat(all_N)
        hn_unit = hn_vec / (torch.norm(hn_vec, dim=1, keepdim=True) + 1e-6)
        ho_unit = diff_ho / (dist_ho.unsqueeze(2))
        cos_ang = torch.sum(hn_unit.unsqueeze(1) * ho_unit, dim=2).clamp(-1.0, 1.0)
        ang = torch.acos(cos_ang)
        dist_w = torch.exp(-((dist_ho - HB_OPT_DIST)**2) / (0.25**2))
        ang_w = torch.exp(-((ang - HB_OPT_ANGLE)**2) / (math.radians(20)**2))
        hb_score = dist_w * ang_w * hb_mask
        hb_energy = -torch.sum(hb_score) * 0.5
        
        # Disulfide Bonds (Cys-Cys)
        # Biological Rule: Cysteines in oxidizing environments tend to form disulfide bridges.
        # We encourage Cys-Cys pairs to be close (~5A CB-CB) if they are reasonably near.
        # This is a soft constraint to guide formation, not a hard bond.
        cys_mask = (full_hydro == 0.0) & (full_polar == 1.0) & (full_charge == 0.0) # Crude mask? No, rely on sequence.
        # Better: derive from sequence explicitly
        cys_indices = torch.nonzero(aa_indices == MJ_INDEX['C']).squeeze()
        loss_disulfide = torch.tensor(0.0, device=device)
        if cys_indices.numel() > 1:
            # Get all pairs
            c_coords = full_CB[cys_indices]
            c_diff = c_coords.unsqueeze(0) - c_coords.unsqueeze(1)
            c_dist = torch.norm(c_diff, dim=2) + 1e-6
            # Mask diagonal
            c_eye = torch.eye(len(cys_indices), device=device)
            # Find potential pairs: distance < 8.0A
            # We want to pull them to ~4.0A (CB-CB ideal for SS bond is ~3.8-4.5A)
            # Use a flat-bottom well or simple harmonic?
            # Biological heuristic: "If close, snap them."
            ss_prox_mask = ((c_dist < 8.0) & (c_dist > 3.0)).float() * (1.0 - c_eye)
            # Potential: (dist - 4.0)^2
            ss_pot = (c_dist - 4.0)**2
            loss_disulfide = torch.sum(ss_pot * ss_prox_mask) * 0.5 # 0.5 for double counting
        
        # Weights (Tuned for Bio-plausibility)
        w_rg = 1.5
        w_rg_tgt = 1.2
        w_clash = 5.0
        w_heavy = 4.5
        w_ss = 2.0
        w_rama = 4.0 # Boosted: Biological backbone preference is strong
        w_hb = 4.0
        w_hydro = 3.0 # Slightly reduced, rely more on burial/MJ
        w_iface_hydro = 2.5
        w_elec = 1.5
        w_catpi = 1.0
        w_pipi = 1.0
        w_lj = 0.5
        w_burial = 4.0
        w_polar = 2.0
        w_smooth = 0.5
        w_mj = 4.0 # Boosted: Statistical potential reflects natural selection
        w_disulfide = 5.0 # Strong bias for SS bonds
        
        # Universal Constraints
        loss_constraints = torch.tensor(0.0, device=device)
        if constraints:
            for c in constraints:
                try:
                    c_type = c.get("type")
                    strength = float(c.get("strength", 5.0))
                    indices = c.get("indices", [])
                    if not indices: continue
                    
                    # Filter indices to valid range
                    valid_indices = [idx for idx in indices if idx < len(full_CB)]
                    if not valid_indices: continue
                    
                    t_indices = torch.tensor(valid_indices, device=device, dtype=torch.long)
                    
                    if c_type == "distance_point":
                        # Logic: Pull selected residues towards their common centroid (simulating coordination center)
                        target_residues = full_CB[t_indices]
                        center = torch.mean(target_residues, dim=0)
                        dists = torch.norm(target_residues - center, dim=1)
                        target_dist = float(c.get("distance", 3.0)) 
                        loss_constraints = loss_constraints + strength * torch.sum((dists - target_dist)**2)
                        
                    elif c_type == "pocket_preservation":
                        target_residues = full_CB[t_indices]
                        center = torch.mean(target_residues, dim=0)
                        rg_sq = torch.sum((target_residues - center)**2) / len(target_residues)
                        target_radius = float(c.get("radius", 10.0))
                        loss_constraints = loss_constraints + strength * (torch.sqrt(rg_sq + 1e-6) - target_radius)**2
                        
                    elif c_type == "surface_exposure":
                        is_exposed = c.get("exposed", True)
                        target_vals = exposure[t_indices]
                        if is_exposed:
                            loss_constraints = loss_constraints + strength * torch.sum((1.0 - target_vals)**2)
                        else:
                            loss_constraints = loss_constraints + strength * torch.sum(target_vals**2)
                            
                    elif c_type == "interface_geometry":
                        geo_type = c.get("geometry_type", "planar")
                        if geo_type == "planar":
                            target_residues = full_CB[t_indices]
                            if len(target_residues) > 3:
                                center = torch.mean(target_residues, dim=0)
                                centered = target_residues - center
                                cov = torch.mm(centered.t(), centered)
                                try:
                                    eigs = torch.linalg.eigvalsh(cov)
                                    min_eig = eigs[0] 
                                    loss_constraints = loss_constraints + strength * min_eig
                                except:
                                    pass
                except Exception:
                    pass

        if num_chains == 1:
            w_iface_hydro = 0.0
            w_mj = 2.0 # Single chain relies more on internal packing
            w_elec = 1.2
            w_rg_tgt = 1.5
        else:
            w_iface_hydro = 3.5
            w_mj = 5.0 # Multi-chain: MJ is critical for docking specificity
            w_elec = 2.0
            w_rg_tgt = 1.0
            w_catpi = 1.5
            w_pipi = 1.5
            w_lj = 1.2
        
        total_loss = (w_rg * rg2 + 
                      w_rg_tgt * loss_rg_target +
                      w_clash * clash_loss +
                      w_heavy * heavy_loss +
                      w_ss * loss_ss + 
                      w_rama * loss_rama + 
                      w_hb * hb_energy + 
                      w_hydro * loss_hydro + 
                      w_iface_hydro * loss_iface_hydro +
                      w_elec * loss_elec +
                      w_catpi * loss_catpi +
                      w_pipi * loss_pipi +
                      w_lj * loss_lj +
                      w_burial * loss_burial +
                      w_polar * loss_polar +
                      w_smooth * loss_smooth +
                      w_mj * loss_mj + 
                      w_disulfide * loss_disulfide)
                      
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
                
                try:
                    if frame_counter > 20 and (frame_counter % 20) == 1:
                        threshold = frame_counter - 1
                        for f in os.listdir(frames_dir):
                            if f.startswith("frame_") and f.endswith(".pdb"):
                                try:
                                    idx = int(f[6:].split(".")[0])
                                except Exception:
                                    idx = None
                                if idx is not None and idx <= threshold:
                                    try:
                                        os.remove(os.path.join(frames_dir, f))
                                    except Exception:
                                        pass
                except Exception:
                    pass
                    
        except Exception as e:
            logger.warning(f"Snapshot failed: {e}")

    save_snapshot("init")
    
    # Bio-Algorithm Initialization Strategy
    # Replaces pure random starts with biologically motivated heuristics.
    # 1. Consensus: Ideal Secondary Structure + Geometric Centering
    # 2. Hydrophobic Collapse: Aligns Hydrophobic Centers of Mass (Simulating the "Fold" event)
    # 3. Ramachandran Sampling: Samples backbone angles from PDB statistics (Natural Variation)
    
    strategies = ["bio_hydro_collapse", "bio_consensus", "bio_rama_sampling"]
    
    # Check for fast mode env var
    if os.environ.get("MINIFOLD_FAST_MODE", "0") == "1":
        strategies = ["bio_hydro_collapse"]
        logger.info("Fast Mode Enabled: Running 'Hydrophobic Collapse' strategy only.")

    num_starts = len(strategies)

    with torch.no_grad():
        best_loss = None
        best_phi = None
        best_psi = None
        best_rb = None
    
    # Helper to calculate centers on the fly
    def get_chain_props(d, p_phi, p_psi):
        # Build temp structure
        t_N, t_CA, t_C, _, _, _ = build_full_structure_tensor(d["seq"], p_phi, p_psi, device)
        # Geo Center
        geo_c = torch.mean(t_CA, dim=0)
        # Hydro Center
        if "hydro_mask" in d:
            w = d["hydro_mask"].unsqueeze(1) + 0.01
            hydro_c = torch.sum(t_CA * w, dim=0) / torch.sum(w)
        else:
            hydro_c = geo_c
        return geo_c, hydro_c

    for s, strategy in enumerate(strategies):
        logger.info(f"  > Start {s+1}/{num_starts}: Strategy '{strategy}'")
        
        with torch.no_grad():
            # 1. Initialize Angles (Phi/Psi)
            for i, d in enumerate(chain_data):
                if strategy == "bio_rama_sampling":
                    # Sample from Ramachandran Distributions
                    # Use simple gaussian sampling based on RAMA_PREF
                    new_phi = d["phi_ref"].clone()
                    new_psi = d["psi_ref"].clone()
                    
                    # Apply noise based on residue type (G, P, General)
                    for j, aa in enumerate(d["seq"]):
                        rtype = "General"
                        if aa == "G": rtype = "GLY"
                        elif aa == "P": rtype = "PRO"
                        
                        centers = RAMA_PREF[rtype]
                        sigmas = RAMA_SIGMA[rtype]
                        
                        # Pick a random center
                        c_idx = torch.randint(0, len(centers), (1,)).item()
                        c_phi, c_psi = centers[c_idx]
                        
                        # Sample
                        # Convert ref to degrees for intuition? No, stick to radians.
                        # Centers in RAMA_PREF are radians.
                        
                        # We mix the "Ideal SS" knowledge with "Rama Preference"
                        # If SS says Helix, we prefer Helix region.
                        # But here we just add biological noise to the SS prediction.
                        noise_phi = torch.randn(1, device=device) * sigmas[0] * 0.5
                        noise_psi = torch.randn(1, device=device) * sigmas[1] * 0.5
                        
                        new_phi[j] += noise_phi
                        new_psi[j] += noise_psi
                        
                    phi_params[i].copy_(new_phi)
                    psi_params[i].copy_(new_psi)
                else:
                    # Consensus / Hydrophobic: Use Ideal SS (Clean Start)
                    # Small noise to break symmetry
                    phi_params[i].copy_(d["phi_ref"] + (torch.rand_like(d["phi_ref"]) - 0.5) * 0.05)
                    psi_params[i].copy_(d["psi_ref"] + (torch.rand_like(d["psi_ref"]) - 0.5) * 0.05)

            # 2. Initialize Rigid Body (Multi-chain)
            if rb_params is not None:
                init_rb = []
                # First chain (idx 0) is fixed at origin.
                # Calculate its props.
                c0_geo, c0_hydro = get_chain_props(chain_data[0], phi_params[0], psi_params[0])
                
                for k in range(1, len(chain_data)):
                    # Calculate current chain props at origin
                    ck_geo, ck_hydro = get_chain_props(chain_data[k], phi_params[k], psi_params[k])
                    
                    if strategy == "bio_hydro_collapse":
                        # Align Hydrophobic Centers
                        # Target: c0_hydro
                        # Current: ck_hydro (assuming no rot/trans yet)
                        # Translation = Target - Current
                        # Add a small offset so they don't clash immediately
                        offset = (torch.rand(3, device=device) - 0.5) * 5.0
                        t = (c0_hydro - ck_hydro) + offset
                        
                        # Rotation: Random is okay, or face-to-face?
                        # Let's use random for now, docking will fix it.
                        r = (torch.rand(3, device=device) * 2 * math.pi) - math.pi
                        init_rb.append(torch.cat([t, r]))
                        
                    elif strategy == "bio_consensus":
                        # Geometric proximity
                        # Place nearby but distinct
                        t = c0_geo - ck_geo + torch.tensor([10.0 * k, 0.0, 0.0], device=device)
                        r = torch.zeros(3, device=device)
                        init_rb.append(torch.cat([t, r]))
                        
                    else: # bio_rama_sampling
                        # Random exploration
                        t = (torch.rand(3, device=device) * 40.0) - 20.0
                        r = (torch.rand(3, device=device) * 2 * math.pi) - math.pi
                        init_rb.append(torch.cat([t, r]))
                        
                rb_params.copy_(torch.stack(init_rb))

        logger.info(f"  > Start {s+1}/{num_starts}: Adam")
        opt_adam = torch.optim.Adam(all_params, lr=0.05)
        for step in range(50): # Reduced from 60
            opt_adam.zero_grad()
            loss = calc_loss()
            loss.backward()
            opt_adam.step()
            if step % 25 == 0 or step == 49: # Snapshot every 25 steps and last
                save_snapshot(f"adam_{s+1}_{step+1}")
        logger.info(f"  > Start {s+1}/{num_starts}: LBFGS")
        opt_lbfgs = torch.optim.LBFGS(all_params, max_iter=60, tolerance_grad=1e-5, tolerance_change=1e-5, history_size=20, line_search_fn="strong_wolfe") # Reduced from 80
        lbfgs_step_count = 0
        def closure():
            nonlocal lbfgs_step_count
            opt_lbfgs.zero_grad()
            l = calc_loss()
            l.backward()
            lbfgs_step_count += 1
            if lbfgs_step_count % 25 == 0: # Snapshot every 25 evaluations
                save_snapshot(f"lbfgs_{s+1}_{lbfgs_step_count}")
            return l
        try:
            opt_lbfgs.step(closure)
        except Exception as e:
            logger.warning(f"Optimization step failed: {e}")
        with torch.no_grad():
            cur_loss = calc_loss().item()
            if (best_loss is None) or (cur_loss < best_loss):
                best_loss = cur_loss
                best_phi = [p.detach().clone() for p in phi_params]
                best_psi = [q.detach().clone() for q in psi_params]
                best_rb = rb_params.detach().clone() if rb_params is not None else None
    with torch.no_grad():
        if best_phi is not None and best_psi is not None:
            for i in range(len(phi_params)):
                phi_params[i].copy_(best_phi[i])
                psi_params[i].copy_(best_psi[i])
        if rb_params is not None and best_rb is not None:
            rb_params.copy_(best_rb)
        
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
    remarks = None
    try:
        phi_all = []
        psi_all = []
        for i, d in enumerate(chain_data):
            phi_all.extend(list(phi_params[i].detach().cpu().numpy()))
            psi_all.extend(list(psi_params[i].detach().cpu().numpy()))
        mj_mat = np.array(MJ_VALUES, dtype=float)
        rp = rama_pass_rate(phi_all, psi_all, sequence)
        ce = contact_energy(final_CA, final_CA, sequence, mj_mat)
        tm = tm_score_proxy(final_CA)
        remarks = {"RAMA_PASS_RATE": f"{rp:.3f}", "CONTACT_ENERGY": f"{ce:.3f}", "TM_SCORE_PROXY": f"{tm:.3f}"}
        out_json = os.path.splitext(output_pdb)[0] + ".metrics.json"
        write_results_json(out_json, {"ramachandran_pass_rate": rp, "contact_energy": ce, "tm_score_proxy": tm, "elapsed_seconds": elapsed})
    except Exception:
        remarks = None
    ok = write_pdb(sequence, final_N, final_CA, final_C, output_pdb, chain_breaks=breaks, remarks=remarks)
    try:
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir, ignore_errors=True)
    except Exception:
        pass
    return ok

from modules.sidechain_builder import build_sidechain

def write_pdb(sequence, N, CA, C, out_path, chain_breaks=None, remarks=None):
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
        
        if remarks:
            for k, v in remarks.items():
                try:
                    f.write(f"REMARK {k}: {v}\n")
                except Exception:
                    pass
        env_atoms = []
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
                env_atoms.append(O_pos)
            except:
                pass
            
            # Sidechain
            if aa != "G":
                # Use sidechain_builder's pack_sidechain to pick best rotamer if possible
                # Or just build with default (first) one for now
                from modules.sidechain_builder import pack_sidechain
                sc_atoms = pack_sidechain(aa, N[i], CA[i], C[i], env_atoms)
                
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
                    env_atoms.append(pos)
            
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
