import math
import logging
import platform
import time
import numpy as np
import torch
try:
    import cpuinfo
except Exception:
    cpuinfo = None
import os
import shutil
import warnings
import subprocess
# Suppress DirectML lerp warning which falls back to CPU (performance warning, not fatal)
warnings.filterwarnings("ignore", message=".*aten::lerp.*")

from modules.quality import rama_pass_rate, contact_energy, tm_score_proxy, write_results_json
from modules.env_loader import load_env
try:
    import torch_directml
    _HAS_DML = True
except Exception:
    _HAS_DML = False

_IPEX = None

def _load_ipex():
    global _IPEX
    if _IPEX is not None:
        return _IPEX, None
    try:
        def _bootstrap_oneapi_env_local():
            try:
                load_env()
            except Exception:
                pass
            try:
                root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                load_env(os.path.join(root, ".env.oneapi"))
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
            except Exception:
                pass
        _bootstrap_oneapi_env_local()
        import intel_extension_for_pytorch as ipex
        _IPEX = ipex
        return _IPEX, None
    except (Exception, SystemExit, OSError) as e:
        return None, e


# Setup logging
logger = logging.getLogger(__name__)
ALGO_VERSION = "igpu_dml_v2.4_phys_mj_pi"

# Precompute bond parameters (PyTorch tensors)
# Engh & Huber 1991
BOND_PARAMS = {
    "n_len": 1.329,
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

OMEGA_SIGMA = {
    "TRANS": math.radians(5.0),
    "PREPRO": math.radians(5.0),
}

OMEGA_CIS_BIAS = 3.0

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

def check_gpu_availability(preferred_backend="auto"):
    """
    Checks for available hardware acceleration devices.
    preferred_backend: "auto", "ipex", "directml", "cuda", "mps", "cpu"
    """
    if preferred_backend in ["cpu", "oneapi_cpu"]:
        name = "Intel oneAPI CPU" if preferred_backend == "oneapi_cpu" else "Forced CPU"
        return False, name, "CPU"

    # 1. IPEX XPU
    if preferred_backend in ["auto", "ipex"]:
        ipex_mod, ipex_err = _load_ipex()
        if ipex_mod is not None:
            try:
                if hasattr(torch, "xpu") and torch.xpu.is_available():
                    name = torch.xpu.get_device_name(0)
                    return True, f"{name} (via IPEX)", "Intel XPU"
            except Exception:
                pass
        elif preferred_backend == "ipex" and ipex_err is not None:
            return False, f"IPEX load failed: {ipex_err}", "CPU"
            
    # 2. DirectML
    if preferred_backend in ["auto", "directml"] and _HAS_DML:
        try:
            dml_dev = torch_directml.device()
            # Verify basic tensor op
            _ = torch.tensor([1.0], device=dml_dev)
            return True, "DirectML (Intel/AMD/NVIDIA/Qualcomm)", "DirectML"
        except Exception as e:
            logger.warning(f"DirectML available but failed init: {e}")

    # 3. CUDA
    if preferred_backend in ["auto", "cuda"] and torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "NVIDIA CUDA"
        return True, name, "NVIDIA Tensor Core"
        
    # 4. MPS
    if preferred_backend in ["auto", "mps"] and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return True, "Apple Neural Engine (MPS)", "Apple NPU"
        
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

def ss_to_phi_psi_tensor(ss, sequence=None):
    """
    Initializes Phi/Psi angles based on Secondary Structure (SS) with DISCRETE SAMPLING from allowed Ramachandran regions.
    
    Allowed Regions (Approximate centers & widths):
    - Alpha Helix (H): phi ~ -60 (-70 to -50), psi ~ -47 (-60 to -35)
    - Beta Sheet (E): phi ~ -120 (-140 to -100), psi ~ 120 (100 to 140)
    - Coil (C): Sample from general allowed regions (Alpha, Beta, PolyProII) to avoid forbidden zones.
      Common Coil regions:
      1. Alpha-like: (-60, -45)
      2. Beta-like: (-120, 120) or (-135, 135)
      3. PolyProII (PPII): (-75, 145)
    """
    phi = []
    psi = []

    seq = sequence if sequence is not None else ("A" * len(ss))
    for i, c in enumerate(ss):
        aa = seq[i] if i < len(seq) else "A"
        if c == "H":
            p = math.radians(-60.0) + np.random.uniform(-1.0, 1.0) * math.radians(10.0)
            q = math.radians(-47.0) + np.random.uniform(-1.0, 1.0) * math.radians(10.0)
        elif c == "E":
            p = math.radians(-120.0) + np.random.uniform(-1.0, 1.0) * math.radians(20.0)
            q = math.radians(120.0) + np.random.uniform(-1.0, 1.0) * math.radians(20.0)
        else:
            grp = "GLY" if aa == "G" else ("PRO" if aa == "P" else "General")
            centers = RAMA_PREF.get(grp, RAMA_PREF["General"])
            sig = RAMA_SIGMA.get(grp, RAMA_SIGMA["General"])
            if grp == "PRO" and len(centers) >= 2:
                k = 0 if (np.random.rand() < 0.7) else 1
            elif grp == "General" and len(centers) >= 2:
                k = 1 if (np.random.rand() < 0.6) else 0
            else:
                k = int(np.random.randint(0, len(centers)))
            cp, cq = centers[k]
            p = float(cp) + np.random.normal(0.0, float(sig[0]))
            q = float(cq) + np.random.normal(0.0, float(sig[1]))

        if aa == "P":
            p = math.radians(-65.0) + np.random.normal(0.0, math.radians(7.0))

        phi.append(float(p))
        psi.append(float(q))

    return torch.tensor(phi, dtype=torch.float32), torch.tensor(psi, dtype=torch.float32)

def _compute_ss_probs(seq):
    seq = seq.upper()
    H = []
    E = []
    C = []
    helix_idx = {c: v for c, v in zip("ACDEFGHIKLMNPQRSTVWY",
        [1.2,0.8,1.0,0.9,1.1,1.0,0.7,1.1,1.2,1.2,1.2,1.1,0.7,1.0,0.6,0.9,0.9,0.8,1.0,1.2])}
    sheet_idx = {c: v for c, v in zip("ACDEFGHIKLMNPQRSTVWY",
        [0.8,0.9,0.9,1.1,1.0,0.9,1.1,0.7,0.8,1.3,1.3,0.8,1.0,1.2,0.6,0.9,0.9,1.3,1.2,1.2])}
    hydro_idx = {c: v for c, v in zip("ACDEFGHIKLMNPQRSTVWY",
        [1.8,-4.5,-3.5,-3.5,2.5,-3.5,-3.5,-0.4,-3.2,4.5,3.8,-3.9,1.9,-1.6,-1.6,-0.8,-0.7,-0.9,-1.3,4.2])}
    for ch in seq:
        h = helix_idx.get(ch, 0.8)
        e = sheet_idx.get(ch, 0.8)
        hp = max(0.0, hydro_idx.get(ch, 0.0))
        h += hp * 0.1
        e += 0.0
        c = max(0.0, 1.0 - (h + e) * 0.3)
        if ch in ("P", "G"):
            h *= 0.6
            e *= 0.8
            c += 0.3
        H.append(h)
        E.append(e)
        C.append(c)
    Ht = torch.tensor(H, dtype=torch.float32)
    Et = torch.tensor(E, dtype=torch.float32)
    Ct = torch.tensor(C, dtype=torch.float32)
    S = Ht + Et + Ct + 1e-9
    pH = Ht / S
    pE = Et / S
    pC = Ct / S
    return torch.stack([pH, pE, pC], dim=1)

def place_atom_tensor(a, b, c, bond_len, bond_angle, dihedral, basis_x=None, basis_y=None):
    bc = c - b
    ab = b - a
    bc_l = torch.norm(bc) + 1e-9
    bc_u = bc / bc_l
    n = torch.linalg.cross(ab, bc_u)
    n_l = torch.norm(n)

    if basis_x is None:
        basis_x = torch.tensor([1.0, 0.0, 0.0], device=a.device, dtype=a.dtype)
    if basis_y is None:
        basis_y = torch.tensor([0.0, 1.0, 0.0], device=a.device, dtype=a.dtype)

    arb = torch.where(torch.abs(bc_u[0]) < 0.9, basis_x, basis_y)
    n_alt = torch.linalg.cross(arb, bc_u)
    n_alt_l = torch.norm(n_alt)

    collinear = n_l < 1e-6
    n = torch.where(collinear, n_alt, n)
    n_l = torch.where(collinear, n_alt_l, n_l)

    n = n / (n_l + 1e-9)
    nb = torch.linalg.cross(n, bc_u)
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

def _get_bond_constants(device):
    p = BOND_PARAMS
    d = {}
    d["n_len"] = torch.tensor(p["n_len"], device=device)
    d["ca_len"] = torch.tensor(p["ca_len"], device=device)
    d["c_len"] = torch.tensor(p["c_len"], device=device)
    d["ang_C_N_CA"] = torch.tensor(p["ang_C_N_CA"], device=device)
    d["ang_N_CA_C"] = torch.tensor(p["ang_N_CA_C"], device=device)
    d["ang_CA_C_N"] = torch.tensor(p["ang_CA_C_N"], device=device)
    d["zero"] = torch.tensor(0.0, device=device)
    d["omega"] = torch.tensor(math.pi, device=device)
    d["bond_NH"] = torch.tensor(1.01, device=device)
    d["bond_CO"] = torch.tensor(1.229, device=device)
    d["bond_CB"] = torch.tensor(1.53, device=device)
    
    d["rad_120_8"] = torch.tensor(math.radians(120.8), device=device)
    d["rad_110_5"] = torch.tensor(math.radians(110.5), device=device)
    d["rad_neg_122_5"] = torch.tensor(math.radians(-122.5), device=device)
    d["rad_122_55"] = torch.tensor(math.radians(122.55), device=device)
    d["rad_119_5"] = torch.tensor(math.radians(119.5), device=device)
    d["pi"] = torch.tensor(math.pi, device=device)
    d["neg_47"] = torch.tensor(math.radians(-47.0), device=device)
    
    d["init_N"] = torch.tensor([0.0, 0.0, 0.0], device=device)
    d["init_CA"] = torch.tensor([p["ca_len"], 0.0, 0.0], device=device)
    d["prev_virtual"] = torch.tensor([-0.5 * p["c_len"], 0.866 * p["c_len"], 0.0], device=device)
    d["dummy_H"] = torch.tensor([0.0, 0.0, 0.0], device=device)
    d["basis_x"] = torch.tensor([1.0, 0.0, 0.0], device=device)
    d["basis_y"] = torch.tensor([0.0, 1.0, 0.0], device=device)
    return d

def _carbonyl_oxygen_tensor(Ci, CAi, Nnext, bond_len):
    v1 = CAi - Ci
    v2 = Nnext - Ci
    u1 = v1 / (torch.norm(v1) + 1e-9)
    u2 = v2 / (torch.norm(v2) + 1e-9)
    d = -(u1 + u2)
    dn = torch.norm(d)
    d = torch.where(dn > 1e-6, d / (dn + 1e-9), -u1)
    return Ci + d * bond_len

def build_full_structure_tensor(seq_len, phi, psi, device, bond_constants=None, omega=None):
    """
    Builds Backbone + CB (Centroid) + H (Amide) + O (Carbonyl).
    Optimized to use precomputed bond constants to avoid CPU-GPU overhead.
    """
    if bond_constants is None:
        bc = _get_bond_constants(device)
    else:
        bc = bond_constants
        
    n_len = bc["n_len"]
    ca_len = bc["ca_len"]
    c_len = bc["c_len"]
    ang_C_N_CA = bc["ang_C_N_CA"]
    ang_N_CA_C = bc["ang_N_CA_C"]
    ang_CA_C_N = bc["ang_CA_C_N"]
    
    # Init
    N = bc["init_N"]
    CA = bc["init_CA"]
    prev_virtual = bc["prev_virtual"]
    
    C = place_atom_tensor(prev_virtual, N, CA, c_len, ang_N_CA_C, bc["zero"])
    
    coords_N = torch.empty((seq_len, 3), device=device, dtype=N.dtype)
    coords_CA = torch.empty((seq_len, 3), device=device, dtype=N.dtype)
    coords_C = torch.empty((seq_len, 3), device=device, dtype=N.dtype)
    coords_O = torch.empty((seq_len, 3), device=device, dtype=N.dtype)
    coords_H = torch.empty((seq_len, 3), device=device, dtype=N.dtype)
    coords_CB = torch.empty((seq_len, 3), device=device, dtype=N.dtype)
    
    omega_const = bc["omega"]
    bond_NH = bc["bond_NH"]
    bond_CO = bc["bond_CO"]
    bond_CB = bc["bond_CB"]
    
    coords_N[0] = N
    coords_CA[0] = CA
    coords_C[0] = C
    coords_H[0] = bc["dummy_H"]
    
    # Place first CB
    basis_x = bc.get("basis_x")
    basis_y = bc.get("basis_y")
    CB0 = place_atom_tensor(N, CA, C, bond_CB, bc["rad_110_5"], bc["rad_neg_122_5"], basis_x=basis_x, basis_y=basis_y)
    coords_CB[0] = CB0

    for i in range(1, seq_len):
        Ni = place_atom_tensor(coords_N[i-1], coords_CA[i-1], coords_C[i-1], n_len, ang_CA_C_N, psi[i-1], basis_x=basis_x, basis_y=basis_y)
        om = omega_const if omega is None else omega[i - 1]
        CAi = place_atom_tensor(coords_CA[i-1], coords_C[i-1], Ni, ca_len, ang_C_N_CA, om, basis_x=basis_x, basis_y=basis_y)
        Ci = place_atom_tensor(coords_C[i-1], Ni, CAi, c_len, ang_N_CA_C, phi[i], basis_x=basis_x, basis_y=basis_y)

        coords_O[i - 1] = _carbonyl_oxygen_tensor(coords_C[i-1], coords_CA[i-1], Ni, bond_CO)
        nrm = torch.linalg.cross(coords_CA[i-1] - coords_N[i-1], coords_C[i-1] - coords_CA[i-1])
        nrm = nrm / (torch.norm(nrm) + 1e-9)
        dproj = torch.sum((coords_O[i - 1] - coords_N[i-1]) * nrm)
        coords_O[i - 1] = coords_O[i - 1] - dproj * nrm

        coords_N[i] = Ni
        coords_CA[i] = CAi
        coords_C[i] = Ci

        coords_H[i] = place_atom_tensor(coords_C[i-1], Ni, CAi, bond_NH, bc["rad_119_5"], bc["pi"], basis_x=basis_x, basis_y=basis_y)
        coords_CB[i] = place_atom_tensor(Ni, CAi, Ci, bond_CB, bc["rad_110_5"], bc["rad_122_55"], basis_x=basis_x, basis_y=basis_y)

    if seq_len > 0:
        last_idx = seq_len - 1
        coords_O[last_idx] = _carbonyl_oxygen_tensor(coords_C[last_idx], coords_CA[last_idx], coords_N[last_idx], bond_CO)
        nrm = torch.linalg.cross(coords_CA[last_idx] - coords_N[last_idx], coords_C[last_idx] - coords_CA[last_idx])
        nrm = nrm / (torch.norm(nrm) + 1e-9)
        dproj = torch.sum((coords_O[last_idx] - coords_N[last_idx]) * nrm)
        coords_O[last_idx] = coords_O[last_idx] - dproj * nrm

    return coords_N, coords_CA, coords_C, coords_O, coords_H, coords_CB

def _get_cpu_threads():
    try:
        val = os.environ.get("ONEAPI_CPU_THREADS") or os.environ.get("MINIFOLD_CPU_THREADS")
        if val:
            return max(1, int(val))
    except Exception:
        pass
    try:
        return max(1, os.cpu_count() or 1)
    except Exception:
        return 1

def optimize_from_ss_gpu(sequence, chain_ss_list, output_pdb, constraints=None, backend="auto"):
    """
    Advanced Full-Atom Joint Optimization Pipeline.
    """
    is_gpu, device_name, device_type = check_gpu_availability(preferred_backend=backend)
    logger.info(f"iGPU Optimization Enabled [{ALGO_VERSION}]. Device: {device_name} ({device_type})")
    
    device = torch.device('cpu')
    use_ipex = False
    ipex_mod = None
    if backend == "oneapi_cpu":
        try:
            th = _get_cpu_threads()
            try:
                torch.set_num_threads(th)
            except Exception:
                pass
            try:
                if hasattr(torch, "set_num_interop_threads"):
                    torch.set_num_interop_threads(max(1, th // 2))
            except Exception:
                pass
            os.environ["OMP_NUM_THREADS"] = str(th)
            os.environ["MKL_NUM_THREADS"] = str(th)
        except Exception:
            pass
    
    if is_gpu:
        if device_type == "Intel XPU":
            ipex_mod, ipex_err = _load_ipex()
            if ipex_mod is not None:
                device = torch.device("xpu")
                use_ipex = True
            else:
                logger.warning(f"IPEX load failed: {ipex_err}")
        elif device_type == "DirectML":
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
        L_ss = len(ss)
        sub_seq = sequence[start_idx : start_idx + L_ss]
        L_seq = len(sub_seq)
        if L_seq < L_ss:
            ss = ss[:L_seq]
        p, q = ss_to_phi_psi_tensor(ss, sub_seq)
        ss_probs = _compute_ss_probs(sub_seq).to(device)
        w_tensor = (ss_probs[:,0] + ss_probs[:,1]).clamp(0.0, 1.0)
        
        hydro_indices = []
        charge_vals = []
        
        HP_MAP = {
            'I': 1.0, 'V': 0.9, 'L': 0.8, 'F': 0.6, 'M': 0.4, 'A': 0.4
        }
        
        CHARGE_MAP = {'D': -1.0, 'E': -1.0, 'K': 1.0, 'R': 1.0, 'H': 0.5}
        POLAR_SET = set(['D','E','K','R','H','N','Q','S','T','Y','C','W'])
        AROMATIC_SET = set(['F','Y','W'])
        
        COLLAPSE_HYDRO_SET = set(['I', 'L', 'V', 'F', 'M'])
        CHARGED_CLASH_SET = set(['D', 'E', 'K', 'R'])

        h_idx_list = []
        c_val_list = []
        polar_list = []
        arom_list = []
        
        col_hydro_list = []
        chg_clash_list = []

        for aa in sub_seq:
            h_idx_list.append(HP_MAP.get(aa, 0.0))
            c_val_list.append(CHARGE_MAP.get(aa, 0.0))
            polar_list.append(1.0 if aa in POLAR_SET else 0.0)
            arom_list.append(1.0 if aa in AROMATIC_SET else 0.0)
            
            col_hydro_list.append(1.0 if aa in COLLAPSE_HYDRO_SET else 0.0)
            chg_clash_list.append(1.0 if aa in CHARGED_CLASH_SET else 0.0)
            
        hydro_mask = torch.tensor(h_idx_list, device=device)
        charge = torch.tensor(c_val_list, device=device)
        polar_mask = torch.tensor(polar_list, device=device)
        aromatic_mask = torch.tensor(arom_list, device=device)
        
        collapse_hydro_mask = torch.tensor(col_hydro_list, device=device)
        charged_clash_mask = torch.tensor(chg_clash_list, device=device)
        
        rama_group_ids = torch.tensor(
            [1 if a == "G" else (2 if a == "P" else 0) for a in sub_seq],
            device=device,
            dtype=torch.long,
        )
        pos_types = []
        if len(ss) > 0:
            start = 0
            while start < len(ss):
                c = ss[start]
                end = start + 1
                while end < len(ss) and ss[end] == c:
                    end += 1
                length = end - start
                for k in range(length):
                    if c == "H":
                        if length <= 2:
                            pos_types.append(1)
                        else:
                            if k == 0 or k == length - 1:
                                pos_types.append(2)
                            else:
                                pos_types.append(1)
                    elif c == "E":
                        if length <= 2:
                            pos_types.append(3)
                        else:
                            if k == 0 or k == length - 1:
                                pos_types.append(4)
                            else:
                                pos_types.append(3)
                    else:
                        pos_types.append(0)
                start = end
        if len(pos_types) < len(sub_seq):
            pos_types.extend([0] * (len(sub_seq) - len(pos_types)))
        pos_type_tensor = torch.tensor(pos_types, device=device, dtype=torch.long)

        chain_data.append({
            "seq": sub_seq,
            "ss": ss,
            "phi_ref": p.to(device),
            "psi_ref": q.to(device),
            "ss_weights": w_tensor,
            "ss_probs": ss_probs,
            "hydro_mask": hydro_mask,
            "charge": charge,
            "polar_mask": polar_mask,
            "aromatic_mask": aromatic_mask,
            "collapse_hydro_mask": collapse_hydro_mask,
            "charged_clash_mask": charged_clash_mask,
            "rama_group_ids": rama_group_ids,
            "prepro_mask": torch.tensor([1.0 if (j + 1 < len(sub_seq) and sub_seq[j + 1] == "P") else 0.0 for j in range(len(sub_seq))], device=device, dtype=torch.float32),
            "pos_type": pos_type_tensor,
        })
        
        start_idx += L_seq
        
    used_sequence = "".join([d["seq"] for d in chain_data])
    num_chains = len(chain_data)
    
    # Initialize Parameters
    phi_params = []
    psi_params = []
    omega_params = []
    
    for d in chain_data:
        p0 = d["phi_ref"] + (torch.rand_like(d["phi_ref"]) * 0.1)
        q0 = d["psi_ref"] + (torch.rand_like(d["psi_ref"]) * 0.1)
        phi_params.append(p0.requires_grad_(True))
        psi_params.append(q0.requires_grad_(True))
        o0 = torch.full_like(d["phi_ref"], math.pi)
        prepro_mask = d.get("prepro_mask")
        if prepro_mask is not None and prepro_mask.numel() == o0.numel():
            cis_sample = (torch.rand_like(o0) < 0.05) & (prepro_mask > 0.5)
            o0 = torch.where(cis_sample, torch.zeros_like(o0), o0)
        omega_params.append(o0.requires_grad_(True))
        
    rb_params = None
    if len(chain_data) > 1:
        init_rb = []
        for _ in range(len(chain_data) - 1):
            t = (torch.rand(3, device=device) * 40.0) - 20.0
            r = (torch.rand(3, device=device) * 2 * math.pi) - math.pi
            init_rb.append(torch.cat([t, r]))
        rb_params = torch.stack(init_rb).requires_grad_(True)
        
    all_params = phi_params + psi_params + omega_params
    if rb_params is not None:
        all_params.append(rb_params)
        
    # LBFGS is powerful but can be slow per step. 
    # For initial convergence, we can use Adam, then refine with LBFGS.
    # Or just optimize LBFGS settings.
    # Reducing history size and relaxing tolerance slightly can help speed without hurting final structure much.
    
    # Strategy: Two-stage optimization
    # Stage 1: Adam (Fast coarse folding)
    # Stage 2: LBFGS (Fine refinement)
    
    # Enable IPEX optimization if available (Intel Arc/Ultra)
    if use_ipex:
        try:
            logger.info("Applying Intel IPEX optimizations (BFloat16/Float16 weights)...")
            # Convert parameters to optimize format if needed, though functional approach makes it tricky.
            # IPEX usually optimizes modules. For functional tensors, we focus on XPU device backend.
            # However, we can use torch.xpu.optimize if available in newer IPEX versions, or just rely on XPU backend.
            # We can try to use ipex.optimize on the optimizer if we had a module.
            pass
        except Exception as e:
            logger.warning(f"IPEX optimization warning: {e}")

    rama_groups = ["General", "GLY", "PRO"]
    rama_group_index = {k: i for i, k in enumerate(rama_groups)}
    rama_kmax = max(len(RAMA_PREF[g]) for g in rama_groups)
    rama_centers_pad = torch.zeros((len(rama_groups), rama_kmax, 2), device=device, dtype=torch.float32)
    rama_valid_pad = torch.zeros((len(rama_groups), rama_kmax), device=device, dtype=torch.float32)
    rama_sigma = torch.zeros((len(rama_groups), 2), device=device, dtype=torch.float32)
    for g in rama_groups:
        gi = rama_group_index[g]
        centers = torch.tensor(RAMA_PREF[g], device=device, dtype=torch.float32)
        k = centers.shape[0]
        rama_centers_pad[gi, :k, :] = centers
        rama_valid_pad[gi, :k] = 1.0
        rama_sigma[gi, :] = torch.tensor(RAMA_SIGMA[g], device=device, dtype=torch.float32)
    
    def _apply_rama_codebook(phi_raw, psi_raw, seq, device_inner):
        phi_eff = phi_raw.clone()
        psi_eff = psi_raw.clone()
        if not seq:
            return phi_eff, psi_eff
        mask_vit = torch.tensor([aa in ("V", "I", "T") for aa in seq], device=device_inner, dtype=torch.bool)
        if mask_vit.any():
            beta_phi = math.radians(-119.0)
            beta_psi = math.radians(120.0)
            beta_phi_t = torch.tensor(beta_phi, device=device_inner)
            beta_psi_t = torch.tensor(beta_psi, device=device_inner)
            phi_eff[mask_vit] = beta_phi_t + 0.5 * torch.tanh(phi_raw[mask_vit] - beta_phi_t)
            psi_eff[mask_vit] = beta_psi_t + 0.5 * torch.tanh(psi_raw[mask_vit] - beta_psi_t)
        mask_pro = torch.tensor([aa == "P" for aa in seq], device=device_inner, dtype=torch.bool)
        if mask_pro.any():
            pro_phi0 = math.radians(-65.0)
            pro_phi_t = torch.tensor(pro_phi0, device=device_inner)
            phi_eff[mask_pro] = pro_phi_t + 0.3 * torch.tanh(phi_raw[mask_pro] - pro_phi_t)
        return phi_eff, psi_eff
        
    # Precompute MJ tensor
    mj_tensor = torch.tensor(MJ_VALUES, dtype=torch.float32, device=device)
    
    # Precompute Bond Constants for reuse
    bond_constants = _get_bond_constants(device)

    chain_lengths = [len(d["seq"]) for d in chain_data]
    total_residues = int(sum(chain_lengths))
    chain_ids = torch.cat(
        [torch.full((L,), i, device=device, dtype=torch.long) for i, L in enumerate(chain_lengths)],
        dim=0,
    )
    aa_indices = torch.tensor(
        [MJ_INDEX.get(a, MJ_INDEX["A"]) for d in chain_data for a in d["seq"]],
        device=device,
        dtype=torch.long,
    )

    full_hydro = torch.cat([d["hydro_mask"] for d in chain_data], dim=0)
    full_charge = torch.cat([d["charge"] for d in chain_data], dim=0)
    full_polar = torch.cat([d["polar_mask"] for d in chain_data], dim=0)
    full_aromatic = torch.cat([d["aromatic_mask"] for d in chain_data], dim=0)
    
    full_collapse_hydro = torch.cat([d["collapse_hydro_mask"] for d in chain_data], dim=0)
    full_charged_clash = torch.cat([d["charged_clash_mask"] for d in chain_data], dim=0)
    full_pos_type = torch.cat([d.get("pos_type", torch.zeros(len(d["seq"]), device=device, dtype=torch.long)) for d in chain_data], dim=0)
    ss_labels = []
    for d in chain_data:
        s = d.get("ss", "")
        if isinstance(s, str):
            ss_labels.extend(list(s))
        else:
            ss_labels.extend(["C"] * len(d["seq"]))
    full_ss_H = torch.tensor([1.0 if c == "H" else 0.0 for c in ss_labels], device=device)
    full_ss_E = torch.tensor([1.0 if c == "E" else 0.0 for c in ss_labels], device=device)

    idx = torch.arange(total_residues, device=device)
    idx_diff = torch.abs(idx.unsqueeze(1) - idx.unsqueeze(0))
    mask_nonlocal = (torch.triu(torch.ones((total_residues, total_residues), device=device), diagonal=3) > 0).float()
    cross_mask = (chain_ids.unsqueeze(0) != chain_ids.unsqueeze(1)).float()
    same_chain = 1.0 - cross_mask

    n_full_atoms = total_residues * 4
    n_heavy_atoms = total_residues * 5
    clash_mask = torch.triu(torch.ones((n_full_atoms, n_full_atoms), device=device), diagonal=2) > 0
    heavy_pair_mask = torch.triu(torch.ones((n_heavy_atoms, n_heavy_atoms), device=device), diagonal=2) > 0

    cys_indices = torch.nonzero(aa_indices == MJ_INDEX["C"]).flatten()
    
    opt_phase = "coarse"
    last_loss_breakdown = None
    def calc_loss():
        nonlocal last_loss_breakdown
        all_N, all_CA, all_C, all_O, all_H, all_CB = [], [], [], [], [], []
        loss_ss = torch.tensor(0.0, device=device)
        loss_rama = torch.tensor(0.0, device=device)
        loss_smooth = torch.tensor(0.0, device=device)
        loss_omega = torch.tensor(0.0, device=device)
        loss_frag = torch.tensor(0.0, device=device)
        loss_beta = torch.tensor(0.0, device=device)
        loss_loop = torch.tensor(0.0, device=device)
        
        for i, d in enumerate(chain_data):
            phi_raw = phi_params[i]
            psi_raw = psi_params[i]
            phi, psi = _apply_rama_codebook(phi_raw, psi_raw, d["seq"], device)
            omega = omega_params[i]
            
            ss_probs = d.get("ss_probs")
            ss_weights = d.get("ss_weights", torch.ones_like(phi, device=device))
            h_phi = torch.full_like(phi, math.radians(-60.0))
            h_psi = torch.full_like(psi, math.radians(-47.0))
            e_phi = torch.full_like(phi, math.radians(-120.0))
            e_psi = torch.full_like(psi, math.radians(120.0))
            d_h = (torch.remainder(phi - h_phi + math.pi, 2*math.pi) - math.pi)**2 + (torch.remainder(psi - h_psi + math.pi, 2*math.pi) - math.pi)**2
            d_e = (torch.remainder(phi - e_phi + math.pi, 2*math.pi) - math.pi)**2 + (torch.remainder(psi - e_psi + math.pi, 2*math.pi) - math.pi)**2
            if ss_probs is not None:
                pH = ss_probs[:,0]
                pE = ss_probs[:,1]
                pC = ss_probs[:,2]
                res_weight = torch.ones_like(pH)
                res_weight = res_weight + 0.2 * d["hydro_mask"].clamp(0.0, 1.0)
                res_weight = res_weight - 0.2 * d["polar_mask"].clamp(0.0, 1.0)
                res_weight = torch.where(d["rama_group_ids"] == 1, res_weight * 0.7, res_weight)
                res_weight = torch.where(d["rama_group_ids"] == 2, res_weight * 0.7, res_weight)
                pos_type_local = d.get("pos_type")
                if pos_type_local is not None:
                    core_mask_local = ((pos_type_local == 1) | (pos_type_local == 3))
                    cap_edge_mask_local = ((pos_type_local == 2) | (pos_type_local == 4))
                    loop_mask_local = (pos_type_local == 0)
                    scale_local = torch.ones_like(res_weight)
                    scale_local = torch.where(core_mask_local, scale_local * 2.0, scale_local)
                    scale_local = torch.where(cap_edge_mask_local, scale_local * 0.8, scale_local)
                    scale_local = torch.where(loop_mask_local, scale_local * 0.4, scale_local)
                    res_weight = res_weight * scale_local
                soft_w = res_weight * (pH + pE) * (1.0 - 0.5 * pC)
                loss_ss = loss_ss + torch.sum(soft_w * (pH * d_h + pE * d_e))
            else:
                d_phi = torch.remainder(phi - d["phi_ref"] + math.pi, 2*math.pi) - math.pi
                d_psi = torch.remainder(psi - d["psi_ref"] + math.pi, 2*math.pi) - math.pi
                loss_ss = loss_ss + torch.sum(ss_weights * (d_phi**2 + d_psi**2))
            
            group_ids = d.get("rama_group_ids")
            if group_ids is None:
                group_ids = torch.zeros((len(d["seq"]),), device=device, dtype=torch.long)

            c_t = rama_centers_pad.index_select(0, group_ids)
            v_t = rama_valid_pad.index_select(0, group_ids)
            s_t = rama_sigma.index_select(0, group_ids)

            phi_ex = phi.unsqueeze(1)
            psi_ex = psi.unsqueeze(1)

            dphi = torch.remainder(phi_ex - c_t[:, :, 0] + math.pi, 2 * math.pi) - math.pi
            dpsi = torch.remainder(psi_ex - c_t[:, :, 1] + math.pi, 2 * math.pi) - math.pi

            term = (dphi / (s_t[:, 0:1] + 1e-9)) ** 2 + (dpsi / (s_t[:, 1:2] + 1e-9)) ** 2
            term = term + (1.0 - v_t) * 1e6

            min_term = torch.min(term, dim=1)[0]
            pos_type_local = d.get("pos_type")
            if pos_type_local is not None:
                scale_rama = torch.ones_like(min_term)
                core_mask_local = ((pos_type_local == 1) | (pos_type_local == 3))
                cap_edge_mask_local = ((pos_type_local == 2) | (pos_type_local == 4))
                loop_mask_local = (pos_type_local == 0)
                scale_rama = torch.where(core_mask_local, scale_rama * 2.0, scale_rama)
                scale_rama = torch.where(cap_edge_mask_local, scale_rama * 0.8, scale_rama)
                scale_rama = torch.where(loop_mask_local, scale_rama * 0.4, scale_rama)
                min_term = min_term * scale_rama
            
            n_res = min_term.size(0)
            n_keep = max(1, int(n_res * 0.97))
            
            sorted_term, _ = torch.sort(min_term)
            valid_term = sorted_term[:n_keep]
            
            rama_soft = torch.sum(torch.relu(valid_term - 1.0))
            loss_rama = loss_rama + rama_soft

            pro_mask = (group_ids == 2).float()
            if pro_mask.numel() > 0:
                pro_phi0 = torch.tensor(math.radians(-65.0), device=device)
                pro_dphi = torch.remainder(phi - pro_phi0 + math.pi, 2 * math.pi) - math.pi
                # Relaxed Proline constraint: Weight 10.0 -> 0.5 to allow kinks
                loss_rama = loss_rama + torch.sum((pro_dphi / math.radians(10.0)) ** 2 * pro_mask) * 0.5

            omega_valid = torch.ones_like(omega)
            if omega_valid.numel() > 0:
                omega_valid[-1] = 0.0
            omega_wrap_trans = torch.remainder(omega - math.pi + math.pi, 2 * math.pi) - math.pi
            trans_term = (omega_wrap_trans / (OMEGA_SIGMA["TRANS"] + 1e-9)) ** 2
            cis_term = (torch.remainder(omega + math.pi, 2 * math.pi) - math.pi) / (OMEGA_SIGMA["PREPRO"] + 1e-9)
            cis_term = cis_term ** 2 + OMEGA_CIS_BIAS
            prepro_mask = d.get("prepro_mask")
            if prepro_mask is None:
                prepro_mask = torch.zeros_like(omega)
            omega_term = torch.where(prepro_mask > 0.5, torch.minimum(trans_term, cis_term), trans_term)
            loss_omega = loss_omega + torch.sum(omega_term * omega_valid) + torch.sum(torch.relu(omega_term - 1.0) ** 2 * omega_valid) * 50.0
            
            diff_phi = phi[1:] - phi[:-1]
            diff_psi = psi[1:] - psi[:-1]
            loss_smooth = loss_smooth + torch.sum(diff_phi**2 + diff_psi**2)
            if ss_probs is not None and len(d["seq"]) >= 3:
                k = 3
                for j in range(len(d["seq"]) - k + 1):
                    pH_w = torch.mean(ss_probs[j:j+k,0])
                    pE_w = torch.mean(ss_probs[j:j+k,1])
                    seg_h = torch.mean(d_h[j:j+k])
                    seg_e = torch.mean(d_e[j:j+k])
                    loss_frag = loss_frag + (pH_w * seg_h + pE_w * seg_e) * 0.5
            
            # Build Structure
            N, CA, C, O, H, CB = build_full_structure_tensor(len(d["seq"]), phi, psi, device, bond_constants, omega=omega)
            
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

            ss_str = d.get("ss", "")
            if isinstance(ss_str, str):
                L_ss = len(ss_str)
            else:
                L_ss = 0
            if L_ss > 1 and CA.shape[0] >= L_ss:
                beta_indices = [j for j, c in enumerate(ss_str) if c == "E"]
                if len(beta_indices) >= 2:
                    idx_tensor = torch.tensor(beta_indices, device=device, dtype=torch.long)
                    coords_ca = CA[idx_tensor]
                    diff_ca = coords_ca.unsqueeze(1) - coords_ca.unsqueeze(0)
                    dist_ca = torch.norm(diff_ca, dim=2) + 1e-6
                    sep = torch.abs(idx_tensor.unsqueeze(1) - idx_tensor.unsqueeze(0))
                    mask = sep > 3
                    if mask.any():
                        dist_sel = dist_ca[mask]
                        d0 = 5.0
                        width = 0.5
                        e_dist = torch.mean(((dist_sel - d0) / width) ** 2)
                    else:
                        e_dist = torch.tensor(0.0, device=device)
                    ca_len = CA.shape[0]
                    if ca_len > 1:
                        tangents = torch.zeros_like(CA)
                        if ca_len > 2:
                            tangents[1:-1] = CA[2:] - CA[:-2]
                        tangents[0] = CA[1] - CA[0]
                        tangents[-1] = CA[-1] - CA[-2]
                        t_norm = torch.norm(tangents, dim=1, keepdim=True) + 1e-6
                        tangents = tangents / t_norm
                        t_beta = tangents[idx_tensor]
                        dot = torch.matmul(t_beta, t_beta.t()).clamp(-1.0, 1.0)
                        m = torch.triu(torch.ones_like(dot, dtype=torch.bool), diagonal=1)
                        if m.any():
                            vals = (torch.abs(dot[m]) - 1.0) ** 2
                            e_orient = torch.mean(vals)
                        else:
                            e_orient = torch.tensor(0.0, device=device)
                    else:
                        e_orient = torch.tensor(0.0, device=device)
                    O_beta = O[idx_tensor]
                    N_beta = N[idx_tensor]
                    diff_on = O_beta.unsqueeze(1) - N_beta.unsqueeze(0)
                    dist_on = torch.norm(diff_on, dim=2) + 1e-6
                    if mask.any():
                        dist_on_sel = dist_on[mask]
                        on_term = torch.mean(torch.abs(dist_on_sel - 2.8))
                    else:
                        on_term = torch.tensor(0.0, device=device)
                    phi_beta = phi[idx_tensor]
                    psi_beta = psi[idx_tensor]
                    beta_phi0 = math.radians(-119.0)
                    beta_psi0 = math.radians(120.0)
                    beta_phi_t = torch.tensor(beta_phi0, device=device)
                    beta_psi_t = torch.tensor(beta_psi0, device=device)
                    dphi_beta = torch.remainder(phi_beta - beta_phi_t + math.pi, 2 * math.pi) - math.pi
                    dpsi_beta = torch.remainder(psi_beta - beta_psi_t + math.pi, 2 * math.pi) - math.pi
                    angle_term = torch.mean(dphi_beta**2 + dpsi_beta**2)
                    loss_beta = loss_beta + (e_dist + 0.3 * e_orient + on_term + 0.5 * angle_term)
            if isinstance(ss_str, str) and L_ss > 0 and CA.shape[0] >= L_ss:
                idx_loop = 0
                while idx_loop < L_ss:
                    if ss_str[idx_loop] != "C":
                        idx_loop += 1
                        continue
                    start_loop = idx_loop
                    while idx_loop < L_ss and ss_str[idx_loop] == "C":
                        idx_loop += 1
                    end_loop = idx_loop - 1
                    prev_i = start_loop - 1
                    next_i = end_loop + 1
                    loop_len = end_loop - start_loop + 1
                    if prev_i >= 0 and next_i < L_ss:
                        ca_prev = CA[prev_i]
                        ca_next = CA[next_i]
                        dist_end = torch.norm(ca_next - ca_prev) + 1e-6
                        path_len = (loop_len + 1) * 3.8
                        d_max = path_len * 0.7
                        loop_term = torch.relu(dist_end - d_max) ** 2
                        loss_loop = loss_loop + loop_term
            
        full_CA = torch.cat(all_CA)
        full_CB = torch.cat(all_CB)
        full_N = torch.cat(all_N)
        full_C = torch.cat(all_C)
        
        # Rg (Compactness)
        centroid = torch.mean(full_CA, dim=0)
        rg2 = torch.sum(torch.sum((full_CA - centroid)**2, dim=1)) / len(full_CA)
        rg = torch.sqrt(rg2 + 1e-9)
        rg_target = 2.2 * (len(full_CA) ** 0.38)
        loss_rg_target = (rg - rg_target) ** 2
        
        full_omega = torch.cat(omega_params)
        diff_ca_adj = full_CA[1:] - full_CA[:-1]
        dist_ca_adj = torch.norm(diff_ca_adj, dim=1)
        adj_same_chain = (chain_ids[1:] == chain_ids[:-1]).float()
        omega_adj = full_omega[:-1]
        cis_w = (1.0 + torch.cos(omega_adj)).clamp(0.0, 1.0) * 0.5
        ca_target = 3.8 * (1.0 - cis_w) + 2.9 * cis_w
        loss_ca_continuity = torch.sum(((dist_ca_adj - ca_target) ** 2) * adj_same_chain) * 10.0
        
        full_O = torch.cat(all_O)
        heavy_per_res = torch.stack([full_N, full_CA, full_C, full_O, full_CB], dim=1)
        heavy_atoms = heavy_per_res.reshape(-1, 3)
        res_ids = torch.arange(total_residues, device=device).repeat_interleave(5)
        atom_radii = torch.tensor([1.55, 1.70, 1.70, 1.52, 1.70], device=device, dtype=heavy_atoms.dtype)
        radii_flat = atom_radii.repeat(total_residues)

        hdist = torch.cdist(heavy_atoms, heavy_atoms, p=2.0)
        pair_mask = torch.triu(torch.ones_like(hdist, dtype=torch.bool), diagonal=1)
        resdiff = torch.abs(res_ids.unsqueeze(0) - res_ids.unsqueeze(1))
        pair_mask = pair_mask & (resdiff >= 2)

        vdW_sum = radii_flat.unsqueeze(0) + radii_flat.unsqueeze(1)
        soft_thr = 0.9 * vdW_sum
        hard_thr = 0.8 * vdW_sum

        soft_pen = torch.relu(soft_thr - hdist) ** 2
        hard_pen = torch.relu(hard_thr - hdist) ** 2

        heavy_loss = torch.sum(soft_pen[pair_mask])
        loss_hard_clash = torch.sum(hard_pen[pair_mask]) * 25.0
        clash_loss = torch.sum((hard_pen > 0.0).float()[pair_mask])
        
        dist_ca_pairs = torch.cdist(full_CA, full_CA, p=2.0)
        pair_mask_ca = torch.triu(torch.ones_like(dist_ca_pairs, dtype=torch.bool), diagonal=1) & (idx_diff >= 2)
        ca_pen = torch.relu(3.8 - dist_ca_pairs) ** 2
        loss_ca_hard36 = torch.sum(ca_pen[pair_mask_ca])
        
        eps_values = torch.tensor([0.12, 0.10, 0.15, 0.20, 0.18], device=device, dtype=heavy_atoms.dtype)
        eps_flat = eps_values.repeat(total_residues)
        sigma_ij = vdW_sum / (2.0 ** (1.0 / 6.0))
        eps_ij = torch.sqrt(eps_flat.unsqueeze(0) * eps_flat.unsqueeze(1))
        hdist_safe = torch.clamp(hdist, min=0.8)
        ratio = sigma_ij / hdist_safe
        ratio_clamped = torch.clamp(ratio, max=5.0)
        lj_heavy = 4.0 * eps_ij * (ratio_clamped**12 - ratio_clamped**6)
        loss_vdw = torch.sum(lj_heavy[pair_mask])
        
        diff_no = full_N.unsqueeze(1) - full_O.unsqueeze(0)
        dist_no = torch.norm(diff_no, dim=2) + 1e-6
        idx_no = torch.arange(total_residues, device=device)
        resdiff_no = torch.abs(idx_no.unsqueeze(1) - idx_no.unsqueeze(0))
        mask_no = (torch.triu(torch.ones_like(dist_no, dtype=torch.bool), diagonal=1) & (resdiff_no >= 2)).float()
        eps_bb = 10.0 + 70.0 * torch.clamp(dist_no / 12.0, min=0.0, max=1.0)
        loss_elec_bb = torch.sum(((0.3 * -0.5) / (eps_bb * dist_no)) * mask_no)
        
        # Hydrophobic Effect
        # Optimize CB distance
        # full_CB: (N_res, 3)
        dist_cb = torch.cdist(full_CB, full_CB, p=2.0) + 1e-6
        diff_cb = full_CB.unsqueeze(0) - full_CB.unsqueeze(1) # Keep for vectors
        
        u_vec = full_CB - full_CA
        u_vec = u_vec / (torch.norm(u_vec, dim=1, keepdim=True) + 1e-6)
        v_unit = (-diff_cb) / (dist_cb.unsqueeze(2))
        dot_i = torch.sum(u_vec.unsqueeze(1) * v_unit, dim=2).clamp(min=0.0)
        dot_j = torch.sum(u_vec.unsqueeze(0) * (-v_unit), dim=2).clamp(min=0.0)
        ori_gate = dot_i * dot_j
        hp_mat = full_hydro.unsqueeze(0) * full_hydro.unsqueeze(1)
        
        hp_dist_sum = torch.sum(dist_cb * hp_mat * mask_nonlocal)
        n_hp_pairs = torch.sum(hp_mat * mask_nonlocal) + 1.0
        loss_hydro = hp_dist_sum / n_hp_pairs
        
        iface_sum = torch.sum(dist_cb * hp_mat * mask_nonlocal * cross_mask * ori_gate)
        iface_pairs = torch.sum(hp_mat * mask_nonlocal * cross_mask) + 1.0
        loss_iface_hydro = iface_sum / iface_pairs
        
        r_min_cb = 2.6
        r_opt_cb = 3.8
        r_cut_cb = 6.0
        r_hard_cb = 2.2
        A_soft_cb = 2.0
        A_hard_cb = 10.0
        B_cb = 3.0
        sigma_cb = 0.6
        base_E_cb = torch.zeros_like(dist_cb)
        mask_cut_cb = (dist_cb <= r_cut_cb) & (dist_cb > 0.0)
        mask_hard_cb = (dist_cb < r_hard_cb) & mask_cut_cb
        mask_soft_cb = (dist_cb >= r_hard_cb) & (dist_cb < r_min_cb) & mask_cut_cb
        mask_well_cb = (dist_cb >= r_min_cb) & (dist_cb <= r_opt_cb) & mask_cut_cb
        if mask_soft_cb.any():
            base_E_cb[mask_soft_cb] = A_soft_cb * (r_min_cb - dist_cb[mask_soft_cb]) ** 2
        if mask_hard_cb.any():
            base_E_cb[mask_hard_cb] = A_hard_cb * (r_hard_cb - dist_cb[mask_hard_cb]) ** 2 + A_soft_cb * (r_min_cb - r_hard_cb) ** 2
        if mask_well_cb.any():
            base_E_cb[mask_well_cb] = -B_cb * torch.exp(-((dist_cb[mask_well_cb] - r_opt_cb) ** 2) / (sigma_cb ** 2))
        type_idx = torch.zeros_like(aa_indices)
        hph_set = set(['A','V','I','L','M','F','W','Y'])
        pos_set = set(['K','R','H'])
        neg_set = set(['D','E'])
        seq_all = used_sequence
        type_list = []
        for a in seq_all:
            if a in hph_set:
                type_list.append(0)
            elif a in pos_set:
                type_list.append(2)
            elif a in neg_set:
                type_list.append(3)
            else:
                type_list.append(1)
        type_tensor = torch.tensor(type_list, device=device, dtype=torch.long)
        ti = type_tensor.unsqueeze(1)
        tj = type_tensor.unsqueeze(0)
        W_tb = torch.tensor([[1.0,0.4,0.2,0.2],
                             [0.4,0.6,0.8,0.8],
                             [0.2,0.8,0.3,1.2],
                             [0.2,0.8,1.2,0.3]], device=device, dtype=dist_cb.dtype)
        wtype_cb = W_tb[ti, tj]
        E_cb = base_E_cb * wtype_cb
        if mask_cut_cb.any():
            loss_mj = torch.sum(E_cb * mask_nonlocal * mask_cut_cb * ((1.0 - cross_mask) + cross_mask * ori_gate)) / (torch.sum(mask_nonlocal * mask_cut_cb) + 1.0)
        else:
            loss_mj = torch.tensor(0.0, device=device)
        
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
        near_mask = ((idx_diff > 2) & (dist_ca < 8.0)).float()
        neighbor_counts = torch.sum(near_mask, dim=1)
        exposure = 1.0 / (1.0 + neighbor_counts)
        loss_burial = torch.sum(full_hydro * exposure)
        loss_polar = torch.sum(full_polar * (1.0 - exposure))
        abs_charge = torch.abs(full_charge)
        core_mask_global = ((full_pos_type == 1) | (full_pos_type == 3)).float()
        charge_weight = 1.0 + core_mask_global
        loss_charge_burial = torch.sum(abs_charge * (1.0 - exposure) * charge_weight)
        
        hydro_mask_cent = full_hydro > 0.0
        if hydro_mask_cent.any():
            cb_h = full_CB[hydro_mask_cent]
            center_h = torch.mean(cb_h, dim=0)
            dist_h = torch.norm(cb_h - center_h, dim=1)
            loss_hydro_centroid = torch.mean(dist_h)
        else:
            loss_hydro_centroid = torch.tensor(0.0, device=device)
        
        hydro_mask_cd = full_hydro > 0.0
        if hydro_mask_cd.any():
            hp_vec = hydro_mask_cd.float()
            pair_hp = hp_vec.unsqueeze(0) * hp_vec.unsqueeze(1)
            r0_cd = 4.0
            rho_mat = torch.exp(-((dist_cb / r0_cd) ** 2)) * pair_hp
            eye_cd = torch.eye(total_residues, device=device, dtype=dist_cb.dtype)
            rho_mat = rho_mat * (1.0 - eye_cd)
            rho = torch.sum(rho_mat, dim=1)
            rho_h = rho[hydro_mask_cd]
            rho_excess = torch.relu(rho_h - 4.5)
            loss_contact_density = torch.mean(rho_excess ** 2)
        else:
            loss_contact_density = torch.tensor(0.0, device=device)
        
        full_H = torch.cat(all_H)
        diff_ho = full_H.unsqueeze(1) - full_O.unsqueeze(0)
        dist_ho = torch.norm(diff_ho, dim=2) + 1e-6
        diff_no = full_N.unsqueeze(1) - full_O.unsqueeze(0)
        dist_no = torch.norm(diff_no, dim=2) + 1e-6
        hb_mask_dist = (dist_no > 2.7) & (dist_no < 3.2)
        helix_pair = (full_ss_H.unsqueeze(1) > 0.5) & (full_ss_H.unsqueeze(0) > 0.5)
        beta_pair = (full_ss_E.unsqueeze(1) > 0.5) & (full_ss_E.unsqueeze(0) > 0.5)
        hb_mask_helix = (idx_diff == 4) & (same_chain > 0.5) & helix_pair
        hb_mask_beta = (idx_diff > 3) & beta_pair
        hb_pair_mask = hb_mask_helix | hb_mask_beta
        hn_vec = full_H - torch.cat(all_N)
        hn_unit = hn_vec / (torch.norm(hn_vec, dim=1, keepdim=True) + 1e-6)
        ho_unit = diff_ho / (dist_ho.unsqueeze(2))
        cos_nho = torch.sum(hn_unit.unsqueeze(1) * ho_unit, dim=2).clamp(-1.0, 1.0)
        ang_nho = torch.acos(cos_nho)
        co_vec = full_C - full_O
        co_unit = co_vec / (torch.norm(co_vec, dim=1, keepdim=True) + 1e-6)
        co_unit_exp = co_unit.unsqueeze(1)
        cos_coh = torch.sum(co_unit_exp * ho_unit, dim=2).clamp(-1.0, 1.0)
        ang_coh = torch.acos(cos_coh)
        ang_mask_nho = ang_nho > HB_MIN_ANGLE
        ang_mask_coh = ang_coh > math.radians(90.0)
        geom_mask = hb_mask_dist & ang_mask_nho & ang_mask_coh
        hb_mask = (hb_pair_mask & geom_mask).float()
        dist_w = torch.exp(-((dist_no - HB_OPT_DIST)**2) / (0.25**2))
        ang_w = torch.exp(-((ang_nho - HB_OPT_ANGLE)**2) / (math.radians(20)**2))
        hb_score = dist_w * ang_w * hb_mask
        hb_energy = -torch.sum(hb_score) * 0.5
        
        # Disulfide Bonds (Cys-Cys)
        # Biological Rule: Cysteines in oxidizing environments tend to form disulfide bridges.
        # We encourage Cys-Cys pairs to be close (~5A CB-CB) if they are reasonably near.
        # This is a soft constraint to guide formation, not a hard bond.
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
        
        # --- Primitive Hydrophobic Collapse & Charged Clash ---
        # 1. Hydrophobic Collapse (I, L, V, F, M)
        # CA-CA < 8 A: negative score
        # CA-CA < 6 A: additional negative score
        hp_col_mat = full_collapse_hydro.unsqueeze(0) * full_collapse_hydro.unsqueeze(1)
        d8_mask = (dist_ca_pairs < 8.0).float()
        d6_mask = (dist_ca_pairs < 6.0).float()
        core_mask_pair_i = ((full_pos_type == 1) | (full_pos_type == 3)).float().unsqueeze(0)
        core_mask_pair_j = ((full_pos_type == 1) | (full_pos_type == 3)).float().unsqueeze(1)
        core_pair_weight = 1.0 + 0.5 * (core_mask_pair_i + core_mask_pair_j)
        loss_hydro_collapse = -torch.sum((d8_mask + d6_mask) * hp_col_mat * pair_mask_ca.float() * core_pair_weight)
        
        # 2. Charged Clash (D, E, K, R)
        # < 4 A clash penalty
        chg_clash_mat = full_charged_clash.unsqueeze(0) * full_charged_clash.unsqueeze(1)
        d4_mask = (dist_ca_pairs < 4.0).float()
        
        loss_charged_clash = torch.sum(d4_mask * chg_clash_mat * pair_mask_ca.float())
        
        rep_mask = dist_ca_pairs < 3.8
        rep_term = torch.relu(3.8 - dist_ca_pairs) ** 2
        rep_energy = torch.sum(rep_term * rep_mask.float() * pair_mask_ca.float())
        attr_mask = (dist_ca_pairs >= 4.5) & (dist_ca_pairs <= 6.5)
        attr_term = torch.exp(-((dist_ca_pairs - 5.5) ** 2) / (2.0 * (0.8 ** 2)))
        attr_energy = torch.sum(attr_term * attr_mask.float() * pair_mask_ca.float())
        loss_ca_lj = rep_energy - 0.2 * attr_energy

        # Weights (Tuned for Bio-plausibility)
        w_rg = 1.5
        w_rg_tgt = 1.2
        w_clash = 5.0
        w_hard_clash = 10.0
        w_heavy = 4.5
        w_ca_cont = 5.0
        w_ss = 2.0
        w_rama = 4.0 # Boosted: Biological backbone preference is strong
        w_omega = 3.0
        w_hb = 4.0
        w_hydro = 2.0
        w_iface_hydro = 2.0
        w_elec = 1.5
        w_catpi = 1.0
        w_pipi = 1.0
        w_lj = 0.5
        w_vdw_heavy = 0.8
        w_burial = 4.0
        w_polar = 2.0
        w_smooth = 0.0 # User Feedback: "Invisible wire" effect. Disable smooth path.
        w_mj = 2.0
        w_disulfide = 5.0 # Strong bias for SS bonds
        w_ca_hard36 = 7.0
        w_ca_lj = 1.0
        w_elec_bb = 0.8
        
        w_hydro_collapse = 2.0
        w_hydro_centroid = 1.0
        w_charged_clash = 35.0
        w_frag = 1.0
        w_beta = 3.0
        w_loop = 1.0
        w_contact_density = 0.4
        w_charge_burial = 3.5
        if opt_phase == "warmup":
            w_ss = 0.5
            w_rama = 1.5
            w_hb = 2.0
            w_hydro = 5.0
            w_iface_hydro = 4.0
            w_mj = 6.0
            w_elec = 2.5
            w_burial = 3.0
            w_polar = 0.5
            w_frag = 0.5
        
        loss = (loss_ss * w_ss + loss_rama * w_rama + loss_omega * w_omega + loss_smooth * w_smooth + loss_rg_target * w_rg_tgt + 
                clash_loss * w_clash + loss_hard_clash * w_hard_clash + heavy_loss * w_heavy + loss_ca_continuity * w_ca_cont +
                loss_hydro * w_hydro + loss_iface_hydro * w_iface_hydro + loss_mj * w_mj + 
                loss_elec * w_elec + loss_elec_bb * w_elec_bb + loss_catpi * w_catpi + loss_pipi * w_pipi + 
                loss_lj * w_lj + loss_vdw * w_vdw_heavy + loss_burial * w_burial + loss_polar * w_polar +
                hb_energy * w_hb + loss_disulfide * w_disulfide + loss_ca_hard36 * w_ca_hard36 + loss_ca_lj * w_ca_lj + loss_hydro_centroid * w_hydro_centroid + loss_frag * w_frag +
                loss_beta * w_beta +
                loss_hydro_collapse * w_hydro_collapse + loss_charged_clash * w_charged_clash +
                loss_loop * w_loop + loss_contact_density * w_contact_density + loss_charge_burial * w_charge_burial)
        
        loss_constraints = torch.tensor(0.0, device=device)
        if constraints:
            for c in constraints:
                try:
                    c_type = c.get("type")
                    strength = float(c.get("strength", 5.0))
                    indices = c.get("indices", [])
                    if not indices:
                        continue
                    valid_indices = [idx for idx in indices if idx < len(full_CB)]
                    if not valid_indices:
                        continue
                    t_indices = torch.tensor(valid_indices, device=device, dtype=torch.long)
                    if c_type == "distance_point":
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
                                except Exception:
                                    pass
                except Exception:
                    pass
        
        loss = loss + loss_constraints
        per_res = float(total_residues)
        if per_res <= 0.0:
            per_res = 1.0
        try:
            last_loss_breakdown = {
                "total": loss.detach(),
                "per_res": (loss / per_res).detach(),
                "ss": (loss_ss * w_ss).detach(),
                "rama": (loss_rama * w_rama).detach(),
                "beta": (loss_beta * w_beta).detach(),
                "hydro": (loss_hydro * w_hydro).detach(),
                "hydro_centroid": (loss_hydro_centroid * w_hydro_centroid).detach(),
                "mj": (loss_mj * w_mj).detach(),
                "elec": (loss_elec * w_elec).detach(),
                "rg": (loss_rg_target * w_rg_tgt).detach(),
                "hb": (hb_energy * w_hb).detach(),
                "ca_lj": (loss_ca_lj * w_ca_lj).detach(),
                "constraints": loss_constraints.detach(),
            }
        except Exception:
            last_loss_breakdown = None
        return loss

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
                      w_ca_hard36 * loss_ca_hard36 +
                      w_disulfide * loss_disulfide)
                      
        return total_loss
    
    if os.environ.get("MINIFOLD_IGPU_COMPILE", "0") == "1" and hasattr(torch, "compile"):
        try:
            calc_loss = torch.compile(calc_loss, mode=os.environ.get("MINIFOLD_IGPU_COMPILE_MODE", "reduce-overhead"))
        except Exception as e:
            logger.warning(f"torch.compile unavailable: {e}")

    snapshots_enabled = os.environ.get("MINIFOLD_IGPU_SNAPSHOTS", "1") == "1"

    frames_dir = None
    frame_counter = 0
    if snapshots_enabled:
        frames_dir = os.path.join(os.path.dirname(output_pdb), "_frames")
        try:
            os.makedirs(frames_dir, exist_ok=True)
        except Exception:
            frames_dir = None

    def save_snapshot(step_name):
        if not snapshots_enabled or frames_dir is None:
            return
        try:
            with torch.no_grad():
                final_N_list, final_CA_list, final_C_list = [], [], []
                for i, d in enumerate(chain_data):
                    phi, psi = phi_params[i], psi_params[i]
                    omega = omega_params[i]
                    N, CA, C, _, _, _ = build_full_structure_tensor(len(d["seq"]), phi, psi, device, bond_constants, omega=omega)
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
                
                lengths = [len(d["seq"]) for d in chain_data]
                breaks = []
                acc = 0
                for L in lengths[:-1]:
                    acc += L
                    breaks.append(acc)
                
                tmp_path = output_pdb + ".tmp"
                # Use global write_pdb
                write_pdb(used_sequence, final_N, final_CA, final_C, tmp_path, chain_breaks=breaks)
                
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
                write_pdb(used_sequence, final_N, final_CA, final_C, frame_tmp, chain_breaks=breaks)
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
    try:
        warmup_steps = int(os.environ.get("MINIFOLD_IGPU_WARMUP_STEPS", "30"))
    except Exception:
        warmup_steps = 30
    warmup_steps = max(0, warmup_steps)
    if warmup_steps > 0:
        opt_phase = "warmup"
        opt_warm = torch.optim.AdamW(all_params, lr=0.08, weight_decay=1e-4, foreach=True)
        for step in range(warmup_steps):
            opt_warm.zero_grad(set_to_none=True)
            loss = calc_loss()
            loss.backward()
            opt_warm.step()
        opt_phase = "coarse"
    
    # Bio-Algorithm Initialization Strategy
    # Replaces pure random starts with biologically motivated heuristics.
    # 1. Consensus: Ideal Secondary Structure + Geometric Centering
    # 2. Hydrophobic Collapse: Aligns Hydrophobic Centers of Mass (Simulating the "Fold" event)
    # 3. Ramachandran Sampling: Samples backbone angles from PDB statistics (Natural Variation)
    
    strategies = ["bio_hydro_collapse", "bio_consensus", "bio_rama_sampling"]
    
    if os.environ.get("MINIFOLD_FAST_MODE", "0") == "1":
        strategies = ["bio_hydro_collapse"]
        logger.info("Fast Mode Enabled: Running 'Hydrophobic Collapse' strategy only.")
    
    try:
        starts_override = int(os.environ.get("MINIFOLD_IGPU_NUM_STARTS", "0"))
    except Exception:
        starts_override = 0
    if starts_override > 0:
        strategies = strategies[: max(1, min(starts_override, len(strategies)))]
    
    num_starts = len(strategies)
    try:
        restarts_per_strategy = int(os.environ.get("MINIFOLD_IGPU_RESTARTS", "1"))
    except Exception:
        restarts_per_strategy = 1
    restarts_per_strategy = max(1, restarts_per_strategy)

    best_loss = None
    best_phi = None
    best_psi = None
    best_omega = None
    best_rb = None
    
    # Helper to calculate centers on the fly
    def get_chain_props(d, p_phi, p_psi, p_omega):
        # Build temp structure
        t_N, t_CA, t_C, _, _, _ = build_full_structure_tensor(len(d["seq"]), p_phi, p_psi, device, bond_constants, omega=p_omega)
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
        for restart_idx in range(restarts_per_strategy):
            logger.info(f"  > Start {s+1}/{num_starts}, Restart {restart_idx+1}/{restarts_per_strategy}: Strategy '{strategy}'")
            
            with torch.no_grad():
                for i, d in enumerate(chain_data):
                    if strategy == "bio_rama_sampling":
                        new_phi, new_psi = ss_to_phi_psi_tensor(d["ss"], d["seq"])
                        phi_params[i].copy_(new_phi.to(device))
                        psi_params[i].copy_(new_psi.to(device))
                    else:
                        phi_params[i].copy_(d["phi_ref"] + (torch.rand_like(d["phi_ref"]) - 0.5) * 0.05)
                        psi_params[i].copy_(d["psi_ref"] + (torch.rand_like(d["psi_ref"]) - 0.5) * 0.05)
                    o0 = torch.full_like(phi_params[i], math.pi)
                    prepro_mask = d.get("prepro_mask")
                    if prepro_mask is not None and prepro_mask.numel() == o0.numel():
                        cis_sample = (torch.rand_like(o0) < 0.05) & (prepro_mask > 0.5)
                        o0 = torch.where(cis_sample, torch.zeros_like(o0), o0)
                    omega_params[i].copy_(o0)
            
                if rb_params is not None:
                    init_rb = []
                    c0_geo, c0_hydro = get_chain_props(chain_data[0], phi_params[0], psi_params[0], omega_params[0])
                    for k in range(1, len(chain_data)):
                        ck_geo, ck_hydro = get_chain_props(chain_data[k], phi_params[k], psi_params[k], omega_params[k])
                        if strategy == "bio_hydro_collapse":
                            offset = (torch.rand(3, device=device) - 0.5) * 5.0
                            t = (c0_hydro - ck_hydro) + offset
                            r = (torch.rand(3, device=device) * 2 * math.pi) - math.pi
                            init_rb.append(torch.cat([t, r]))
                        elif strategy == "bio_consensus":
                            t = c0_geo - ck_geo + torch.tensor([10.0 * k, 0.0, 0.0], device=device)
                            r = torch.zeros(3, device=device)
                            init_rb.append(torch.cat([t, r]))
                        else:
                            t = (torch.rand(3, device=device) * 40.0) - 20.0
                            r = (torch.rand(3, device=device) * 2 * math.pi) - math.pi
                            init_rb.append(torch.cat([t, r]))
                    rb_params.copy_(torch.stack(init_rb))

            logger.info(f"  > Start {s+1}/{num_starts}, Restart {restart_idx+1}/{restarts_per_strategy}: AdamW (Coarse Folding)")
            try:
                adam_lr = float(os.environ.get("MINIFOLD_IGPU_ADAM_LR", "0.05"))
            except Exception:
                adam_lr = 0.05
            try:
                adam_wd = float(os.environ.get("MINIFOLD_IGPU_ADAM_WD", "1e-4"))
            except Exception:
                adam_wd = 1e-4
            opt_adam = torch.optim.AdamW(all_params, lr=adam_lr, weight_decay=adam_wd, foreach=True)
        
        # IPEX Optimizer Fuse (if applicable)
            if use_ipex and ipex_mod is not None:
                try:
                    opt_adam, _ = ipex_mod.optimize(opt_adam, dtype=torch.bfloat16 if torch.xpu.has_bf16_support() else torch.float32, inplace=True)
                    logger.info("  > IPEX Fused Optimizer enabled")
                except Exception:
                    pass

            try:
                adam_steps = int(os.environ.get("MINIFOLD_IGPU_ADAM_STEPS", "350"))
            except Exception:
                adam_steps = 350
            adam_steps = max(1, adam_steps)
            try:
                log_every = int(os.environ.get("MINIFOLD_IGPU_LOG_EVERY", "5"))
            except Exception:
                log_every = 5

            for step in range(adam_steps): 
                opt_adam.zero_grad(set_to_none=True)
                if use_ipex:
                    with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16 if torch.xpu.has_bf16_support() else torch.float16):
                        loss = calc_loss()
                elif device.type == "cuda":
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        loss = calc_loss()
                else:
                    loss = calc_loss()
                loss.backward()
                opt_adam.step()
                
                if log_every > 0 and (step + 1) % log_every == 0:
                    msg = f"[PROGRESS_DETAIL] Phase 1 (Coarse): Step {step+1}/{adam_steps} | Loss: {loss.item():.2f}"
                    if last_loss_breakdown is not None:
                        try:
                            br = last_loss_breakdown
                            msg += (
                                f" | per_res={br.get('per_res', loss / max(1, total_residues)).item():.2f}"
                                f", ss={br.get('ss', torch.tensor(0.0, device=device)).item():.1f}"
                                f", rama={br.get('rama', torch.tensor(0.0, device=device)).item():.1f}"
                                f", beta={br.get('beta', torch.tensor(0.0, device=device)).item():.1f}"
                                f", hydro={br.get('hydro', torch.tensor(0.0, device=device)).item():.1f}"
                                f", mj={br.get('mj', torch.tensor(0.0, device=device)).item():.1f}"
                                f", elec={br.get('elec', torch.tensor(0.0, device=device)).item():.1f}"
                            )
                        except Exception:
                            pass
                    logger.info(msg)

                if step == adam_steps - 1: 
                    save_snapshot(f"adam_{s+1}_{step+1}")
                
            logger.info(f"  > Start {s+1}/{num_starts}, Restart {restart_idx+1}/{restarts_per_strategy}: LBFGS (Fine Tuning)")
            try:
                lbfgs_max_iter = int(os.environ.get("MINIFOLD_IGPU_LBFGS_MAX_ITER", "960"))
            except Exception:
                lbfgs_max_iter = 960
            lbfgs_max_iter = max(1, lbfgs_max_iter)
            try:
                lbfgs_hist = int(os.environ.get("MINIFOLD_IGPU_LBFGS_HISTORY", "10"))
            except Exception:
                lbfgs_hist = 10
            try:
                lbfgs_tol_grad = float(os.environ.get("MINIFOLD_IGPU_LBFGS_TOL_GRAD", "1e-5"))
            except Exception:
                lbfgs_tol_grad = 1e-5
            try:
                lbfgs_tol_change = float(os.environ.get("MINIFOLD_IGPU_LBFGS_TOL_CHANGE", "1e-5"))
            except Exception:
                lbfgs_tol_change = 1e-5

            opt_lbfgs = torch.optim.LBFGS(
                all_params,
                max_iter=lbfgs_max_iter,
                tolerance_grad=lbfgs_tol_grad,
                tolerance_change=lbfgs_tol_change,
                history_size=lbfgs_hist,
                line_search_fn="strong_wolfe",
            )
            
            lbfgs_step_count = 0
            amp_lbfgs = os.environ.get("MINIFOLD_IGPU_AMP_LBFGS", "0") == "1"
            def closure():
                nonlocal lbfgs_step_count
                opt_lbfgs.zero_grad(set_to_none=True)
                if amp_lbfgs and use_ipex:
                    with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16 if torch.xpu.has_bf16_support() else torch.float16):
                        l = calc_loss()
                elif amp_lbfgs and device.type == "cuda":
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        l = calc_loss()
                else:
                    l = calc_loss()
                l.backward()
                lbfgs_step_count += 1
                if log_every > 0 and lbfgs_step_count % log_every == 0:
                    msg2 = f"[PROGRESS_DETAIL] Phase 2 (Fine): Step {lbfgs_step_count} | Loss: {l.item():.2f}"
                    if last_loss_breakdown is not None:
                        try:
                            br2 = last_loss_breakdown
                            msg2 += (
                                f" | per_res={br2.get('per_res', l / max(1, total_residues)).item():.2f}"
                                f", ss={br2.get('ss', torch.tensor(0.0, device=device)).item():.1f}"
                                f", rama={br2.get('rama', torch.tensor(0.0, device=device)).item():.1f}"
                                f", beta={br2.get('beta', torch.tensor(0.0, device=device)).item():.1f}"
                                f", hydro={br2.get('hydro', torch.tensor(0.0, device=device)).item():.1f}"
                                f", mj={br2.get('mj', torch.tensor(0.0, device=device)).item():.1f}"
                                f", elec={br2.get('elec', torch.tensor(0.0, device=device)).item():.1f}"
                            )
                        except Exception:
                            pass
                    logger.info(msg2)
                
                if lbfgs_step_count % 40 == 0: 
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
                    best_omega = [o.detach().clone() for o in omega_params]
                    best_rb = rb_params.detach().clone() if rb_params is not None else None
    with torch.no_grad():
        if best_phi is not None and best_psi is not None and best_omega is not None:
            for i in range(len(phi_params)):
                phi_params[i].copy_(best_phi[i])
                psi_params[i].copy_(best_psi[i])
                omega_params[i].copy_(best_omega[i])
        if rb_params is not None and best_rb is not None:
            rb_params.copy_(best_rb)
        
    # Reconstruct Final
    final_N_list = []
    final_CA_list = []
    final_C_list = []
    final_CB_list = []
    
    with torch.no_grad():
        for i, d in enumerate(chain_data):
            phi = phi_params[i]
            psi = psi_params[i]
            omega = omega_params[i]
            N, CA, C, _, _, CB = build_full_structure_tensor(len(d["seq"]), phi, psi, device, bond_constants, omega=omega)
            
            if i > 0 and rb_params is not None:
                params = rb_params[i-1]
                t, r = params[:3], params[3:]
                N = _apply_transform_tensor(N, t, r, device)
                CA = _apply_transform_tensor(CA, t, r, device)
                C = _apply_transform_tensor(C, t, r, device)
                CB = _apply_transform_tensor(CB, t, r, device)
                
            final_N_list.append(N)
            final_CA_list.append(CA)
            final_C_list.append(C)
            final_CB_list.append(CB)
            
    final_N = torch.cat(final_N_list).cpu().numpy()
    final_CA = torch.cat(final_CA_list).cpu().numpy()
    final_C = torch.cat(final_C_list).cpu().numpy()
    final_CB = torch.cat(final_CB_list).cpu().numpy()
    
    elapsed = time.time() - start_time
    logger.info(f"iGPU Optimization completed in {elapsed:.4f}s [{ALGO_VERSION}]")
    
    # Breaks
    lengths = [len(d["seq"]) for d in chain_data]
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
        rp = rama_pass_rate(phi_all, psi_all, used_sequence)
        ce = contact_energy(final_CA, final_CB, used_sequence, mj_mat)
        tm = tm_score_proxy(final_CA)
        remarks = {"RAMA_PASS_RATE": f"{rp:.3f}", "CONTACT_ENERGY": f"{ce:.3f}", "TM_SCORE_PROXY": f"{tm:.3f}"}
        out_json = os.path.splitext(output_pdb)[0] + ".metrics.json"
        write_results_json(out_json, {"ramachandran_pass_rate": rp, "contact_energy": ce, "tm_score_proxy": tm, "elapsed_seconds": elapsed})
    except Exception:
        remarks = None
    ok = write_pdb(used_sequence, final_N, final_CA, final_C, output_pdb, chain_breaks=breaks, remarks=remarks)
    try:
        if frames_dir is not None and os.path.exists(frames_dir):
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
            
            try:
                if (i + 1) < len(sequence) and (i + 1) < next_break:
                    Nnext = N[i + 1]
                else:
                    Nnext = N[i]
                v1 = CA[i] - C[i]
                v2 = Nnext - C[i]
                u1 = v1 / (np.linalg.norm(v1) + 1e-9)
                u2 = v2 / (np.linalg.norm(v2) + 1e-9)
                d = -(u1 + u2)
                dn = np.linalg.norm(d)
                if dn > 1e-6:
                    d = d / (dn + 1e-9)
                else:
                    d = -u1
                O_pos = C[i] + 1.229 * d
                v1p = CA[i] - N[i]
                v2p = C[i] - CA[i]
                nrm = np.cross(v1p, v2p)
                nl = np.linalg.norm(nrm)
                if nl > 1e-6:
                    nrm = nrm / nl
                    dproj = np.dot(O_pos - N[i], nrm)
                    O_pos = O_pos - dproj * nrm
                f.write(f"ATOM  {atom_idx:5d}  O   {resn:>3s} {chain_id}{res_idx:4d}    {O_pos[0]:8.3f}{O_pos[1]:8.3f}{O_pos[2]:8.3f}  1.00  0.00           O\n")
                atom_idx += 1
                env_atoms.append(O_pos)
            except Exception:
                pass
            
            if aa != "G":
                from modules.sidechain_builder import pack_sidechain
                sc_atoms = pack_sidechain(aa, N[i], CA[i], C[i], env_atoms)
                if isinstance(sc_atoms, dict):
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
def run_backbone_fold_multichain(sequence, chain_ss_list, output_pdb, chain_slices=None, constraints=None, backend="auto"):
    try:
        return optimize_from_ss_gpu(sequence, chain_ss_list, output_pdb, constraints=constraints, backend=backend)
    except Exception as e:
        logger.error(f"iGPU Optimization failed: {e}. Falling back to standard predictor.")
        return False
