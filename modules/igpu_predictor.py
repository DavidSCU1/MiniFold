import math
import logging
import platform
import time
import numpy as np
import torch
import cpuinfo
try:
    import torch_directml
    _HAS_DML = True
except Exception:
    _HAS_DML = False

# Setup logging
logger = logging.getLogger(__name__)
ALGO_VERSION = "igpu_dml_v1.0"

# Precompute bond parameters (PyTorch tensors)
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

three_letter = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"
}

def check_gpu_availability():
    """
    Detects GPU/iGPU accelerator availability with priority to DirectML on Windows.
    Returns: (is_available, device_name, device_type)
    """
    # Prefer DirectML on Windows for broad NPU/GPU coverage
    if _HAS_DML:
        try:
            dml_dev = torch_directml.device()
            return True, "DirectML", "DirectML"
        except Exception:
            pass

    # CUDA/ROCm
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "NVIDIA CUDA"
        devtype = "NVIDIA Tensor Core"
        try:
            if hasattr(torch.version, 'hip') and torch.version.hip:
                devtype = "AMD ROCm"
        except Exception:
            pass
        return True, name, devtype
    
    # Apple MPS
    if torch.backends.mps.is_available():
        return True, "Apple Neural Engine (MPS)", "Apple NPU"

    return False, "No dedicated GPU detected", "CPU"

def _normalize_chains_to_seq_len(chain_ss_list, seq_len):
    """Normalize chain SS list so total length equals seq_len by cropping or padding last chain."""
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
    # pad
    add = seq_len - total
    parts[-1] = parts[-1] + ("C" * add)
    return parts

def ss_to_phi_psi_tensor(ss):
    """
    Converts secondary structure string to phi/psi angles using PyTorch tensors.
    """
    phi = []
    psi = []
    for c in ss:
        if c == "H":
            phi.append(-math.radians(62.0))
            psi.append(-math.radians(41.0))
        elif c == "E":
            phi.append(-math.radians(135.0))
            psi.append(math.radians(135.0))
        else:
            # Random coil: vectorizable later, for now consistent with baseline
            phi.append(math.radians(-60.0 + np.random.uniform(-15.0, 15.0)))
            psi.append(math.radians(140.0 + np.random.uniform(-15.0, 15.0)))
    
    # Return as tensors
    return torch.tensor(phi, dtype=torch.float32), torch.tensor(psi, dtype=torch.float32)

def place_atom_tensor(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, bond_len: torch.Tensor, bond_angle: torch.Tensor, dihedral: torch.Tensor) -> torch.Tensor:
    ab = b - a
    bc = c - b
    ab = ab / (torch.norm(ab) + 1e-9)
    bc = bc / (torch.norm(bc) + 1e-9)
    
    # Custom cross product to avoid aten::linalg_cross on DirectML
    n = torch.stack([
        ab[1] * bc[2] - ab[2] * bc[1],
        ab[2] * bc[0] - ab[0] * bc[2],
        ab[0] * bc[1] - ab[1] * bc[0]
    ], dim=0)
    
    n = n / (torch.norm(n) + 1e-9)
    
    m = torch.stack([
        n[1] * bc[2] - n[2] * bc[1],
        n[2] * bc[0] - n[0] * bc[2],
        n[0] * bc[1] - n[1] * bc[0]
    ], dim=0)
    
    m = m / (torch.norm(m) + 1e-9)
    x = bond_len * torch.cos(bond_angle)
    y = bond_len * torch.sin(bond_angle) * torch.cos(dihedral)
    z = bond_len * torch.sin(bond_angle) * torch.sin(dihedral)
    d = -bc * x + m * y + n * z
    return b + d

def build_backbone_tensor(sequence: str, phi: torch.Tensor, psi: torch.Tensor, omega: torch.Tensor, device: torch.device) -> tuple:
    """
    Builds backbone using PyTorch operations. 
    """
    p = BOND_PARAMS
    n_len = torch.tensor(p["n_len"], device=device)
    ca_len = torch.tensor(p["ca_len"], device=device)
    c_len = torch.tensor(p["c_len"], device=device)
    ang_C_N_CA = torch.tensor(p["ang_C_N_CA"], device=device)
    ang_N_CA_C = torch.tensor(p["ang_N_CA_C"], device=device)
    ang_CA_C_N = torch.tensor(p["ang_CA_C_N"], device=device)
    N = torch.tensor([0.0, 0.0, 0.0], device=device)
    CA = torch.tensor([ca_len.item(), 0.0, 0.0], device=device)
    C = torch.tensor([ca_len.item() + c_len.item() * p["cos_N_CA_C"], c_len.item() * p["sin_N_CA_C"], 0.0], device=device)

    coords_N = [N]
    coords_CA = [CA]
    coords_C = [C]
    
    for i in range(1, len(sequence)):
        Ni = place_atom_tensor(coords_C[-1], coords_N[-1], coords_CA[-1], n_len, ang_CA_C_N, omega[i-1])
        CAi = place_atom_tensor(Ni, coords_C[-1], coords_N[-1], ca_len, ang_C_N_CA, phi[i])
        Ci = place_atom_tensor(CAi, Ni, coords_C[-1], c_len, ang_N_CA_C, psi[i])
        
        coords_N.append(Ni)
        coords_CA.append(CAi)
        coords_C.append(Ci)
        
    return torch.stack(coords_N), torch.stack(coords_CA), torch.stack(coords_C)

def _optimize_single_chain_gpu(sequence, ss_str):
    """
    Optimizes a single chain using iGPU.
    Returns N, CA, C numpy arrays.
    """
    is_gpu, device_name, device_type = check_gpu_availability()
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
        
    p, q = ss_to_phi_psi_tensor(ss_str)
    phi_init = p.to(device).requires_grad_(True)
    psi_init = q.to(device).requires_grad_(True)
    
    optimizer = torch.optim.LBFGS([phi_init, psi_init],
                                  max_iter=50,
                                  tolerance_grad=1e-5,
                                  tolerance_change=1e-5,
                                  history_size=20)
                                  
    def closure():
        optimizer.zero_grad()
        N_co, CA_co, C_co = build_backbone_tensor(sequence, phi_init, psi_init, torch.full_like(phi_init, math.pi), device)
        
        ca_diff = CA_co[1:] - CA_co[:-1]
        ca_dist = torch.norm(ca_diff, dim=1)
        loss_bond = torch.sum((ca_dist - 3.80)**2)
        
        diff_mat = CA_co.unsqueeze(0) - CA_co.unsqueeze(1)
        dist_mat = torch.norm(diff_mat, dim=2)
        
        mask = torch.triu(torch.ones_like(dist_mat), diagonal=2) > 0
        clash_dists = dist_mat[mask]
        clash_loss = torch.sum(torch.relu(2.5 - clash_dists)**2)
        
        loss_reg = torch.sum((phi_init - phi_init.detach())**2 + (psi_init - psi_init.detach())**2) * 0.001
        
        total_loss = loss_bond + 5.0 * clash_loss + loss_reg
        total_loss.backward()
        return total_loss

    try:
        optimizer.step(closure)
    except Exception as e:
        logger.warning(f"Optimization step failed: {e}")

    with torch.no_grad():
        N_final, CA_final, C_final = build_backbone_tensor(sequence, phi_init, psi_init, torch.full_like(phi_init, math.pi), device)
        N_np = N_final.cpu().numpy()
        CA_np = CA_final.cpu().numpy()
        C_np = C_final.cpu().numpy()
        
    return N_np, CA_np, C_np

def optimize_from_ss_gpu(sequence, chain_ss_list, output_pdb):
    """
    iGPU-optimized version of optimize_from_ss.
    Uses PyTorch L-BFGS optimizer which is often faster and supports GPU acceleration.
    """
    is_gpu, device_name, device_type = check_gpu_availability()
    logger.info(f"iGPU Optimization Enabled [{ALGO_VERSION}]. Device: {device_name} ({device_type})")
    
    start_time = time.time()
    
    # Normalize chains to sequence length
    chain_ss_list = _normalize_chains_to_seq_len(chain_ss_list, len(sequence))
    
    all_N = []
    all_CA = []
    all_C = []
    
    start_idx = 0
    
    logger.info("Starting iGPU optimization loop for chains...")
    
    for i, ss in enumerate(chain_ss_list):
        L = len(ss)
        sub_seq = sequence[start_idx : start_idx + L]
        start_idx += L
        
        try:
            N, CA, C = _optimize_single_chain_gpu(sub_seq, ss)
            
            # Offset
            offset = np.array([30.0 * i, 0.0, 0.0])
            all_N.append(N + offset)
            all_CA.append(CA + offset)
            all_C.append(C + offset)
        except Exception as e:
            logger.error(f"Failed to optimize chain {i}: {e}")
            return False

    elapsed = time.time() - start_time
    logger.info(f"iGPU Optimization completed in {elapsed:.4f}s [{ALGO_VERSION}]")
    
    # Concatenate coordinates
    if not all_N:
        return False
        
    final_N = np.concatenate(all_N)
    final_CA = np.concatenate(all_CA)
    final_C = np.concatenate(all_C)
    
    # Write PDB
    lengths = [len(s) for s in chain_ss_list]
    breaks = []
    acc = 0
    for L in lengths[:-1]:
        acc += L
        breaks.append(acc)
        
    return write_pdb(sequence, final_N, final_CA, final_C, output_pdb, chain_breaks=breaks)

from modules.sidechain_builder import build_sidechain

def write_pdb(sequence, N, CA, C, out_path, chain_breaks=None):
    # Same implementation as backbone_predictor to ensure consistency
    # We could refactor this into a shared utility, but for now duplicate to keep modules standalone-ish
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
        
        current_interval_idx = 0
        next_break = intervals[0][1]
        
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
            
            # Sidechain
            if aa != "G":
                sc_atoms = build_sidechain(aa, N[i], CA[i], C[i])
                
                # Approximate O (carbonyl)
                try:
                    # Need place_atom function locally or re-implement simple O placement
                    # Simple vector math:
                    # O is in plane of CA-C-N(virtual next)
                    # Let's assume O is opposite to bisector of N-CA-C projected?
                    # Or simpler:
                    u_ca_c = (C[i] - CA[i]) / np.linalg.norm(C[i] - CA[i])
                    u_c_n = (N[i] - C[i]) # Wait, N is previous.
                    # We want to place O relative to C[i].
                    # Bond C=O.
                    # We used psi to place C[i] relative to N[i], CA[i].
                    # O is fixed relative to CA[i], C[i].
                    # We can use NeRF logic here too if we import place_atom_nerf or use numpy.
                    
                    # Let's import place_atom_nerf from sidechain_builder for O placement?
                    # Or just use the one in backbone_predictor if we could import it.
                    # igpu_predictor doesn't have place_atom exposed easily.
                    # Let's use simple geometric construction for O.
                    
                    # O direction roughly: bisect angle N-CA-C? No.
                    # It's in the peptide plane.
                    # Vector CA->C is u1.
                    # Vector CA->N is u2.
                    # Normal n = cross(u1, u2).
                    # O lies in plane (u1, n) rotated?
                    # Angle N-CA-C is ~111.
                    # O is attached to C.
                    # Angle CA-C-O is ~121.
                    # Dihedral N-CA-C-O is usually near 180 (trans) or so?
                    # Actually, O and CA are cis or trans?
                    # C=O is roughly parallel to N-H (previous).
                    
                    # Let's use specific vector construction:
                    # O = C + 1.23 * (normalized(C-CA) rotated by 120 deg in plane)
                    # Which plane? The plane defined by N, CA, C?
                    # No, psi rotation defines C position. O rotates with C.
                    # So O, C, CA, and N(next) are planar.
                    # Since we don't have N(next), we can assume standard trans planar peptide.
                    # But we are building residue i.
                    # We have N, CA, C.
                    # O is in the plane of CA-C-N(next).
                    # Without N(next), we can't be sure where the plane is unless we assume psi=180?
                    # No, psi is the rotation around CA-C bond.
                    # So C-N(next) is determined by psi.
                    # O is also determined by psi?
                    # Actually, the entire peptide unit (C_i, O_i, N_i+1, H_i+1) is rigid planar.
                    # This plane rotates around CA-C bond by angle psi.
                    # So we need to place O in this rotated plane.
                    # We can construct a "reference" N_next at psi=0 (cis) or 180 (trans) and place O relative to it.
                    
                    # Simpler:
                    # We just need to place O such that C=O vector makes 123 deg with C-CA.
                    # And lies in the plane perpendicular to the N-CA-C plane? No.
                    
                    # Let's skip precise O placement for now in iGPU module or use sidechain builder to get O?
                    # sidechain builder currently doesn't build backbone O.
                    # Let's just omit O or use a placeholder direction if critical.
                    # Visualization usually infers O or it's not strictly needed for sidechain vis.
                    # But for "Full Atom", O is nice.
                    
                    # Let's try to calculate it simply:
                    # Vector v = C - CA
                    # Vector w = N - CA
                    # Normal n = cross(v, w)
                    # O is C + 1.23 * (v rotated 120 deg around n)?
                    # This places O in the N-CA-C plane.
                    # For Beta sheets this is roughly correct. For Alpha helices, O points differently.
                    # But this is better than nothing.
                    
                    v = C[i] - CA[i]
                    v /= np.linalg.norm(v)
                    w = N[i] - CA[i]
                    w /= np.linalg.norm(w)
                    n = np.cross(v, w)
                    n /= np.linalg.norm(n)
                    
                    # Rotate v around n by 120 degrees (2.09 rad)
                    # Rodrigues formula
                    theta = 2.1
                    v_rot = v * math.cos(theta) + np.cross(n, v) * math.sin(theta) + n * np.dot(n, v) * (1 - math.cos(theta))
                    
                    O_pos = C[i] + 1.23 * v_rot
                    f.write(f"ATOM  {atom_idx:5d}  O   {resn:>3s} {chain_id}{res_idx:4d}    {O_pos[0]:8.3f}{O_pos[1]:8.3f}{O_pos[2]:8.3f}  1.00  0.00           O\n")
                    atom_idx += 1
                except:
                    pass

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
