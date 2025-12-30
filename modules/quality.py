import math
import json
import numpy as np
from typing import Dict, Any, List, Tuple
from .ss_generator import _compute_propensity_arrays

def rama_pass_rate(phi, psi, seq):
    phi = np.asarray(phi, dtype=float)
    psi = np.asarray(psi, dtype=float)
    n = len(phi)
    if n == 0:
        return 0.0
    def centers(group):
        if group == "GLY":
            return [(1.4, -2.8), (-1.4, 2.8), (1.4, 2.8), (-1.4, -2.8)]
        if group == "PRO":
            return [(-1.0, 2.6), (-1.0, -0.5)]
        return [(-1.2, 2.4), (-1.0, -0.8)]
    def sigma(group):
        if group == "GLY":
            return (0.6, 0.6)
        if group == "PRO":
            return (0.5, 0.5)
        return (0.4, 0.4)
    count = 0
    for i in range(n):
        aa = seq[i] if i < len(seq) else "A"
        g = "GLY" if aa == "G" else ("PRO" if aa == "P" else "General")
        cs = centers(g)
        sg = sigma(g)
        ok = False
        for cphi, cpsi in cs:
            val = ((phi[i] - cphi) ** 2) / (sg[0] ** 2) + ((psi[i] - cpsi) ** 2) / (sg[1] ** 2)
            if val < 4.0:
                ok = True
                break
        if ok:
            count += 1
    return count / float(n)

def contact_energy(CA, CB, seq, mj_matrix):
    CA = np.asarray(CA, dtype=float)
    CB = np.asarray(CB, dtype=float)
    n = len(CA)
    if n == 0:
        return 0.0
    seq = str(seq or "")
    diff = CB[np.newaxis, :, :] - CB[:, np.newaxis, :]
    dist = np.linalg.norm(diff, axis=2) + 1e-6
    r_min = 2.6
    r_opt = 3.8
    r_cut = 6.0
    r_hard = 2.2
    A_soft = 2.0
    A_hard = 10.0
    B = 3.0
    sigma = 0.6
    base_E = np.zeros_like(dist)
    idx = np.arange(n)
    pair_mask = (np.abs(idx[:, np.newaxis] - idx[np.newaxis, :]) > 2)
    mask_cut = (dist <= r_cut) & pair_mask
    mask_hard = (dist < r_hard) & mask_cut
    mask_soft = (dist >= r_hard) & (dist < r_min) & mask_cut
    mask_well = (dist >= r_min) & (dist <= r_opt) & mask_cut
    if np.any(mask_soft):
        base_E[mask_soft] = A_soft * (r_min - dist[mask_soft]) ** 2
    if np.any(mask_hard):
        base_E[mask_hard] = A_hard * (r_hard - dist[mask_hard]) ** 2 + A_soft * (r_min - r_hard) ** 2
    if np.any(mask_well):
        base_E[mask_well] = -B * np.exp(-((dist[mask_well] - r_opt) ** 2) / (sigma ** 2))
    atom_type = np.zeros((n,), dtype=int)
    seq_upper = [c.upper() for c in seq]
    for i in range(min(n, len(seq_upper))):
        aa = seq_upper[i]
        if aa in ("A", "V", "I", "L", "M", "F", "W", "Y"):
            atom_type[i] = 0
        elif aa in ("K", "R", "H"):
            atom_type[i] = 2
        elif aa in ("D", "E"):
            atom_type[i] = 3
        else:
            atom_type[i] = 1
    W = np.array([
        [1.0, 0.4, 0.2, 0.2],
        [0.4, 0.6, 0.8, 0.8],
        [0.2, 0.8, 0.3, 1.2],
        [0.2, 0.8, 1.2, 0.3],
    ], dtype=float)
    ti = atom_type[:, np.newaxis]
    tj = atom_type[np.newaxis, :]
    wtype = W[ti, tj]
    E = base_E * wtype
    if not np.any(mask_cut):
        return 0.0
    score = np.sum(E[mask_cut])
    pairs = float(np.sum(mask_cut))
    return float(score / pairs)

def tm_score_proxy(CA):
    CA = np.asarray(CA, dtype=float)
    if len(CA) == 0:
        return 0.0
    centroid = np.mean(CA, axis=0)
    rg2 = np.sum(np.sum((CA - centroid) ** 2, axis=1)) / len(CA)
    rg = math.sqrt(max(rg2, 1e-9))
    target = 2.2 * (len(CA) ** 0.38)
    x = abs(rg - target) / max(target, 1e-6)
    return float(np.clip(1.0 - x, 0.0, 1.0))

def write_results_json(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False

def _parse_pdb_coords(pdb_path: str) -> Tuple[List[int], np.ndarray, np.ndarray]:
    ca = []
    cb = []
    resseq = []
    with open(pdb_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            name = line[12:16].strip()
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            rid = int(line[22:26].strip())
            if name == "CA":
                ca.append([x, y, z])
                resseq.append(rid)
            elif name == "CB":
                cb.append([x, y, z])
    CA = np.asarray(ca, dtype=float)
    CB = np.asarray(cb, dtype=float)
    return resseq, CA, CB

def _exposure_array(CA: np.ndarray) -> np.ndarray:
    if len(CA) == 0:
        return np.zeros((0,), dtype=float)
    diff = CA[np.newaxis, :, :] - CA[:, np.newaxis, :]
    dist = np.linalg.norm(diff, axis=2) + 1e-6
    idx = np.arange(len(CA))
    mask = (np.abs(idx[np.newaxis, :] - idx[:, np.newaxis]) > 2) & (dist < 8.0)
    counts = np.sum(mask, axis=1)
    exp = 1.0 / (1.0 + counts.astype(float))
    return exp

def ss_conf(sequence: str, ss_str: str) -> float:
    H, E, C = _compute_propensity_arrays(sequence)
    n = len(sequence)
    s = 0.0
    for i in range(n):
        h = H[i]
        e = E[i]
        c = C[i]
        tot = h + e + c + 1e-9
        ph = h / tot
        pe = e / tot
        pc = c / tot
        lab = ss_str[i] if i < len(ss_str) else "C"
        if lab == "H":
            s += ph
        elif lab == "E":
            s += pe
        else:
            s += pc
    return float(s / max(n, 1))

def summarize_structure(pdb_path: str, sequence: str, ss_str: str) -> Dict[str, Any]:
    resseq, CA, CB = _parse_pdb_coords(pdb_path)
    exp = _exposure_array(CA)
    hyd_set = set(['I','L','V','F','M','A'])
    polar_set = set(['D','E','K','R','H','N','Q','S','T','Y','C','W'])
    hyd_buried = 0
    hyd_total = 0
    polar_buried = 0
    polar_total = 0
    for i, aa in enumerate(sequence):
        if i >= len(exp):
            break
        if aa in hyd_set:
            hyd_total += 1
            if exp[i] < 0.5:
                hyd_buried += 1
        elif aa in polar_set:
            polar_total += 1
            if exp[i] < 0.5:
                polar_buried += 1
    core_stability = float(hyd_buried / max(hyd_total, 1))
    loop_uncertainty = float(sum(1 for i,c in enumerate(ss_str) if c == "C" and (i < len(exp) and exp[i] < 0.3)) / max(sum(1 for c in ss_str if c == "C"), 1))
    hl = []
    el = []
    cl = []
    cur = None
    cnt = 0
    for c in ss_str:
        if cur is None:
            cur = c
            cnt = 1
        elif c == cur:
            cnt += 1
        else:
            if cur == "H":
                hl.append(cnt)
            elif cur == "E":
                el.append(cnt)
            else:
                cl.append(cnt)
            cur = c
            cnt = 1
    if cur is not None:
        if cur == "H":
            hl.append(cnt)
        elif cur == "E":
            el.append(cnt)
        else:
            cl.append(cnt)
    mj = None
    try:
        aa_order = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
        mj_matrix = np.zeros((20,20), dtype=float)
        mj = contact_energy(CA, CB if len(CB)==len(CA) else CA, sequence, mj_matrix)
    except Exception:
        mj = 0.0
    return {
        "helix_lengths": hl,
        "strand_lengths": el,
        "loop_lengths": cl,
        "buried_exposed_ratio": float(np.mean(exp)) if len(exp)>0 else 0.0,
        "core_stability": core_stability,
        "loop_uncertainty": loop_uncertainty,
        "ss_conf": ss_conf(sequence, ss_str),
        "mj_contact_energy": mj
    }
