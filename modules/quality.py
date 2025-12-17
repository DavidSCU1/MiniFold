import math
import json
import numpy as np

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
    aa_order = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
    idx = {aa:i for i,aa in enumerate(aa_order)}
    ids = np.array([idx.get(a, 0) for a in seq], dtype=int)
    diff = CB[np.newaxis, :, :] - CB[:, np.newaxis, :]
    dist = np.linalg.norm(diff, axis=2) + 1e-6
    w = np.exp(-((dist - 6.0) ** 2) / (2.0 * (2.0 ** 2)))
    mask = (dist < 10.0) & (dist > 3.0)
    M = mj_matrix[np.ix_(ids, ids)]
    score = np.sum(M * w * mask)
    pairs = np.sum(mask)
    if pairs < 1:
        return 0.0
    return float(score) / float(pairs)

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
