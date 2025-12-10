import math
import os

import numpy as np

three_letter = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"
}

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
    n1 = ab / np.linalg.norm(ab)
    n2 = bc / np.linalg.norm(bc)
    n = np.cross(n1, n2)
    n /= np.linalg.norm(n)
    m = np.cross(n, n2)
    x = bond_len * math.cos(bond_angle)
    y = bond_len * math.sin(bond_angle) * math.cos(dihedral)
    z = bond_len * math.sin(bond_angle) * math.sin(dihedral)
    return b + (-n2 * x) + (m * y) + (n * z)

def build_backbone(sequence, phi, psi, omega=None):
    n_len = 1.329
    ca_len = 1.458
    c_len = 1.525
    ang_C_N_CA = math.radians(121.7)
    ang_N_CA_C = math.radians(111.2)
    ang_CA_C_N = math.radians(116.2)
    if omega is None:
        omega = np.full_like(phi, math.pi)
    N = []
    CA = []
    C = []
    N.append(np.array([0.0, 0.0, 0.0]))
    CA.append(np.array([ca_len, 0.0, 0.0]))
    C.append(np.array([ca_len + c_len * math.cos(ang_N_CA_C), c_len * math.sin(ang_N_CA_C), 0.0]))
    for i in range(1, len(sequence)):
        Ni = place_atom(C[i-1], N[i-1], CA[i-1], n_len, ang_CA_C_N, omega[i-1])
        CAi = place_atom(Ni, C[i-1], N[i-1], ca_len, ang_C_N_CA, phi[i])
        Ci = place_atom(CAi, Ni, C[i-1], c_len, ang_N_CA_C, psi[i])
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

def run_backbone_fold_multichain(sequence, chain_ss_list, output_pdb, chain_slices=None):
    seq_len = len(sequence)
    if not chain_ss_list:
        return False
    lengths = [len(s) for s in chain_ss_list]
    if sum(lengths) != seq_len:
        return False
    phi = []
    psi = []
    for s in chain_ss_list:
        p, q = ss_to_phi_psi(s)
        phi.append(p)
        psi.append(q)
    phi = np.concatenate(phi)
    psi = np.concatenate(psi)
    N, CA, C = build_backbone(sequence, phi, psi)
    breaks = []
    acc = 0
    for L in lengths[:-1]:
        acc += L
        breaks.append(acc)
    return write_pdb(sequence, N, CA, C, output_pdb, chain_breaks=breaks)

