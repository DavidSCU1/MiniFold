import os
import math
import numpy as np
from typing import Optional, Dict, Any

def _read_backbone(pdb_path):
    chains = []
    N = []
    CA = []
    C = []
    last_chain = None
    with open(pdb_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("TER"):
                if len(CA) > 0:
                    chains.append({"N": np.array(N), "CA": np.array(CA), "C": np.array(C)})
                N = []
                CA = []
                C = []
                last_chain = None
                continue
            if not line.startswith("ATOM"):
                continue
            chain_id = line[21:22]
            if last_chain is None:
                last_chain = chain_id
            if chain_id != last_chain:
                if len(CA) > 0:
                    chains.append({"N": np.array(N), "CA": np.array(CA), "C": np.array(C)})
                N = []
                CA = []
                C = []
                last_chain = chain_id
            atom = line[12:16].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            v = np.array([x, y, z], dtype=float)
            if atom == "N":
                N.append(v)
            elif atom == "CA":
                CA.append(v)
            elif atom == "C":
                C.append(v)
    if len(CA) > 0:
        chains.append({"N": np.array(N), "CA": np.array(CA), "C": np.array(C)})
    return chains

def _dihedral(a, b, c, d):
    b0 = a - b
    b1 = c - b
    b2 = d - c
    b1 /= np.linalg.norm(b1) + 1e-9
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return math.degrees(math.atan2(y, x))

def _rotate_block(points, axis_point, axis_dir, angle):
    axis = axis_dir / (np.linalg.norm(axis_dir) + 1e-9)
    c = math.cos(angle)
    s = math.sin(angle)
    u = axis
    R = np.array([
        [c + u[0]*u[0]*(1-c), u[0]*u[1]*(1-c) - u[2]*s, u[0]*u[2]*(1-c) + u[1]*s],
        [u[1]*u[0]*(1-c) + u[2]*s, c + u[1]*u[1]*(1-c), u[1]*u[2]*(1-c) - u[0]*s],
        [u[2]*u[0]*(1-c) - u[1]*s, u[2]*u[1]*(1-c) + u[0]*s, c + u[2]*u[2]*(1-c)]
    ])
    pts = points - axis_point
    rot = pts @ R.T
    return rot + axis_point

def _ramachandran_adjust(chains, max_delta_deg=10.0):
    for ch in chains:
        N = ch["N"]
        CA = ch["CA"]
        C = ch["C"]
        n = len(CA)
        for i in range(n):
            if i > 0:
                phi = _dihedral(C[i-1], N[i], CA[i], C[i])
                target_phi = -60.0
                dphi = math.radians(max(-max_delta_deg, min(max_delta_deg, target_phi - phi)))
                axis_dir = CA[i] - N[i]
                if i < n:
                    block = C[i:].copy()
                    C[i:] = _rotate_block(block, N[i], axis_dir, dphi)
            if i < n - 1:
                psi = _dihedral(N[i], CA[i], C[i], N[i+1])
                target_psi = -45.0
                dpsi = math.radians(max(-max_delta_deg, min(max_delta_deg, target_psi - psi)))
                axis_dir = C[i] - CA[i]
                if i + 1 < n:
                    blockN = N[i+1:].copy()
                    N[i+1:] = _rotate_block(blockN, C[i], axis_dir, dpsi)
        ch["N"] = N
        ch["CA"] = CA
        ch["C"] = C
    return chains

def _write_backbone_only(pdb_in, chains, pdb_out):
    lines = []
    serial = 1
    rid = 1
    chain_id_base = ord("A")
    seq_ptr = 0
    with open(pdb_in, "r", encoding="utf-8", errors="replace") as f:
        remarks = []
        for line in f:
            if line.startswith("REMARK"):
                remarks.append(line.rstrip("\r\n"))
            if line.startswith("ATOM"):
                break
    for r in remarks:
        lines.append(r)
    for ci, ch in enumerate(chains):
        chain_id = chr(chain_id_base + (ci % 26))
        N = ch["N"]
        CA = ch["CA"]
        C = ch["C"]
        length = len(CA)
        for i in range(length):
            lines.append(f"ATOM  {serial:5d}  N   UNK {chain_id}{rid:4d}    {N[i][0]:8.3f}{N[i][1]:8.3f}{N[i][2]:8.3f}  1.00  0.00           N")
            serial += 1
            lines.append(f"ATOM  {serial:5d}  CA  UNK {chain_id}{rid:4d}    {CA[i][0]:8.3f}{CA[i][1]:8.3f}{CA[i][2]:8.3f}  1.00  0.00           C")
            serial += 1
            lines.append(f"ATOM  {serial:5d}  C   UNK {chain_id}{rid:4d}    {C[i][0]:8.3f}{C[i][1]:8.3f}{C[i][2]:8.3f}  1.00  0.00           C")
            serial += 1
            rid += 1
        lines.append("TER")
        seq_ptr += length
    lines.append("END")
    with open(pdb_out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")
    return True

def _openmm_min_md(pdb_in, pdb_out, steps):
    try:
        try:
            import openmm
            import openmm.app as app
            import openmm.unit as unit
        except Exception:
            from simtk import openmm
            from simtk.openmm import app
            from simtk import unit
        pdb = app.PDBFile(pdb_in)
        ff = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
        modeller = app.Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(ff)
        system = ff.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
        integrator = openmm.LangevinIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.002*unit.picoseconds)
        platform = openmm.Platform.getPlatformByName("CPU")
        sim = app.Simulation(modeller.topology, system, integrator, platform)
        sim.context.setPositions(modeller.positions)
        sim.minimizeEnergy()
        if steps and steps > 0:
            sim.step(int(steps))
        with open(pdb_out, "w") as outp:
            app.PDBFile.writeFile(sim.topology, sim.context.getState(getPositions=True).getPositions(), outp)
        return True
    except Exception:
        return False

def analyze_ubiquitin_core(pdb_path, sequence) -> Optional[Dict[str, Any]]:
    seq = sequence.upper()
    if "UBIQUITIN" in seq or len(seq) in (76, 77):
        coords = {}
        with open(pdb_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                resn = line[17:20].strip()
                chain = line[21:22]
                resid = int(line[22:26])
                atom = line[12:16].strip()
                if resid in (44, 70) and resn in ("ILE", "VAL"):
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                    coords.setdefault((chain, resid), {}).setdefault(atom, np.array([x, y, z], dtype=float))
        p44 = coords.get(("A", 44)) or {}
        p70 = coords.get(("A", 70)) or {}
        cands = []
        for a in ("CB", "CG", "CG1", "CG2"):
            for b in ("CB", "CG", "CG1", "CG2"):
                if a in p44 and b in p70:
                    d = float(np.linalg.norm(p44[a] - p70[b]))
                    cands.append(d)
        if cands:
            return {"pair": {"distance": min(cands)}}
    return None

def run_refinements(pdb_path, sequence, do_ramachandran=False, do_hydrophobic=False, repack_long=False, md_engine=None, md_steps=0):
    updated = False
    if do_ramachandran:
        chains = _read_backbone(pdb_path)
        chains = _ramachandran_adjust(chains, max_delta_deg=10.0)
        _write_backbone_only(pdb_path, chains, pdb_path)
        updated = True
    if md_engine == "openmm" and md_steps and md_steps > 0:
        ok = _openmm_min_md(pdb_path, pdb_path, md_steps)
        if ok:
            updated = True
    return updated
