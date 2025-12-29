import numpy as np
import math
import os
import json

# Basic Sidechain Templates (Chi angles will be used to rotate them)
# Coordinates relative to CA at origin, N on -x axis, C in xy plane.
# Simplified representation: CB only for now, or full atom templates.
# To do this properly without a huge library, we can use average bond lengths/angles
# and construct atom by atom using NeRF (Natural Extension Reference Frame) method.

# Bond lengths and angles for sidechains (approximate)
SC_BONDS = {
    'CB': {'len': 1.53, 'angle': math.radians(110.1)}, # CA-CB
    'CG': {'len': 1.52, 'angle': math.radians(113.8)}, # CB-CG
    'CD': {'len': 1.52, 'angle': math.radians(113.0)}, # CG-CD
    'CE': {'len': 1.52, 'angle': math.radians(113.0)}, # CD-CE
    'CZ': {'len': 1.52, 'angle': math.radians(113.0)}, # CE-CZ
    'OD': {'len': 1.23, 'angle': math.radians(120.0)}, # C=O type
    'ND': {'len': 1.33, 'angle': math.radians(120.0)}, # C-N type
    'SD': {'len': 1.81, 'angle': math.radians(110.0)}, # C-S type
}

# Advanced Rotamer Library (Approximation of Dunbrack 2010 top probabilities)
# Format: {AA: [[chi1, chi2, ...], [chi1, chi2, ...], ...]}
ROTAMER_LIBRARY = {
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

try:
    _base_dir = os.path.dirname(os.path.dirname(__file__))
    _dun_path = os.path.join(_base_dir, "data", "dunbrack2010.json")
    if os.path.exists(_dun_path):
        with open(_dun_path, "r", encoding="utf-8") as _f:
            _lib = json.load(_f)
            for k, v in _lib.items():
                if isinstance(v, list) and all(isinstance(x, list) for x in v):
                    ROTAMER_LIBRARY[k] = v
except Exception:
    pass

# Compatibility alias for single-rotamer usage
ROTAMERS = {k: v[0] for k, v in ROTAMER_LIBRARY.items()}

def _wrap_deg(x):
    return (x + 180.0) % 360.0 - 180.0

def _circ_diff_deg(a, b):
    d = _wrap_deg(a - b)
    return abs(d)

def _nearest_chi1_state(chi1_deg):
    x = _wrap_deg(float(chi1_deg))
    if x <= -150.0:
        x = 180.0
    states = (-60.0, 60.0, 180.0)
    best = states[0]
    best_d = _circ_diff_deg(x, best)
    for s in states[1:]:
        d = _circ_diff_deg(x, s)
        if d < best_d:
            best = s
            best_d = d
    return best

def _sample_rotamer_by_chi1(res_name, options):
    if not options or len(options) == 1:
        return options[0] if options else []
    if res_name in ("ALA", "GLY", "PRO"):
        return options[0]

    pref = {
        "LEU": (0.45, 0.45, 0.10),
        "ILE": (0.45, 0.45, 0.10),
        "VAL": (0.45, 0.45, 0.10),
        "PHE": (0.20, 0.20, 0.60),
        "TYR": (0.20, 0.20, 0.60),
        "TRP": (0.20, 0.20, 0.60),
        "HIS": (0.20, 0.20, 0.60),
    }.get(res_name, (0.34, 0.33, 0.33))

    r = np.random.rand()
    if r < pref[0]:
        target = -60.0
    elif r < pref[0] + pref[1]:
        target = 60.0
    else:
        target = 180.0

    filtered = []
    for rot in options:
        if not rot:
            continue
        if _nearest_chi1_state(rot[0]) == target:
            filtered.append(rot)
    if not filtered:
        filtered = options
    return filtered[int(np.random.randint(0, len(filtered)))]

def pack_sidechain(aa_code, n_coord, ca_coord, c_coord, local_environment_atoms=None, forced_chi1=None, forced_chi2=None):
    """
    Selects the best rotamer from the library by checking for clashes.
    Currently checks self-consistency and simple steric clash if env provided.
    If no env, returns the most probable (first) rotamer.
    """
    three_letter = {
        "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
        "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
        "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
        "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"
    }
    res_name = three_letter.get(aa_code, "ALA")
    options = ROTAMER_LIBRARY.get(res_name, [[]])
    base = options[0] if options and options[0] else []
    chi1_options = []
    if base:
        for s in (-60.0, 60.0, 180.0):
            chi1_options.append([s] + base[1:])
    options = chi1_options if chi1_options else options
    
    if forced_chi1 is not None:
        try:
            target = _nearest_chi1_state(forced_chi1)
            filtered = []
            for rot in options:
                if not rot:
                    continue
                if _nearest_chi1_state(rot[0]) == target:
                    r = rot[:]
                    if forced_chi2 is not None and len(r) > 1:
                        r[1] = float(forced_chi2)
                    filtered.append(r)
            if filtered:
                options = filtered
        except Exception:
            pass
    if local_environment_atoms is None or (isinstance(local_environment_atoms, np.ndarray) and local_environment_atoms.size == 0):
        rot = _sample_rotamer_by_chi1(res_name, options)
        if forced_chi2 is not None and rot and len(rot) > 1:
            rot = rot[:]
            rot[1] = float(forced_chi2)
        return build_sidechain(aa_code, n_coord, ca_coord, c_coord, rot)
    if len(options) == 1:
        return build_sidechain(aa_code, n_coord, ca_coord, c_coord, options[0])
        
    best_atoms = None
    min_clash = 1e9
    
    # Iterate through rotamers
    for rot in options:
        atoms = build_sidechain(aa_code, n_coord, ca_coord, c_coord, rot)
        clash_score = 0.0
        heavy_loss = 0.0
        lj_sum = 0.0
        
        # Check clashes against environment
        if local_environment_atoms is not None and len(local_environment_atoms) > 0:
            sc_coords = np.array(list(atoms.values()))
            if len(sc_coords) == 0: 
                if min_clash > 0:
                     min_clash = 0
                     best_atoms = atoms
                continue

            env = local_environment_atoms
            if not isinstance(env, np.ndarray):
                env = np.array(env)
            score = _score_atoms(aa_code, sc_coords, env)
        else:
            score = 0.0
        if score < min_clash:
            min_clash = score
            best_atoms = atoms
            best_rot = rot
        
        # If perfect score, break early (greedy)
        if min_clash == 0:
            break
            
    if best_atoms is not None:
        if local_environment_atoms is not None and len(local_environment_atoms) > 0:
            refined = _refine_chi(aa_code, n_coord, ca_coord, c_coord, best_rot, local_environment_atoms)
            if refined is not None:
                return refined
        return best_atoms

def _score_atoms(aa_code, sc_coords, env):
    diff = sc_coords[:, np.newaxis, :] - env[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2) + 1e-6
    clashes = np.sum((dists < 1.5) & (dists > 0.1))
    heavy_loss = float(np.sum(np.maximum(0.0, 2.6 - dists) ** 2))
    sigma = 4.0
    epsilon = 0.05
    r = np.clip(dists, 0.5, None)
    lj = epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
    lj_sum = float(np.sum(lj))
    contact_mask = (dists < 4.5) & (dists > 1.5)
    contact_count = float(np.sum(contact_mask))
    hydrophobic_set = set(["A", "V", "I", "L", "M", "F", "W", "Y"])
    polar_charged_set = set(["R", "N", "D", "C", "Q", "E", "G", "H", "K", "S", "T"])
    exposure_term = 0.0
    if aa_code in hydrophobic_set:
        exposure_term = -0.1 * contact_count
    elif aa_code in polar_charged_set:
        exposure_term = 0.1 * contact_count
    return clashes * 5.0 + heavy_loss * 1.0 + lj_sum * 0.2 + exposure_term

def _refine_chi(aa_code, n_coord, ca_coord, c_coord, chi_angles, env):
    if chi_angles is None or len(chi_angles) == 0:
        return None
    env_arr = env if isinstance(env, np.ndarray) else np.array(env)
    base_atoms = build_sidechain(aa_code, n_coord, ca_coord, c_coord, chi_angles)
    base_coords = np.array(list(base_atoms.values()))
    best_score = _score_atoms(aa_code, base_coords, env_arr)
    best_atoms = base_atoms
    chis = np.array(chi_angles, dtype=float)
    for scale in [10.0, 5.0]:
        for _ in range(16):
            trial = chis + (np.random.rand(len(chis)) - 0.5) * 2.0 * scale
            atoms = build_sidechain(aa_code, n_coord, ca_coord, c_coord, list(trial))
            sc = np.array(list(atoms.values()))
            if len(sc) == 0:
                continue
            s = _score_atoms(aa_code, sc, env_arr)
            if s < best_score:
                best_score = s
                best_atoms = atoms
                chis = trial
    return best_atoms

def place_atom_nerf(a, b, c, bond_len, bond_angle, torsion):
    """
    NeRF (Natural Extension Reference Frame) method to place atom d.
    a, b, c are coordinates of previous 3 atoms.
    """
    bc = c - b
    ab = b - a
    bc = bc / np.linalg.norm(bc)
    
    n = np.cross(ab, bc)
    n = n / np.linalg.norm(n)
    
    # Rotation matrix to align with BC frame
    M = np.array([
        bc,
        np.cross(n, bc),
        n
    ]).T
    
    # D coordinates in local frame
    d_local = np.array([
        -bond_len * np.cos(bond_angle),
        bond_len * np.sin(bond_angle) * np.cos(torsion),
        bond_len * np.sin(bond_angle) * np.sin(torsion)
    ])
    
    return c + M @ d_local

def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v
    return v / n

def _rotate_around_axis(v, axis, angle):
    axis = _normalize(axis)
    c = math.cos(angle)
    s = math.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * (np.dot(axis, v)) * (1.0 - c)

def _orthonormal_frame(axis1, ref):
    a1 = _normalize(axis1)
    r = ref - np.dot(ref, a1) * a1
    if np.linalg.norm(r) < 1e-9:
        if abs(a1[0]) < 0.9:
            r = np.array([1.0, 0.0, 0.0])
        else:
            r = np.array([0.0, 1.0, 0.0])
        r = r - np.dot(r, a1) * a1
    a2 = _normalize(r)
    a3 = _normalize(np.cross(a1, a2))
    a2 = _normalize(np.cross(a3, a1))
    return a1, a2, a3

def _rotate_plane_axes(axis1, axis2, axis3, angle):
    c = math.cos(angle)
    s = math.sin(angle)
    a2 = axis2 * c + axis3 * s
    a3 = -axis2 * s + axis3 * c
    return axis1, a2, a3

def _place_planar_template(origin, axis2, axis3, coords_2d):
    out = {}
    for k, (x, y) in coords_2d.items():
        out[k] = origin + axis2 * float(x) + axis3 * float(y)
    return out

def _his_template_2d():
    s = 1.37
    R = s / (2.0 * math.sin(math.pi / 5.0))
    ang = [0.0, 2.0 * math.pi / 5.0, 4.0 * math.pi / 5.0, 6.0 * math.pi / 5.0, 8.0 * math.pi / 5.0]
    verts = [(R * math.cos(a), R * math.sin(a)) for a in ang]
    v0x, v0y = verts[0]
    rel = [(x - v0x, y - v0y) for (x, y) in verts]
    return {
        "ND1": rel[1],
        "CE1": rel[2],
        "NE2": rel[3],
        "CD2": rel[4],
    }

_TRP_TEMPLATE_2D = {
    "CD1": (-1.35, 1.20),
    "NE1": (-2.55, 0.50),
    "CE2": (-1.90, -0.75),
    "CD2": (-0.55, -0.60),
    "CE3": (0.45, -1.55),
    "CZ3": (1.75, -1.05),
    "CH2": (2.05, 0.25),
    "CZ2": (0.85, 0.95),
}

def _build_his_ring(n_coord, ca_coord, cb_coord, chi1, chi2):
    atoms = {"CB": cb_coord}
    cg = place_atom_nerf(n_coord, ca_coord, cb_coord, 1.50, math.radians(110.5), chi1)
    atoms["CG"] = cg
    axis1 = cg - cb_coord
    ref = ca_coord - cb_coord
    a1, a2, a3 = _orthonormal_frame(axis1, ref)
    _, a2r, a3r = _rotate_plane_axes(a1, a2, a3, chi2)
    atoms.update(_place_planar_template(cg, a2r, a3r, _his_template_2d()))
    return atoms

def _build_trp_ring(n_coord, ca_coord, cb_coord, chi1, chi2):
    atoms = {"CB": cb_coord}
    cg = place_atom_nerf(n_coord, ca_coord, cb_coord, 1.50, math.radians(110.5), chi1)
    atoms["CG"] = cg
    axis1 = cg - cb_coord
    ref = ca_coord - cb_coord
    a1, a2, a3 = _orthonormal_frame(axis1, ref)
    _, a2r, a3r = _rotate_plane_axes(a1, a2, a3, chi2)
    atoms.update(_place_planar_template(cg, a2r, a3r, _TRP_TEMPLATE_2D))
    return atoms

def _build_phe_tyr_ring(res_name, n_coord, ca_coord, c_coord, cb_coord, chi1, chi2):
    atoms = {"CB": cb_coord}
    cg_coord = place_atom_nerf(n_coord, ca_coord, cb_coord, 1.50, math.radians(110.5), chi1)
    atoms["CG"] = cg_coord

    cd1 = place_atom_nerf(ca_coord, cb_coord, cg_coord, 1.39, math.radians(120.0), chi2)
    cd2 = place_atom_nerf(ca_coord, cb_coord, cg_coord, 1.39, math.radians(120.0), chi2 + math.pi)
    atoms["CD1"] = cd1
    atoms["CD2"] = cd2

    nrm = np.cross(cd1 - cg_coord, cd2 - cg_coord)
    if np.linalg.norm(nrm) < 1e-6:
        nrm = np.cross(cb_coord - cg_coord, cd1 - cg_coord)
    nrm = _normalize(nrm)

    u1 = _normalize(cg_coord - cd1)
    u2 = _normalize(cg_coord - cd2)
    e1a = _normalize(_rotate_around_axis(u1, nrm, 2.0 * math.pi / 3.0))
    e1b = _normalize(_rotate_around_axis(u1, nrm, -2.0 * math.pi / 3.0))
    ce1 = cd1 + 1.39 * (e1a if np.linalg.norm((cd2 - (cd1 + 1.39 * e1a))) > np.linalg.norm((cd2 - (cd1 + 1.39 * e1b))) else e1b)

    e2a = _normalize(_rotate_around_axis(u2, nrm, 2.0 * math.pi / 3.0))
    e2b = _normalize(_rotate_around_axis(u2, nrm, -2.0 * math.pi / 3.0))
    ce2 = cd2 + 1.39 * (e2a if np.linalg.norm((cd1 - (cd2 + 1.39 * e2a))) > np.linalg.norm((cd1 - (cd2 + 1.39 * e2b))) else e2b)

    atoms["CE1"] = ce1
    atoms["CE2"] = ce2

    u3 = _normalize(cd1 - ce1)
    cz_a = ce1 + 1.39 * _normalize(_rotate_around_axis(u3, nrm, 2.0 * math.pi / 3.0))
    cz_b = ce1 + 1.39 * _normalize(_rotate_around_axis(u3, nrm, -2.0 * math.pi / 3.0))
    cz = cz_a if np.linalg.norm(cz_a - ce2) < np.linalg.norm(cz_b - ce2) else cz_b
    atoms["CZ"] = cz

    if res_name == "TYR":
        out_dir = _normalize(-((ce1 - cz) + (ce2 - cz)))
        oh = cz + 1.37 * out_dir
        atoms["OH"] = oh
    return atoms

def _project_atoms_to_plane(atoms, anchor_names):
    pts = [atoms.get(n) for n in anchor_names if n in atoms]
    if len(pts) < 3:
        return atoms
    p0, p1, p2 = pts[0], pts[1], pts[2]
    nrm = np.cross(p1 - p0, p2 - p0)
    if np.linalg.norm(nrm) < 1e-6:
        return atoms
    nrm = _normalize(nrm)
    for k, v in list(atoms.items()):
        if v is None:
            continue
        d = np.dot(v - p0, nrm)
        atoms[k] = v - d * nrm
    return atoms

def build_sidechain(aa_code, n_coord, ca_coord, c_coord, chi_angles=None):
    """
    Builds sidechain atoms for a given amino acid.
    aa_code: 1-letter code (e.g. 'A')
    n, ca, c: coordinates of backbone atoms
    chi_angles: optional list of torsion angles. If None, uses defaults.
    
    Returns: dict {atom_name: np.array([x,y,z])}
    """
    three_letter = {
        "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
        "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
        "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
        "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"
    }
    
    res_name = three_letter.get(aa_code, "ALA")
    if res_name == "GLY":
        return {}
        
    atoms = {}
    
    # 1. Build CB
    # Torsion N-C-CA-CB is roughly -122.5 deg (or 122.5 depending on def)
    # Actually, standard geometry: N-CA-CB angle ~ 110.5
    # Torsion defined by N, C, CA? No.
    # We can use N, C, CA to define frame.
    # CB is built off N-C-CA? No, usually off N-CA-C.
    # Let's use N, C, CA to place CB.
    # "The pseudo-torsion angle N-C-CA-CB is -120 degrees for L-amino acids"
    
    # Using NeRF: a=N, b=C, c=CA (Wait, sequence order matters for NeRF usually)
    # Standard: N(i) -> CA(i) -> C(i). 
    # To place CB(i):
    # Bond CA-CB
    # Angle N-CA-CB ~ 110.5
    # Torsion C(i-1)-N(i)-CA(i)-CB(i)? No.
    # Usually we use N, C, CA atoms of the SAME residue to define the tetrahedron at CA.
    # Imputed CB position:
    # 1. Find vector bisecting N-CA-C angle.
    # 2. CB is along that bisector but tilted out of plane.
    
    # Simple method:
    # 1. Calculate unit vectors u_ca_n and u_ca_c
    u_ca_n = (n_coord - ca_coord) / np.linalg.norm(n_coord - ca_coord)
    u_ca_c = (c_coord - ca_coord) / np.linalg.norm(c_coord - ca_coord)
    
    # 2. Average vector (bisector)
    # Note: angle N-CA-C is ~111 deg.
    # Direction in plane: -(u_ca_n + u_ca_c)
    v_bisect = -(u_ca_n + u_ca_c)
    v_bisect = v_bisect / np.linalg.norm(v_bisect)
    
    # 3. Normal to plane N-CA-C
    n_plane = np.cross(u_ca_n, u_ca_c)
    n_plane = n_plane / np.linalg.norm(n_plane)
    
    # 4. Construct CB vector
    # Rotate bisector by some angle around axis perpendicular to bisector and normal?
    # Or just use precomputed geometry relative to frame.
    # For L-amino acids:
    # CB lies roughly in the direction of (n_plane + 1/3 * v_bisect)? 
    # Standard formula:
    # cb = -0.58273431*n - 0.56802827*c - 0.54067466*cross(n,c) + ca
    # where n = N-CA, c = C-CA (normalized)
    
    b = 1.53333333 # Bond length CA-CB
    
    # Formula from "Evaluation of a novel method for the reconstruction of protein side-chains"
    # Actually, simple geometric construction:
    # Tetrahedral geometry.
    # Rotate u_ca_n 120 deg around u_ca_c? No.
    # Rotate n_plane around v_bisect?
    
    # Let's use the formula from OpenFold/AlphaFold logic (rigid body frames)
    # Or simpler:
    # cb = ca + b * ( -0.58*u_ca_n - 0.58*u_ca_c - 0.54*n_plane )?
    # Actually: 
    # For perfect tetrahedron, bisector is -1/sqrt(3) * (u1+u2+u3) = 0
    
    # Standard approximation:
    # CB = CA + 1.53 * ( (u_ca_n + u_ca_c).normalized * sin(alpha/2) + n_plane * cos(alpha/2) )?
    # Let's use the hardcoded formula which is robust.
    
    cb_vec = -0.5366 * u_ca_n - 0.5366 * u_ca_c - 0.6517 * n_plane
    cb_coord = ca_coord + 1.53 * (cb_vec / np.linalg.norm(cb_vec))
    atoms['CB'] = cb_coord

    if not chi_angles:
        chi_angles = ROTAMERS.get(res_name, [])
    chis = [math.radians(x) for x in chi_angles]
    if res_name in ("PHE", "TYR"):
        chi1 = chis[0] if len(chis) > 0 else math.radians(-60.0)
        chi2 = chis[1] if len(chis) > 1 else math.radians(90.0)
        atoms.update(_build_phe_tyr_ring(res_name, n_coord, ca_coord, c_coord, cb_coord, chi1, chi2))
        return atoms
    if res_name == "HIS":
        chi1 = chis[0] if len(chis) > 0 else math.radians(-60.0)
        chi2 = chis[1] if len(chis) > 1 else math.radians(90.0)
        return _build_his_ring(n_coord, ca_coord, cb_coord, chi1, chi2)
    if res_name == "TRP":
        chi1 = chis[0] if len(chis) > 0 else math.radians(-60.0)
        chi2 = chis[1] if len(chis) > 1 else math.radians(90.0)
        return _build_trp_ring(n_coord, ca_coord, cb_coord, chi1, chi2)
    
    if res_name == "ALA":
        return atoms
        
    # 2. Build remaining atoms using NeRF and Chi angles
    # Start chain: N -> CA -> CB -> ...
    
    current_chis = 0
    
    # Generic logic for linear chains (simplified)
    # We need specific topology for branched AAs (VAL, THR, ILE, LEU, ASP, GLU, ASN, GLN, ARG, LYS, MET, PHE, TYR, TRP, HIS)
    
    # Define topology: (AtomName, Parent1, Parent2, Parent3, BondLen, BondAngle, ChiIndex)
    # P1 is immediate parent, P2 is P1's parent...
    # CB is P1.
    
    topology = []
    
    if res_name == "VAL":
        # Branched at CB
        topology.append(('CG1', 'CB', 'CA', 'N', 1.52, 110.5, 0)) # Chi1 controls CG1
        topology.append(('CG2', 'CB', 'CA', 'N', 1.52, 110.5, 0)) # Chi1+120? No, distinct positions fixed relative to CB frame
        # Actually for VAL, Chi1 rotates the whole isopropyl group.
        # CG1 and CG2 are fixed relative to each other.
        # We can simulate this by adding an offset to torsion for CG2.
    elif res_name == "THR":
        topology.append(('OG1', 'CB', 'CA', 'N', 1.43, 109.5, 0))
        topology.append(('CG2', 'CB', 'CA', 'N', 1.52, 110.5, 0))
    elif res_name == "ILE":
        topology.append(('CG1', 'CB', 'CA', 'N', 1.52, 110.5, 0))
        topology.append(('CG2', 'CB', 'CA', 'N', 1.52, 110.5, 0)) # Fixed branch? No, CG2 is on CB.
        topology.append(('CD1', 'CG1', 'CB', 'CA', 1.52, 110.5, 1)) # Chi2
    elif res_name == "LEU":
        topology.append(('CG', 'CB', 'CA', 'N', 1.52, 110.5, 0))
        topology.append(('CD1', 'CG', 'CB', 'CA', 1.52, 110.5, 1))
        topology.append(('CD2', 'CG', 'CB', 'CA', 1.52, 110.5, 1))
    elif res_name == "SER":
        topology.append(('OG', 'CB', 'CA', 'N', 1.43, 109.5, 0))
    elif res_name == "CYS":
        topology.append(('SG', 'CB', 'CA', 'N', 1.81, 109.5, 0))
    elif res_name == "MET":
        topology.append(('CG', 'CB', 'CA', 'N', 1.52, 110.5, 0))
        topology.append(('SD', 'CG', 'CB', 'CA', 1.81, 110.5, 1))
        topology.append(('CE', 'SD', 'CG', 'CB', 1.78, 100.0, 2))
    elif res_name == "PHE" or res_name == "TYR":
        topology.append(('CG', 'CB', 'CA', 'N', 1.50, 110.5, 0))
        topology.append(('CD1', 'CG', 'CB', 'CA', 1.39, 120.0, 1))
        topology.append(('CD2', 'CG', 'CB', 'CA', 1.39, 120.0, 1)) # +180
        # Ring closure implied
        topology.append(('CE1', 'CD1', 'CG', 'CB', 1.39, 120.0, None)) # Fixed planar
        topology.append(('CE2', 'CD2', 'CG', 'CB', 1.39, 120.0, None))
        topology.append(('CZ', 'CE1', 'CD1', 'CG', 1.39, 120.0, None))
        if res_name == "TYR":
            topology.append(('OH', 'CZ', 'CE1', 'CD1', 1.37, 120.0, None))
    elif res_name == "TRP":
        topology.append(('CG', 'CB', 'CA', 'N', 1.50, 110.5, 0))
        topology.append(('CD1', 'CG', 'CB', 'CA', 1.37, 125.0, 1))
        topology.append(('CD2', 'CG', 'CB', 'CA', 1.43, 125.0, 1)) # +180 approx
        # Indole ring... complex. Simplified:
        topology.append(('NE1', 'CD1', 'CG', 'CB', 1.38, 110.0, None))
        topology.append(('CE2', 'CD2', 'CG', 'CB', 1.40, 110.0, None))
        topology.append(('CE3', 'CD2', 'CG', 'CB', 1.40, 120.0, None))
        topology.append(('CZ2', 'CE2', 'CD2', 'CG', 1.40, 120.0, None))
        topology.append(('CZ3', 'CE3', 'CD2', 'CG', 1.40, 120.0, None))
        topology.append(('CH2', 'CZ2', 'CE2', 'CD2', 1.40, 120.0, None))
    elif res_name == "ASP":
        topology.append(('CG', 'CB', 'CA', 'N', 1.52, 110.5, 0))
        topology.append(('OD1', 'CG', 'CB', 'CA', 1.25, 120.0, 1))
        topology.append(('OD2', 'CG', 'CB', 'CA', 1.25, 120.0, 1)) # +180
    elif res_name == "ASN":
        topology.append(('CG', 'CB', 'CA', 'N', 1.52, 110.5, 0))
        topology.append(('OD1', 'CG', 'CB', 'CA', 1.23, 120.0, 1))
        topology.append(('ND2', 'CG', 'CB', 'CA', 1.33, 120.0, 1)) # +180
    elif res_name == "GLU":
        topology.append(('CG', 'CB', 'CA', 'N', 1.52, 110.5, 0))
        topology.append(('CD', 'CG', 'CB', 'CA', 1.52, 110.5, 1))
        topology.append(('OE1', 'CD', 'CG', 'CB', 1.25, 120.0, 2))
        topology.append(('OE2', 'CD', 'CG', 'CB', 1.25, 120.0, 2)) # +180
    elif res_name == "GLN":
        topology.append(('CG', 'CB', 'CA', 'N', 1.52, 110.5, 0))
        topology.append(('CD', 'CG', 'CB', 'CA', 1.52, 110.5, 1))
        topology.append(('OE1', 'CD', 'CG', 'CB', 1.23, 120.0, 2))
        topology.append(('NE2', 'CD', 'CG', 'CB', 1.33, 120.0, 2)) # +180
    elif res_name == "LYS":
        topology.append(('CG', 'CB', 'CA', 'N', 1.52, 110.5, 0))
        topology.append(('CD', 'CG', 'CB', 'CA', 1.52, 110.5, 1))
        topology.append(('CE', 'CD', 'CG', 'CB', 1.52, 110.5, 2))
        topology.append(('NZ', 'CE', 'CD', 'CG', 1.49, 110.5, 3))
    elif res_name == "ARG":
        topology.append(('CG', 'CB', 'CA', 'N', 1.52, 110.5, 0))
        topology.append(('CD', 'CG', 'CB', 'CA', 1.52, 110.5, 1))
        topology.append(('NE', 'CD', 'CG', 'CB', 1.46, 110.5, 2))
        topology.append(('CZ', 'NE', 'CD', 'CG', 1.33, 120.0, 3))
        topology.append(('NH1', 'CZ', 'NE', 'CD', 1.33, 120.0, None)) # Planar
        topology.append(('NH2', 'CZ', 'NE', 'CD', 1.33, 120.0, None)) # Planar
    elif res_name == "PRO":
        # Ring fixed
        topology.append(('CG', 'CB', 'CA', 'N', 1.50, 104.0, 0))
        topology.append(('CD', 'CG', 'CB', 'CA', 1.50, 104.0, 1))
        # Note: CD should close to N. Simplified here.

    # Build atoms based on topology
    # Need to handle branching angles (e.g. +120, -120)
    # Simplified logic:
    
    # Store coordinates for lookup
    coords = {'N': n_coord, 'CA': ca_coord, 'CB': cb_coord, 'C': c_coord}
    
    for atom_name, p1, p2, p3, length, angle_deg, chi_idx in topology:
        a = coords[p3]
        b = coords[p2]
        c = coords[p1]
        
        angle = math.radians(angle_deg)
        
        # Determine torsion
        torsion = 0.0
        if chi_idx is not None and chi_idx < len(chis):
            torsion = chis[chi_idx]
            
        # Hardcoded branching logic offsets
        if res_name == "VAL":
            if atom_name == "CG2": torsion += math.radians(120)
        elif res_name == "THR":
            if atom_name == "CG2": torsion += math.radians(120)
        elif res_name == "ILE":
            if atom_name == "CG2": torsion += math.radians(120)
        elif res_name == "LEU":
            if atom_name == "CD2": torsion += math.radians(120)
        elif res_name in ["ASP", "ASN", "GLU", "GLN", "ARG", "PHE", "TYR", "TRP", "HIS"]:
            # Planar or symmetric branches
            if atom_name in ["OD2", "ND2", "OE2", "NE2", "CD2", "NH2", "CE2"]: 
                torsion += math.pi # 180 degrees opposite
        
        new_coord = place_atom_nerf(a, b, c, length, angle, torsion)
        atoms[atom_name] = new_coord
        coords[atom_name] = new_coord

    if res_name in ("TRP", "HIS"):
        if res_name == "TRP":
            _project_atoms_to_plane(atoms, ["CG", "CD1", "CD2"])
        else:
            _project_atoms_to_plane(atoms, ["CG", "ND1", "CD2"])

    return atoms
