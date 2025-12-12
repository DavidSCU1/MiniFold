import numpy as np
import math

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

# Chi angles (rotamers) - simplified defaults
# Format: {AA: [chi1, chi2, chi3, chi4]}
# In a real predictor, these would come from the model.
# Here we use most common rotamer values.
ROTAMERS = {
    'ALA': [],
    'ARG': [180, 180, 180, 180],
    'ASN': [-60, -60],
    'ASP': [-60, -60],
    'CYS': [-60],
    'GLN': [-60, -60, 180],
    'GLU': [-60, -60, 180],
    'GLY': [],
    'HIS': [-60, 180],
    'ILE': [-60, 180],
    'LEU': [-60, 180],
    'LYS': [-60, 180, 180, 180],
    'MET': [-60, 180, 180],
    'PHE': [-60, 90],
    'PRO': [0, 0], # Special case ring
    'SER': [-60],
    'THR': [-60],
    'TRP': [-60, 90],
    'TYR': [-60, 90],
    'VAL': [180]
}

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
    
    if res_name == "ALA":
        return atoms
        
    # 2. Build remaining atoms using NeRF and Chi angles
    # Start chain: N -> CA -> CB -> ...
    
    if not chi_angles:
        chi_angles = ROTAMERS.get(res_name, [])
        
    # Convert degrees to radians
    chis = [math.radians(x) for x in chi_angles]
    
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
        
    return atoms
