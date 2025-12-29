import os
import sys
import json
import numpy as np
import torch


def select_device(backend="auto"):
    b = (backend or "auto").lower()
    if b == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda"), "cuda"
        return torch.device("cpu"), "cpu"
    if b == "directml":
        try:
            import torch_directml

            return torch_directml.device(), "directml"
        except Exception:
            return torch.device("cpu"), "cpu"
    if b in ("ipex", "oneapi_cpu"):
        try:
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                return torch.xpu.device("xpu"), "ipex"
        except Exception:
            pass
        return torch.device("cpu"), "cpu"
    if b == "cpu":
        return torch.device("cpu"), "cpu"
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    try:
        import torch_directml

        return torch_directml.device(), "directml"
    except Exception:
        return torch.device("cpu"), "cpu"


def load_sequence_from_fasta(fasta_path):
    try:
        from modules.input_handler import load_fasta
    except Exception:
        import sys

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if root not in sys.path:
            sys.path.insert(0, root)
        from modules.input_handler import load_fasta

    seqs = load_fasta(fasta_path)
    if not seqs:
        raise ValueError("No sequence found in FASTA")
    return seqs[0][1]


def load_ss_from_file(ss_path):
    if not ss_path:
        return None
    if not os.path.exists(ss_path):
        raise FileNotFoundError(ss_path)
    with open(ss_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    return text


def _encode_sequence_to_features(sequence, dim, device):
    aa_order = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_idx = {a: i for i, a in enumerate(aa_order)}
    L = len(sequence)
    x = torch.zeros((1, L, dim), dtype=torch.float32, device=device)
    for i, c in enumerate(sequence):
        idx = aa_to_idx.get(c.upper(), 0)
        if idx < dim:
            x[0, i, idx] = 1.0
    mask = torch.ones((1, L), dtype=torch.bool, device=device)
    return x, mask


def _ca_to_backbone(ca):
    L = ca.shape[0]
    N = np.zeros_like(ca)
    C = np.zeros_like(ca)
    for i in range(L):
        if L == 1:
            direction = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            if i == 0:
                forward = ca[i + 1] - ca[i]
                backward = forward
            elif i == L - 1:
                backward = ca[i] - ca[i - 1]
                forward = backward
            else:
                backward = ca[i] - ca[i - 1]
                forward = ca[i + 1] - ca[i]
            v = forward + backward
            norm = np.linalg.norm(v)
            if norm < 1e-6:
                v = forward
                norm = np.linalg.norm(v)
            if norm < 1e-6:
                v = np.array([1.0, 0.0, 0.0], dtype=float)
            direction = v / (np.linalg.norm(v) + 1e-8)
        N[i] = ca[i] - direction * 1.46
        C[i] = ca[i] + direction * 1.52
    return N, ca, C


def _write_backbone_pdb_simple(sequence, N, CA, C, out_path):
    three_letter = {
        "A": "ALA",
        "R": "ARG",
        "N": "ASN",
        "D": "ASP",
        "C": "CYS",
        "Q": "GLN",
        "E": "GLU",
        "G": "GLY",
        "H": "HIS",
        "I": "ILE",
        "L": "LEU",
        "K": "LYS",
        "M": "MET",
        "F": "PHE",
        "P": "PRO",
        "S": "SER",
        "T": "THR",
        "W": "TRP",
        "Y": "TYR",
        "V": "VAL",
    }
    with open(out_path, "w", encoding="utf-8") as f:
        atom_idx = 1
        res_idx = 1
        chain_id = "A"
        chain_idx = 0
        for i, aa in enumerate(sequence):
            resn = three_letter.get(aa, "UNK")
            f.write(
                f"ATOM  {atom_idx:5d}  N   {resn:>3s} {chain_id}{res_idx:4d}    {N[i][0]:8.3f}{N[i][1]:8.3f}{N[i][2]:8.3f}  1.00  0.00           N\n"
            )
            atom_idx += 1
            f.write(
                f"ATOM  {atom_idx:5d}  CA  {resn:>3s} {chain_id}{res_idx:4d}    {CA[i][0]:8.3f}{CA[i][1]:8.3f}{CA[i][2]:8.3f}  1.00  0.00           C\n"
            )
            atom_idx += 1
            f.write(
                f"ATOM  {atom_idx:5d}  C   {resn:>3s} {chain_id}{res_idx:4d}    {C[i][0]:8.3f}{C[i][1]:8.3f}{C[i][2]:8.3f}  1.00  0.00           C\n"
            )
            atom_idx += 1
            res_idx += 1
        f.write("TER\n")
        f.write("END\n")
    return True


def load_esmfold_model(model_path=None, backend="auto"):
    device, actual_backend = select_device(backend)
    ckpt_path = model_path
    if ckpt_path is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ckpt_path = os.path.join(root, "3d_model", "best_model_gpu.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path_legacy = os.path.join(root, "3d_moudel", "best_model_gpu.pt")
            if os.path.exists(ckpt_path_legacy):
                ckpt_path = ckpt_path_legacy
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise RuntimeError("ESM model checkpoint not found.")
    state = torch.load(ckpt_path, map_location="cpu")
    model_type = None
    if isinstance(state, dict):
        from model import ProteinRegressor

        reg = ProteinRegressor(input_dim=480)
        reg.load_state_dict(state, strict=True)
        reg = reg.to(device)
        reg.eval()
        model = reg
        model_type = "regressor_state_dict"
    elif hasattr(state, "infer_pdb"):
        model = state
        if isinstance(model, torch.nn.Module):
            model = model.to(device)
            model.eval()
        model_type = "full_model_infer_pdb"
    else:
        raise RuntimeError("Unsupported ESM checkpoint format.")
    return model, model_type, device, actual_backend


def predict_structure_with_esm(
    fasta_path,
    ss_path,
    output_pdb_path,
    model_path=None,
    npz_output_path=None,
    backend="auto",
):
    sequence = load_sequence_from_fasta(fasta_path)
    model, model_type, device, actual_backend = load_esmfold_model(
        model_path=model_path, backend=backend
    )
    out_dir = os.path.dirname(os.path.abspath(output_pdb_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    if model_type == "full_model_infer_pdb":
        with torch.no_grad():
            try:
                pdb_str = model.infer_pdb(sequence)
            except TypeError:
                pdb_str = model.infer_pdb(sequence, None)
        with open(output_pdb_path, "w", encoding="utf-8") as f:
            f.write(pdb_str)
    elif model_type == "regressor_state_dict":
        x, mask = _encode_sequence_to_features(sequence, getattr(model, "input_dim", 480), device)
        with torch.no_grad():
            coords = model(x, mask)
        ca = coords[0].detach().cpu().numpy()
        N, CA, C = _ca_to_backbone(ca)
        _write_backbone_pdb_simple(sequence, N, CA, C, output_pdb_path)
    else:
        raise RuntimeError("Unknown ESM model type.")
    return {"backend": actual_backend, "device": str(device)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--ss", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--npz", default=None)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    try:
        info = predict_structure_with_esm(
            fasta_path=args.fasta,
            ss_path=args.ss,
            output_pdb_path=args.output,
            model_path=args.model,
            npz_output_path=args.npz,
            backend=args.backend,
        )
        print(json.dumps(info))
    except Exception as e:
        print(f"[ESM Runner] Error: {e}", file=sys.stderr)
        sys.exit(1)
