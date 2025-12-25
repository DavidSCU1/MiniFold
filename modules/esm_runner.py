import os
import sys
import json
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
    if hasattr(state, "infer_pdb"):
        model = state
    else:
        raise RuntimeError("ESM checkpoint must be a full model object with infer_pdb.")
    if isinstance(model, torch.nn.Module):
        model = model.to(device)
        model.eval()
    return model, None, device, actual_backend


def predict_structure_with_esm(
    fasta_path,
    ss_path,
    output_pdb_path,
    model_path=None,
    npz_output_path=None,
    backend="auto",
):
    sequence = load_sequence_from_fasta(fasta_path)
    model, alphabet, device, actual_backend = load_esmfold_model(
        model_path=model_path, backend=backend
    )
    with torch.no_grad():
        pdb_str = model.infer_pdb(sequence)
    out_dir = os.path.dirname(os.path.abspath(output_pdb_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(output_pdb_path, "w", encoding="utf-8") as f:
        f.write(pdb_str)
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
