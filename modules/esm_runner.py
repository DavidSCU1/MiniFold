import os
import sys
import json
import numpy as np
import torch
import importlib
import importlib.util

esm = None


def _ensure_esm_loaded():
    global esm
    if esm is not None:
        return
    try:
        import esm as _esm

        esm = _esm
        return
    except Exception:
        pass
    try:
        hub_dir = torch.hub.get_dir()
        repo_dir = os.path.join(hub_dir, "facebookresearch_esm_main")
        init_path = os.path.join(repo_dir, "esm", "__init__.py")
        if os.path.exists(init_path):
            spec = importlib.util.spec_from_file_location("esm", init_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["esm"] = module
            spec.loader.exec_module(module)
            esm = module
            return
    except Exception:
        pass
    esm = None


def _load_esm2_model():
    _ensure_esm_loaded()
    if esm is not None:
        pretrained_mod = getattr(esm, "pretrained", None)
        if pretrained_mod is not None and hasattr(pretrained_mod, "esm2_t33_650M_UR50D"):
            return pretrained_mod.esm2_t33_650M_UR50D()
    raise RuntimeError(
        "ESM-2 model definition not available. Ensure either fair-esm is installed\n"
        "or torch.hub has cloned facebookresearch/esm into the local cache."
    )


def _load_esmfold_core():
    _ensure_esm_loaded()
    if esm is not None:
        pretrained_mod = getattr(esm, "pretrained", None)
        if pretrained_mod is not None and hasattr(pretrained_mod, "esmfold_v1"):
            return pretrained_mod.esmfold_v1()
    raise RuntimeError(
        "ESMFold model definition not available. Ensure either fair-esm is installed\n"
        "or torch.hub has cloned facebookresearch/esm into the local cache."
    )


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


def generate_npz_from_esm(fasta_path, ss_path, npz_output_path, backend="auto"):
    sequence = load_sequence_from_fasta(fasta_path)
    ss_text = load_ss_from_file(ss_path) if ss_path else None
    device, actual_backend = select_device(backend)
    esm_model, alphabet = _load_esm2_model()
    esm_model = esm_model.to(device)
    esm_model.eval()
    batch_converter = alphabet.get_batch_converter()
    data = [("protein", sequence)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)
    with torch.no_grad():
        out = esm_model(tokens, repr_layers=[33], return_contacts=False)
        rep = out["representations"][33].cpu().numpy()
    tokens_np = tokens.cpu().numpy()
    ss_arr = None
    if ss_text:
        ss_arr = np.frombuffer(ss_text.encode("utf-8"), dtype=np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(npz_output_path)), exist_ok=True)
    np.savez(npz_output_path, tokens=tokens_np, representation=rep, ss=ss_arr, backend=actual_backend)
    return {"backend": actual_backend}


def load_esmfold_model(model_path=None, backend="auto"):
    device, actual_backend = select_device(backend)
    model, alphabet = _load_esmfold_core()
    ckpt_path = model_path
    if ckpt_path is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ckpt_path = os.path.join(root, "3d_model", "best_model_gpu.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path_legacy = os.path.join(root, "3d_moudel", "best_model_gpu.pt")
            if os.path.exists(ckpt_path_legacy):
                ckpt_path = ckpt_path_legacy
    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict):
            model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    return model, alphabet, device, actual_backend


def predict_structure_with_esm(
    fasta_path,
    ss_path,
    output_pdb_path,
    model_path=None,
    npz_output_path=None,
    backend="auto",
):
    sequence = load_sequence_from_fasta(fasta_path)
    if npz_output_path:
        generate_npz_from_esm(fasta_path, ss_path, npz_output_path, backend=backend)
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
    info = predict_structure_with_esm(
        fasta_path=args.fasta,
        ss_path=args.ss,
        output_pdb_path=args.output,
        model_path=args.model,
        npz_output_path=args.npz,
        backend=args.backend,
    )
    print(json.dumps(info))
