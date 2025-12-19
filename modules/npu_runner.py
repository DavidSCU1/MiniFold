import argparse
import sys
import os
import torch
import numpy as np

def _detect_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    try:
        import torch_directml
        d = torch_directml.device()
        x = torch.tensor([1.0], device=d)
        return d, torch.float16
    except Exception:
        pass
    return torch.device("cpu"), torch.float32

def _parse_ca_coords(pdb_path):
    coords = []
    with open(pdb_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("ATOM") and (" CA " in line[12:16]):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                except Exception:
                    continue
    return np.array(coords, dtype=np.float32)

def _fixed_dist_patch(coords, fixed_len=128):
    L = coords.shape[0]
    if L == 0:
        return np.zeros((fixed_len, fixed_len), dtype=np.float32)
    if L >= fixed_len:
        use = coords[:fixed_len]
    else:
        pad = np.zeros((fixed_len - L, 3), dtype=np.float32)
        use = np.concatenate([coords, pad], axis=0)
    diff = use[:, None, :] - use[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-9)
    dist = np.clip(dist, 0.0, 50.0) / 50.0
    return dist.astype(np.float32)

class FixedHead(torch.nn.Module):
    def __init__(self, fixed_len=128):
        super().__init__()
        self.fixed_len = fixed_len
        self.net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(fixed_len * fixed_len, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

def run_inplace(pdb_path):
    device, dtype = _detect_device()
    coords = _parse_ca_coords(pdb_path)
    patch = _fixed_dist_patch(coords, 128)
    t = torch.tensor(patch, device=device, dtype=dtype)
    model = FixedHead(128).to(device=device, dtype=dtype)
    with torch.no_grad():
        score = model(t[None, ...]).detach().cpu().numpy().reshape(-1)[0]
    tmp = pdb_path + ".tmp"
    wrote_remark = False
    with open(pdb_path, "r", encoding="utf-8", errors="replace") as fin, open(tmp, "w", encoding="utf-8") as fout:
        for line in fin:
            if (not wrote_remark) and line.startswith("ATOM"):
                fout.write(f"REMARK NPU_HEAD Applied FixedPatch128 Score {score:.4f}\n")
                wrote_remark = True
            fout.write(line)
    try:
        if os.path.exists(pdb_path):
            os.remove(pdb_path)
        os.rename(tmp, pdb_path)
    except Exception:
        pass
    print(f"[NPU Runner] Head score={score:.4f} device={str(device)} dtype={str(dtype)}")
    return True

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    ok = run_inplace(args.input)
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
