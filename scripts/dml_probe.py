import sys
import torch

print("PY", sys.version)

# DirectML probe
try:
    import torch_directml as tdml
    dev = tdml.device()
    a = torch.randn(256, 256).to(dev)
    b = torch.randn(256, 256).to(dev)
    s = (a @ b).sum().item()
    print("DML_OK", dev, s)
except Exception as e:
    print("DML_ERR", repr(e))

# XPU (Intel GPU) probe
try:
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        name = None
        try:
            name = torch.xpu.get_device_name(0)
        except Exception:
            name = 'Intel XPU'
        print("XPU_OK", name)
    else:
        print("XPU_NO")
except Exception as e:
    print("XPU_ERR", repr(e))

