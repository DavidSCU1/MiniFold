import os
from typing import Iterable, List, Optional

def _iter_candidate_env_files(path: Optional[str]) -> List[str]:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if path:
        base_dir = path if os.path.isdir(path) else os.path.dirname(path)
        return [
            os.path.join(base_dir, ".env"),
            os.path.join(base_dir, ".env.oneapi"),
            os.path.join(base_dir, ".env.local"),
        ]
    cwd = os.getcwd()
    return [
        os.path.join(cwd, ".env"),
        os.path.join(cwd, ".env.oneapi"),
        os.path.join(cwd, ".env.local"),
        os.path.join(root, ".env"),
        os.path.join(root, ".env.oneapi"),
        os.path.join(root, ".env.local"),
    ]

def _normalize_env_line(line: str) -> str:
    s = line.strip()
    if not s:
        return ""
    if s.startswith("#"):
        return ""
    if s.startswith("-"):
        s = s[1:].lstrip()
    if s.lower().startswith("export "):
        s = s[7:].lstrip()
    return s

def load_env(path: Optional[str] = None) -> bool:
    candidates: List[str] = []
    seen = set()
    for p in _iter_candidate_env_files(path):
        if p not in seen:
            seen.add(p)
            candidates.append(p)
    loaded = False
    for p in candidates:
        if not os.path.exists(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    s = _normalize_env_line(line)
                    if not s or "=" not in s:
                        continue
                    k, v = s.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k:
                        os.environ[k] = v
            loaded = True
        except Exception:
            continue
    return loaded
