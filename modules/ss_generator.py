import re
import random
from typing import List, Dict, Any, Tuple

def _safe_import_pybiomed():
    try:
        from PyBioMed.PyProtein.AAIndex import GetAAIndex1
        return GetAAIndex1
    except Exception:
        return None

def _get_index(GetAAIndex1, name: str, default: float = 0.0):
    try:
        idx = GetAAIndex1(name)
        if isinstance(idx, dict):
            return {k.upper(): float(v) for k, v in idx.items()}
    except Exception:
        pass
    return {c: default for c in "ACDEFGHIKLMNPQRSTVWY"}

def _compute_propensity_arrays(sequence: str) -> Tuple[list, list, list]:
    seq = sequence.upper()
    GetAAIndex1 = _safe_import_pybiomed()
    
    # Debug log (usually hidden, but good to know)
    # print(f"DEBUG: PyBioMed Loaded: {bool(GetAAIndex1)}")

    if GetAAIndex1:
        helix_idx = _get_index(GetAAIndex1, "CHOP780201", 0.5)
        sheet_idx = _get_index(GetAAIndex1, "CHOP780202", 0.5)
        hydro_idx = _get_index(GetAAIndex1, "KYTJ820101", 0.0)
        bulk_idx = _get_index(GetAAIndex1, "CHAM830102", 0.0)
    else:
        helix_idx = {c: v for c, v in zip("ACDEFGHIKLMNPQRSTVWY",
            [1.2,0.8,1.0,0.9,1.1,1.0,0.7,1.1,1.2,1.2,1.2,1.1,0.7,1.0,0.6,0.9,0.9,0.8,1.0,1.2])}
        sheet_idx = {c: v for c, v in zip("ACDEFGHIKLMNPQRSTVWY",
            [0.8,0.9,0.9,1.1,1.0,0.9,1.1,0.7,0.8,1.3,1.3,0.8,1.0,1.2,0.6,0.9,0.9,1.3,1.2,1.2])}
        hydro_idx = {c: v for c, v in zip("ACDEFGHIKLMNPQRSTVWY",
            [1.8,-4.5,-3.5,-3.5,2.5,-3.5,-3.5,-0.4,-3.2,4.5,3.8,-3.9,1.9,-1.6,-1.6,-0.8,-0.7,-0.9,-1.3,4.2])}
        bulk_idx = {c: v for c, v in zip("ACDEFGHIKLMNPQRSTVWY",
            [88,173,114,111,135,148,138,60,153,166,166,146,124,189,112,99,122,205,181,140])}
    helix = []
    sheet = []
    coil = []
    for ch in seq:
        h = helix_idx.get(ch, 0.8)
        e = sheet_idx.get(ch, 0.8)
        hyd = hydro_idx.get(ch, 0.0)
        bulk = bulk_idx.get(ch, 120.0)
        hp = h + max(0.0, hyd * 0.1)
        ep = e + max(0.0, (bulk - 120.0) * 0.002)
        cp = max(0.0, 1.0 - (hp + ep) * 0.3)
        if ch in ("P", "G"):
            hp *= 0.6
            ep *= 0.8
            cp += 0.3
        helix.append(hp)
        sheet.append(ep)
        coil.append(cp)
    return helix, sheet, coil

def _labels_from_propensity(sequence: str, win: int, h_bias: float, e_bias: float, stochastic: bool = False) -> List[str]:
    seq = sequence.upper()
    H, E, C = _compute_propensity_arrays(seq)
    n = len(seq)
    half = max(1, win // 2)
    out = []
    for i in range(n):
        l = max(0, i - half)
        r = min(n, i + half + 1)
        h_score = sum(H[l:r]) / max(1, (r - l))
        e_score = sum(E[l:r]) / max(1, (r - l))
        c_score = sum(C[l:r]) / max(1, (r - l))
        
        # Apply bias
        h_score *= (1.0 + h_bias)
        e_score *= (1.0 + e_bias)
        
        if stochastic:
            # 强化随机性：
            # 1. 扩大噪声范围至 +/- 30%
            # 2. 引入极小概率的“随机突变”，模拟局部能量扰动
            h_score *= random.uniform(0.7, 1.3)
            e_score *= random.uniform(0.7, 1.3)
            c_score *= random.uniform(0.7, 1.3)
            
            # 5% 概率强制随机反转某个状态，打破局部极小值
            if random.random() < 0.05:
                choice = random.choice(["H", "E", "C"])
                if choice == "H": h_score += 10.0
                elif choice == "E": e_score += 10.0
                else: c_score += 10.0

        if h_score >= e_score and h_score >= c_score:
            out.append("H")
        elif e_score >= h_score and e_score >= c_score:
            out.append("E")
        else:
            out.append("C")
    return out

def _match_fraction_assign(seq_len: int, labels: List[str], frac: Tuple[float, float, float]) -> List[str]:
    h_target = int(round(frac[0] * seq_len))
    t_target = int(round(frac[1] * seq_len))
    e_target = int(round(frac[2] * seq_len))
    c_target = max(0, seq_len - h_target - e_target)
    arr = labels[:]
    idxs = list(range(seq_len))
    if sum(1 for x in arr if x == "H") > h_target:
        need = sum(1 for x in arr if x == "H") - h_target
        for i in idxs:
            if need <= 0:
                break
            if arr[i] == "H":
                arr[i] = "C"
                need -= 1
    if sum(1 for x in arr if x == "E") > e_target:
        need = sum(1 for x in arr if x == "E") - e_target
        for i in idxs:
            if need <= 0:
                break
            if arr[i] == "E":
                arr[i] = "C"
                need -= 1
    while sum(1 for x in arr if x == "H") < h_target:
        for i in idxs:
            if sum(1 for x in arr if x == "H") >= h_target:
                break
            if arr[i] == "C":
                arr[i] = "H"
    while sum(1 for x in arr if x == "E") < e_target:
        for i in idxs:
            if sum(1 for x in arr if x == "E") >= e_target:
                break
            if arr[i] == "C":
                arr[i] = "E"
    while sum(1 for x in arr if x == "C") > c_target:
        for i in idxs[::-1]:
            if sum(1 for x in arr if x == "C") <= c_target:
                break
            if arr[i] == "C":
                arr[i] = "H" if sum(1 for x in arr if x == "H") < h_target else "E"
    return arr

def _split_chains(ss: str, target_count: int = None) -> List[str]:
    n = len(ss)
    cuts = []
    i = 0
    
    # If target_chains is set and > 1, we try to find the best cut points
    if target_count is not None and target_count > 1:
        # Collect all potential cut points (coil regions)
        possible_cuts = []
        i = 0
        while i < n:
            if ss[i] == "C":
                j = i
                while j < n and ss[j] == "C":
                    j += 1
                # (start, end, length)
                possible_cuts.append((i, j, j - i))
                i = j
            else:
                i += 1
        
        # Sort by length descending to pick most likely linkers
        possible_cuts.sort(key=lambda x: x[2], reverse=True)
        
        # Select top N-1 cuts
        num_cuts_needed = target_count - 1
        selected_cuts = possible_cuts[:num_cuts_needed]
        
        # If not enough cuts found, fallback to even splitting (naive approach)
        if len(selected_cuts) < num_cuts_needed:
            # We need more cuts. Let's just split evenly based on length for the remaining
            # This is a fallback and might break SS, but respects the user's "target_chains" constraint
            needed = num_cuts_needed - len(selected_cuts)
            # This is complex to mix with existing cuts. 
            # Simplification: If we can't find natural cuts, we force cuts at uniform intervals
            # ignoring structure. Or we accept fewer chains?
            # User said "人工置顶", implies strict enforcement.
            pass

        # Sort selected cuts by position
        selected_cuts.sort(key=lambda x: x[0])
        
        parts = []
        last = 0
        # For strict cutting, we remove the linker (standard behavior)
        for l, r, _ in selected_cuts:
            if l > last:
                parts.append(ss[last:l])
            last = r
        if last < n:
            parts.append(ss[last:n])
            
        parts = [p for p in parts if p]
        
        # If we still don't have enough parts (because we ran out of Coil regions), 
        # force split the longest parts until we reach target
        while len(parts) < target_count:
            # Find longest part
            longest_idx = -1
            max_len = -1
            for idx, p in enumerate(parts):
                if len(p) > max_len:
                    max_len = len(p)
                    longest_idx = idx
            
            if longest_idx == -1: break
            
            # Split it in half
            p = parts[longest_idx]
            if len(p) < 2: break # Can't split
            
            mid = len(p) // 2
            p1, p2 = p[:mid], p[mid:]
            
            # Replace
            parts.pop(longest_idx)
            parts.insert(longest_idx, p2)
            parts.insert(longest_idx, p1)
            
        return parts

    if target_count == 1:
        return [ss]

    # Default logic (Automatic / Stochastic)
    # Improved splitting logic:
    # 1. Minimum linker length is 5 (unchanged)
    # 2. Add randomness to the split point to simulate flexibility
    
    while i < n:
        if ss[i] == "C":
            j = i
            while j < n and ss[j] == "C":
                j += 1
            
            linker_len = j - i
            if linker_len >= 5:
                # Instead of always cutting the whole linker, we might:
                # a) Cut the whole linker (standard)
                # b) Leave some coil on the ends (randomized)
                
                # Standard cut: remove the coil region effectively by splitting around it
                # In current logic, we append ss[last:l] so the coil region itself is EXCLUDED from chains?
                # Wait, looking at original logic:
                # parts.append(ss[last:l]) -> This drops the coil region [l:r] completely!
                # This means "C" regions > 5 are treated as "break points" and discarded.
                
                # Let's enhance this. Sometimes we want to KEEP part of the linker.
                # Stochastic choice:
                # 80%: Standard split (drop linker)
                # 20%: Keep linker attached to one side or split in middle
                
                choice = random.random()
                if choice < 0.8:
                    # Standard behavior: cut out the linker
                    cuts.append((i, j))
                elif choice < 0.9:
                    # Split in the middle of linker
                    mid = (i + j) // 2
                    cuts.append((mid, mid)) # Split at mid point, keeping C's
                else:
                    # Don't split here! Treat this flexible region as part of a single chain
                    pass

            i = j
        else:
            i += 1
            
    if not cuts:
        # Fallback split if no natural cuts found
        # Original: Split in half
        # Enhanced: Split at random position between 40%-60%
        mid = int(n * random.uniform(0.4, 0.6))
        mid = max(1, min(mid, n - 1))
        return [ss[:mid], ss[mid:]]
        
    parts = []
    last = 0
    for l, r in cuts:
        if l == r:
            # Special case: split without dropping residues
            parts.append(ss[last:l])
            last = l
        else:
            # Standard case: drop residues between l and r
            if l - last > 0:
                parts.append(ss[last:l])
            last = r
            
    if last < n:
        parts.append(ss[last:n])
        
    parts = [p for p in parts if p]
    
    # Safety fallback
    if len(parts) < 2 and parts:
        mid = max(1, min(len(parts[0]) // 2, len(parts[0]) - 1))
        parts = [parts[0][:mid], parts[0][mid:]]
        
    return parts

def pybiomed_ss_candidates(sequence: str, environment: str = None, num: int = 5, target_chains: int = None) -> Dict[str, Any]:
    seq = sequence.strip().upper()
    seq_len = len(seq)
    frac = (0.34, 0.32, 0.34)
    windows = [7, 9, 11, 13, 5]
    hs = [0.0, 0.1, 0.2]
    es = [0.0, 0.1, 0.2]
    cases = []
    raw_lines = []
    attempts = 0
    logs = []
    
    # Check PyBioMed status once
    has_pybiomed = bool(_safe_import_pybiomed())
    if has_pybiomed:
        logs.append("Info: Using PyBioMed for AAIndex properties.")
    else:
        logs.append("Info: PyBioMed not found, using internal fallback dictionary.")
        
    if target_chains is not None and target_chains > 0:
        logs.append(f"Info: Enforcing target chain count = {target_chains}")

    # Store all generated candidates to fallback if needed
    all_generated = []

    # Dynamic loop: Keep trying until we have enough candidates or hit max attempts
    # We randomize parameters in each iteration to explore the space
    max_total_attempts = num * 20  # Allow plenty of tries
    current_attempt = 0
    
    while len(cases) < num and current_attempt < max_total_attempts:
        current_attempt += 1
        attempts += 1
        
        # Randomly pick parameters for this attempt
        win = random.choice([5, 7, 9, 11, 13, 15]) # Expanded window sizes
        h_thr = random.choice([0.0, 0.05, 0.1, 0.15, 0.2, 0.25])
        e_thr = random.choice([0.0, 0.05, 0.1, 0.15, 0.2, 0.25])
        
        # Enable stochastic mode
        labels = _labels_from_propensity(seq, win, h_thr, e_thr, stochastic=True)
        ss = "".join(labels)
        chains = _split_chains(ss, target_count=target_chains)
        total = sum(len(p) for p in chains)
        
        # Length correction
        if total != seq_len:
            if total > seq_len:
                over = total - seq_len
                for k in range(len(chains) - 1, -1, -1):
                    if over <= 0:
                        break
                    cut = min(over, len(chains[k]))
                    chains[k] = chains[k][:-cut]
                    over -= cut
                chains = [p for p in chains if p]
            else:
                add = seq_len - total
                if chains:
                    chains[-1] = chains[-1] + ("C" * add)
                else:
                    chains = ["C" * seq_len]
        
        if not chains:
            # logs.append(f"Attempt {attempts}: No chains after split/correction.")
            continue
        
        ss_str = "".join(chains)
        # Check diversity
        is_monochromatic = len(set(ss_str)) < 2
        
        cand = {"chains": chains, "ss": ss_str, "win": win, "h": h_thr, "e": e_thr}
        all_generated.append(cand)

        if is_monochromatic:
            # logs.append(f"Attempt {attempts}: Monochromatic ({ss_str[:10]}...)")
            continue
        
        # Avoid duplicates in cases
        if any(c["chains"] == chains for c in cases):
            continue
            
        cases.append({"chains": chains})
        raw_lines.append("|".join(chains))
        
    # If we still don't have enough cases, try to fill with unique monochromatic ones
    if len(cases) < num and all_generated:
        logs.append(f"Warning: Only found {len(cases)} diverse candidates. Filling with top unique monochromatic ones.")
        seen = set(tuple(c["chains"]) for c in cases)
        for cand in all_generated:
            t = tuple(cand["chains"])
            if t not in seen:
                seen.add(t)
                cases.append({"chains": cand["chains"]})
                raw_lines.append("|".join(cand["chains"]))
                if len(cases) >= num:
                    break
                    
    return {"cases": cases, "raw": "\n".join(raw_lines), "attempts": attempts, "lines": len(raw_lines), "logs": logs}
