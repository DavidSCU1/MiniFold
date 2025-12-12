import os
import json
from openai import OpenAI
import json as _json
import re

def qwen_filter_candidates(sequence, environment, candidates, threshold=0.5, api_key=None):
    api_key = api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
    if not api_key:
        print("Warning: QWEN_API_KEY/DASHSCOPE_API_KEY not found.")
        return []
    base_url = os.environ.get("QWEN_API_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model = os.environ.get("QWEN_MODEL", "qwen3-max")
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=300)
    items = [{"idx": i, "ss": c.get("ss", ""), "reason": c.get("reason", "")} for i, c in enumerate(candidates) if c.get("ss")]
    prompt = {
        "sequence": sequence,
        "environment": environment or "",
        "items": items,
        "instruction": (
            "For each item, estimate a probability (0-1) that this secondary structure matches the given sequence and environment. "
            "Return strict JSON with key 'results' where each entry has 'idx', 'ss', and 'p'. Do not modify 'ss'. "
            "Ensure one-to-one mapping using 'idx'."
        )
    }
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Return only JSON."},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}
            ],
            stream=False
        )
        content = completion.choices[0].message.content
        try:
            obj = json.loads(content)
            results = obj.get("results", [])
            keeps = []
            # Map back by idx for one-to-one correspondence
            for it in results:
                idx = it.get("idx")
                ss = (it.get("ss") or "").strip()
                p = it.get("p")
                if isinstance(idx, int) and 0 <= idx < len(candidates):
                    if ss and isinstance(p, (int, float)) and p >= threshold and len(ss) == len(sequence) and set(ss) <= set("HEC"):
                        # Ensure ss exactly matches the candidate's ss for integrity
                        orig_ss = (candidates[idx].get("ss") or "").strip()
                        if ss == orig_ss:
                            keeps.append({"idx": idx, "ss": ss, "p": float(p)})
            return keeps
        except Exception:
            # Fallback: parse line-wise JSON
            keeps = []
            for line in content.splitlines():
                try:
                    it = json.loads(line)
                    idx = it.get("idx")
                    ss = (it.get("ss") or "").strip()
                    p = it.get("p")
                    if isinstance(idx, int) and 0 <= idx < len(candidates):
                        if ss and isinstance(p, (int, float)) and p >= threshold and len(ss) == len(sequence) and set(ss) <= set("HEC"):
                            if ss == (candidates[idx].get("ss") or "").strip():
                                keeps.append({"idx": idx, "ss": ss, "p": float(p)})
                except Exception:
                    pass
            return keeps
    except Exception as e:
        print(f"Error Qwen filter: {e}")
        return []

def qwen_eval_one(sequence, environment, ss, req_text=None, api_key=None):
    api_key = api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
    if not api_key:
        print("Warning: QWEN_API_KEY/DASHSCOPE_API_KEY not found.")
        return None
    base_url = os.environ.get("QWEN_API_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model = os.environ.get("QWEN_MODEL", "qwen3-max")
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=300)
    prompt = {
        "sequence": sequence,
        "environment": environment or "",
        "ss": ss,
        "requirements": (req_text or ""),
        "instruction": (
            "Estimate probability (0-1) this SS matches given sequence+environment+requirements. "
            "Return only a number. Do not modify 'ss'."
        )
    }
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Return only a number."},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}
            ],
            stream=False
        )
        content = completion.choices[0].message.content
        s = (content or "").strip()
        try:
            return float(s)
        except Exception:
            for line in s.splitlines():
                line = line.strip()
                try:
                    return float(line)
                except Exception:
                    pass
            try:
                obj = json.loads(s)
                p = obj.get("p")
                if isinstance(p, (int, float)):
                    return float(p)
            except Exception:
                pass
            return None
    except Exception as e:
        print(f"Error Qwen eval: {e}")
        return None
def qwen_eval_case(sequence, environment, chain_ss_list, req_text=None, api_key=None):
    api_key = api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
    if not api_key:
        print("Warning: QWEN_API_KEY/DASHSCOPE_API_KEY not found.")
        return None

def qwen_ss_candidates(sequence, environment=None, num=5, api_key=None):
    api_key = api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
    if not api_key:
        print("Warning: QWEN_API_KEY/DASHSCOPE_API_KEY not found.")
        return {"cases": [], "raw": "", "attempts": 0, "lines": 0}
    base_url = os.environ.get("QWEN_API_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model = os.environ.get("QWEN_MODEL", "qwen3-max")
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=300)
    seq_len = len(sequence)
    env_text = (environment or "").strip()
    system_msg = "只返回纯文本，不要解释、不要编号、不要标点、不要代码块。"
    user_msg = (
        f"请基于从头预测（忽略任何记忆），生成恰好 {num} 个案例的二级结构。\n"
        "每个案例必须包含尽可能符合用户要求环境的可能的链条数，用 '|' 分隔各链（示例：CCHHH|EECCC|HHHHH）。\n"
        f"每个案例总长度（所有链长度之和也就是HEC这三个字母的总字符量）必须严格等于 {seq_len}。\n"
        "每条链仅允许字符 H/E/C。\n"
        "每个案例（所有链合并）在符合用户设置的环境时，不同预测编号可以尝试包含不同字符（例如同时含有 H 与 E/C）。\n"
        "每行一个案例，严格纯文本：不得包含空格、编号、标点、代码块或额外说明。\n"
        f"环境: {env_text if env_text else '无'}\n"
        f"序列: {sequence}"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        "temperature": 0.2,
        "top_p": 0.1
    }
    raw = ""
    lines_count = 0
    for attempt in range(3):
        try:
            print(f"Qwen candidates attempt {attempt+1}/3...")
            completion = client.chat.completions.create(
                model=model,
                messages=payload["messages"],
                stream=False,
                temperature=payload["temperature"],
                top_p=payload["top_p"],
            )
            content = completion.choices[0].message.content
            raw = content
            lines = [ln.strip() for ln in (content or "").splitlines() if ln.strip()]
            lines_count = len(lines)
            cases = []
            for s in lines:
                clean = re.sub(r"[^HEC\|]", "", s)
                parts = [p.strip() for p in clean.split('|') if p.strip()]
                if not parts:
                    continue
                if len(parts) < 2:
                    single = parts[0]
                    L = len(single)
                    if L == 0:
                        continue
                    if L != seq_len:
                        if L > seq_len:
                            single = single[:seq_len]
                        else:
                            single = single + ("C" * (seq_len - L))
                        L = len(single)
                    if L < 2:
                        continue
                    left = max(1, min(L // 2, L - 1))
                    parts = [single[:left], single[left:]]
                total = 0
                valid = True
                for p in parts:
                    if not set(p) <= set("HEC") or len(p) == 0:
                        valid = False
                        break
                    total += len(p)
                if not valid:
                    continue
                # enforce diversity across the whole case
                if len(set("".join(parts))) < 2:
                    # will try normalization below only for length issues; otherwise skip
                    pass
                if total != seq_len:
                    if total > seq_len:
                        over = total - seq_len
                        new_parts = parts[:]
                        for idx in range(len(new_parts) - 1, -1, -1):
                            if over <= 0:
                                break
                            cut = min(over, len(new_parts[idx]))
                            if cut > 0:
                                new_parts[idx] = new_parts[idx][:-cut]
                                over -= cut
                        # drop empty chains
                        new_parts = [p for p in new_parts if p]
                        parts = new_parts
                    else:
                        # pad last chain with 'C'
                        add = seq_len - total
                        parts[-1] = parts[-1] + ("C" * add)
                    # ensure at least two chains
                    if len(parts) < 2 and parts:
                        single = parts[0]
                        if len(single) >= 2:
                            left = max(1, min(len(single) // 2, len(single) - 1))
                            parts = [single[:left], single[left:]]
                        else:
                            continue
                    # recompute total
                    total = sum(len(p) for p in parts)
                    if total != seq_len:
                        continue
                # after normalization, ensure diversity
                if len(set("".join(parts))) < 2:
                    continue
                cases.append({"chains": parts})
                if len(cases) >= num:
                    break
            if cases:
                return {"cases": cases, "raw": raw, "attempts": attempt+1, "lines": lines_count}
            else:
                payload["messages"][1]["content"] = (
                    f"上一次输出无效。请严格输出 {num} 行、仅包含 H/E/C 与 '|'，总长度必须等于 {seq_len}。"
                )
        except Exception as e:
            print(f"Qwen candidates error on attempt {attempt+1}: {e}")
    return {"cases": [], "raw": raw, "attempts": 3, "lines": lines_count}
