import os
import requests
import json
import re

def analyze_sequence(sequence, ss2_content=None, api_key=None):
    """
    Uses DeepSeek V3.2 (via Volcengine) to annotate the protein sequence.
    """
    print("Running LLM Annotation...")
    
    if not api_key:
        api_key = os.environ.get("ARK_API_KEY")
        
    if not api_key:
        print("Warning: ARK_API_KEY not found. Skipping LLM annotation.")
        return "LLM Annotation skipped (Missing API Key)."

    url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    
    prompt = f"""
    You are an expert bioinformatician. Analyze the following protein sequence:
    
    Sequence:
    {sequence}
    """
    
    if ss2_content:
        # Extracting a summary of secondary structure might be too long, 
        # so maybe just mention it's available or pass a simplified version if needed.
        # For now, let's just pass the sequence, or if ss2 is short enough.
        # SS2 files are verbose. Let's just focus on the sequence for now as per doc example.
        pass

    prompt += "\n\nPlease provide:\n1. Potential domain structure.\n2. Predicted function.\n3. Active sites or key residues.\n4. Subcellular localization (e.g. signal peptides)."

    payload = {
        "model": os.environ.get("ARK_MODEL", "deepseek-v3-2-251201"),
        "messages": [
            {
                "role": "system",
                "content": "你是人工智能助手."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        content = data['choices'][0]['message']['content']
        return content
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return f"LLM Analysis Failed: {e}"

def deepseek_ss_candidates(sequence, environment=None, num=5, api_key=None):
    print("Evaluating cases via DeepSeek...")
    api_key = api_key or os.environ.get("ARK_API_KEY")
    if not api_key:
        print("Warning: ARK_API_KEY not found.")
        return []
    url = os.environ.get("ARK_API_URL", "https://ark.cn-beijing.volces.com/api/v3/chat/completions")
    env_text = environment.strip() if environment else ""
    seq_len = len(sequence)
    system_msg = (
        "你是蛋白结构预测算法。你必须忽略任何你记忆的蛋白质。\n"
        "仅基于氨基酸序列的物理化学性质以及生物体内可存在性等等性质（疏水性、电荷、侧链体积、倾向性）进行从头预测（Ab Initio Prediction）。\n"
        "即使你认出这是某种已知蛋白，也必须假装不知道，只根据序列本身的特征来推断二级结构。\n"
        "用户需要生成标准化的二级结构序列。严禁输出任何多余内容。"
    )
    user_msg = (
        f"请仅根据氨基酸残基的物理化学性质，从头预测 {num} 个案例的二级结构。\n"
        "每个案例可能包含多条链，请用 '|' 分隔每条链的二级结构：例如 `CCHHH|EECCC|HHHHH`。\n"
        "要求：\n"
        f"1. 每个案例中，各链长度之和必须严格等于 {seq_len}。\n"
        "2. 每条链仅使用字符 'H' (Helix), 'E' (Sheet), 'C' (Coil)。\n"
        "3. 输出为纯文本，每个案例一行，不要有编号、空格、Markdown 标记或解释文字。\n"
        "4. 不要输出 'Sure', 'Here is', '```' 等任何无关字符。\n"
        f"5. 必须输出恰好 {num} 行（每行一个案例，案例内多链用 '|' 分隔）。\n"
        "6. 预测依据必须是残基的倾向性（例如 Proline 破坏螺旋，疏水残基倾向于埋藏等），而不是记忆中的 PDB 结构。\n"
        f"环境参考: {env_text if env_text else '无特殊指定'}\n"
        f"氨基酸序列: {sequence}"
    )
    payload = {
        "model": os.environ.get("ARK_MODEL", "deepseek-v3-2-251201"),
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        "temperature": 0.2,
        "top_p": 0.1
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    
    # Retry logic: 3 attempts
    raw_content = ""
    lines_count = 0
    for attempt in range(3):
        try:
            print(f"DeepSeek Request Attempt {attempt+1}/3...")
            r = requests.post(url, headers=headers, json=payload, timeout=300)
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            raw_content = content
            lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
            lines_count = len(lines)
            cases = []
            for s in lines:
                # Clean to HEC and '|'
                clean = re.sub(r"[^HEC\|]", "", s)
                parts = [p.strip() for p in clean.split('|') if p.strip()]
                if not parts:
                    continue
                # Validate each chain and total length
                total = 0
                valid = True
                for p in parts:
                    if not set(p) <= set("HEC") or len(p) == 0:
                        valid = False
                        break
                    total += len(p)
                if not valid:
                    continue
                if total != seq_len:
                    continue
                cases.append({"chains": parts})
                if len(cases) >= num:
                    break
            if len(cases) >= 1:
                return {"cases": cases, "raw": raw_content, "attempts": attempt+1, "lines": lines_count}
            else:
                # corrective prompt for next attempt
                user_msg = (
                    f"上一次输出未能解析出有效案例。请严格输出恰好 {num} 行，每行仅包含 'H','E','C' 和分隔符 '|'。\n"
                    f"每行所有链的长度之和必须等于 {seq_len}。不要输出任何其他字符、注释或标点。\n"
                    f"示例：CCHHH|EECCC|HHHHH\n"
                    f"氨基酸序列: {sequence}"
                )
                payload["messages"][1]["content"] = user_msg
        except requests.exceptions.Timeout:
            print(f"DeepSeek Attempt {attempt+1} timed out.")
        except Exception as e:
            print(f"Error DeepSeek SS (Attempt {attempt+1}): {e}")
            # If it's not a timeout (e.g. auth error), maybe break? 
            # But simple retry is safer for network glitches.
            
    print("DeepSeek failed after 3 attempts.")
    return {"cases": [], "raw": raw_content, "attempts": 3, "lines": lines_count}

def deepseek_eval_case(sequence, environment=None, chain_ss_list=None, req_text=None, api_key=None):
    api_key = api_key or os.environ.get("ARK_API_KEY")
    if not api_key:
        print("Warning: ARK_API_KEY not found.")
        return None
    url = os.environ.get("ARK_API_URL", "https://ark.cn-beijing.volces.com/api/v3/chat/completions")
    model = os.environ.get("ARK_MODEL", "deepseek-v3-2-251201")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "仅返回一个0-1的数字。"},
            {"role": "user", "content": json.dumps({
                "sequence": sequence,
                "environment": environment or "",
                "chains": chain_ss_list or [],
                "requirements": req_text or "",
                "instruction": "Estimate probability (0-1) for the CASE; return only a number"
            }, ensure_ascii=False)}
        ]
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"]
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
        print(f"Error DeepSeek case eval: {e}")
        return None
