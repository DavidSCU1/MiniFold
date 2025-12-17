import os
import json
from typing import List, Dict, Any, Optional, Tuple
import requests

def _ark_endpoint() -> str:
    return os.environ.get("ARK_API_URL", "https://ark.cn-beijing.volces.com/api/v3/chat/completions")

def _ark_headers(api_key: Optional[str]) -> Dict[str, str]:
    key = api_key or os.environ.get("ARK_API_KEY", "")
    return {"Content-Type": "application/json", "Authorization": f"Bearer {key}"} if key else {"Content-Type": "application/json"}

def ark_analyze_sequence(sequence: str, api_key: Optional[str] = None) -> str:
    """
    Uses Ark API to annotate the protein sequence.
    """
    url = _ark_endpoint()
    headers = _ark_headers(api_key)
    
    prompt = f"""
    You are an expert bioinformatician. Analyze the following protein sequence:
    
    Sequence:
    {sequence}
    
    Please provide:
    1. Potential domain structure.
    2. Predicted function.
    3. Active sites or key residues.
    4. Subcellular localization (e.g. signal peptides).
    """

    payload = {
        "model": "deepseek-v3-2-251201", # Default model for annotation
        "messages": [
            {"role": "system", "content": "你是人工智能助手."},
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
             return f"API Error: {json.dumps(data['error'])}"
        content = data['choices'][0]['message']['content']
        return content
    except Exception as e:
        return f"LLM Analysis Failed: {e}"

def get_default_models() -> List[str]:
    env_models = os.environ.get("ARK_MODELS", "")
    if env_models:
        return [m.strip() for m in env_models.split(",") if m.strip()]
    return [
        "doubao-seed-1-6-251015",
        "deepseek-v3-2-251201",
        "doubao-1-5-pro-256k-250115",
        "kimi-k2-thinking-251104",
        "deepseek-r1-250528",
    ]

def get_model_weights(models: List[str]) -> List[float]:
    env_w = os.environ.get("ARK_MODEL_WEIGHTS", "")
    if env_w:
        parts = [p.strip() for p in env_w.split(",") if p.strip()]
        vals = []
        for i in range(len(models)):
            try:
                vals.append(float(parts[i]))
            except Exception:
                vals.append(1.0)
        s = sum(vals) or 1.0
        return [v / s for v in vals]
    return [1.0 / max(1, len(models))] * len(models)

def ark_eval_case(model: str, sequence: str, environment: Optional[str], chains: List[str], req_text: Optional[str] = None, api_key: Optional[str] = None, timeout: int = 120) -> Tuple[Optional[float], Optional[str]]:
    url = _ark_endpoint()
    headers = _ark_headers(api_key)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "仅返回一个0-1的数字。"},
            {"role": "user", "content": json.dumps({
                "sequence": sequence,
                "environment": environment or "",
                "chains": chains or [],
                "requirements": req_text or "",
                "instruction": "Estimate probability (0-1) for the CASE; return only a number"
            }, ensure_ascii=False)}
        ]
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}: {r.text[:100]}"
            
        data = r.json()
        if "error" in data:
             return None, f"API Error: {json.dumps(data['error'])}"
             
        s = (data["choices"][0]["message"]["content"] or "").strip()
        try:
            return float(s), None
        except Exception:
            # Try parsing line by line
            for line in s.splitlines():
                line = line.strip()
                try:
                    return float(line), None
                except Exception:
                    pass
            # Try parsing JSON
            try:
                obj = json.loads(s)
                p = obj.get("p")
                if isinstance(p, (int, float)):
                    return float(p), None
            except Exception:
                pass
            return None, f"Parse Error: Could not extract float from '{s[:50]}...'"
    except Exception as e:
        return None, f"Network Error: {str(e)}"

def ark_refine_structure(model: str, sequence: str, environment: str, chains: List[str], api_key: Optional[str] = None, timeout: int = 120) -> Tuple[Optional[List[str]], Optional[str]]:
    """
    Asks the model to refine the SS structure based on environment description.
    Returns (refined_chains, error_msg).
    """
    url = _ark_endpoint()
    headers = _ark_headers(api_key)
    
    prompt = f"""
Given the protein sequence (length {len(sequence)}) and environment description: "{environment}", 
please REFINE the following secondary structure prediction to be more physically realistic for this environment.

Current prediction (chains):
{json.dumps(chains)}

Rules:
1. Return ONLY a JSON list of strings (chains).
2. The total length of residues (H/E/C) must exactly match the input sequence length.
3. Consider the environment: e.g., if membrane, helices might be preferred; if high temp, structure might be more compact.
4. Do not output any explanation, just the JSON list.
"""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a protein structure expert. Output only valid JSON."},
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if r.status_code != 200:
            return None, f"Refine HTTP {r.status_code}: {r.text[:100]}"
            
        data = r.json()
        if "error" in data:
             return None, f"Refine API Error: {json.dumps(data['error'])}"
             
        s = (data["choices"][0]["message"]["content"] or "").strip()
        
        # Try to parse JSON list from response
        # It might be wrapped in ```json ... ```
        if "```" in s:
            s = s.split("```")[1]
            if s.startswith("json"): s = s[4:]
            
        try:
            refined_chains = json.loads(s)
            if isinstance(refined_chains, list) and all(isinstance(x, str) for x in refined_chains):
                return refined_chains, None
            else:
                return None, "Invalid format: Not a list of strings"
        except Exception:
            return None, f"Refine Parse Error: Could not parse JSON from response"
            
    except Exception as e:
        return None, f"Refine Network Error: {str(e)}"

import concurrent.futures

def ark_vote_cases(models: List[str], sequence: str, environment: Optional[str], cases: List[Dict[str, Any]], req_text: Optional[str] = None, api_key: Optional[str] = None) -> Dict[str, Any]:
    weights = get_model_weights(models)
    per_case = []
    
    # Flatten all tasks: (case_idx, model_idx, model_name)
    tasks = []
    for c_idx, case in enumerate(cases):
        for m_idx, m_name in enumerate(models):
            tasks.append((c_idx, m_idx, m_name))
            
    results_map = {} # (c_idx, m_idx) -> (score, error_msg)

    # Parallel execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_task = {}
        for (c_idx, m_idx, m_name) in tasks:
            chains = cases[c_idx].get("chains") or []
            f = executor.submit(ark_eval_case, m_name, sequence, environment, chains, req_text=req_text, api_key=api_key)
            future_to_task[f] = (c_idx, m_idx)
            
        for future in concurrent.futures.as_completed(future_to_task):
            c_idx, m_idx = future_to_task[future]
            try:
                score, err = future.result()
                results_map[(c_idx, m_idx)] = (score, err)
            except Exception as e:
                results_map[(c_idx, m_idx)] = (None, str(e))

    # Aggregate results
    for idx, case in enumerate(cases):
        chains = case.get("chains") or []
        scores = []
        per_model = []
        
        for m_i, m in enumerate(models):
            res = results_map.get((idx, m_i))
            if res:
                p, err = res
                if p is not None:
                    w = weights[m_i]
                    per_model.append({"model": m, "p": float(p), "w": w})
                    scores.append((float(p), w))
                else:
                    # Record failure for visibility
                    per_model.append({"model": m, "p": -1.0, "error": err or "Unknown"})
            else:
                per_model.append({"model": m, "p": -1.0, "error": "No result"})
                
        if scores:
            total_w = sum(w for _, w in scores)
            avg = sum(s * w for s, w in scores) / (total_w if total_w > 0 else 1.0)
            med = sorted([s for s, _ in scores])[len(scores) // 2]
        else:
            avg = 0.0
            med = 0.0
        per_case.append({"idx": idx, "chains": len(chains), "avg": avg, "med": med, "models": per_model})

    best_idx = -1
    best_score = -1.0
    for it in per_case:
        sc = (it["avg"] + it["med"]) / 2.0
        if sc > best_score:
            best_score = sc
            best_idx = it["idx"]
    return {"cases": per_case, "best_idx": best_idx, "best_score": best_score}

def ark_extract_constraints(model: str, sequence: str, environment: str, api_key: Optional[str] = None, timeout: int = 120) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Extracts physical constraints from the environment description using the LLM.
    Returns (constraints_list, error_msg).
    """
    url = _ark_endpoint()
    headers = _ark_headers(api_key)
    
    prompt = f"""
Analyze the protein environment and extract physical constraints.
Sequence Length: {len(sequence)}
Environment: "{environment}"

Output a JSON list of constraints using these 4 primitives:

1. "distance_point": Constrain residue(s) to a point (e.g., metal ion).
   {{ "type": "distance_point", "indices": [i, j...], "distance": float, "strength": float }}
   *Note: "indices" are 0-based.*

2. "pocket_preservation": Maintain a binding pocket shape around residues.
   {{ "type": "pocket_preservation", "indices": [i, j...], "radius": float, "strength": float }}

3. "surface_exposure": Force residues to be exposed (e.g., for glycosylation) or buried.
   {{ "type": "surface_exposure", "indices": [i...], "exposed": bool, "strength": float }}

4. "interface_geometry": Constrain geometry for DNA/RNA/Membrane interaction.
   {{ "type": "interface_geometry", "indices": [i...], "geometry_type": "planar"|"helical"|"membrane", "params": {{...}}, "strength": float }}

Return ONLY the valid JSON list.
"""
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a biophysics expert. Extract constraints as JSON."},
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if r.status_code != 200:
            return [], f"HTTP {{r.status_code}}: {{r.text[:100]}}"
            
        data = r.json()
        if "error" in data:
             return [], f"API Error: {{json.dumps(data['error'])}}"
             
        content = (data["choices"][0]["message"]["content"] or "").strip()
        
        # Simple parsing to find JSON list
        start = content.find('[')
        end = content.rfind(']') + 1
        if start != -1 and end != -1:
            json_str = content[start:end]
            try:
                constraints = json.loads(json_str)
                if isinstance(constraints, list):
                    return constraints, None
            except:
                pass
        return [], "No JSON list found"
            
    except Exception as e:
        return [], str(e)
