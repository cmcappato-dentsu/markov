import pandas as pd
from collections import Counter, defaultdict
from typing import Dict, Tuple, List
from model.preprocessing import clean_steps, clean_tokens
from model.core import STATE_START, ABSORB_CONV, ABSORB_NULL


def compute_touch_rate_steps(df_paths, sep): # MODELO.py
    total_paths = len(df_paths)
    
    if total_paths <= 0:
        return {}

    touch = Counter()

    for p in df_paths["path"]:
        for t in set(clean_steps(p, sep, {}, dedup_consecutive=True)):
            if t not in {STATE_START, ABSORB_CONV, ABSORB_NULL}:
                touch[t] += 1

    return {c: touch[c] / total_paths for c in touch}

def compute_touch_rate_tokens(df_paths: pd.DataFrame, sep: str) -> Dict[str, float]:
    total_paths = len(df_paths)
    
    if total_paths <= 0:
        return {}
    touch = Counter()
    
    for p in df_paths["path"]:
        for t in set(clean_tokens(p, sep, strip_dupes=True)):
            if t not in {STATE_START, ABSORB_CONV, ABSORB_NULL}:
                touch[t] += 1
                
    return {c: touch[c] / total_paths for c in touch}

def compute_avg_position_steps(df_paths, sep):
    pos_sum = {}
    pos_cnt = {}
    
    for p in df_paths["path"]:
        toks = [
            t for t in clean_steps(p, sep, {}, dedup_consecutive=True)
            if t not in {STATE_START, ABSORB_CONV, ABSORB_NULL}
        ]

        for i, ch in enumerate(toks):
            pos_sum[ch] = pos_sum.get(ch, 0) + i
            pos_cnt[ch] = pos_cnt.get(ch, 0) + 1

    return {ch: pos_sum[ch] / pos_cnt[ch] for ch in pos_sum}

def compute_avg_position_tokens(df_paths: pd.DataFrame, sep: str) -> Dict[str, float]:
    pos_sum, pos_cnt = {}, {}
    for p in df_paths["path"]:
        toks = [t for t in clean_tokens(p, sep, strip_dupes=True)
                if t not in {STATE_START, ABSORB_CONV, ABSORB_NULL}]
        for i, ch in enumerate(toks):
            pos_sum[ch] = pos_sum.get(ch, 0) + i
            pos_cnt[ch] = pos_cnt.get(ch, 0) + 1
    return {ch: pos_sum[ch] / pos_cnt[ch] for ch in pos_sum}


def compute_channel_transitions(df_paths: pd.DataFrame, sep: str) -> Tuple[Dict[Tuple[str, str], float], Dict[str, float]]:
    links = defaultdict(float)
    node_weight: Dict[str, float] = {}

    for _, row in df_paths.iterrows():
        conv = float(row["conversions"])
        steps = clean_tokens(row["path"], sep, strip_dupes=True)

        if not steps:
            # (NULL) explícito como ruta
            links[(STATE_START, ABSORB_NULL)] += conv
            node_weight[ABSORB_NULL] += conv
            continue

        # Conteo de presencia por canal (ponderado por conversions)
        for c in set(steps):
            node_weight[c] = node_weight.get(c, 0.0) + conv

        full = [STATE_START] + steps + [ABSORB_CONV]
        for a, b in zip(full, full[1:]):
            links[(a, b)] += conv

    return links, dict(node_weight)

def top_k_nodes(node_weight: Dict[str, float], k: int) -> List[str]:
    return [n for n, _ in sorted(node_weight.items(), key=lambda x: x[1], reverse=True)[:k]]

def classify_risk_and_action(
    df_chan: pd.DataFrame,
    baseline: float,
    basis_cfg: str,
    thr_abs: dict,
    thr_share: dict
) -> pd.DataFrame:
    if basis_cfg.lower() == "auto":
        basis = "share" if baseline < 0.20 else "abs"
    else:
        basis = basis_cfg.lower()

    levels, actions = [], []

    for _, r in df_chan.iterrows():
        tr = float(r.get("touch_rate", 0.0))
        if basis == "abs":
            val = float(r.get("removal_drop_abs", 0.0))
            hi_min = float(thr_abs.get("high", {}).get("removal_drop_min", 0.06))
            hi_tr  = float(thr_abs.get("high", {}).get("touch_rate_min", 0.50))
            med_min = float(thr_abs.get("medium", {}).get("removal_drop_min", 0.03))
        else:
            val = float(r.get("removal_effect_share", 0.0))
            hi_min = float(thr_share.get("high", {}).get("removal_share_min", 0.25))
            hi_tr  = float(thr_share.get("high", {}).get("touch_rate_min", 0.35))
            med_min = float(thr_share.get("medium", {}).get("removal_share_min", 0.12))

        if (val >= hi_min) and (tr >= hi_tr):
            risk = "HIGH"
        elif val >= med_min:
            risk = "MEDIUM"
        else:
            risk = "LOW"
        levels.append(risk)

        pc = float(r.get("peso_canal", 0.0))
        if risk == "HIGH":
            act = "PROTECT" if pc >= 0.08 else "OPTIMIZE"
        elif risk == "MEDIUM":
            act = "OPTIMIZE" if pc >= 0.05 else "EXPAND"
        else:
            act = "EXPAND" if pc < 0.03 else "MONITOR"
        actions.append(act)

    out = df_chan.copy()
    out["risk_basis"] = basis
    out["risk_level"] = levels
    out["accion_sugerida"] = actions
    
    return out

