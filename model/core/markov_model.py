from collections import defaultdict
from typing import Dict
import pandas as pd
import numpy as np
from model.core import STATE_START, ABSORB_CONV, ABSORB_NULL, RESERVED
from model.preprocessing import clean_steps, interpret_null_aliases

def build_transition_matrix_from_df(
    df: pd.DataFrame,
    sep: str,
    cleaning_cfg: dict,
    null_aliases: list,
    reserved_suffix: str,
    eps_num: float
):
    trans_counts = defaultdict(float)
    first_touch: Dict[str, float] = {}

    for _, row in df.iterrows():
        conv = float(row["conversions"])
        steps_raw = clean_steps(row["path"], sep, cleaning_cfg, dedup_consecutive=True)

        if interpret_null_aliases(steps_raw, null_aliases):
            trans_counts[(STATE_START, ABSORB_NULL)] += conv
            continue

        if not steps_raw:
            continue

        tokens = [(s + reserved_suffix) if s in RESERVED else s for s in steps_raw]
        first_touch[tokens[0]] = first_touch.get(tokens[0], 0.0) + conv

        states_path = [STATE_START] + tokens + [ABSORB_CONV]
        for a, b in zip(states_path, states_path[1:]):
            trans_counts[(a, b)] += conv

    channels = sorted(set(
        [a for (a, b) in trans_counts.keys() if a not in RESERVED] +
        [b for (a, b) in trans_counts.keys() if b not in RESERVED]
    ))
    states = [STATE_START] + channels + [ABSORB_CONV, ABSORB_NULL]
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    P = np.zeros((n, n), dtype=float)

    outgoing = defaultdict(float)
    for (a, b), c in trans_counts.items():
        outgoing[a] += c
    for (a, b), c in trans_counts.items():
        i, j = idx[a], idx[b]
        P[i, j] += c / outgoing[a]

    # Absorbentes
    P[idx[ABSORB_CONV], :] = 0.0; P[idx[ABSORB_CONV], idx[ABSORB_CONV]] = 1.0
    P[idx[ABSORB_NULL], :] = 0.0; P[idx[ABSORB_NULL], idx[ABSORB_NULL]] = 1.0

    # Completar residuo a NULL / normalizar
    for s in states:
        if s in {ABSORB_CONV, ABSORB_NULL}:
            continue
        i = idx[s]
        row_sum = P[i, :].sum()
        if row_sum < 1.0 - eps_num:
            P[i, idx[ABSORB_NULL]] += (1.0 - row_sum)
        elif row_sum > 1.0 + eps_num:
            P[i, :] /= row_sum

    return states, idx, P, channels, first_touch


def p_conv_from_P(P: np.ndarray, idx: dict) -> float:
    n = P.shape[0]
    i_start, i_conv, i_null = idx[STATE_START], idx[ABSORB_CONV], idx[ABSORB_NULL]
    absorbing = [i_conv, i_null]
    transient = [i for i in range(n) if i not in absorbing]
    if not transient:
        return 0.0

    Q = P[np.ix_(transient, transient)]
    R = P[np.ix_(transient, absorbing)]
    I = np.eye(Q.shape[0])
    try:
        N = np.linalg.inv(I - Q)
    except np.linalg.LinAlgError:
        N = np.linalg.pinv(I - Q)
    B = N @ R
    start_rel = transient.index(i_start)
    conv_rel = 0
    val = float(B[start_rel, conv_rel])
    return max(0.0, min(1.0, val))


def build_model(
    df,
    sep,
    cleaning_cfg,
    null_aliases,
    reserved_suffix,
    eps_num
):
    states, idx, P, channels, first_touch = build_transition_matrix_from_df(
        df, sep, cleaning_cfg, null_aliases, reserved_suffix, eps_num
    )

    baseline = p_conv_from_P(P, idx)

    return {
        "states": states,
        "idx": idx,
        "P": P,
        "channels": channels,
        "first_touch": first_touch,
        "baseline": baseline,
    }