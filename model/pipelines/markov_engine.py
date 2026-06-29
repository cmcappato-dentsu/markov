import pandas as pd
import numpy as np
import logging
import time
from model.utils import get_progress_bar, compile_exclusion_predicate
from model.core import build_model
from model.attribution import rebuild_df_without_channel
from model.preprocessing import add_non_converting

def compute_markov_v3(
    df_raw: pd.DataFrame,
    sep: str,
    cleaning_cfg: dict,
    null_aliases: list,
    reserved_suffix: str,
    mode_non_conv: str,
    global_rate: float,
    absolute_count: int,
    removal_w: float,
    first_w: float,
    min_attrib: float,
    eps_num: float,
    eps_drop: float,
    exclude_list,
    exclude_regex,
    progress_enabled: bool = True,
    progress_log_every: int = 5
):
    t0 = time.perf_counter()
    # 1) Agregar no-conversoras si corresponde
    df0 = add_non_converting(df_raw, mode_non_conv, global_rate, absolute_count)
    t1 = time.perf_counter()
    logging.info("Paso 1/4 OK: NO conversoras (%.2fs)", t1 - t0)

    # 2) Modelo base
    model0 = build_model(df0, sep, cleaning_cfg, null_aliases, reserved_suffix, eps_num)
    channels = model0["channels"]
    t2 = time.perf_counter()
    logging.info("Paso 2/4 OK: Modelo base (%.2fs) - canales=%d", t2 - t1, len(channels))

    # 3) Shares de first touch (ponderado por conversions)
    total_ft = float(sum(model0["first_touch"].values())) or 1.0
    ft_share = {ch: float(model0["first_touch"].get(ch, 0.0)) / total_ft for ch in channels}

    # 4) Removal Effect por reconstrucción
    is_excluded = compile_exclusion_predicate(exclude_list, exclude_regex)
    removal_drop = {}
    total = len(channels)
    logging.info("Paso 3/4: Removal por reconstrucción (canales=%d)", total)

    progress = get_progress_bar(total_steps=total, mode="console")

    t_removal_start = time.perf_counter()
    for i, ch in enumerate(channels, 1):
        if is_excluded(ch):
            removal_drop[ch] = 0.0
        else:
            df_minus = rebuild_df_without_channel(df0, ch, sep, cleaning_cfg)
            model_minus = build_model(df_minus, sep, cleaning_cfg, null_aliases, reserved_suffix, eps_num)
            drop = max(0.0, model0["baseline"] - model_minus["baseline"])
            removal_drop[ch] = 0.0 if drop < eps_drop else drop

        progress.update(i, f"Procesando canal {i}/{total}")

    t3 = time.perf_counter()
    logging.info("Paso 3/4 OK: Removal completo (%.2fs)", t3 - t2)

    # 5) Normalización de drops -> shares
    total_drop = sum(removal_drop.values())
    if total_drop <= eps_drop:
        removal_share = {ch: 0.0 for ch in channels}
    else:
        removal_share = {ch: removal_drop[ch] / total_drop for ch in channels}

    # 6) Combinación final (modelo híbrido: RE + FT)
    combined = {ch: removal_w * removal_share.get(ch, 0.0) + first_w * ft_share.get(ch, 0.0) for ch in channels}
    if (min_attrib or 0.0) > 0.0:
        combined = {ch: max(min_attrib, v) for ch, v in combined.items()}
    s = sum(combined.values()) or 1.0
    combined = {ch: v / s for ch, v in combined.items()}

    # 7) DataFrame de salida (mismo esquema v3)
    df_out = (
        pd.DataFrame({
            "channel": channels,
            "first_touch_share": [ft_share[c] for c in channels],
            "removal_effect_share": [removal_share[c] for c in channels],
            "peso_canal": [combined[c] for c in channels]
        })
        .sort_values("peso_canal", ascending=False)
        .reset_index(drop=True)
    )
    df_out.insert(0, "rank", np.arange(1, len(df_out) + 1))

    # 8) Artifacts internos para Script2
    artifacts = {
        "baseline_structural_probability": model0["baseline"],
        "channels": channels,
        "first_touch_share": ft_share,
        "removal_drop_absolute": removal_drop,  # clave para riesgo categórico
        "removal_effect_share": removal_share
    }

    t4 = time.perf_counter()
    logging.info("Paso 4/4 OK: Combinación y armado de outputs (%.2fs)", t4 - t3)
    logging.info("Tiempo total compute_markov_v3: %.2fs", t4 - t0)

    progress.finish()
    
    return df_out, artifacts, df0
