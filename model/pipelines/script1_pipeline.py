import pandas as pd
from model.utils import setup_logger
from model.preprocessing import normalize_and_remap_paths
from model.attribution.removal import rebuild_df_without_channel
from model.pipelines import compute_markov_v3
from model.in_out import load_config, load_paths_csv


def run_markov_pipeline(
    config_path: str, 
    df_input: pd.DataFrame | None = None,
    channel_to_remove: str | None = None 
):
    cfg = load_config(config_path)
    setup_logger(cfg.get("logging", {}).get("verbosity", "info"))

    # INPUT
    if df_input is not None:
        df = df_input.copy()
    else:
        csv_path = cfg["io"]["input_csv"]
        df = load_paths_csv(csv_path)


    # CONFIG
    parsing_cfg = cfg.get("parsing", {})
    sep = parsing_cfg.get("path_separator", ">")
    cleaning_cfg = parsing_cfg.get("cleaning", {})
    channel_remap = cfg.get("mapping", {}).get("channel_remap", {})

    # ✅ NORMALIZACIÓN (CRÍTICO)
    df = normalize_and_remap_paths(df, sep, cleaning_cfg, channel_remap)
    
    if channel_to_remove and channel_to_remove != "Ninguno":
        df = rebuild_df_without_channel(
            df,
            channel_to_remove,
            sep,
            cleaning_cfg
        )


    # RUN ENGINE
    df_attr, artifacts, df_base = compute_markov_v3(
        df_raw=df,
        sep=sep,
        cleaning_cfg=cleaning_cfg,
        null_aliases=parsing_cfg.get("null_aliases", []),
        reserved_suffix=parsing_cfg.get("reserved_state_suffix", "_CHAN"),
        mode_non_conv=cfg["non_converting_paths"]["mode"],
        global_rate=cfg["non_converting_paths"]["global_rate"],
        absolute_count=cfg["non_converting_paths"]["absolute_count"],
        removal_w=cfg["weights"]["removal_w"],
        first_w=cfg["weights"]["first_w"],
        min_attrib=cfg["weights"]["min_attrib"],
        eps_num=cfg["numeric"]["eps_num"],
        eps_drop=cfg["numeric"]["eps_drop"],
        exclude_list=cfg["exclusions"]["exclude_for_removal"],
        exclude_regex=cfg["exclusions"]["exclude_regex"]
    )

    return df_attr, artifacts, df_base