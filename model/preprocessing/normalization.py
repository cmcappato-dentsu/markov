import pandas as pd
from typing import List
from .cleaning import clean_steps

def apply_channel_remap(tokens: List[str], channel_remap: dict) -> List[str]:
    if not channel_remap:
        return tokens
    return [channel_remap.get(t, t) for t in tokens]

def normalize_and_remap_paths(df: pd.DataFrame, sep: str, cleaning_cfg: dict, channel_remap: dict) -> pd.DataFrame:
    def _normalize_row(p):
        toks = clean_steps(p, sep, cleaning_cfg, dedup_consecutive=True)
        toks = apply_channel_remap(toks, channel_remap)
        
        return sep.join(toks)

    out = df.copy()
    out["path"] = out["path"].astype(str).apply(_normalize_row)
    
    return out