import re
from typing import List

def canon(s: str, strip_quotes=True, trim=True, collapse_spaces=True) -> str:
    s = str(s)
    if strip_quotes:
        s = re.sub(r'["\']', "", s)
    if trim:
        s = s.strip()
    if collapse_spaces:
        s = re.sub(r"\s+", " ", s)
    return s

def clean_steps(raw: str, sep: str, cleaning_cfg: dict, dedup_consecutive=True) -> List[str]:
    toks = [
        canon(
            x,
            strip_quotes=cleaning_cfg.get("strip_quotes", True),
            trim=cleaning_cfg.get("trim", True),
            collapse_spaces=cleaning_cfg.get("collapse_spaces", True)
        )
        for x in str(raw).split(sep)
        if str(x).strip()
    ]
    
    if not toks:
        return []
    
    if dedup_consecutive and cleaning_cfg.get("dedup_consecutive_steps", True):
        out = [toks[0]]
        for s in toks[1:]:
            if s != out[-1]:
                out.append(s)
        toks = out
    
    return toks

def clean_tokens(path: str, sep: str, strip_dupes: bool = True) -> List[str]:
    toks = [canon(t) for t in str(path).split(sep) if str(t).strip()]
    
    if not toks:
        return []

    if strip_dupes:
        out = [toks[0]]
        for t in toks[1:]:
            if t != out[-1]:
                out.append(t)
        toks = out

    return toks

def interpret_null_aliases(tokens: List[str], null_aliases: list) -> bool:
    if len(tokens) != 1:
        return False

    tok = tokens[0].lower()
    
    for alias in (null_aliases or []):
        if tok == str(alias).lower():
            return True

    return False