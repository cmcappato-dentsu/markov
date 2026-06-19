import logging
import pandas as pd

def add_non_converting(df: pd.DataFrame, mode: str, global_rate: float, absolute_count: int) -> pd.DataFrame:
    total_conv = float(df["conversions"].sum())
    add = 0.0
    why = None
    m = (mode or "global_rate").lower()

    if m == "global_rate" and (global_rate is not None) and (0.0 < float(global_rate) < 1.0):
        total_paths = total_conv / float(global_rate)
        add = max(0.0, total_paths - total_conv)
        why = f"por tasa global ({float(global_rate):.2%})"
    elif m == "absolute_count" and (absolute_count is not None) and int(absolute_count) > 0:
        add = float(int(absolute_count))
        why = f"por cantidad declarada ({int(absolute_count)})"

    if add > 0:
        df = pd.concat([df, pd.DataFrame({"path": ["(NULL)"], "conversions": [add]})], ignore_index=True)
        logging.info(f"Se agregaron {add:.0f} rutas NO conversoras sintetizadas {why}.")
    else:
        logging.info("No se agregaron rutas NO conversoras (modo desactivado o sin valores).")

    return df
