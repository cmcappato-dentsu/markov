import json
import pandas as pd

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg

def load_paths_csv(file) -> pd.DataFrame:
    """
    Carga CSV (path o uploaded_file) y devuelve
    DataFrame con columnas path y conversions.
    """

    df = pd.read_csv(
        file,              # ✅ ahora acepta ambos
        sep=",",
        skiprows=9,
        usecols=[0, 2],
        engine="python"
    )

    df.rename(
        columns={
            df.columns[0]: "path",
            df.columns[1]: "conversions"
        },
        inplace=True
    )

    df["path"] = (
        df["path"]
        .astype(str)
        .str.replace(",", ">", regex=False)
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
        .str.replace('"', "", regex=False)
    )

    df = df[df["path"].str.strip().ne("")].copy()

    df["conversions"] = (
        pd.to_numeric(df["conversions"], errors="coerce")
        .fillna(0.0)
    )

    return df