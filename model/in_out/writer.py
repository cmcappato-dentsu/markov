import json

def save_csv(df, path: str):
    df.to_csv(path, index=False, encoding="utf-8")

def save_json(obj: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)