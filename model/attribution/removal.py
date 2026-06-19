import pandas as pd
from model.preprocessing import clean_steps, canon

def rebuild_df_without_channel(df, channel, sep, cleaning_cfg):
    ch = canon(
        channel,
        strip_quotes=cleaning_cfg.get("strip_quotes", True),
        trim=cleaning_cfg.get("trim", True),
        collapse_spaces=cleaning_cfg.get("collapse_spaces", True),
    )

    rows = []

    for _, row in df.iterrows():
        conv = float(row["conversions"])

        toks = clean_steps(row["path"], sep, cleaning_cfg, dedup_consecutive=True)
        toks = [t for t in toks if t != ch]

        if not toks:
            rows.append({"path": "(NULL)", "conversions": conv})
        else:
            toks2 = [toks[0]]
            for t in toks[1:]:
                if t != toks2[-1]:
                    toks2.append(t)

            rows.append({"path": sep.join(toks2), "conversions": conv})

    return pd.DataFrame(rows)