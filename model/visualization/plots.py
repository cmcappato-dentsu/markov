import pandas as pd
from typing import Dict, Tuple, List
from model.core import STATE_START, ABSORB_CONV, ABSORB_NULL
import plotly.graph_objects as go
import plotly.express as px

def safe_float(x, default: float = 0.0) -> float:
    try:
        val = pd.to_numeric(x, errors="coerce")
        return float(val) if pd.notna(val) else default
    except Exception:
        return default

def build_sankey_fig(
    links: Dict[Tuple[str, str], float],
    node_order: List[str],
    df_attr: pd.DataFrame,
    df_sum: pd.DataFrame,
    min_link_share: float
):
    labels = [STATE_START] + node_order + [ABSORB_CONV]

    has_null = any(ABSORB_NULL in pair for pair in links.keys())
    if has_null:
        labels.append(ABSORB_NULL)

    idx = {lab: i for i, lab in enumerate(labels)}

    total_val = sum(links.values()) or 1.0
    min_val = float(min_link_share or 0.0) * total_val

    src, dst, val = [], [], []

    for (a, b), v in links.items():
        if a in idx and b in idx and v >= min_val:
            src.append(idx[a])
            dst.append(idx[b])
            val.append(v)

    weight_map = {
        row["channel"]: float(row["peso_canal"])
        for _, row in df_attr.iterrows()
    }

    min_w, max_w = 0.02, 0.25

    def scale_w(x: float) -> float:
        return min_w + (max_w - min_w) * max(0.0, min(1.0, x))

    risk_only = df_sum.set_index("channel")[["risk_level"]]
    meta = df_attr.set_index("channel").join(risk_only, how="left")

    risk_color = {
        "HIGH": "#d62728",
        "MEDIUM": "#ff7f0e",
        "LOW": "#2ca02c"
    }

    sizes, colors, hover = [], [], []

    attr_ix = df_attr.set_index("channel")

    assert df_attr["channel"].is_unique, "channel no es único"

    for lab in labels:

        if lab in {STATE_START, ABSORB_CONV, ABSORB_NULL}:
            sizes.append(0.15 if lab == STATE_START else 0.12)
            colors.append("#7f7f7f" if lab != ABSORB_CONV else "#2ca02c")
            hover.append(lab)
            continue

        pc = float(weight_map.get(lab, 0.0))
        sizes.append(scale_w(pc))

        rk = "LOW"
        if lab in meta.index:
            val_rk = meta.at[lab, "risk_level"]

            rk = str(val_rk) if pd.notna(val_rk) else "LOW"

        colors.append(risk_color.get(rk, "#7f7f7f"))

        if lab in attr_ix.index:

            ft = safe_float(attr_ix.at[lab, "first_touch_share"])
            re = safe_float(attr_ix.at[lab, "removal_effect_share"])
            da = safe_float(attr_ix.at[lab, "removal_drop_abs"])
            tr = safe_float(attr_ix.at[lab, "touch_rate"])

            ap_val = attr_ix.at[lab, "avg_position"]
            ap = safe_float(ap_val, default=float("nan"))

        else:
            ft = re = da = tr = 0.0
            ap = float("nan")

        hover.append(
            f"<b>{lab}</b><br>"
            f"peso_canal={pc:.3f}<br>"
            f"first_touch_share={ft:.3f}<br>"
            f"removal_effect_share={re:.3f}<br>"
            f"removal_drop_abs={da:.4f}<br>"
            f"touch_rate={tr:.3f}<br>"
            f"avg_pos={ap:.2f}<br>"
            f"riesgo={rk}"
        )

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            label=labels,
            pad=20,
            thickness=18,
            color=colors,
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>"
        ),
        link=dict(
            source=src,
            target=dst,
            value=val,
            color="rgba(150,150,150,0.35)"
        )
    )])

    fig.update_layout(
        title="Flujo de canales (START → ... → CONVERSION/NULL)",
        height=700
    )

    return fig

def build_scatter_fig(df_sum: pd.DataFrame):
    basis = df_sum["risk_basis"].iloc[0] if len(df_sum) else "share"
    x_col = "removal_effect_share" if basis == "share" else "removal_drop_abs"

    fig = px.scatter(
        df_sum,
        x=x_col,
        y="touch_rate",
        size="peso_canal",
        color="risk_level",
        hover_name="channel",
        title="Mapa de dependencias por canal",
        color_discrete_map={
            "HIGH": "#d62728",
            "MEDIUM": "#ff7f0e",
            "LOW": "#2ca02c"
        }
    )

    fig.update_layout(
        height=650,
        legend_title="Riesgo"
    )

    fig.update_traces(
        marker=dict(
            opacity=0.85,
            line=dict(width=0.5, color="#444")
        )
    )

    return fig
