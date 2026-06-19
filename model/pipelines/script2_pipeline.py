from model.attribution import compute_touch_rate_tokens, compute_avg_position_tokens, classify_risk_and_action, compute_channel_transitions, top_k_nodes
from model.visualization import build_sankey_fig, build_scatter_fig
from model.utils import build_summary_text


def run_analysis_pipeline(df_paths, df_attr, artifacts, cfg):
    sep = cfg.get("parsing", {}).get("path_separator", " > ")

    baseline = float(artifacts.get("baseline_structural_probability", 0.0))
    rem_drop_abs = artifacts.get("removal_drop_absolute", {})

    # ✅ métricas
    touch_rate = compute_touch_rate_tokens(df_paths, sep)
    avg_pos = compute_avg_position_tokens(df_paths, sep)

    df_attr = df_attr.copy()

    df_attr["removal_drop_abs"] = df_attr["channel"].map(rem_drop_abs).fillna(0.0)
    df_attr["touch_rate"] = df_attr["channel"].map(touch_rate).fillna(0.0)
    df_attr["avg_position"] = df_attr["channel"].map(avg_pos)

    # ✅ riesgo
    df_sum = classify_risk_and_action(
        df_attr,
        baseline,
        cfg.get("post_analysis", {}).get("risk_basis", "auto"),
        cfg.get("post_analysis", {}).get("risk_thresholds_abs", {}),
        cfg.get("post_analysis", {}).get("risk_thresholds_share", {})
    )

    # ✅ Sankey
    links, node_weight = compute_channel_transitions(df_paths, sep)

    node_order = top_k_nodes(
        node_weight,
        cfg.get("sankey", {}).get("top_n_channels", 30)
    )

    sankey_fig = build_sankey_fig(
        links,
        node_order,
        df_attr,
        df_sum,
        cfg.get("sankey", {}).get("min_link_share", 0.0005)
    )

    # Scatter
    scatter_fig = build_scatter_fig(df_sum)

    # Summary
    summary = build_summary_text(df_sum, baseline)

    return {
        "df_summary": df_sum,
        "sankey": sankey_fig,
        "scatter": scatter_fig,
        "summary": summary
    }