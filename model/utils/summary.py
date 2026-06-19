import pandas as pd

def build_summary_text(
    df_sum: pd.DataFrame,
    baseline: float,
    scenario_header: str | None = None
) -> str:
    lines = []
    if scenario_header:
        lines.append(scenario_header + "\n")

    lines.append("BASELINE estructural (prob. de conversión): {:.2%}\n".format(baseline))
    top = df_sum.sort_values("peso_canal", ascending=False).head(5)
    lines.append("Top canales por peso_canal:")
    for _, r in top.iterrows():
        lines.append(
            "  - {}: peso={:.2%}, RE_share={:.2%}, FT={:.2%}, riesgo={} ({})"
            .format(
                r["channel"],
                float(r["peso_canal"]),
                float(r["removal_effect_share"]),
                float(r["first_touch_share"]),
                r["risk_level"],
                r["accion_sugerida"],
            )
        )

    lines.append("\nACCIONABLES:")
    for lvl in ["HIGH", "MEDIUM", "LOW"]:
        subset = df_sum[df_sum["risk_level"] == lvl].sort_values("peso_canal", ascending=False).head(6)
        if len(subset):
            if lvl == "HIGH":
                lines.append(" • PROTEGER (alto riesgo):")
            elif lvl == "MEDIUM":
                lines.append(" • OPTIMIZAR (riesgo medio):")
            else:
                lines.append(" • ESCALAR/MONITOREAR (riesgo bajo):")
            for _, r in subset.iterrows():
                lines.append(
                    "    - {}: {} (peso={:.2%}, touch_rate={:.1%}, avg_pos={:.2f})"
                    .format(
                        r["channel"],
                        r["accion_sugerida"],
                        float(r["peso_canal"]),
                        float(r["touch_rate"]),
                        float(r["avg_position"]) if not pd.isna(r["avg_position"]) else float("nan")
                    )
                )

    lines.append("\nNota: riesgo describe dependencia estructural del funnel; no implica causalidad.")
    
    return "\n".join(lines)