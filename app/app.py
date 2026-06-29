import streamlit as st
import pandas as pd
import json
import sys
import os

# 👉 agregar la carpeta raíz del proyecto al path
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_PATH)

# Pipelines
from model.pipelines import run_markov_pipeline, run_analysis_pipeline
from model.in_out import load_paths_csv
from model.utils import get_progress_bar

CONFIG_PATH = "model/config/config.json"

# ----------------------------
# CONFIG INICIAL
# ----------------------------
st.set_page_config(
    page_title="Markov Attribution",
    layout="wide"
)

st.title("📊 Markov Attribution Model")

# ----------------------------
# CARGA CONFIG
# ----------------------------
@st.cache_data
def load_config(path):
    with open(path, "r") as f:
        return json.load(f)

cfg = load_config(CONFIG_PATH)

# Valor por defecto para la simulación (se sobreescribe desde la sidebar más abajo)
if "channels_to_remove" not in st.session_state:
    st.session_state.channels_to_remove = []
    
if "available_channels" not in st.session_state:
    st.session_state.available_channels = []

if "model_executed" not in st.session_state:
    st.session_state.model_executed = False
    
if "results" not in st.session_state:
    st.session_state.results = None

if "artifacts" not in st.session_state:
    st.session_state.artifacts = None

if "df_sum" not in st.session_state:
    st.session_state.df_sum = None

if "df_base" not in st.session_state:
    st.session_state.df_base = None

if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

# Removed pre-population of available_channels from CSV to enforce
# explicit model run before populating dropdowns.


# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.header("⚙️ Configuración")

uploaded_file = st.sidebar.file_uploader(
    "Subir CSV de rutas",
    type=["csv"]
)

# Reiniciar el estado si cambió el archivo
current_file = (
    uploaded_file.file_id
    if uploaded_file is not None
    else None
)

if current_file != st.session_state.last_uploaded_file:

    st.session_state.last_uploaded_file = current_file

    # Limpiar simulación
    st.session_state.channels_to_remove = []
    st.session_state.available_channels = []

    # Limpiar resultados
    st.session_state.results = None
    st.session_state.artifacts = None
    st.session_state.df_sum = None
    st.session_state.df_base = None

    st.session_state.model_executed = False

    # Reiniciar selectbox
    if "channel_to_remove" in st.session_state:
        del st.session_state["channel_to_remove"]


# ----------------------------
# SLIDERS MODELO
# ----------------------------
st.sidebar.subheader("📐 Parámetros del modelo")

removal_w = st.sidebar.slider(
    "Peso Removal Effect",
    0.0, 1.0,
    value=cfg["weights"]["removal_w"],
    step=0.05
)

first_w = st.sidebar.slider(
    "Peso First Touch",
    0.0, 1.0,
    value=cfg["weights"]["first_w"],
    step=0.05
)

global_rate = st.sidebar.slider(
    "Probabilidad de conversión",
    0.0, 1.0,
    value=cfg["non_converting_paths"]["global_rate"],
    step=0.05
)

# ----------------------------
# SLIDERS VISUAL
# ----------------------------
# st.sidebar.subheader("📊 Visualización")


# top_n_channels = st.sidebar.slider(
#     "Top canales en Sankey",
#     10, 100,
#     value=cfg["sankey"]["top_n_channels"],
#     step=5
# )


# actualizar config dinámico
cfg["weights"]["removal_w"] = removal_w
cfg["weights"]["first_w"] = first_w
cfg["non_converting_paths"]["global_rate"] = global_rate
# cfg["sankey"]["top_n_channels"] = top_n_channels

# validar pesos
if abs(removal_w + first_w - 1.0) > 0.05:
    st.sidebar.warning("⚠️ Los pesos deberían sumar aproximadamente 1")

def remove_channels_from_paths(df, channels_to_remove, sep=">"):

    def normalize_token(x):
        return (
            str(x)
            .replace('"', '')
            .replace("'", "")
            .strip()
            .capitalize()
        )

    if not channels_to_remove:
        return df.copy()

    channels_norm = {
        normalize_token(c)
        for c in channels_to_remove
    }

    def clean_path(path):

        tokens = [
            normalize_token(t)
            for t in str(path).split(sep)
        ]

        tokens = [
            t for t in tokens
            if t not in channels_norm
        ]

        return " > ".join(tokens)

    df_new = df.copy()
    df_new["path"] = df_new["path"].apply(clean_path)

    return df_new[df_new["path"].str.strip() != ""]

run_button = st.sidebar.button("🚀 Ejecutar modelo")

st.sidebar.subheader("🧪 Simulación de eliminación")

selected_channel = None
rerun_button = False

if st.session_state.model_executed:

    selected_channel = st.sidebar.selectbox(
        "Eliminar canal",
        options=["Ninguno"] + st.session_state.available_channels,
        key="channel_to_remove"
    )

    rerun_button = st.sidebar.button(
        "🔄 Recalcular con eliminación"
    )

# ----------------------------
# MAIN
# ----------------------------
if run_button or rerun_button:
    
    if run_button:
        st.session_state.channels_to_remove = []


    progress = get_progress_bar(
        total_steps=6,
        mode="streamlit"
    )

    # ----------------------------
    # INPUT
    # ----------------------------
    
    progress.update(1, "Cargando datos...")
    
    if uploaded_file is not None:
        df_input = load_paths_csv(uploaded_file)
        st.info("📁 Usando archivo subido")
    else:
        csv_path = os.path.join(
            ROOT_PATH,
            cfg["io"]["input_csv"]
            .replace("\\", os.sep)
            .replace("/", os.sep)
        )
        df_input = load_paths_csv(csv_path)
        st.info("📁 Usando archivo del config")

    st.success("✅ Datos cargados")
    
    if rerun_button:

        if selected_channel != "Ninguno":
            st.session_state.channels_to_remove = [selected_channel]
        else:
            st.session_state.channels_to_remove = []

    # ----------------------------
    # SIMULACIÓN DE REMOCIÓN
    # ----------------------------
    progress.update(2, "Aplicando simulación...")
    
    channels_to_remove = [
        ch
        for ch in st.session_state.channels_to_remove
        if ch is not None
    ]

    if channels_to_remove:

        df_input_sim = remove_channels_from_paths(
            df_input,
            channels_to_remove
        )

        remaining_text = " ".join(df_input_sim["path"].astype(str).tolist())

        for ch in channels_to_remove:
            if str(ch).capitalize() in remaining_text.capitalize():
                st.error(f"❌ ERROR: {ch} sigue presente en paths")

        st.info(
            "🧪 Simulación: eliminando "
            + ", ".join(channels_to_remove)
        )

    else:
        df_input_sim = df_input

    # ----------------------------
    # MODELO
    # ----------------------------
    progress.update(3, "Ejecutando modelo Markov...")
    
    df_attr, artifacts, df_base = run_markov_pipeline(
        cfg=cfg,
        df_input=df_input_sim
    )

    st.success("✅ Modelo ejecutado")

    # ----------------------------
    # ANALYSIS
    # ----------------------------
    progress.update(4, "Calculando métricas...")
    
    results = run_analysis_pipeline(
        df_paths=df_base,
        df_attr=df_attr,
        artifacts=artifacts,
        cfg=cfg
    )
    
    if channels_to_remove:
        still_exists = results["df_summary"][
            results["df_summary"]["channel"]
            .str.capitalize()
            .isin([c.capitalize() for c in channels_to_remove])
        ]

        if not still_exists.empty:
            st.error("❌ El modelo sigue incluyendo canales eliminados")

    # ----------------------------
    # SELECCIÓN DE CANALES
    # ----------------------------
    df_sum = results["df_summary"]

    progress.update(5, "Guardando resultados...")
    
    st.session_state.results = results
    st.session_state.artifacts = artifacts
    st.session_state.df_sum = df_sum
    st.session_state.df_base = df_base

    st.session_state.available_channels = (
        df_sum["channel"]
        .dropna()
        .sort_values()
        .tolist()
    )
    
    progress.update(6, "Finalizando...")
    
    st.session_state.model_executed = True
    progress.finish()
    st.rerun()
            
# ----------------------------
# VISUALIZACIÓN
# ----------------------------
if (st.session_state.results is not None and st.session_state.artifacts is not None and st.session_state.df_sum is not None):

    results = st.session_state.results
    artifacts = st.session_state.artifacts
    df_sum = st.session_state.df_sum

    # ----------------------------
    # TABS
    # ----------------------------
    
    tab0, tab1, tab2, tab3, tab4 = st.tabs([
        "📊 KPIs",
        "📈 Sankey",
        "📊 Dependencias",
        "📋 Tabla",
        "🧠 Insights"
    ])
        
    # ----------------------------------
    # TAB 0 - KPI
    # ----------------------------------
    with tab0:

        st.markdown("## 📊 KPIs del modelo")

        baseline = artifacts.get("baseline_structural_probability", None)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Prob. conversión",
            f"{baseline:.2%}" if baseline else "N/A"
        )

        col2.metric(
            "Canales activos",
            len(df_sum)
        )

        col3.metric(
            "Peso total",
            f"{df_sum['peso_canal'].sum():.2%}"
        )

        high_risk = df_sum[df_sum["risk_level"] == "HIGH"]

        col4.metric(
            "Canales críticos",
            len(high_risk)
        )

        # progreso
        st.caption("Probabilidad estructural de conversión")
        st.progress(min(1.0, baseline if baseline else 0.0))

        # ----------------------------------
        # TOP CHANNEL
        # ----------------------------------
        top_channel = df_sum.sort_values(
            "peso_canal", ascending=False
        ).iloc[0]

        st.markdown("## 🏆 Canal principal")

        col1, col2, col3 = st.columns(3)

        col1.metric("Canal", top_channel["channel"])
        col2.metric("Peso", f"{top_channel['peso_canal']:.2%}")
        col3.metric(
            "Impacto",
            f"{top_channel['removal_effect_share']:.2%}"
        )

    # Sankey
    with tab1:
        st.subheader("Flujo de canales")
        st.plotly_chart(results["sankey"], use_container_width=True)

    # Scatter
    with tab2:
        st.subheader("Mapa de dependencias")
        st.plotly_chart(results["scatter"], use_container_width=True)

    # Tabla
    with tab3:
        st.subheader("Tabla de canales")

        basis = df_sum["risk_basis"].iloc[0]

        if basis == "share":
            st.info(
                "**Criterio de evaluación del riesgo:** Impacto porcentual (**Removal Effect**).\n\n"
                "El nivel de riesgo se calcula según cuánto disminuye la probabilidad de conversión al eliminar un canal."
            )
        else:
            st.info(
                "**Criterio de evaluación del riesgo:** Pérdida absoluta de conversiones (**Removal Drop**).\n\n"
                "El nivel de riesgo se calcula según la cantidad de conversiones que se perderían al eliminar un canal."
            )

        MIN_EFFECT = 1e-4

        df_table = df_sum[
            (df_sum["removal_effect_share"] >= MIN_EFFECT) |
            (df_sum["channel"] == "Direct")
        ].copy()
        
        df_table = df_table.drop(columns=["risk_basis"], errors="ignore")
        
        # Mostrar únicamente la métrica utilizada para evaluar el riesgo
        if basis == "share":
            df_table = df_table.drop(columns=["removal_drop_abs"], errors="ignore")
        else:
            df_table = df_table.drop(columns=["removal_effect_share"], errors="ignore")
        
        column_config = {
            "rank": "Ranking",
            "channel": "Canal",
            "peso_canal": st.column_config.ProgressColumn(
                "Peso del canal",
                min_value=0,
                max_value=df_sum["peso_canal"].max(),
                format="%.4f%%"
            ),
            "first_touch_share": st.column_config.NumberColumn(
                "First Touch",
                format="%.4f"
            ),
            "touch_rate": st.column_config.NumberColumn(
                "Touch Rate",
                format="%.4f"
            ),
            "avg_position": st.column_config.NumberColumn(
                "Posición promedio",
                format="%.4f"
            ),
            "risk_level": st.column_config.TextColumn("Nivel de riesgo"),
            "accion_sugerida": st.column_config.TextColumn("Acción sugerida"),
        }

        if basis == "share":
            column_config["removal_effect_share"] = st.column_config.NumberColumn(
                "Removal Effect",
                format="%.4f"
            )
        else:
            column_config["removal_drop_abs"] = st.column_config.NumberColumn(
                "Pérdida de conversiones",
                format="%.4f"
            )

        st.dataframe(
            df_table,
            use_container_width=True,
            hide_index=True,
            column_config=column_config
        )

        csv = df_sum.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Descargar CSV",
            csv,
            "markov_summary.csv",
            "text/csv"
        )

    # Summary
    with tab4:
        st.subheader("Resumen accionable")

        # Baseline
        st.metric(
            "Probabilidad de conversión (Baseline)",
            f"{results['baseline']:.2%}"
        )

        st.divider()

        # Top canales
        st.markdown("### 🏆 Top 5 canales por peso")

        top = (
            df_sum
            .sort_values("peso_canal", ascending=False)
            .head(5)
        )

        st.dataframe(
            top[[
                "channel",
                "peso_canal",
                "removal_effect_share",
                "first_touch_share",
                "risk_level",
                "accion_sugerida"
            ]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "channel": "Canal",
                "peso_canal": st.column_config.NumberColumn(
                    "Peso",
                    format="%.2f%%"
                ),
                "removal_effect_share": st.column_config.NumberColumn(
                    "Removal Effect",
                    format="%.2f%%"
                ),
                "first_touch_share": st.column_config.NumberColumn(
                    "First Touch",
                    format="%.2f%%"
                ),
                "risk_level": "Riesgo",
                "accion_sugerida": "Acción"
            }
        )

        st.divider()

        niveles = [
            ("HIGH", "🔴 Canales a proteger", st.error),
            ("MEDIUM", "🟡 Canales a optimizar", st.warning),
            ("LOW", "🟢 Canales a escalar / monitorear", st.success),
        ]

        for nivel, titulo, box in niveles:

            subset = (
                df_sum[df_sum["risk_level"] == nivel]
                .sort_values("peso_canal", ascending=False)
            )

            if subset.empty:
                continue

            box(f"**{titulo}**")

            st.dataframe(
                subset[[
                    "channel",
                    "peso_canal",
                    "touch_rate",
                    "avg_position",
                    "accion_sugerida"
                ]],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "channel": "Canal",
                    "peso_canal": st.column_config.NumberColumn(
                        "Peso",
                        format="%.2f%%"
                    ),
                    "touch_rate": st.column_config.NumberColumn(
                        "Touch Rate",
                        format="%.2f%%"
                    ),
                    "avg_position": st.column_config.NumberColumn(
                        "Posición promedio",
                        format="%.2f"
                    ),
                    "accion_sugerida": "Acción sugerida"
                }
            )

        st.info(
            "El nivel de riesgo representa la dependencia estructural del funnel respecto a cada canal. "
            "No implica necesariamente una relación causal."
        )