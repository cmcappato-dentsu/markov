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
from model.in_out.reader import load_paths_csv

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

# Removed pre-population of available_channels from CSV to enforce
# explicit model run before populating dropdowns.

# Limpieza/sanitización de session_state para evitar valores obsoletos
# Si hay claves previas que contengan valores que no existen en available_channels,
# las eliminamos para evitar errores de Streamlit sobre defaults/keys duplicadas.
def _sanitize_state():
    try:
        av = st.session_state.get("available_channels", []) or []

        # Only clear stored selections if we have no available channels at all.
        # Avoid deleting user selections when available_channels is present,
        # since that causes widget/default mismatches and unexpected reruns.
        if not av:
            if "selected_channels_display" in st.session_state:
                try:
                    del st.session_state["selected_channels_display"]
                except Exception:
                    pass
            if "channels_multiselect" in st.session_state:
                try:
                    del st.session_state["channels_multiselect"]
                except Exception:
                    pass
    except Exception:
        # No queremos que la sanitización falle la carga de la app
        pass

_sanitize_state()

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.header("⚙️ Configuración")

uploaded_file = st.sidebar.file_uploader(
    "Subir CSV de rutas",
    type=["csv"]
)

# Debug toggle: muestra session_state en la barra lateral (útil para diagnosticar widgets)
show_debug = st.sidebar.checkbox("Mostrar session_state (debug)", value=False)
if show_debug:
    try:
        st.sidebar.write(dict(st.session_state))
    except Exception:
        st.sidebar.write("No se pudo mostrar session_state")

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

# ----------------------------
# SLIDERS VISUAL
# ----------------------------
st.sidebar.subheader("📊 Visualización")

top_n_channels = st.sidebar.slider(
    "Top canales en Sankey",
    10, 100,
    value=cfg["sankey"]["top_n_channels"],
    step=5
)


# actualizar config dinámico
cfg["weights"]["removal_w"] = removal_w
cfg["weights"]["first_w"] = first_w
cfg["sankey"]["top_n_channels"] = top_n_channels

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
            .lower()
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

# Simulación de eliminación: multiselect persistente en sidebar
st.sidebar.subheader("🧪 Simulación de eliminación")
# Inicializar la clave del multiselect si aún no existe y hay canales precargados
if "channels_multiselect" not in st.session_state and st.session_state.get("available_channels"):
    st.session_state.channels_multiselect = list(st.session_state.available_channels)

channels_to_remove = st.sidebar.multiselect(
    "Eliminar canales del modelo",
    options=st.session_state.get("available_channels", []),
    key="channels_multiselect"
)

# guardar selección simplificada
st.session_state.channels_to_remove = channels_to_remove

run_button = st.sidebar.button("🚀 Ejecutar modelo")

# ----------------------------
# MAIN
# ----------------------------
if run_button:

    with st.spinner("Ejecutando modelo Markov..."):

        # ----------------------------
        # INPUT
        # ----------------------------
        if uploaded_file is not None:
            df_input = load_paths_csv(uploaded_file)
            st.info("📁 Usando archivo subido")
        else:
            csv_path = cfg["io"]["input_csv"]
            df_input = load_paths_csv(csv_path)
            st.info("📁 Usando archivo del config")

        st.success("✅ Datos cargados")
        
        # ----------------------------
        # SIMULACIÓN DE REMOCIÓN
        # ----------------------------
        channels_to_remove = st.session_state.channels_to_remove
        
        if channels_to_remove:
            df_input_sim = remove_channels_from_paths(
                df_input,
                channels_to_remove
            )

            remaining_text = " ".join(df_input_sim["path"].astype(str).tolist())

            for ch in channels_to_remove:
                if ch.lower() in remaining_text.lower():
                    st.error(f"❌ ERROR: {ch} sigue presente en paths")

            st.info("🧪 Simulación: eliminando " + ", ".join(channels_to_remove))

        else:
            df_input_sim = df_input

        # ----------------------------
        # MODELO
        # ----------------------------
        df_attr, artifacts, df_base = run_markov_pipeline(
            CONFIG_PATH,
            df_input=df_input_sim
        )

        st.success("✅ Modelo ejecutado")

        # ----------------------------
        # ANALYSIS
        # ----------------------------
        results = run_analysis_pipeline(
            df_paths=df_base,
            df_attr=df_attr,
            artifacts=artifacts,
            cfg=cfg
        )
        
        if channels_to_remove:
            still_exists = results["df_summary"][
                results["df_summary"]["channel"]
                .str.lower()
                .isin([c.lower() for c in channels_to_remove])
            ]

            if not still_exists.empty:
                st.error("❌ El modelo sigue incluyendo canales eliminados")

        # ----------------------------
        # SELECCIÓN DE CANALES
        # ----------------------------
        df_sum = results["df_summary"]

        available_channels = df_sum["channel"].tolist()
        st.session_state.available_channels = available_channels
        st.session_state.model_executed = True
        

        # Mostrar multiselect de filtro con opciones y default
        st.sidebar.subheader("🎯 Filtro de canales")
        selected_channels = st.sidebar.multiselect(
            "Elegir canales para visualizar",
            options=available_channels,
            default=available_channels,
            key="selected_channels_display"
        )

        if not selected_channels:
            selected_channels = available_channels

        df_sum_filtered = df_sum[df_sum["channel"].isin(selected_channels)]

        if df_sum_filtered.empty:
            st.warning("⚠️ El filtro dejó sin datos")
            st.stop()

        st.sidebar.caption(f"{len(selected_channels)} canales seleccionados")

        df_sum = df_sum_filtered


        
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

            st.dataframe(df_sum, use_container_width=True)

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
            st.text(results["summary"])