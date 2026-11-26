import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Riesgo sísmico por zona", page_icon="🌎", layout="wide")

DATA_PATH = "Dataset_Final_Modelamiento.csv"
MODEL_PATH = "model.joblib"
TARGET = "target_m5_prox_trim"


@st.cache_resource
def cargar_modelo():
    return joblib.load(MODEL_PATH)


@st.cache_data
def cargar_datos():
    return pd.read_csv(DATA_PATH)


def siguiente_trimestre(t):
    anio, q = int(t[:4]), int(t[-1])
    return f"{anio + 1}Q1" if q == 4 else f"{anio}Q{q + 1}"


def puntuar(df, art):
    """Calcula el puntaje de riesgo usando exactamente las 26 variables del entrenamiento."""
    X = df[art["features"]]
    out = df.copy()
    out["puntaje"] = art["pipeline"].predict_proba(X)[:, 1]
    out["alerta"] = (out["puntaje"] >= art["umbral"]).astype(int)
    return out


art = cargar_modelo()
df = cargar_datos()

# Validación de esquema: el dataset debe traer las mismas columnas que vio el modelo
faltantes = [c for c in art["features"] if c not in df.columns]
if faltantes:
    st.error(f"Faltan variables que el modelo necesita: {faltantes}")
    st.stop()

df = puntuar(df, art)

# ---------------- Encabezado ----------------
st.title("🌎 Riesgo sísmico por zona y trimestre")
st.write(
    "Estima si en una zona ocurrirá **al menos un sismo de magnitud ≥ 5.0 en el trimestre siguiente (T+1)**, "
    "usando solo información histórica hasta el trimestre elegido. Datos: catálogo USGS · Grupo 04 MDS, UPC."
)

# ---------------- Controles ----------------
trimestres = sorted(df["trimestre"].unique())
zonas = sorted(df["zona_id"].unique())

with st.sidebar:
    st.header("Consulta")
    zona = st.selectbox("Zona", zonas)
    trimestre = st.selectbox("Trimestre de referencia (T)", trimestres, index=len(trimestres) - 1)
    st.caption(f"Se predice el trimestre **{siguiente_trimestre(trimestre)}**.")
    st.divider()
    st.caption(f"Modelo: {art['modelo']} · umbral de alerta: {art['umbral']}")

tab_pred, tab_rank, tab_mon = st.tabs(["Predicción", "Ranking de zonas", "Desempeño y monitoreo"])

# ---------------- Tab 1: Predicción ----------------
with tab_pred:
    fila = df[(df.zona_id == zona) & (df.trimestre == trimestre)].iloc[0]
    sig = siguiente_trimestre(trimestre)

    c1, c2, c3 = st.columns(3)
    c1.metric("Puntaje de riesgo (0 a 1)", f"{fila['puntaje']:.2f}",
              help="Puntaje del modelo. Se entrenó con pesos balanceados, así que no es una probabilidad "
                   "calibrada: sirve para ordenar zonas y activar la alerta.")
    c2.metric("Tasa histórica M≥5 de la zona", f"{fila['tasa_hist_m5_zona']:.1%}")
    c3.metric("Sismos M≥5 en los últimos 4 trimestres", int(fila["n_m5_4t"]))

    if fila["alerta"] == 1:
        st.error(f"⚠️ **Riesgo ALTO**: el modelo anticipa al menos un sismo M ≥ 5.0 en {zona} durante {sig}.")
    else:
        st.success(f"✅ **Riesgo BAJO**: el modelo no anticipa sismos M ≥ 5.0 en {zona} durante {sig}.")

    real = "Sí ocurrió" if fila[TARGET] == 1 else "No ocurrió"
    st.caption(f"Resultado real registrado en {sig}: **{real}** (dato del catálogo, para contrastar).")
    st.caption("Es una estimación estadística, no un pronóstico oficial ni una garantía.")

    with st.expander("Ver las variables que usó el modelo"):
        st.dataframe(fila[art["features"]].rename("valor").to_frame(), use_container_width=True)

# ---------------- Tab 2: Ranking ----------------
with tab_rank:
    st.subheader(f"Zonas ordenadas por riesgo · base {trimestre} → predicción {siguiente_trimestre(trimestre)}")
    rank = (df[df.trimestre == trimestre]
            .sort_values("puntaje", ascending=False)
            [["zona_id", "puntaje", "alerta", "n_m5_4t", "tasa_hist_m5_zona", TARGET]]
            .rename(columns={"zona_id": "Zona", "puntaje": "Puntaje", "alerta": "Alerta",
                             "n_m5_4t": "M≥5 últimos 4T", "tasa_hist_m5_zona": "Tasa histórica",
                             TARGET: "Ocurrió (real)"}))
    rank["Alerta"] = rank["Alerta"].map({1: "ALTO", 0: "bajo"})
    rank["Ocurrió (real)"] = rank["Ocurrió (real)"].map({1: "Sí", 0: "No"})
    st.dataframe(rank.style.format({"Puntaje": "{:.2f}", "Tasa histórica": "{:.1%}"}),
                 use_container_width=True, hide_index=True)

# ---------------- Tab 3: Desempeño y monitoreo ----------------
with tab_mon:
    st.subheader("Desempeño en prueba (2021–2026, datos que el modelo no vio)")
    m = art["metricas_test"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall", f"{m['recall']:.0%}")
    c2.metric("Precision", f"{m['precision']:.0%}")
    c3.metric("F1", f"{m['f1']:.2f}")
    c4.metric("PR-AUC", f"{m['pr_auc']:.2f}", help=f"Línea base (tasa histórica): {art['linea_base_test']['pr_auc']}")
    cm = m["matriz_confusion"]
    st.write(f"Matriz de confusión → TP: {cm['TP']} · FN: {cm['FN']} · FP: {cm['FP']} · TN: {cm['TN']}")

    st.subheader("Monitoreo por año (conjunto de prueba)")
    test = df[df.split == "test"].copy()
    test["anio"] = test["trimestre"].str[:4]
    mon = test.groupby("anio").apply(lambda g: pd.Series({
        "Casos reales M≥5": int(g[TARGET].sum()),
        "Alertas emitidas": int(g["alerta"].sum()),
        "Aciertos": int(((g["alerta"] == 1) & (g[TARGET] == 1)).sum()),
        "Puntaje medio": round(g["puntaje"].mean(), 3),
    }), include_groups=False)
    mon["Recall"] = (mon["Aciertos"] / mon["Casos reales M≥5"].replace(0, pd.NA)).astype(float).round(2)
    st.dataframe(mon, use_container_width=True)
    st.caption("Si el recall cae de forma sostenida o el puntaje medio se aleja mucho del histórico, "
               "es señal de reentrenar el modelo con datos actualizados del catálogo.")
