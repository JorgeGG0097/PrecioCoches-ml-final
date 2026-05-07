import os
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

AÑO_ACTUAL = 2026
RUTA_BASE  = os.path.dirname(os.path.abspath(__file__))

AZUL           = "#0070f3"
AZUL_L         = "#3B82F6"
AZUL_LL        = "#BFDBFE"
VERDE          = "#00c072"
VERDE_L        = "#D1FAE5"
NARANJA        = "#f59e0b"
ROJO           = "#DC2626"
OSCURO         = "#0F172A"
GRIS           = "#64748B"
COLORES_LINEAS = ["#0070f3","#f59e0b","#00c072","#7928ca","#C62828","#00838F"]

ETIQUETAS = {
    "precio_eur":         "Precio (€)",
    "año":                "Año de fabricación",
    "kilometraje_km":     "Kilometraje (km)",
    "potencia_cv":        "Potencia (CV)",
    "antiguedad":         "Antigüedad (años)",
    "km_por_año":         "Kilómetros por año",
    "marca":              "Marca",
    "modelo":             "Modelo",
    "combustible":        "Combustible",
    "transmision":        "Transmisión",
    "etiqueta_ambiental": "Etiqueta ambiental",
    "tipo_venta":         "Tipo de venta",
    "ubicacion":          "Provincia",
}

VARS_CATEGORICAS = ["marca", "combustible", "transmision", "etiqueta_ambiental", "tipo_venta"]
VARS_NUMERICAS   = ["precio_eur", "año", "kilometraje_km", "potencia_cv", "antiguedad", "km_por_año"]

CONTEXTOS_CAT = {
    ("combustible", "precio_eur"): (
        "Los vehículos eléctricos e híbridos enchufables tienen precios más elevados por el coste de sus baterías. "
        "Los de gasolina representan la gama más amplia y asequible del mercado de ocasión."
    ),
    ("transmision", "precio_eur"): (
        "La transmisión automática se asocia a vehículos de gama media-alta. "
        "En el mercado español, la preferencia histórica por el manual hace que los automáticos tengan un componente de gama clara."
    ),
    ("etiqueta_ambiental", "precio_eur"): (
        "Las etiquetas 0 EMISIONES y ECO corresponden a vehículos con propulsión más avanzada, "
        "lo que eleva su valor. Los incentivos fiscales vinculados a estas etiquetas sostienen los precios."
    ),
    ("tipo_venta", "precio_eur"): (
        "Los vehículos km 0 y casi nuevos proceden de concesionario con garantía completa, "
        "lo que eleva su precio respecto al mercado de segunda mano convencional. "
        "Los de demostración ofrecen un equilibrio entre precio y garantía oficial."
    ),
    ("marca", "precio_eur"): (
        "El posicionamiento de marca impacta directamente en el precio de segunda mano. "
        "Las marcas premium mantienen valores significativamente más altos que las generalistas."
    ),
    ("ubicacion", "precio_eur"): (
        "Las diferencias por provincia reflejan el poder adquisitivo local y la oferta-demanda regional. "
        "Las grandes ciudades suelen mostrar precios más altos por mayor rotación."
    ),
    ("combustible", "kilometraje_km"): (
        "Los diésel acumulan más kilómetros porque históricamente han sido preferidos para trayectos largos. "
        "Los de gasolina se usan más en entorno urbano con recorridos más cortos."
    ),
    ("transmision", "kilometraje_km"): (
        "Los automáticos se asocian más al uso urbano y distancias cortas, "
        "mientras que los manuales son más comunes en desplazamientos interurbanos."
    ),
    ("marca", "potencia_cv"): (
        "La potencia media por marca refleja el posicionamiento de gama: "
        "las marcas premium ofrecen motorizaciones más potentes como propuesta de valor diferencial."
    ),
    ("etiqueta_ambiental", "potencia_cv"): (
        "Los vehículos ECO y 0 EMISIONES tienen motorizaciones modernas que combinan eficiencia con buena potencia, "
        "mientras que los sin etiqueta corresponden mayoritariamente a modelos más antiguos."
    ),
}

CONTEXTOS_NUM = {
    ("año", "precio_eur"): (
        "A mayor año de fabricación, mayor precio. Es el efecto principal de la depreciación temporal: "
        "un vehículo pierde valor progresivamente, siendo más pronunciada en los primeros años."
    ),
    ("kilometraje_km", "precio_eur"): (
        "Más kilómetros implican mayor desgaste mecánico y menor vida útil esperada. "
        "Este efecto se combina con el del año para explicar gran parte de la depreciación."
    ),
    ("potencia_cv", "precio_eur"): (
        "La potencia es uno de los factores más determinantes del precio. Los vehículos de alta potencia "
        "pertenecen a gamas superiores con mayor equipamiento y tecnología."
    ),
    ("antiguedad", "precio_eur"): (
        "Relación negativa directa: cada año adicional supone una reducción del valor. "
        "La depreciación es especialmente intensa en los primeros 3-5 años."
    ),
    ("km_por_año", "precio_eur"): (
        "Un coche con muchos km/año tiene mayor desgaste relativo a su edad, "
        "lo que se traduce en un valor más bajo incluso si es relativamente reciente."
    ),
    ("año", "kilometraje_km"): (
        "Los vehículos más antiguos han tenido más tiempo para acumular kilometraje. "
        "La pendiente indica los km promedio que acumula un vehículo por año en España."
    ),
    ("potencia_cv", "kilometraje_km"): (
        "La correlación entre potencia y kilometraje suele ser débil: un vehículo potente no "
        "necesariamente acumula más km. Refleja el tipo de uso más que la distancia total."
    ),
    ("antiguedad", "kilometraje_km"): (
        "Relación directa esperada: a mayor antigüedad, más km acumulados. "
        "La dispersión refleja la variabilidad de uso entre conductores."
    ),
}


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PrecioCoches ML",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"], .stMarkdown, p, h1, h2, h3, h4, label {
    font-family: 'Inter', sans-serif !important;
}

/* ── Cursor mano en step buttons y expanders ── */
div[data-testid="stNumberInput"] button,
button[aria-label="Increment"], button[aria-label="Decrement"],
button[aria-label="increment"], button[aria-label="decrement"] { cursor: pointer !important; }
[data-testid="stExpander"] details > summary, details > summary, summary { cursor: pointer !important; }

/* ── Barra superior ── */
.topbar {
    position: fixed; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #0070f3, #00c072, #f59e0b);
    z-index: 9999;
}

/* ── Título de sección ── */
.page-title { font-size: 1.9rem; font-weight: 700; color: #111827; margin: 0 0 4px 0; line-height: 1.2; }
.page-subtitle { font-size: 0.92rem; color: #6B7280; margin: 0 0 24px 0; }

/* ── Cards ── */
.card {
    background: #ffffff;
    border-radius: 12px; padding: 24px;
    border: 1px solid #e5e7eb;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.home-card {
    background: #ffffff; border-radius: 16px; padding: 28px 24px 20px;
    border: 1px solid #e5e7eb; margin-bottom: 4px;
    position: relative; overflow: hidden;
    transition: border-color 0.2s, box-shadow 0.2s; cursor: pointer;
    min-height: 170px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.home-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #0070f3, #3B82F6);
}
.home-card:hover { border-color: #0070f3; box-shadow: 0 4px 12px rgba(0,112,243,0.12); }
.home-card .card-icon { font-size: 2rem; margin-bottom: 10px; display: block; }
.home-card h3 { font-size: 1.05rem !important; font-weight: 700 !important; color: #111827 !important; margin: 0 0 6px 0 !important; }
.home-card p  { font-size: 0.87rem !important; color: #6B7280 !important; margin: 0 !important; line-height: 1.5 !important; }

/* ── Stat cards ── */
.stat-card {
    background: #ffffff; border-radius: 12px; padding: 20px 24px;
    border: 1px solid #e5e7eb; text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.stat-card .stat-val { font-size: 1.75rem; font-weight: 800; line-height: 1; }
.stat-card .stat-lbl { font-size: 0.78rem; color: #6B7280; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }

/* ── Precio principal ── */
.precio-principal { font-size: 3rem; font-weight: 800; color: #111827; line-height: 1; }
.precio-rango { font-size: 0.9rem; color: #6B7280; margin-top: 6px; }
.precio-diff-pos {
    display: inline-block; background: rgba(245,158,11,0.1); color: #d97706;
    border-radius: 6px; padding: 5px 12px; font-size: 0.85rem; font-weight: 500; margin-top: 10px;
}
.precio-diff-neg {
    display: inline-block; background: rgba(0,192,114,0.1); color: #059669;
    border-radius: 6px; padding: 5px 12px; font-size: 0.85rem; font-weight: 500; margin-top: 10px;
}

/* ── Insight ── */
.insight {
    background: rgba(0,112,243,0.06); border-left: 3px solid #0070f3;
    border-radius: 0 8px 8px 0; padding: 14px 18px; margin-top: 16px;
    font-size: 0.88rem; color: #1d4ed8; line-height: 1.6;
}

/* ── Confidence badge ── */
.confidence-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(5,150,105,0.08); border: 1px solid rgba(5,150,105,0.25);
    border-radius: 20px; padding: 4px 10px;
    font-size: 0.78rem; font-weight: 600; color: #059669; margin-top: 10px;
}
.confidence-badge .dot {
    width: 7px; height: 7px; background: #059669; border-radius: 50%;
    box-shadow: 0 0 6px rgba(5,150,105,0.5); display: inline-block; flex-shrink: 0;
}

/* ── Panel label (explorador) ── */
.panel-label {
    font-size: 0.68rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.09em; color: #9CA3AF; margin: 16px 0 6px 0;
}

/* ── Ocultar sidebar ── */
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* ── Ocultar botones estrella y lapiz de la toolbar ── */
.stToolbarActions button:not(:first-child) { display: none !important; }
[data-testid="stToolbarActions"] button:not(:first-child) { display: none !important; }
button[title="Star"], button[title="Edit on Github"],
button[title="Edit"], button[aria-label*="star" i],
button[aria-label*="edit" i], button[aria-label*="pencil" i] { display: none !important; }
header button[kind="headerNoPadding"]:not(:first-of-type) { display: none !important; }

/* ── Misc ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.main .block-container { padding-top: 1.5rem; max-width: 1200px; }
</style>

<div class="topbar"></div>
""", unsafe_allow_html=True)


# ── Cache ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    ruta = os.path.join(RUTA_BASE, "modelo_gbm.pkl")
    return joblib.load(ruta) if os.path.exists(ruta) else None

_MARCA_MAP = {"Alfa": "Alfa Romeo", "Mercedes": "Mercedes-Benz", "Land": "Land Rover"}

def _normalizar_marca(m):
    s = str(m)
    if s.lower().startswith("citro"):
        return "Citroen"
    return _MARCA_MAP.get(s, s)

def _norm_fuel(s):
    import unicodedata
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii").lower().strip()

def _derivar_etiqueta(combustible, año):
    c = _norm_fuel(combustible)
    if "electric" in c:
        return "0_EMISIONES"
    if "brid" in c:
        return "ECO"
    if "gasolina" in c:
        if año >= 2006: return "C"
        if año >= 2001: return "B"
        return "Sin etiqueta"
    if "diesel" in c or "sel" in c:
        if año >= 2015: return "C"
        if año >= 2006: return "B"
        return "Sin etiqueta"
    return "ECO"

_COLOR_ETIQUETA = {
    "0_EMISIONES": ("#065F46", "#D1FAE5"),
    "ECO":         ("#0E7490", "#CFFAFE"),
    "C":           ("#166534", "#DCFCE7"),
    "B":           ("#92400E", "#FEF3C7"),
    "Sin etiqueta":("#475569", "#F1F5F9"),
}

@st.cache_data
def cargar_datos():
    ruta = os.path.join(RUTA_BASE, "Cars_combinado_limpio.csv")
    if not os.path.exists(ruta):
        return None
    df = pd.read_csv(ruta, encoding="utf-8-sig")
    df["marca"] = df["marca"].apply(_normalizar_marca)
    df["antiguedad"] = AÑO_ACTUAL - df["año"]
    df["km_por_año"] = (df["kilometraje_km"] / df["antiguedad"].clip(lower=1)).round(0).astype(int)
    if "etiqueta_ambiental" not in df.columns:
        df["etiqueta_ambiental"] = df.apply(
            lambda r: _derivar_etiqueta(r["combustible"], r["año"]), axis=1
        )
    return df

@st.cache_data
def cargar_json(nombre):
    ruta = os.path.join(RUTA_BASE, nombre)
    if not os.path.exists(ruta):
        return None
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


# ── Carga ──────────────────────────────────────────────────────────────────────
modelo_ml  = cargar_modelo()
df         = cargar_datos()
categorias = cargar_json("categorias.json")
rangos     = cargar_json("rangos_numericos.json")
info_model = cargar_json("features_info.json")

if modelo_ml is None or df is None or categorias is None or rangos is None:
    st.error("Archivos del modelo no encontrados. Ejecuta primero **entrenar_modelo.py**.")
    st.stop()

MAE_MODELO   = info_model["mae_test"] if info_model else 1772
MAPE_FACTOR  = MAE_MODELO / df["precio_eur"].median()   # ≈ 0.108 → intervalo escala con el precio


# ── Secciones ──────────────────────────────────────────────────────────────────
SECCIONES = [
    "🏠  Inicio",
    "🔍  Tasador",
    "💰  Presupuesto",
    "📉  Depreciación",
    "📊  Explorador",
    "🎯  Chollos",
]

if "seccion_idx" not in st.session_state:
    st.session_state["seccion_idx"] = 0
if "_nav_radio" not in st.session_state:
    st.session_state["_nav_radio"] = SECCIONES[0]


# ── Navegación superior ────────────────────────────────────────────────────────
seccion = st.radio(
    "nav",
    SECCIONES,
    key="_nav_radio",
    horizontal=True,
    label_visibility="collapsed",
)
st.session_state["seccion_idx"] = SECCIONES.index(seccion)
st.divider()


# ── Helpers Plotly ─────────────────────────────────────────────────────────────
def estilo_fig(fig, height=420):
    fig.update_layout(
        template="simple_white",
        height=height,
        font=dict(family="Inter, sans-serif", color="#374151", size=12),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor="#f9fafb", bordercolor="#e5e7eb",
                    borderwidth=1, font=dict(size=11, color="#6B7280")),
    )
    fig.update_xaxes(showgrid=False, linecolor="#e5e7eb", zeroline=False,
                     tickfont=dict(size=11, color="#6B7280"))
    fig.update_yaxes(gridcolor="#f3f4f6", showline=False, zeroline=False,
                     tickfont=dict(size=11, color="#6B7280"))
    return fig

def fmt_eur_axis(fig, axis="y"):
    if axis == "y":
        fig.update_yaxes(tickformat=",.0f", ticksuffix=" €")
    else:
        fig.update_xaxes(tickformat=",.0f", ticksuffix=" €")
    return fig


# ── Explicaciones automáticas ──────────────────────────────────────────────────
def generar_explicacion(df_data, var_x, var_y, tipo_x):
    if tipo_x == "categorica":
        stats = df_data.groupby(var_x)[var_y].median()
        if len(stats) < 2:
            return "No hay suficientes categorías para comparar."
        mejor, peor = stats.idxmax(), stats.idxmin()
        val_max, val_min = stats.max(), stats.min()
        diff_pct = ((val_max - val_min) / val_min * 100) if val_min > 0 else 0
        base = (
            f"<b>{mejor}</b> presenta el valor más alto (mediana: <b>{val_max:,.0f}</b>), "
            f"mientras que <b>{peor}</b> registra el más bajo (<b>{val_min:,.0f}</b>). "
            f"La brecha entre extremos es del <b>{diff_pct:.0f}%</b>. "
        )
        return base + CONTEXTOS_CAT.get((var_x, var_y), "")
    else:
        valid = df_data[[var_x, var_y]].dropna()
        if len(valid) < 10:
            return "No hay suficientes datos para calcular la correlación."
        corr = valid[var_x].corr(valid[var_y])
        intensidad = "fuerte" if abs(corr) > 0.6 else "moderada" if abs(corr) > 0.3 else "débil"
        direccion  = "positiva" if corr > 0 else "negativa"
        base = f"La relación muestra una correlación <b>{direccion} y {intensidad}</b> (r = {corr:.2f}). "
        return base + CONTEXTOS_NUM.get((var_x, var_y), CONTEXTOS_NUM.get((var_y, var_x), ""))


# ── Generador de PDF de tasación ──────────────────────────────────────────────
def _generar_pdf_tasacion(
    marca, modelo, año, km, cv, combustible, transmision,
    etiqueta, tipo_venta, precio, p_min, p_max, pct_intervalo,
    diff, signo, mae,
    edades_rng, precios_edad, ant,
    km_rng, precios_km_suave, km_sel,
    tabla_sim,
):
    import io, matplotlib, matplotlib.pyplot as plt
    from fpdf import FPDF
    from datetime import datetime
    matplotlib.use("Agg")

    # ── Gráfico 1: depreciación ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 3.6))
    band_hi = [p * (1 + MAPE_FACTOR) for p in precios_edad]
    band_lo = [p * (1 - MAPE_FACTOR) for p in precios_edad]
    ax.fill_between(edades_rng, band_lo, band_hi, alpha=0.15, color="#0070f3")
    ax.plot(edades_rng, precios_edad, color="#0070f3", linewidth=2.2)
    ax.axvline(x=ant, color="#00c072", linestyle="--", linewidth=1.5,
               label=f"Tu vehiculo ({ant} anos)")
    ax.set_xlabel("Antiguedad (anos)"); ax.set_ylabel("Precio estimado (EUR)")
    ax.set_title("Depreciacion segun la antiguedad")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    buf1 = io.BytesIO(); fig.savefig(buf1, format="png", dpi=130); plt.close(); buf1.seek(0)

    # ── Gráfico 2: impacto km ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot([k / 1000 for k in km_rng], precios_km_suave, color="#f59e0b", linewidth=2.2)
    ax.axvline(x=km_sel / 1000, color="#00c072", linestyle="--", linewidth=1.5,
               label=f"{km_sel:,} km")
    ax.set_xlabel("Kilometraje (miles de km)"); ax.set_ylabel("Precio estimado (EUR)")
    ax.set_title("Impacto del kilometraje en el precio")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    buf2 = io.BytesIO(); fig.savefig(buf2, format="png", dpi=130); plt.close(); buf2.seek(0)

    # ── Construcción del PDF ───────────────────────────────────────────────────
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=14)
    AZUL_PDF = (0, 112, 243); GRIS_PDF = (100, 116, 139); NEGRO = (17, 24, 39)

    def cabecera():
        pdf.set_fill_color(*AZUL_PDF)
        pdf.rect(0, 0, 210, 14, "F")
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(255, 255, 255)
        pdf.set_xy(10, 3)
        pdf.cell(0, 8, "PrecioCoches ML  |  Tasacion de vehiculo", ln=False)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_xy(0, 3)
        pdf.cell(200, 8, datetime.now().strftime("%d/%m/%Y"), align="R")
        pdf.set_text_color(*NEGRO)

    def pie():
        pdf.set_auto_page_break(auto=False)   # evitar que el pie salte a página nueva
        pdf.set_y(-10)
        pdf.set_font("Helvetica", "I", 7.5)
        pdf.set_text_color(*GRIS_PDF)
        pdf.cell(0, 5,
            "Estimacion generada por modelo de ML (GBM). No constituye una tasacion oficial. "
            "Verificar siempre el estado real del vehiculo antes de cualquier decision de compra.",
            align="C")
        pdf.set_text_color(*NEGRO)
        pdf.set_auto_page_break(auto=True, margin=14)

    # ── PAGINA 1 — Resumen ─────────────────────────────────────────────────────
    pdf.add_page()
    cabecera()

    # Titulo seccion
    pdf.set_xy(10, 18)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(*AZUL_PDF)
    pdf.cell(0, 8, "Resumen de la tasacion", ln=True)
    pdf.set_text_color(*NEGRO)

    # Datos del vehiculo — caja gris
    pdf.set_xy(10, 28)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*GRIS_PDF)
    pdf.cell(0, 5, "DATOS DEL VEHICULO", ln=True)
    pdf.set_text_color(*NEGRO)

    pdf.set_fill_color(249, 250, 251)
    pdf.set_draw_color(229, 231, 235)
    pdf.rect(10, 34, 190, 36, "FD")

    datos = [
        ("Marca / Modelo", f"{marca}  {modelo}"),
        ("Ano de fabricacion", str(año)),
        ("Kilometraje", f"{km:,} km"),
        ("Potencia", f"{cv} CV"),
        ("Combustible", combustible),
        ("Transmision", transmision),
        ("Tipo de venta", tipo_venta),
        ("Etiqueta DGT", etiqueta.replace("_", " ")),
    ]
    col_w = 95
    for i, (k, v) in enumerate(datos):
        col = i % 2
        row = i // 2
        x = 14 + col * col_w
        y = 36 + row * 8.5
        pdf.set_font("Helvetica", "B", 8.5); pdf.set_xy(x, y)
        pdf.cell(40, 6, k + ":", ln=False)
        pdf.set_font("Helvetica", "", 8.5)
        pdf.cell(col_w - 42, 6, v, ln=False)

    # Precio estimado — caja azul
    pdf.set_xy(10, 74)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*GRIS_PDF)
    pdf.cell(0, 5, "PRECIO ESTIMADO DE MERCADO", ln=True)
    pdf.set_text_color(*NEGRO)

    pdf.set_fill_color(239, 246, 255)
    pdf.set_draw_color(*AZUL_PDF)
    pdf.rect(10, 80, 190, 28, "FD")

    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*AZUL_PDF)
    pdf.set_xy(10, 82)
    pdf.cell(190, 12, f"{precio:,.0f} EUR", align="C", ln=True)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*GRIS_PDF)
    pdf.set_xy(10, 95)
    pdf.cell(190, 6,
        f"Intervalo de mercado: {p_min:,.0f} EUR  -  {p_max:,.0f} EUR   (+/-{pct_intervalo}%)",
        align="C", ln=True)

    signo_es = "por encima" if diff >= 0 else "por debajo"
    pdf.set_xy(10, 101)
    pdf.cell(190, 6,
        f"{'+'if diff>=0 else ''}{diff:,.0f} EUR {signo_es} de la mediana de {marca}",
        align="C", ln=True)
    pdf.set_text_color(*NEGRO)

    # Fiabilidad del modelo
    pdf.set_xy(10, 113)
    pdf.set_fill_color(240, 253, 244)
    pdf.set_draw_color(134, 239, 172)
    pdf.rect(10, 113, 190, 14, "FD")
    pdf.set_font("Helvetica", "B", 8.5)
    pdf.set_xy(10, 115)
    pdf.cell(190, 5, "Fiabilidad del modelo:", align="C", ln=True)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(*GRIS_PDF)
    pdf.cell(190, 5,
        f"Algoritmo GBM (HistGradientBoosting)  |  R2 = 0.941  |  MAE = +/-{mae:,} EUR  |  Entrenado con 80.528 anuncios",
        align="C", ln=True)
    pdf.set_text_color(*NEGRO)

    # Tabla coches similares
    if tabla_sim is not None and len(tabla_sim) > 0:
        pdf.set_xy(10, 132)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*GRIS_PDF)
        pdf.cell(0, 5, "COCHES SIMILARES EN EL DATASET", ln=True)
        pdf.set_text_color(*NEGRO)
        pdf.ln(1)

        headers = ["Ano", "Km", "CV", "Combustible", "Precio real", "vs. estimado"]
        col_ws  = [16,    30,   16,   38,             30,             30]
        pdf.set_fill_color(*AZUL_PDF); pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 7.5)
        for h, w in zip(headers, col_ws):
            pdf.cell(w, 6, h, border=1, fill=True)
        pdf.ln()

        pdf.set_text_color(*NEGRO); pdf.set_font("Helvetica", "", 7.5)
        fill = False
        for _, row in tabla_sim.head(8).iterrows():
            pdf.set_fill_color(249, 250, 251) if fill else pdf.set_fill_color(255, 255, 255)
            vals = [str(row.get("Ano", row.get("año",""))),
                    str(row.get("Km", row.get("kilometraje_km",""))),
                    str(row.get("CV", row.get("potencia_cv",""))),
                    str(row.get("Combustible", row.get("combustible",""))),
                    str(row.get("Precio real", row.get("precio_eur",""))),
                    str(row.get("vs. estimado",""))]
            for v, w in zip(vals, col_ws):
                pdf.cell(w, 5.5, v[:20], border=1, fill=True)
            pdf.ln(); fill = not fill

        pdf.set_font("Helvetica", "I", 7)
        pdf.set_text_color(*GRIS_PDF)
        pdf.ln(1)
        pdf.cell(0, 4, f"Fuente: dataset de 80.528 anuncios del mercado espanol de segunda mano.", ln=True)
        pdf.set_text_color(*NEGRO)

    pie()

    # ── PAGINA 2 — Graficos ────────────────────────────────────────────────────
    pdf.add_page()
    cabecera()

    pdf.set_xy(10, 18)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(*AZUL_PDF)
    pdf.cell(0, 8, "Analisis grafico", ln=True)
    pdf.set_text_color(*NEGRO)

    pdf.set_xy(10, 28)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*GRIS_PDF)
    pdf.cell(0, 5, "DEPRECIACION SEGUN LA ANTIGUEDAD", ln=True)
    pdf.set_text_color(*NEGRO)
    pdf.image(buf1, x=10, y=34, w=190)

    pdf.set_xy(10, 120)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*GRIS_PDF)
    pdf.cell(0, 5, "IMPACTO DEL KILOMETRAJE EN EL PRECIO", ln=True)
    pdf.set_text_color(*NEGRO)
    pdf.image(buf2, x=10, y=126, w=190)

    pdf.set_xy(10, 212)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*GRIS_PDF)
    pdf.multi_cell(190, 4.5,
        "La linea verde discontinua indica el vehiculo introducido. "
        "La banda azul representa el intervalo de confianza del modelo (+/-" + str(pct_intervalo) + "%). "
        "Las estimaciones son medias estadisticas; dos vehiculos identicos en papel pueden diferir "
        "en precio segun su estado real, historial y condiciones de venta.")
    pdf.set_text_color(*NEGRO)

    pie()
    return bytes(pdf.output())


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 0 — PORTADA
# ══════════════════════════════════════════════════════════════════════════════
if seccion == SECCIONES[0]:
    st.markdown("""
    <div style="margin-bottom:28px;">
        <div style="display:inline-block;background:#EFF6FF;color:#1d4ed8;border-radius:20px;
                    padding:4px 14px;font-size:0.78rem;font-weight:600;margin-bottom:12px;">
            TFG &middot; Universidad Europea de Madrid &middot; 2026
        </div>
        <h1 style="font-size:2.2rem;font-weight:800;color:#111827;margin:0 0 8px 0;">🚗 PrecioCoches ML</h1>
        <p style="font-size:1rem;color:#6B7280;max-width:620px;margin:0;">
            Plataforma de análisis del mercado de vehículos de segunda mano en España,
            impulsada por Gradient Boosting Machine entrenado sobre más de 80.000 anuncios reales.
        </p>
    </div>
    """, unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4, gap="small")
    for col, val, lbl, desc, color in [
        (s1, "80.528",
             "Vehículos en dataset",
             "Anuncios reales del mercado español usados para entrenar el modelo.",
             "#0070f3"),
        (s2, "0.941",
             "R² del modelo GBM",
             "Precisión global: 1.0 sería perfección absoluta. 0.941 indica que el modelo explica el 94% de la variación de precios.",
             "#00d084"),
        (s3, f"±{MAE_MODELO:,} €",
             "Error medio (MAE)",
             "En promedio, la estimación se desvía ±1.732 € del precio real del anuncio.",
             "#ff9500"),
        (s4, "5",
             "Herramientas de análisis",
             "Tasador, buscador por presupuesto, depreciación, explorador de variables y detector de chollos.",
             "#7928ca"),
    ]:
        col.markdown(f"""
        <div class="stat-card" style="border-top: 3px solid {color};">
            <div class="stat-val" style="color:{color};">{val}</div>
            <div class="stat-lbl">{lbl}</div>
            <div style="font-size:0.75rem;color:#6B7280;margin-top:6px;line-height:1.4;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ¿Qué puedes hacer?")
    st.markdown("Elige una herramienta para empezar:")

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown("""
        <div class="home-card">
            <span class="card-icon">🔍</span>
            <h3>Tasador de precio</h3>
            <p>Introduce las características de un vehículo y obtén una estimación
               del precio de mercado con el modelo GBM, más coches similares reales.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Ir al Tasador de precio", key="btn_sec1", use_container_width=True, type="primary"):
            st.session_state["seccion_idx"] = 1
            st.session_state["_nav_radio"] = SECCIONES[1]
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class="home-card">
            <span class="card-icon">📉</span>
            <h3>¿Qué coche pierde menos valor?</h3>
            <p>Compara la retención de valor de hasta 6 marcas distintas
               y analiza cuál conserva mejor su precio con el paso del tiempo.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Analizar depreciación", key="btn_sec3", use_container_width=True, type="primary"):
            st.session_state["seccion_idx"] = 3
            st.session_state["_nav_radio"] = SECCIONES[3]
            st.rerun()

    with col_b:
        st.markdown("""
        <div class="home-card">
            <span class="card-icon">💰</span>
            <h3>¿Qué me puedo permitir?</h3>
            <p>Define tu presupuesto, marca, potencia y kilometraje deseado para
               descubrir qué opciones existen en el mercado de segunda mano.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Explorar por presupuesto", key="btn_sec2", use_container_width=True, type="primary"):
            st.session_state["seccion_idx"] = 2
            st.session_state["_nav_radio"] = SECCIONES[2]
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class="home-card">
            <span class="card-icon">📊</span>
            <h3>Relación entre características</h3>
            <p>Analiza cualquier par de variables del dataset — precio, potencia,
               kilometraje, combustible… — y obtén gráficas e interpretaciones automáticas.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Ver relación entre características", key="btn_sec4", use_container_width=True, type="primary"):
            st.session_state["seccion_idx"] = 4
            st.session_state["_nav_radio"] = SECCIONES[4]
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="home-card" style="border-left:4px solid #f59e0b;">
        <span class="card-icon">🎯</span>
        <h3>Detector de Chollos</h3>
        <p>El modelo analiza los anuncios reales scrapeados de Coches.net y detecta aquellos cuyo precio
           está significativamente por debajo del valor estimado — chollos reales con enlace directo al anuncio.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Ver detector de chollos", key="btn_sec5", use_container_width=True, type="primary"):
        st.session_state["seccion_idx"] = 5
        st.session_state["_nav_radio"] = SECCIONES[5]
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="insight">
        💡 <b>Sobre el modelo:</b> Se ha entrenado un Gradient Boosting Machine (GBM) sobre 80.528 anuncios
        reales del mercado español de segunda mano. Las variables más relevantes son el año de fabricación,
        el kilometraje y la potencia. El modelo alcanza un R² de 0.941 con un error medio absoluto de
        ±{MAE_MODELO:,} €.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — TASADOR
# ══════════════════════════════════════════════════════════════════════════════
elif seccion == SECCIONES[1]:
    st.markdown('<p class="page-title">🔍 Tasador de precio</p><p class="page-subtitle">Introduce las características del vehículo y obtén una estimación basada en 80.000 anuncios reales.</p>', unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("#### Características del vehículo")

        marca_sel = st.selectbox("Marca", sorted(df["marca"].unique()))
        modelos_disp = sorted(df[df["marca"] == marca_sel]["modelo"].unique())
        modelo_sel = st.selectbox("Modelo", modelos_disp)

        c1, c2 = st.columns(2)
        with c1:
            año_sel = st.number_input("Año de fabricación",
                min_value=int(rangos["año"]["min"]), max_value=AÑO_ACTUAL,
                value=2019, step=1)
        with c2:
            km_sel = st.number_input("Kilometraje (km)",
                min_value=0, max_value=600_000, value=75_000, step=5_000)

        c3, c4 = st.columns(2)
        with c3:
            cv_sel = st.number_input("Potencia (CV)",
                min_value=int(rangos["potencia_cv"]["min"]),
                max_value=int(rangos["potencia_cv"]["max"]),
                value=110, step=5)
        with c4:
            combustible_sel = st.selectbox("Combustible", sorted(df["combustible"].unique()))

        transmision_sel = st.selectbox("Transmisión", sorted(df["transmision"].unique()))
        tipo_venta_sel  = st.selectbox("Tipo de venta", ["Usado", "Km 0", "Casi nuevo", "Demo"])
        calcular = st.button("Calcular precio estimado", type="primary", use_container_width=True)

    with col_result:
        if calcular:
            ant = AÑO_ACTUAL - año_sel
            kpa = int(km_sel // max(ant, 1))
            etiqueta_sel = _derivar_etiqueta(combustible_sel, año_sel)

            entrada = pd.DataFrame([{
                "año": año_sel, "potencia_cv": cv_sel, "kilometraje_km": km_sel,
                "antiguedad": ant, "km_por_año": kpa,
                "marca": marca_sel, "modelo": modelo_sel,
                "combustible": combustible_sel, "transmision": transmision_sel,
                "etiqueta_ambiental": etiqueta_sel,
                "tipo_venta": tipo_venta_sel,
            }])

            precio = modelo_ml.predict(entrada)[0]
            p_min  = max(500, precio * (1 - MAPE_FACTOR))
            p_max  = precio * (1 + MAPE_FACTOR)
            pct_intervalo = round(MAPE_FACTOR * 100)

            media_marca = df[df["marca"] == marca_sel]["precio_eur"].median()
            diff = precio - media_marca
            signo = "por encima" if diff > 0 else "por debajo"
            clase_diff = "precio-diff-pos" if diff > 0 else "precio-diff-neg"
            icono_diff = "▲" if diff > 0 else "▼"
            etq_fg, etq_bg = _COLOR_ETIQUETA.get(etiqueta_sel, ("#475569", "#F1F5F9"))

            st.markdown(f"""
            <div class="card">
                <p style="font-size:0.8rem;text-transform:uppercase;letter-spacing:.05em;color:{GRIS};margin:0 0 8px;">
                    Precio estimado de mercado
                </p>
                <div class="precio-principal">{precio:,.0f} €</div>
                <div class="precio-rango">Intervalo de mercado: {p_min:,.0f} € — {p_max:,.0f} € &nbsp;<span style="font-size:0.8rem;color:#9CA3AF;">(±{pct_intervalo}%)</span></div>
                <div class="confidence-badge"><span class="dot"></span> GBM · R² 0.941 · MAE del modelo ±{MAE_MODELO:,} €</div>
                <br><span class="{clase_diff}">{icono_diff} {abs(diff):,.0f} € {signo} de la mediana de {marca_sel}</span>
                <br><span style="display:inline-block;margin-top:10px;background:{etq_bg};color:{etq_fg};
                    border-radius:6px;padding:4px 10px;font-size:0.82rem;font-weight:600;">
                    Etiqueta DGT: {etiqueta_sel.replace('_', ' ')}
                </span>
            </div>
            """, unsafe_allow_html=True)

            # ── Curva depreciación ─────────────────────────────────────────────
            st.markdown("**Depreciación según la antigüedad**")
            edades_rng   = list(range(0, 19))
            precios_edad = []
            for edad in edades_rng:
                e = entrada.copy()
                e["año"]        = AÑO_ACTUAL - edad
                e["antiguedad"] = edad
                e["km_por_año"] = int(km_sel // max(edad, 1))
                precios_edad.append(modelo_ml.predict(e)[0])

            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=edades_rng + edades_rng[::-1],
                y=[p * (1 + MAPE_FACTOR) for p in precios_edad] + [p * (1 - MAPE_FACTOR) for p in precios_edad][::-1],
                fill="toself", fillcolor="rgba(0,112,243,0.1)",
                line=dict(color="rgba(0,0,0,0)"), name="Intervalo estimado",
            ))
            fig1.add_trace(go.Scatter(
                x=edades_rng, y=precios_edad,
                mode="lines", line=dict(color=AZUL, width=2.5),
                name="Precio estimado",
                hovertemplate="<b>%{x} años</b><br>%{y:,.0f} €<extra></extra>",
            ))
            fig1.add_vline(x=ant, line_dash="dash", line_color=VERDE, line_width=1.5,
                           annotation_text=f"Tu vehículo ({ant} años)",
                           annotation_font_color=VERDE, annotation_font_size=11)
            estilo_fig(fig1)
            fmt_eur_axis(fig1, "y")
            fig1.update_xaxes(title_text="Antigüedad (años)")
            fig1.update_yaxes(title_text="Precio estimado (€)")
            st.plotly_chart(fig1, use_container_width=True)

            # ── Curva impacto kilómetros ───────────────────────────────────────
            st.markdown("**Impacto del kilometraje en el precio**")
            km_rng    = list(range(0, 300_001, 3_000))
            precios_km = []
            for k in km_rng:
                e = entrada.copy()
                e["kilometraje_km"] = k
                e["km_por_año"]     = int(k // max(ant, 1))
                precios_km.append(modelo_ml.predict(e)[0])

            precios_km_suave = (
                pd.Series(precios_km)
                .rolling(window=7, center=True, min_periods=1).mean().tolist()
            )
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=[k / 1000 for k in km_rng], y=precios_km_suave,
                mode="lines", line=dict(color=NARANJA, width=2.5),
                name="Precio estimado",
                hovertemplate="<b>%{x:.0f}k km</b><br>%{y:,.0f} €<extra></extra>",
            ))
            fig2.add_vline(x=km_sel / 1000, line_dash="dash", line_color=VERDE, line_width=1.5,
                           annotation_text=f"{km_sel:,} km",
                           annotation_font_color=VERDE, annotation_font_size=11)
            estilo_fig(fig2)
            fmt_eur_axis(fig2, "y")
            fig2.update_xaxes(title_text="Kilometraje (miles de km)")
            fig2.update_yaxes(title_text="Precio estimado (€)")
            st.plotly_chart(fig2, use_container_width=True)

            # ── Coches similares en el dataset ────────────────────────────────
            st.markdown("**Coches similares en el dataset**")
            mask_sim = (
                (df["marca"] == marca_sel) &
                (df["año"].between(año_sel - 2, año_sel + 2)) &
                (df["kilometraje_km"].between(max(0, km_sel - 50_000), km_sel + 50_000)) &
                (df["tipo_venta"] == tipo_venta_sel)
            )
            similares = df[mask_sim].copy()
            aviso_tipo = ""
            if similares.empty:
                # Si no hay del mismo tipo, ampliar sin filtro de tipo_venta
                mask_sim = (
                    (df["marca"] == marca_sel) &
                    (df["año"].between(año_sel - 2, año_sel + 2)) &
                    (df["kilometraje_km"].between(max(0, km_sel - 50_000), km_sel + 50_000))
                )
                similares = df[mask_sim].copy()
                aviso_tipo = f" (sin coincidencias de tipo '{tipo_venta_sel}', mostrando todos los tipos)"
            if similares.empty:
                st.caption("No se encontraron anuncios con características similares en el dataset.")
            else:
                similares = similares.sort_values(
                    by="precio_eur",
                    key=lambda s: (s - precio).abs()
                ).head(8)
                similares["vs. estimado"] = (similares["precio_eur"] - precio).apply(
                    lambda x: f"+{x:,.0f} €" if x >= 0 else f"{x:,.0f} €"
                )
                cols_tabla = ["año", "kilometraje_km", "potencia_cv",
                              "combustible", "transmision", "tipo_venta", "precio_eur", "vs. estimado"]
                tabla_sim = similares[cols_tabla].copy()
                tabla_sim["precio_eur"]     = tabla_sim["precio_eur"].apply(lambda x: f"{x:,.0f} €")
                tabla_sim["kilometraje_km"] = tabla_sim["kilometraje_km"].apply(lambda x: f"{x:,.0f} km")
                tabla_sim["potencia_cv"]    = tabla_sim["potencia_cv"].apply(lambda x: f"{int(x)} CV")
                tabla_sim.columns = ["Año", "Km", "CV", "Combustible", "Transmisión", "Tipo venta", "Precio real", "vs. estimado"]
                st.dataframe(tabla_sim.reset_index(drop=True), use_container_width=True, hide_index=True)
                st.caption(f"{len(mask_sim[mask_sim])} anuncios de {marca_sel} con ±2 años y ±50.000 km encontrados{aviso_tipo}.")

            # ── Exportar PDF ───────────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            try:
                pdf_bytes = _generar_pdf_tasacion(
                    marca=marca_sel, modelo=modelo_sel, año=año_sel,
                    km=km_sel, cv=cv_sel, combustible=combustible_sel,
                    transmision=transmision_sel, etiqueta=etiqueta_sel,
                    tipo_venta=tipo_venta_sel,
                    precio=precio, p_min=p_min, p_max=p_max,
                    pct_intervalo=pct_intervalo,
                    diff=diff, signo=signo, mae=MAE_MODELO,
                    edades_rng=edades_rng, precios_edad=precios_edad, ant=ant,
                    km_rng=km_rng, precios_km_suave=precios_km_suave, km_sel=km_sel,
                    tabla_sim=tabla_sim if not similares.empty else None,
                )
                nombre_pdf = f"tasacion_{marca_sel}_{modelo_sel[:15].replace(' ','_')}_{año_sel}.pdf"
                st.download_button(
                    label="Descargar tasacion en PDF",
                    data=pdf_bytes,
                    file_name=nombre_pdf,
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.caption(f"No se pudo generar el PDF: {e}")

        else:
            st.markdown("""
            <div class="card" style="text-align:center;padding:60px 24px;">
                <div style="font-size:3rem;margin-bottom:16px;">🚗</div>
                <p style="font-size:1rem;color:#6B7280;">
                    Rellena el formulario y pulsa <strong>Calcular precio estimado</strong>
                </p>
                <p style="font-size:0.85rem;color:#9CA3AF;">
                    El modelo analizará las características del vehículo<br>
                    y estimará su precio de mercado actual.
                </p>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — ¿QUÉ ME PUEDO PERMITIR?
# ══════════════════════════════════════════════════════════════════════════════
elif seccion == SECCIONES[2]:
    st.markdown('<p class="page-title">💰 ¿Qué me puedo permitir?</p><p class="page-subtitle">Define tu presupuesto y preferencias para descubrir qué opciones encajan en el mercado de segunda mano.</p>', unsafe_allow_html=True)

    col_filtros, col_res = st.columns([1, 2], gap="large")

    with col_filtros:
        st.markdown("#### Tu presupuesto")
        p_min_u, p_max_u = st.slider("Rango de precio (€)",
            min_value=int(df["precio_eur"].min()),
            max_value=int(df["precio_eur"].max()),
            value=(5_000, 20_000), step=500)

        st.markdown("#### Kilometraje")
        ck1, ck2 = st.columns(2)
        with ck1:
            min_km = st.number_input("Km mínimos", min_value=0, max_value=600_000, value=0, step=5_000)
        with ck2:
            max_km = st.number_input("Km máximos", min_value=0, max_value=600_000, value=200_000, step=5_000)
        if min_km > max_km:
            st.warning("El mínimo de km no puede superar el máximo.")

        st.markdown("#### Potencia (CV)")
        cc1, cc2 = st.columns(2)
        with cc1:
            min_cv = st.number_input("CV mínimos", min_value=0, max_value=700, value=0, step=10)
        with cc2:
            max_cv = st.number_input("CV máximos", min_value=0, max_value=700, value=500, step=10)
        if min_cv > max_cv:
            st.warning("El mínimo de CV no puede superar el máximo.")

        st.markdown("#### Año de fabricación")
        ca1, ca2 = st.columns(2)
        with ca1:
            min_año = st.number_input("Año mínimo",
                min_value=int(rangos["año"]["min"]), max_value=AÑO_ACTUAL,
                value=2015, step=1)
        with ca2:
            max_año = st.number_input("Año máximo",
                min_value=int(rangos["año"]["min"]), max_value=AÑO_ACTUAL,
                value=AÑO_ACTUAL, step=1)
        if min_año > max_año:
            st.warning("El año mínimo no puede superar el máximo.")

        st.markdown("#### Otros filtros")
        marca_fil = st.multiselect("Marca", sorted(df["marca"].unique()))
        comb_sel  = st.multiselect("Combustible", sorted(df["combustible"].unique()))
        solo_electrico = (
            len(comb_sel) > 0
            and all("lectric" in c.lower() or "léctric" in c.lower() for c in comb_sel)
        )
        if solo_electrico:
            st.caption("Los vehículos eléctricos tienen transmisión automática.")
        trans_sel = st.multiselect("Transmisión", sorted(df["transmision"].unique()), disabled=solo_electrico)

        buscar = st.button("Buscar opciones", type="primary", use_container_width=True)

    with col_res:
        if buscar:
            km_min_f = min(min_km, max_km)
            km_max_f = max(min_km, max_km)
            cv_min_f = min(min_cv, max_cv)
            cv_max_f = max(min_cv, max_cv)
            yr_min_f = min(min_año, max_año)
            yr_max_f = max(min_año, max_año)

            mask = (
                (df["precio_eur"] >= p_min_u) & (df["precio_eur"] <= p_max_u) &
                (df["potencia_cv"] >= cv_min_f) & (df["potencia_cv"] <= cv_max_f) &
                (df["kilometraje_km"] >= km_min_f) & (df["kilometraje_km"] <= km_max_f) &
                (df["año"] >= yr_min_f) & (df["año"] <= yr_max_f)
            )
            if marca_fil: mask &= df["marca"].isin(marca_fil)
            if comb_sel:  mask &= df["combustible"].isin(comb_sel)
            if trans_sel: mask &= df["transmision"].isin(trans_sel)
            filtrado = df[mask]

            if filtrado.empty:
                st.warning("No se encontraron vehículos. Prueba a ampliar el rango o reducir filtros.")
            else:
                st.success(f"**{len(filtrado):,} vehículos** encajan con tu búsqueda.")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Precio mediano",  f"{filtrado['precio_eur'].median():,.0f} €")
                c2.metric("Km medianos",      f"{filtrado['kilometraje_km'].median():,.0f}")
                c3.metric("Año mediano",      f"{filtrado['año'].median():.0f}")
                c4.metric("CV medianos",      f"{filtrado['potencia_cv'].median():.0f}")

                st.divider()

                # Marcas más frecuentes
                st.markdown("#### Marcas más frecuentes en tu presupuesto")
                top_marcas = filtrado["marca"].value_counts().head(12).reset_index()
                top_marcas.columns = ["Marca", "Anuncios"]
                fig = px.bar(top_marcas.sort_values("Anuncios"), x="Anuncios", y="Marca",
                             orientation="h", color_discrete_sequence=[AZUL_L])
                fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x:,} vehículos<extra></extra>")
                estilo_fig(fig, height=380)
                fig.update_layout(title=f"Oferta entre {p_min_u:,} € y {p_max_u:,} €")
                st.plotly_chart(fig, use_container_width=True)

                # Configuraciones típicas
                st.markdown("#### Configuraciones más habituales")
                config = (
                    filtrado
                    .groupby(["marca", "combustible", "transmision"])
                    .agg(anuncios=("precio_eur","count"),
                         precio_mediano=("precio_eur","median"),
                         año_tipico=("año","median"),
                         km_tipicos=("kilometraje_km","median"),
                         cv_tipicos=("potencia_cv","median"))
                    .sort_values("anuncios", ascending=False)
                    .head(10).reset_index()
                )
                config["precio_mediano"] = config["precio_mediano"].apply(lambda x: f"{x:,.0f} €")
                config["año_tipico"]     = config["año_tipico"].apply(lambda x: str(int(x)))
                config["km_tipicos"]     = config["km_tipicos"].apply(lambda x: f"{x:,.0f} km")
                config["cv_tipicos"]     = config["cv_tipicos"].apply(lambda x: f"{int(x)} CV")
                config.columns = ["Marca","Combustible","Transmisión","Anuncios",
                                   "Precio mediano","Año típico","Km típicos","CV típicos"]
                st.dataframe(config, use_container_width=True, hide_index=True)

                # ── Link a Coches.net ──────────────────────────────────────────
                st.divider()
                st.markdown("#### Buscar en Coches.net")
                st.caption(
                    f"Abre la búsqueda con tu rango de precio ({p_min_u:,} € – {p_max_u:,} €) "
                    f"y kilometraje ({km_min_f:,} – {km_max_f:,} km) ya aplicados. "
                    f"Una vez en la web, escribe la marca en el buscador y te saldrán las opciones disponibles."
                )
                url_coches = (
                    f"https://www.coches.net/segunda-mano/"
                    f"?MinPrice={p_min_u}&MaxPrice={p_max_u}"
                    f"&MinKms={km_min_f}&MaxKms={km_max_f}"
                )
                st.link_button(
                    "🔗 Ver coches en Coches.net",
                    url_coches,
                    use_container_width=True,
                )

        else:
            st.markdown("""
            <div class="card" style="text-align:center;padding:60px 24px;">
                <div style="font-size:3rem;margin-bottom:16px;">💰</div>
                <p style="font-size:1rem;color:#6B7280;">
                    Define tu presupuesto y pulsa <strong>Buscar opciones</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — DEPRECIACIÓN
# ══════════════════════════════════════════════════════════════════════════════
elif seccion == SECCIONES[3]:
    st.markdown('<p class="page-title">📉 ¿Qué coche pierde menos valor?</p><p class="page-subtitle">Analiza cómo pierden valor las diferentes marcas a lo largo del tiempo.</p>', unsafe_allow_html=True)

    marcas_sel = st.multiselect(
        "Selecciona marcas para comparar (máximo 6)",
        sorted(df["marca"].unique()),
        default=["Volkswagen", "Seat", "Renault", "Bmw"],
    )
    if len(marcas_sel) > 6:
        st.warning("Selecciona como máximo 6 marcas.")
        marcas_sel = marcas_sel[:6]

    with st.expander("Filtros opcionales: potencia y kilometraje"):
        fd1, fd2 = st.columns(2)
        with fd1:
            st.markdown("**Potencia (CV)**")
            fc1, fc2 = st.columns(2)
            with fc1:
                dep_min_cv = st.number_input("CV mínimos", min_value=0, max_value=700, value=0, step=10, key="dep_min_cv")
            with fc2:
                dep_max_cv = st.number_input("CV máximos", min_value=0, max_value=700, value=700, step=10, key="dep_max_cv")
        with fd2:
            st.markdown("**Kilometraje (km)**")
            fk1, fk2 = st.columns(2)
            with fk1:
                dep_min_km = st.number_input("Km mínimos", min_value=0, max_value=600_000, value=0, step=5_000, key="dep_min_km")
            with fk2:
                dep_max_km = st.number_input("Km máximos", min_value=0, max_value=600_000, value=600_000, step=5_000, key="dep_max_km")

    if not marcas_sel:
        st.info("Selecciona al menos una marca para ver la comparativa.")
    else:
        df_marc = df[
            df["marca"].isin(marcas_sel) &
            (df["potencia_cv"] >= min(dep_min_cv, dep_max_cv)) &
            (df["potencia_cv"] <= max(dep_min_cv, dep_max_cv)) &
            (df["kilometraje_km"] >= min(dep_min_km, dep_max_km)) &
            (df["kilometraje_km"] <= max(dep_min_km, dep_max_km))
        ]

        if df_marc.empty:
            st.warning("Los filtros aplicados dejan el dataset vacío. Amplía los rangos de CV o km.")
        else:
            curvas    = df_marc.groupby(["marca","año"])["precio_eur"].median().reset_index()
            color_map = {m: COLORES_LINEAS[i % len(COLORES_LINEAS)] for i, m in enumerate(marcas_sel)}

            col_graf, col_tabla = st.columns([2, 1], gap="large")

            with col_graf:
                # ── Curva evolución del precio ──────────────────────────────────
                fig = go.Figure()
                años_min = curvas["año"].min()
                años_max = curvas["año"].max()
                for i, marca in enumerate(marcas_sel):
                    datos = curvas[curvas["marca"] == marca].sort_values("año")
                    if len(datos) >= 3:
                        fig.add_trace(go.Scatter(
                            x=datos["año"], y=datos["precio_eur"],
                            mode="lines+markers",
                            name=marca,
                            line=dict(color=color_map[marca], width=2.5),
                            marker=dict(size=4),
                            hovertemplate=f"<b>{marca}</b><br>Año: %{{x}}<br>%{{y:,.0f}} €<extra></extra>",
                        ))
                estilo_fig(fig, height=380)
                fmt_eur_axis(fig, "y")
                fig.update_layout(
                    title="Evolución del precio por antigüedad",
                    annotations=[dict(
                        text=f"{años_min} – {años_max}", x=1, y=1.06,
                        xref="paper", yref="paper", showarrow=False,
                        font=dict(size=10, color="#9CA3AF"),
                    )],
                )
                fig.update_xaxes(title_text="Año de fabricación")
                fig.update_yaxes(title_text="Precio mediano (€)")
                st.plotly_chart(fig, use_container_width=True)

                # ── Uso promedio por marca (km/año) ────────────────────────────
                km_data = (df_marc[df_marc["marca"].isin(marcas_sel)]
                           .groupby("marca")["km_por_año"].median().reset_index())
                km_data = km_data.set_index("marca").reindex(marcas_sel).dropna().reset_index()
                km_data = km_data.sort_values("km_por_año")
                media_km = float(km_data["km_por_año"].mean())

                fig_km = go.Figure()
                for _, row in km_data.iterrows():
                    bar_color = NARANJA if row["km_por_año"] > media_km else VERDE
                    fig_km.add_trace(go.Bar(
                        x=[row["km_por_año"]], y=[row["marca"]],
                        orientation="h",
                        marker_color=bar_color,
                        showlegend=False,
                        text=[f" {row['km_por_año']:,.0f}"],
                        textposition="outside",
                        textfont=dict(size=11, color="#374151"),
                        hovertemplate=f"<b>{row['marca']}</b><br>{row['km_por_año']:,.0f} km/año<extra></extra>",
                    ))
                fig_km.add_vline(
                    x=media_km, line_dash="dot", line_color="#94A3B8", line_width=1.5,
                    annotation_text=f"Media: {media_km:,.0f}",
                    annotation_position="top right",
                    annotation_font_size=10, annotation_font_color="#6B7280",
                )
                estilo_fig(fig_km, height=max(200, len(km_data) * 54))
                fig_km.update_layout(
                    title="Uso promedio por marca (km/año)",
                    bargap=0.35,
                )
                fig_km.update_xaxes(
                    tickformat=",.0f",
                    range=[0, km_data["km_por_año"].max() * 1.4],
                )
                st.plotly_chart(fig_km, use_container_width=True)

            with col_tabla:
                # ── Precio mediano actual con puntos de color ──────────────────
                st.markdown('<p style="font-size:0.9rem;font-weight:600;color:#111827;margin:0 0 10px 0;">Precio mediano actual</p>', unsafe_allow_html=True)
                precios_act = (df_marc.groupby("marca")["precio_eur"]
                               .median().sort_values(ascending=False).reset_index())
                html_precios = '<div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:10px;padding:4px 0;">'
                for _, row in precios_act.iterrows():
                    color = color_map.get(row["marca"], "#6B7280")
                    html_precios += f"""
                    <div style="display:flex;justify-content:space-between;align-items:center;
                                padding:10px 16px;border-bottom:1px solid #f3f4f6;">
                        <div style="display:flex;align-items:center;gap:10px;">
                            <span style="width:9px;height:9px;border-radius:50%;background:{color};
                                         display:inline-block;flex-shrink:0;"></span>
                            <span style="color:#374151;font-size:0.88rem;">{row['marca']}</span>
                        </div>
                        <span style="color:{color};font-weight:700;font-size:0.92rem;
                                     font-variant-numeric:tabular-nums;">{row['precio_eur']:,.0f} €</span>
                    </div>"""
                html_precios += "</div>"
                st.markdown(html_precios, unsafe_allow_html=True)

                # ── Pérdida de valor acumulada ─────────────────────────────────
                st.markdown('<p style="font-size:0.9rem;font-weight:600;color:#111827;margin:16px 0 10px 0;">Pérdida de valor acumulada</p>', unsafe_allow_html=True)
                html_dep = '<div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:10px;overflow:hidden;">'
                html_dep += """<div style="display:grid;grid-template-columns:1fr 44px 44px 54px;
                                           padding:8px 14px;border-bottom:1px solid #f3f4f6;">
                    <span style="font-size:0.67rem;text-transform:uppercase;letter-spacing:.07em;color:#9CA3AF;">MARCA</span>
                    <span style="font-size:0.67rem;text-transform:uppercase;letter-spacing:.07em;color:#9CA3AF;text-align:right;">3A</span>
                    <span style="font-size:0.67rem;text-transform:uppercase;letter-spacing:.07em;color:#9CA3AF;text-align:right;">5A</span>
                    <span style="font-size:0.67rem;text-transform:uppercase;letter-spacing:.07em;color:#9CA3AF;text-align:right;">10A</span>
                </div>"""

                for marca in marcas_sel:
                    datos = curvas[curvas["marca"] == marca].sort_values("año")
                    if len(datos) < 4:
                        continue
                    año_max  = datos["año"].max()
                    p_ref_v  = datos[datos["año"] == año_max]["precio_eur"].values
                    if len(p_ref_v) == 0:
                        continue
                    p_ref = p_ref_v[0]

                    def _dep(h, d=datos, r=p_ref):
                        v = d[d["año"] == d["año"].max() - h]["precio_eur"].values
                        return round((r - v[0]) / r * 100, 1) if len(v) else None

                    d3, d5, d10 = _dep(3), _dep(5), _dep(10)
                    color = color_map.get(marca, "#6B7280")

                    def _fmt(val, alpha):
                        if val is None: return '<span style="color:#D1D5DB;">—</span>'
                        return f'<span style="color:rgba(220,38,38,{alpha});font-weight:600;">-{val:.0f}%</span>'

                    html_dep += f"""<div style="display:grid;grid-template-columns:1fr 44px 44px 54px;
                                               padding:9px 14px;border-bottom:1px solid #f3f4f6;align-items:center;">
                        <div style="display:flex;align-items:center;gap:8px;">
                            <span style="width:8px;height:8px;border-radius:50%;background:{color};
                                         display:inline-block;"></span>
                            <span style="color:#374151;font-size:0.84rem;">{marca}</span>
                        </div>
                        <div style="text-align:right;">{_fmt(d3, 0.55)}</div>
                        <div style="text-align:right;">{_fmt(d5, 0.75)}</div>
                        <div style="text-align:right;">{_fmt(d10, 1.0)}</div>
                    </div>"""

                html_dep += "</div>"
                st.markdown(html_dep, unsafe_allow_html=True)

                st.markdown("""
                <div class="insight" style="font-size:0.82rem;margin-top:14px;">
                    ⓘ Las marcas premium mantienen mejor el valor absoluto pero pueden mostrar mayor depreciación porcentual. Considera ambos factores al comparar.
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — EXPLORADOR DE MERCADO
# ══════════════════════════════════════════════════════════════════════════════
elif seccion == SECCIONES[4]:
    st.markdown('<p class="page-title">📊 Explorador</p><p class="page-subtitle">Analiza relaciones entre variables del mercado de segunda mano.</p>', unsafe_allow_html=True)

    col_ctrl, col_viz = st.columns([1, 2], gap="large")

    with col_ctrl:
        opciones_x_cat   = {ETIQUETAS[v]: v for v in VARS_CATEGORICAS}
        opciones_x_num   = {ETIQUETAS[v]: v for v in VARS_NUMERICAS}
        opciones_x_todas = {**opciones_x_cat, **opciones_x_num}
        opciones_y       = {ETIQUETAS[v]: v for v in VARS_NUMERICAS}

        st.markdown('<p class="panel-label">Variable X (eje horizontal)</p>', unsafe_allow_html=True)
        etiqueta_x = st.selectbox("varx", list(opciones_x_todas.keys()),
            index=list(opciones_x_todas.keys()).index("Combustible"),
            label_visibility="collapsed")

        st.markdown('<p class="panel-label">Variable Y (eje vertical)</p>', unsafe_allow_html=True)
        etiqueta_y = st.selectbox("vary", list(opciones_y.keys()), index=0,
            label_visibility="collapsed")

        var_x  = opciones_x_todas[etiqueta_x]
        var_y  = opciones_y[etiqueta_y]
        tipo_x = "categorica" if var_x in VARS_CATEGORICAS else "numerica"

        df_viz = df

        tiene_ubicacion = "ubicacion" in df.columns
        if tiene_ubicacion:
            st.markdown('<p class="panel-label">Filtrar por provincia</p>', unsafe_allow_html=True)
            provs_activo = st.multiselect("prov", sorted(df["ubicacion"].unique()),
                                          label_visibility="collapsed")
            if provs_activo:
                df_viz = df_viz[df_viz["ubicacion"].isin(provs_activo)]

        st.markdown(f'<p class="panel-label" style="margin-top:20px;">Muestra</p><p style="font-size:1.4rem;font-weight:700;color:#111827;margin:0;">{len(df_viz):,}</p><p style="font-size:0.75rem;color:#6B7280;margin:0;">vehículos analizados</p>', unsafe_allow_html=True)

    with col_viz:
        if var_x == var_y:
            st.warning("Selecciona variables diferentes para X e Y.")
        elif len(df_viz) < 10:
            st.warning("El filtro es demasiado restrictivo. Amplía los criterios.")
        else:
            tipo_graf = "Distribución (Barras)" if tipo_x == "categorica" else "Dispersión"
            mediana_label = "MEDIANA" if tipo_x == "categorica" else "CORRELACIÓN"

            if tipo_x == "categorica":
                orden = (df_viz.groupby(var_x)[var_y].median()
                         .sort_values(ascending=False).index[:15].tolist())
                df_plot = df_viz[df_viz[var_x].isin(orden)].copy()
                fig = px.bar(
                    df_plot.groupby(var_x)[var_y].median().reindex(orden).reset_index(),
                    x=var_x, y=var_y,
                    color_discrete_sequence=[AZUL],
                )
                fig.update_traces(
                    marker_color=AZUL,
                    hovertemplate="<b>%{x}</b><br>Mediana: %{y:,.0f}<extra></extra>",
                )
                fig.update_layout(
                    title=f"{ETIQUETAS[var_y]} por {ETIQUETAS[var_x]}",
                    annotations=[dict(text=mediana_label, x=1, y=1.06, xref="paper",
                                      yref="paper", showarrow=False,
                                      font=dict(size=10, color="#9CA3AF"))],
                    xaxis_title="", yaxis_title=ETIQUETAS[var_y],
                )
                if var_y in ("precio_eur", "km_por_año", "kilometraje_km"):
                    fig.update_yaxes(tickformat=",.0f")
            else:
                sample = df_viz[[var_x, var_y]].dropna().sample(
                    min(6000, len(df_viz)), random_state=42)
                fig = px.scatter(sample, x=var_x, y=var_y,
                                 opacity=0.15, color_discrete_sequence=[AZUL])
                valid = df_viz[[var_x, var_y]].dropna()
                if len(valid) >= 2:
                    z      = np.polyfit(valid[var_x], valid[var_y], 1)
                    x_line = np.linspace(valid[var_x].min(), valid[var_x].max(), 200)
                    fig.add_trace(go.Scatter(
                        x=x_line, y=np.poly1d(z)(x_line),
                        mode="lines",
                        line=dict(color=VERDE, width=2, dash="dash"),
                        name="Tendencia lineal",
                    ))
                fig.update_layout(
                    title=f"{ETIQUETAS[var_x]}  vs  {ETIQUETAS[var_y]}",
                    xaxis_title=ETIQUETAS[var_x], yaxis_title=ETIQUETAS[var_y],
                )

            estilo_fig(fig, height=420)
            fig.update_yaxes(tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)

            explicacion = generar_explicacion(df_viz, var_x, var_y, tipo_x)
            st.markdown(
                f'<div class="insight">ⓘ {explicacion}</div>',
                unsafe_allow_html=True)

            # ── Estadísticos siempre visibles ──────────────────────────────────
            st.markdown('<p style="font-size:0.9rem;font-weight:600;color:#111827;margin:20px 0 10px 0;">Estadísticos descriptivos</p>', unsafe_allow_html=True)
            if tipo_x == "categorica":
                tabla = (df_viz.groupby(var_x)[var_y]
                         .agg(["count","median","mean","std"]).round(1)
                         .sort_values("median", ascending=False)
                         .rename(columns={"count":"N","median":"Mediana",
                                          "mean":"Media","std":"Desv. típica"}))
                st.dataframe(tabla, use_container_width=True)
            else:
                corr = df_viz[var_x].corr(df_viz[var_y])
                c1, c2, c3 = st.columns(3)
                c1.metric("Correlación (r)", f"{corr:.3f}")
                c2.metric(f"Media {ETIQUETAS[var_x]}", f"{df_viz[var_x].mean():,.1f}")
                c3.metric(f"Media {ETIQUETAS[var_y]}", f"{df_viz[var_y].mean():,.1f}")


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — DETECTOR DE CHOLLOS
# ══════════════════════════════════════════════════════════════════════════════
elif seccion == SECCIONES[5]:
    st.markdown("""
    <h2 style="margin-bottom:4px;">🎯 Detector de Chollos</h2>
    <p style="color:#6B7280;margin-bottom:12px;">
        Anuncios reales de Coches.net donde el precio pedido está por debajo del valor estimado por el modelo.
    </p>
    <div style="background:#FFF7ED;border-left:3px solid #f59e0b;padding:10px 14px;border-radius:4px;margin-bottom:20px;font-size:0.85rem;color:#78350F;line-height:1.6;">
        <b>¿Qué descuento es realista?</b> El modelo tiene un error medio de ±11 %, por lo que descuentos
        inferiores al 15 % pueden ser simplemente ruido estadístico. Un descuento del <b>15-25 %</b> empieza
        a ser significativo (2.500-5.000 € en un coche de 20.000 €) y difícil de justificar por desgaste normal.
        Por eso solo se muestran anuncios con <b>descuento entre el 15 % y el 25 %</b>: por debajo es ruido,
        por encima suele indicar que el modelo no conoce bien ese vehículo concreto.
    </div>
    """, unsafe_allow_html=True)

    RUTA_SCRAPEADOS = os.path.join(RUTA_BASE, "coches_scrapeados.csv")

    if not os.path.exists(RUTA_SCRAPEADOS):
        st.info(
            "No se encontró **coches_scrapeados.csv**. "
            "Ejecuta `python scraper_coches.py` para generar los datos."
        )
        st.stop()

    @st.cache_data
    def cargar_scrapeados():
        d = pd.read_csv(RUTA_SCRAPEADOS, encoding="utf-8-sig")
        return d

    @st.cache_data
    def _modelos_por_marca():
        """Devuelve dict marca -> lista de modelos conocidos en el training."""
        return (
            df.groupby("marca")["modelo"]
            .apply(lambda s: s.dropna().unique().tolist())
            .to_dict()
        )

    def _modelo_mas_cercano(marca, modelo_scr, conocidos_por_marca):
        from difflib import get_close_matches
        candidatos = conocidos_por_marca.get(marca, [])
        if not candidatos:
            return modelo_scr
        modelo_l  = str(modelo_scr).lower().strip()
        marca_l   = marca.lower()

        # Quitar prefijo de marca de los modelos conocidos ("BMW X1" -> "x1")
        def _quitar_prefijo(s):
            s2 = s.lower().strip()
            return s2[len(marca_l):].strip() if s2.startswith(marca_l) else s2

        pares = [(c, _quitar_prefijo(c)) for c in candidatos]  # (original, sin_prefijo)

        # 1. Exacto
        for c, cn in pares:
            if cn == modelo_l or c.lower() == modelo_l:
                return c
        # 2. Nombre corto conocido contenido en el modelo scrapeado ("x1" en "ix1 xdrive30")
        for c, cn in sorted(pares, key=lambda x: len(x[1]), reverse=True):
            if cn and cn in modelo_l:
                return c
        # 3. Primer token del scrapeado contiene o está contenido en el nombre corto
        primer_token = modelo_l.split()[0] if modelo_l.split() else modelo_l
        for c, cn in sorted(pares, key=lambda x: len(x[1]), reverse=True):
            if cn and (cn in primer_token or primer_token in cn):
                return c
        # 4. Fuzzy del primer token contra nombres cortos
        nombres_cortos = [cn for _, cn in pares if cn]
        matches = get_close_matches(primer_token, nombres_cortos, n=1, cutoff=0.6)
        if matches:
            return pares[nombres_cortos.index(matches[0])][0]
        # 5. Fuzzy del modelo completo contra nombres cortos
        matches = get_close_matches(modelo_l, nombres_cortos, n=1, cutoff=0.45)
        if matches:
            return pares[nombres_cortos.index(matches[0])][0]
        return None  # sin coincidencia: descartar este anuncio

    @st.cache_data
    def predecir_chollos(hash_key):
        d = cargar_scrapeados().copy()
        conocidos = _modelos_por_marca()
        FEATURES_M = ["año", "potencia_cv", "kilometraje_km", "antiguedad", "km_por_año",
                      "marca", "modelo", "combustible", "transmision",
                      "etiqueta_ambiental", "tipo_venta"]
        # Mapear cada modelo al más cercano conocido; None = sin coincidencia → descartar
        d["modelo_pred"] = d.apply(
            lambda r: _modelo_mas_cercano(r["marca"], r["modelo"], conocidos), axis=1
        )
        d = d[d["modelo_pred"].notna()].copy()
        X = d[FEATURES_M].copy()
        X["modelo"] = d["modelo_pred"]
        preds = modelo_ml.predict(X)
        d["precio_modelo"] = preds.round(0)
        d["ahorro_eur"]    = (d["precio_modelo"] - d["precio_eur"]).round(0)
        d["descuento_pct"] = (d["ahorro_eur"] / d["precio_modelo"].replace(0, 1) * 100).round(1)
        return d.drop(columns=["modelo_pred"])

    import hashlib
    mtime = str(os.path.getmtime(RUTA_SCRAPEADOS))
    df_pred = predecir_chollos(mtime)

    col_filt, col_main = st.columns([1, 3], gap="large")

    with col_filt:
        st.markdown("**Filtros**")
        umbral    = st.slider("Descuento mínimo (%)", 15, 25, 20, 1)
        precio_max_c = st.number_input("Precio máximo (€)", value=50000, step=1000, min_value=1000)
        marcas_c  = st.multiselect("Marca", sorted(df_pred["marca"].dropna().unique()), default=[])
        comb_c    = st.multiselect("Combustible", sorted(df_pred["combustible"].dropna().unique()), default=[])
        fecha_scr = df_pred["fecha_scraping"].iloc[0] if "fecha_scraping" in df_pred.columns else "—"
        st.caption(f"Datos scrapeados: {fecha_scr} · {len(df_pred):,} anuncios")

    with col_main:
        mask = (
            (df_pred["descuento_pct"] >= umbral) &
            (df_pred["descuento_pct"] <= 25) &
            (df_pred["precio_eur"]    <= precio_max_c) &
            (df_pred["precio_eur"]    > 500)
        )
        if marcas_c:
            mask &= df_pred["marca"].isin(marcas_c)
        if comb_c:
            mask &= df_pred["combustible"].isin(comb_c)

        df_c = df_pred[mask].sort_values("descuento_pct", ascending=False).reset_index(drop=True)

        k1, k2, k3 = st.columns(3)
        k1.metric("Chollos encontrados", f"{len(df_c):,}")
        k2.metric("Descuento medio",
                  f"{df_c['descuento_pct'].mean():.1f} %" if len(df_c) else "—")
        k3.metric("Ahorro medio",
                  f"{df_c['ahorro_eur'].mean():,.0f} €" if len(df_c) else "—")

        st.markdown("<br>", unsafe_allow_html=True)

        if len(df_c) == 0:
            st.info("No se encontraron chollos con estos filtros. Reduce el descuento mínimo o amplía los criterios.")
        else:
            st.markdown("""
            <div style="background:#FEF2F2;border-left:3px solid #DC2626;padding:10px 14px;
                        border-radius:4px;margin-bottom:14px;font-size:0.82rem;color:#7F1D1D;line-height:1.7;">
            <b>⚠️ Antes de tomar cualquier decisión, ten en cuenta:</b><br>
            &bull; El modelo predice el precio a partir de marca, modelo, año, km, potencia y combustible.
            <b>No tiene en cuenta</b> el estado real del vehículo, el historial de accidentes, las reparaciones previas,
            el número de propietarios ni si tiene la ITV en vigor.<br>
            &bull; Un precio bajo puede deberse a <b>daños estéticos o mecánicos no declarados</b>,
            a una situación urgente de venta o a diferencias fiscales regionales
            (p. ej. IGIC en Canarias frente al IVA peninsular).<br>
            &bull; La estimación es una <b>media estadística</b>: dos coches idénticos en papel
            pueden diferir miles de euros según su mantenimiento real.<br>
            &bull; <b>Solicita siempre el informe de historial del vehículo</b> (DGT, Carfax o similar)
            y realiza una inspección presencial o con un mecánico de confianza antes de comprar.<br>
            &bull; Esta herramienta es un <b>punto de partida para detectar oportunidades</b>,
            no un sustituto de la debida diligencia antes de una compra.
            </div>
            """, unsafe_allow_html=True)
            df_show = df_c[[
                "marca", "modelo", "año", "kilometraje_km", "potencia_cv", "combustible",
                "precio_eur", "precio_modelo", "descuento_pct", "ahorro_eur",
                "provincia", "url",
            ]].rename(columns={
                "año":            "Año",
                "kilometraje_km": "Km",
                "potencia_cv":    "CV",
                "combustible":    "Combustible",
                "precio_eur":     "Precio (€)",
                "precio_modelo":  "Modelo estima (€)",
                "descuento_pct":  "Descuento %",
                "ahorro_eur":     "Ahorro (€)",
                "provincia":      "Provincia",
                "url":            "Anuncio",
                "marca":          "Marca",
                "modelo":         "Modelo",
            })
            st.dataframe(
                df_show,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Anuncio": st.column_config.LinkColumn("Anuncio", display_text="Ver"),
                    "Descuento %": st.column_config.NumberColumn(format="%.1f %%"),
                    "Precio (€)":     st.column_config.NumberColumn(format="%,.0f €"),
                    "Modelo estima (€)": st.column_config.NumberColumn(format="%,.0f €"),
                    "Ahorro (€)":     st.column_config.NumberColumn(format="%,.0f €"),
                    "Km":             st.column_config.NumberColumn(format="%,.0f"),
                    "CV":             st.column_config.NumberColumn(format="%d"),
                },
            )
