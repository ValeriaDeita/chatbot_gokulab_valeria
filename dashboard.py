import os
from itertools import product as iproduct

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
from pymongo import MongoClient
from streamlit_autorefresh import st_autorefresh

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Gōku Lab",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .main { background-color: #0f0f0f; }
    .block-container { padding: 2rem 2.5rem; }

    .metric-card {
        background: #E1F5FE;
        border: 1px solid #B3E5FC;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        text-align: center;
    }
    .metric-value {
        font-family: 'Space Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: #00AEEF;
        line-height: 1;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #888;
        margin-top: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .section-title {
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        color: #555;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.75rem;
        border-left: 3px solid #00AEEF;
        padding-left: 0.75rem;
    }
    .drill-header {
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        color: #00AEEF;
        margin-bottom: 0.5rem;
    }
    div[data-testid="stSelectbox"] label {
        font-size: 0.78rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# AUTO REFRESCO (cada 60 segundos)
# ─────────────────────────────────────────────

st_autorefresh(interval=60_000, key="autorefresh")


@st.cache_resource
def conectar_mongo():
    uri = os.getenv("MONGO_URI") or st.secrets.get("MONGO_URI")
    if not uri:
        st.error("No se encontró MONGO_URI. Agrégala en secrets o como variable de entorno.")
        st.stop()
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    return client["chatbot_Goku_lab"]

db = conectar_mongo()

@st.cache_data(ttl=60)
def cargar_datos():
    docs = list(db["conversaciones"].find(
        {},
        {"_id": 0, "numero": 1, "mensaje": 1, "intencion": 1,
         "confianza": 1, "sentimiento": 1, "uso_rag": 1,
         "respuesta": 1, "timestamp": 1}
    ))
    if not docs:
        return pd.DataFrame()

    df = pd.DataFrame(docs)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["hora"]      = df["timestamp"].dt.hour
    df["fecha"]     = df["timestamp"].dt.date
    df["confianza"] = pd.to_numeric(df["confianza"], errors="coerce")
    return df

df_raw = cargar_datos()


EXCLUIR = ["captura_numero"]

COLORES = {
    "Saludo":                   "#00EF34",
    "Despedida":                "#E63329",
    "Consultar_Cursos":         "#00AEEF",
    "Consultar_Costos":         "#E63329",
    "Consultar_Horarios":       "#4ecdc4",
    "Consultar_Modalidad":      "#F5A800",
    "Consultar_Certificacion":  "#a29bfe",
    "Consultar_ClaseDemo":      "#fd79a8",
    "Consultar_FormasPago":     "#fdcb6e",
    "Consultar_RequisitosEdad": "#81ecec",
    "Consultar_Duracion":       "#6c5ce7",
    "Consultar_Ubicacion":      "#00cec9",
    "Desconocido":              "#636e72",
}

COLORES_SENT = {
    "positivo": "#00AEEF",
    "neutral":  "#4ecdc4",
    "negativo": "#E63329",
}

PATRONES_NO_SABE = [
    "no tengo información",
    "no tengo información sobre",
    "no puedo responder",
    "no entiendo",
    "no sé cómo ayudarte",
    "no cuento con información",
    "lo siento, no",
    "disculpa, no",
    "no encuentro información",
]


col_logo, col_titulo, col_update = st.columns([1, 6, 2])
with col_logo:
    st.markdown("## 🎮")
with col_titulo:
    st.markdown("# Gōku Lab ")
    st.markdown(
        "<p style='color:#555; font-size:0.8rem; margin-top:-12px'>Chatbot · Atención académica</p>",
        unsafe_allow_html=True,
    )
with col_update:
    st.markdown(
        f"<p style='color:#444; font-size:0.75rem; text-align:right; margin-top:1.5rem'>"
        f"↻ actualiza cada 60s<br>{datetime.now().strftime('%H:%M:%S')}</p>",
        unsafe_allow_html=True,
    )

st.divider()


with st.sidebar:
    st.markdown("### 🎛️ Filtros")

    if not df_raw.empty:
        fecha_min = df_raw["timestamp"].min().date()
        fecha_max = df_raw["timestamp"].max().date()
    else:
        fecha_min = fecha_max = datetime.now().date()

    rango = st.date_input(
        "Rango de fechas",
        value=(fecha_min, fecha_max),
        min_value=fecha_min,
        max_value=fecha_max,
    )

    sentimientos     = ["Todos", "positivo", "neutral", "negativo"]
    filtro_sentimiento = st.selectbox("Sentimiento", sentimientos)
    incluir_ruido    = st.checkbox("Incluir Saludo / Despedida", value=False)

    st.divider()
    st.markdown(
        "<p style='color:#444; font-size:0.75rem'>Gōku Lab · Prácticas UNAM</p>",
        unsafe_allow_html=True,
    )


if df_raw.empty:
    st.warning("No hay datos en la colección todavía.")
    st.stop()

df = df_raw.copy()

if isinstance(rango, tuple) and len(rango) == 2:
    df = df[(df["fecha"] >= rango[0]) & (df["fecha"] <= rango[1])]

if filtro_sentimiento != "Todos":
    df = df[df["sentimiento"] == filtro_sentimiento]

if not incluir_ruido:
    df = df[~df["intencion"].isin(EXCLUIR)]

df_consultas = df[~df["intencion"].isin(EXCLUIR)]

# ─────────────────────────────────────────────
# MÉTRICAS PRINCIPALES
# ─────────────────────────────────────────────

st.markdown('<p class="section-title">Resumen general</p>', unsafe_allow_html=True)

total_msgs     = len(df)
total_leads    = len(df_raw[df_raw["intencion"] == "captura_numero"])
tasa_rag       = df["uso_rag"].mean() * 100 if not df.empty else 0
confianza_prom = df["confianza"].mean() * 100 if not df.empty else 0
usuarios_uniq  = df["numero"].nunique() if not df.empty else 0

c1, c2, c3, c4, c5 = st.columns(5)
for col, valor, label, color in [
    (c1, total_msgs,              "mensajes totales",    "#00AEEF"),
    (c2, usuarios_uniq,           "usuarios únicos",     "#EFEF00"),
    (c3, total_leads,             "leads capturados",    "#F5A800"),
    (c4, f"{tasa_rag:.1f}%",      "tasa RAG fallback",   "#E63329"),
    (c5, f"{confianza_prom:.1f}%","confianza promedio",  "#EFE700"),
]:
    with col:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value" style="color:{color}">{valor}</div>'
            f'<div class="metric-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)


col_izq, col_der = st.columns([3, 2])

with col_izq:
    st.markdown('<p class="section-title">Consultas por intención</p>', unsafe_allow_html=True)

    if not df_consultas.empty:
        conteo = (
            df_consultas["intencion"]
            .value_counts()
            .reset_index()
            .rename(columns={"intencion": "Intención", "count": "Mensajes"})
        )

        fig_bar = px.bar(
            conteo,
            x="Mensajes",
            y="Intención",
            orientation="h",
            color="Intención",
            color_discrete_map=COLORES,
            template="plotly_dark",
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            height=320,
            yaxis=dict(categoryorder="total ascending"),
            font=dict(family="DM Sans", color="#aaa", size=12),
        )
        fig_bar.update_traces(marker_line_width=0)

        evento = st.plotly_chart(fig_bar, use_container_width=True, on_select="rerun", key="bar_intencion")
        intencion_seleccionada = None
        if evento and evento.get("selection") and evento["selection"].get("points"):
            intencion_seleccionada = evento["selection"]["points"][0].get("label")
    else:
        st.info("Sin datos para mostrar.")
        intencion_seleccionada = None

with col_der:
    st.markdown('<p class="section-title">Distribución de sentimientos</p>', unsafe_allow_html=True)

    if not df.empty:
        sent_conteo = df["sentimiento"].value_counts().reset_index()
        sent_conteo.columns = ["Sentimiento", "Total"]

        fig_donut = px.pie(
            sent_conteo,
            names="Sentimiento",
            values="Total",
            hole=0.6,
            color="Sentimiento",
            color_discrete_map=COLORES_SENT,
            template="plotly_dark",
        )
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(font=dict(color="#aaa", size=11)),
            margin=dict(l=0, r=0, t=0, b=0),
            height=320,
            font=dict(family="DM Sans", color="#aaa"),
        )
        fig_donut.update_traces(textfont_color="white")
        st.plotly_chart(fig_donut, use_container_width=True)

# ─────────────────────────────────────────────
# FILA 2 — MENSAJES POR HORA + RAG vs CLF
# ─────────────────────────────────────────────

col_a, col_b = st.columns([3, 2])

with col_a:
    st.markdown('<p class="section-title">Mensajes por hora del día</p>', unsafe_allow_html=True)

    if not df.empty:
        por_hora   = df.groupby("hora").size().reset_index(name="Mensajes")
        todas_horas = pd.DataFrame({"hora": range(24)})
        por_hora   = todas_horas.merge(por_hora, on="hora", how="left").fillna(0)

        fig_hora = px.area(
            por_hora, x="hora", y="Mensajes",
            template="plotly_dark",
            color_discrete_sequence=["#00AEEF"],
        )
        fig_hora.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
            height=240,
            xaxis=dict(title="Hora", tickmode="linear", dtick=2, color="#555"),
            yaxis=dict(title="", color="#555"),
            font=dict(family="DM Sans", color="#aaa"),
        )
        fig_hora.update_traces(line_color="#00AEEF", fillcolor="rgba(0,174,239,0.1)")
        st.plotly_chart(fig_hora, use_container_width=True)

with col_b:
    st.markdown('<p class="section-title">RAG vs Clasificador SVM</p>', unsafe_allow_html=True)

    if not df.empty:
        rag_conteo = df["uso_rag"].value_counts().reset_index()
        rag_conteo.columns = ["Tipo", "Total"]
        rag_conteo["Tipo"] = rag_conteo["Tipo"].map(
            {True: "RAG fallback", False: "Clasificador SVM"}
        )

        fig_rag = px.pie(
            rag_conteo,
            names="Tipo", values="Total",
            hole=0.6,
            color="Tipo",
            color_discrete_map={"RAG fallback": "#E63329", "Clasificador SVM": "#00AEEF"},
            template="plotly_dark",
        )
        fig_rag.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(font=dict(color="#aaa", size=11)),
            margin=dict(l=0, r=0, t=0, b=0),
            height=240,
            font=dict(family="DM Sans", color="#aaa"),
        )
        fig_rag.update_traces(textfont_color="white")
        st.plotly_chart(fig_rag, use_container_width=True)

# ─────────────────────────────────────────────
# CONFIANZA PROMEDIO POR INTENCIÓN
# ─────────────────────────────────────────────

st.markdown('<p class="section-title">Confianza promedio por intención</p>', unsafe_allow_html=True)

if not df_consultas.empty:
    conf_prom = (
        df_consultas.groupby("intencion")["confianza"]
        .mean()
        .reset_index()
        .rename(columns={"intencion": "Intención", "confianza": "Confianza"})
        .sort_values("Confianza")
    )
    conf_prom["color"] = conf_prom["Confianza"].apply(
        lambda x: "#E63329" if x < 0.6 else "#F5A800" if x < 0.8 else "#00AEEF"
    )

    fig_conf = go.Figure(go.Bar(
        x=conf_prom["Confianza"],
        y=conf_prom["Intención"],
        orientation="h",
        marker_color=conf_prom["color"],
        text=conf_prom["Confianza"].apply(lambda x: f"{x:.0%}"),
        textposition="outside",
        textfont=dict(color="#aaa", size=11),
    ))
    fig_conf.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=60, t=0, b=0),
        height=280,
        xaxis=dict(range=[0, 1.1], tickformat=".0%", color="#555"),
        yaxis=dict(color="#aaa"),
        font=dict(family="DM Sans", color="#aaa"),
    )
    st.plotly_chart(fig_conf, use_container_width=True)
    st.caption("🔴 < 60%  necesita más datos de entrenamiento  ·  🟡 60-80%  aceptable  ·  🔵 > 80%  excelente")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEATMAP — hora × día de la semana
# ─────────────────────────────────────────────

st.markdown('<p class="section-title">Heatmap de actividad — hora vs día</p>', unsafe_allow_html=True)

if not df.empty:
    df_heat = df.copy()
    df_heat["dia_semana"] = df_heat["timestamp"].dt.day_name()
    df_heat["hora"]       = df_heat["timestamp"].dt.hour

    ORDEN_DIAS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    NOMBRES_DIAS = {
        "Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles",
        "Thursday": "Jueves", "Friday": "Viernes", "Saturday": "Sábado", "Sunday": "Domingo",
    }

    pivot = (
        df_heat.groupby(["dia_semana", "hora"])
        .size()
        .reset_index(name="consultas")
    )

    todas = pd.DataFrame(
        list(iproduct(ORDEN_DIAS, range(24))),
        columns=["dia_semana", "hora"],
    )
    pivot = todas.merge(pivot, on=["dia_semana", "hora"], how="left").fillna(0)
    pivot["dia_nombre"] = pivot["dia_semana"].map(NOMBRES_DIAS)
    pivot["dia_semana"] = pd.Categorical(pivot["dia_semana"], categories=ORDEN_DIAS, ordered=True)
    pivot = pivot.sort_values(["dia_semana", "hora"])

    colorscale_heat = [
        [0.00, "#0f0f0f"],
        [0.20, "#003D6B"],
        [0.50, "#00AEEF"],
        [0.80, "#F5A800"],
        [1.00, "#E63329"],
    ]

    fig_heat = go.Figure(go.Heatmap(
        x=pivot["dia_nombre"],
        y=pivot["hora"].apply(lambda h: f"{h:02d}:00"),
        z=pivot["consultas"],
        colorscale=colorscale_heat,
        showscale=True,
        hoverongaps=False,
        hovertemplate="<b>%{x}</b><br>%{y}<br>%{z:.0f} consultas<extra></extra>",
        colorbar=dict(
            title=dict(text="Consultas", font=dict(color="#888", size=11)),
            tickfont=dict(color="#888", size=10),
            thickness=12,
            len=0.8,
        ),
        xgap=2,
        ygap=1,
    ))
    fig_heat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#1a1a1a",
        margin=dict(l=0, r=0, t=0, b=0),
        height=420,
        xaxis=dict(
            title="",
            side="top",
            tickfont=dict(color="#aaa", size=12),
            gridcolor="rgba(0,0,0,0)",
        ),
        yaxis=dict(
            title="Hora",
            autorange="reversed",
            tickfont=dict(color="#888", size=10),
            dtick=2,
            gridcolor="rgba(255,255,255,0.04)",
        ),
        font=dict(family="DM Sans", color="#aaa"),
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption("Negro = sin actividad · Azul = actividad normal · Amarillo/Rojo = hora pico")

else:
    st.info("Sin datos suficientes para el heatmap.")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TOP INTENCIONES CON PEOR SENTIMIENTO
# ─────────────────────────────────────────────

st.markdown('<p class="section-title">Intenciones con mayor negatividad</p>', unsafe_allow_html=True)

if not df_consultas.empty and "sentimiento" in df_consultas.columns:

    sent_intent   = (
        df_consultas.groupby(["intencion", "sentimiento"])
        .size()
        .reset_index(name="n")
    )
    total_intent  = df_consultas.groupby("intencion").size().reset_index(name="total")
    sent_intent   = sent_intent.merge(total_intent, on="intencion")
    sent_intent["pct"] = sent_intent["n"] / sent_intent["total"] * 100

    negativos = (
        sent_intent[sent_intent["sentimiento"] == "negativo"]
        .copy()
        .sort_values("pct", ascending=True)
        .tail(8)
    )

    if not negativos.empty:
        fig_neg = go.Figure()

        fig_neg.add_trace(go.Bar(
            x=[100] * len(negativos),
            y=negativos["intencion"],
            orientation="h",
            marker_color="rgba(255,255,255,0.04)",
            showlegend=False,
            hoverinfo="skip",
        ))

        fig_neg.add_trace(go.Bar(
            x=negativos["pct"],
            y=negativos["intencion"],
            orientation="h",
            marker=dict(
                color=negativos["pct"],
                colorscale=[
                    [0.0, "#F5A800"],
                    [0.5, "#E67300"],
                    [1.0, "#E63329"],
                ],
                cmin=0, cmax=100,
                showscale=False,
            ),
            text=negativos["pct"].apply(lambda x: f"{x:.0f}%"),
            textposition="outside",
            textfont=dict(color="#aaa", size=11),
            hovertemplate="<b>%{y}</b><br>%{x:.1f}% mensajes negativos<br>de %{customdata} totales<extra></extra>",
            customdata=negativos["total"],
        ))

        fig_neg.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            barmode="overlay",
            margin=dict(l=0, r=60, t=0, b=0),
            height=max(240, len(negativos) * 42),
            xaxis=dict(
                range=[0, 115],
                ticksuffix="%",
                color="#555",
                gridcolor="rgba(255,255,255,0.05)",
            ),
            yaxis=dict(color="#aaa"),
            font=dict(family="DM Sans", color="#aaa"),
            showlegend=False,
        )
        st.plotly_chart(fig_neg, use_container_width=True)
        st.caption("% de mensajes con sentimiento negativo dentro de cada intención · prioriza las más altas para mejorar las respuestas")
    else:
        st.success("Sin mensajes negativos registrados en el período seleccionado.")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MENSAJES SIN COBERTURA
# ─────────────────────────────────────────────

st.markdown('<p class="section-title">Mensajes sin cobertura — entrenamiento pendiente</p>', unsafe_allow_html=True)

if not df_raw.empty and "respuesta" in df_raw.columns:
    mascara = df_raw["respuesta"].str.lower().str.contains(
        "|".join(PATRONES_NO_SABE), na=False, regex=True
    )
    df_sin = df_raw[mascara].copy()

    if isinstance(rango, tuple) and len(rango) == 2:
        df_sin = df_sin[(df_sin["fecha"] >= rango[0]) & (df_sin["fecha"] <= rango[1])]

    total_sin   = len(df_sin)
    pct_sin     = (total_sin / len(df_raw) * 100) if len(df_raw) > 0 else 0
    temas_uniq  = df_sin["intencion"].nunique() if "intencion" in df_sin.columns else 0

    cs1, cs2, cs3 = st.columns(3)
    for col, valor, label, color in [
        (cs1, total_sin,          "mensajes sin cobertura",    "#E63329"),
        (cs2, f"{pct_sin:.1f}%",  "del total de mensajes",     "#F5A800"),
        (cs3, temas_uniq,         "intenciones involucradas",  "#00AEEF"),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value" style="color:{color}">{valor}</div>'
                f'<div class="metric-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    if total_sin > 0:
        freq_sin = (
            df_sin.groupby("fecha")
            .size()
            .reset_index(name="sin_cobertura")
        )
        fig_sin = px.bar(
            freq_sin, x="fecha", y="sin_cobertura",
            template="plotly_dark",
            color_discrete_sequence=["#E63329"],
        )
        fig_sin.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
            height=160,
            xaxis=dict(title="", color="#555"),
            yaxis=dict(title="mensajes", color="#555"),
            font=dict(family="DM Sans", color="#aaa"),
        )
        fig_sin.update_traces(marker_line_width=0)
        st.plotly_chart(fig_sin, use_container_width=True)

        st.markdown(
            "<p style='color:#888; font-size:0.78rem; margin-bottom:0.5rem'>"
            "Lee estos mensajes para identificar qué temas agregar al entrenamiento:</p>",
            unsafe_allow_html=True,
        )

        df_tabla_sin = df_sin.sort_values("timestamp", ascending=False).copy()
        df_tabla_sin["timestamp"] = df_tabla_sin["timestamp"].dt.strftime("%Y-%m-%d %H:%M")

        columnas_mostrar = ["timestamp", "mensaje", "respuesta"]
        renombrar = {
            "timestamp": "Fecha/Hora",
            "mensaje":   "Pregunta del usuario",
            "respuesta": "Respuesta del bot",
        }
        if "intencion" in df_tabla_sin.columns:
            columnas_mostrar.insert(2, "intencion")
            renombrar["intencion"] = "Intención detectada"

        st.dataframe(
            df_tabla_sin[columnas_mostrar].rename(columns=renombrar),
            use_container_width=True,
            height=320,
            hide_index=True,
        )
        st.caption(f"{total_sin} mensajes sin cobertura · descarga para agregar al dataset de entrenamiento")

        csv_sin = (
            df_tabla_sin[columnas_mostrar]
            .rename(columns=renombrar)
            .to_csv(index=False)
            .encode("utf-8")
        )
        st.download_button(
            label="⬇️  Descargar mensajes sin cobertura (.csv)",
            data=csv_sin,
            file_name="mensajes_sin_cobertura.csv",
            mime="text/csv",
        )
    else:
        st.success("Sin mensajes sin cobertura en el período seleccionado.")

else:
    st.info("No hay campo 'respuesta' en la colección o no hay datos.")

st.divider()

# ─────────────────────────────────────────────
# DRILL DOWN — tabla de mensajes por intención
# ─────────────────────────────────────────────

if intencion_seleccionada:
    intencion_drill = intencion_seleccionada
    st.markdown(
        f'<p class="drill-header">📋 Mensajes de: {intencion_drill}</p>',
        unsafe_allow_html=True,
    )
else:
    opciones = (
        ["— Ver todos —"] + sorted(df_consultas["intencion"].unique().tolist())
        if not df_consultas.empty else ["— Ver todos —"]
    )
    intencion_drill = st.selectbox("📋 Detalle de mensajes por intención", opciones)
    if intencion_drill == "— Ver todos —":
        intencion_drill = None

df_tabla = df[df["intencion"] == intencion_drill].copy() if intencion_drill else df.copy()

if not df_tabla.empty:
    df_tabla = df_tabla.sort_values("timestamp", ascending=False)
    df_tabla["timestamp"] = df_tabla["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    df_tabla["confianza"] = df_tabla["confianza"].apply(
        lambda x: f"{x:.0%}" if pd.notna(x) else "—"
    )

    st.dataframe(
        df_tabla[["timestamp","intencion","mensaje","sentimiento","confianza","uso_rag"]].rename(columns={
            "timestamp":   "Fecha/Hora",
            "intencion":   "Intención",
            "mensaje":     "Mensaje del usuario",
            "sentimiento": "Sentimiento",
            "confianza":   "Confianza",
            "uso_rag":     "RAG",
        }),
        use_container_width=True,
        height=350,
        hide_index=True,
    )
    st.caption(f"{len(df_tabla)} mensajes · haz click en una barra arriba para filtrar automáticamente")
else:
    st.info("Sin mensajes para esta selección.")