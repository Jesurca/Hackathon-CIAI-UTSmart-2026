"""
ui.py — estilo visual, componentes y figuras reutilizables de la app.
(Las mismas figuras se usan en las páginas, en el modo presentación y en el reporte.)
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from agente_rl.variables import VARIABLES

ACENTO = "#FF7A45"
AZUL = "#4C9AFF"
COLOR_NIVEL = {"Bajo": "#3FB68B", "Medio": "#F2A541", "Alto": "#E5484D"}
COLOR_CONF = {"Alta": "#3FB68B", "Media": "#F2A541", "Baja": "#E5484D"}
ICONO_CONF = {"Alta": "🟢 Alta", "Media": "🟡 Media", "Baja": "🔴 Baja"}
COLOR_GRUPO_VAR = {"Sísmica": ACENTO, "Exposición": AZUL, "Vulnerabilidad": "#A78BFA", "Capacidad": "#2EC4B6",
                   "Territorial": "#8FA3BF", "Modelos del equipo": "#F2A541", "Decisión": "#E5484D"}
PALETA = ["#4C9AFF", "#FF7A45", "#2EC4B6", "#F2A541", "#A78BFA", "#E5484D", "#8FA3BF", "#3FB68B"]
COLOR_POLITICA = {"Agente RL (ensamble)": ACENTO}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, p, li { font-family: 'Inter', sans-serif; }
h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.02em; }
.block-container { padding-top: 2.6rem; max-width: 1380px; }
.hero { background: radial-gradient(circle at 85% 20%, rgba(255,122,69,.35), transparent 45%),
        linear-gradient(135deg, #1A1F2E 0%, #121620 100%);
        border: 1px solid rgba(255,255,255,.08); border-radius: 18px; padding: 28px 32px; margin-bottom: 18px; }
.hero h1 { margin: 0 0 6px 0; font-size: 2.1rem; color: #F5F7FA; }
.hero p  { margin: 0; color: #AEB6C5; font-size: 1.02rem; }
.chip { display:inline-block; padding:3px 11px; border-radius:999px; font-size:.78rem; font-weight:600;
        margin-right:6px; margin-top:10px; border:1px solid rgba(255,255,255,.12); }
.tarjeta { background: #161B26; border: 1px solid rgba(255,255,255,.07); border-radius: 14px;
           padding: 18px 20px; min-height: 215px; }
.tarjeta h4 { margin: 0 0 6px 0; font-size: 1.02rem; }
.tarjeta p  { margin: 0; color: #AEB6C5; font-size: .92rem; line-height: 1.45; }
.paso-num { color: #FF7A45; font-family: 'Space Grotesk'; font-weight: 700; font-size: .8rem; letter-spacing:.04em; }
div[data-testid="stMetric"] { background: #161B26; border: 1px solid rgba(255,255,255,.07);
                              border-radius: 14px; padding: 14px 18px; }
div[data-testid="stMetricValue"] { font-family: 'Space Grotesk'; }
.aviso { padding: 9px 14px; border-radius: 8px; font-size: .86rem; margin-bottom: 14px; }
.aviso.demo { background: rgba(242,165,65,.10); border-left: 3px solid #F2A541; color: #E8C48A; }
.aviso.real { background: rgba(63,182,139,.10); border-left: 3px solid #3FB68B; color: #A9E2CB; }
.aviso.info { background: rgba(76,154,255,.10); border-left: 3px solid #4C9AFF; color: #B9D4FF; }
.badge { display:inline-block; padding:2px 10px; border-radius:999px; font-size:.75rem; font-weight:600; }
.b-archivo { background: rgba(63,182,139,.15); color:#6FD3A8; border:1px solid rgba(63,182,139,.4); }
.b-modelo { background: rgba(76,154,255,.15); color:#8DBBFF; border:1px solid rgba(76,154,255,.4); }
.b-sustituto { background: rgba(242,165,65,.15); color:#F2C27A; border:1px solid rgba(242,165,65,.4); }
.resultado { border-radius: 14px; padding: 22px; text-align: center; }
.resultado h2 { margin: 4px 0; font-size: 2.2rem; }
.slide { background: linear-gradient(160deg, #161B26 0%, #0F131B 100%); border: 1px solid rgba(255,255,255,.08);
         border-radius: 22px; padding: 34px 40px; min-height: 120px; margin-bottom: 14px; }
.slide .eyebrow { color:#FF7A45; font-family:'Space Grotesk'; font-weight:700; letter-spacing:.08em; font-size:.85rem; }
.slide h1 { font-size: 2.6rem; margin: 6px 0 10px 0; color:#F5F7FA; }
.slide p { font-size: 1.15rem; color:#C3CAD6; margin:0; }
.big { font-family:'Space Grotesk'; font-size: 2.8rem; font-weight:700; color:#F5F7FA; line-height:1; }
.big-l { color:#AEB6C5; font-size:.95rem; margin-top:6px; }
.puntos { text-align:center; color:#5A6475; letter-spacing:.4em; font-size:1.2rem; }
section[data-testid="stSidebar"] { background: #0F131B; }
.estado { font-size:.82rem; color:#AEB6C5; line-height:1.6; }
</style>
"""


def aplicar_estilo():
    st.markdown(CSS, unsafe_allow_html=True)


def estilo_fig(fig, alto=380):
    fig.update_layout(template="plotly_dark", height=alto, margin=dict(l=10, r=10, t=46, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, sans-serif", size=13, color="#D6DBE4"),
                      legend=dict(bgcolor="rgba(0,0,0,0)"), title_font=dict(family="Space Grotesk, Inter, sans-serif", size=16))
    fig.update_xaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False)
    return fig


def _scatter_map(df, **args):
    args = {k: v for k, v in args.items() if v is not None}
    if hasattr(px, "scatter_map"):
        return px.scatter_map(df, map_style="carto-darkmatter", **args)
    return px.scatter_mapbox(df, mapbox_style="carto-darkmatter", **args)


def _layout_mapa(fig):
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"),
                      legend=dict(bgcolor="rgba(15,19,27,.8)", y=0.98, x=0.01))
    return fig


def mapa(df, color, alto=480, tamano=None, hover=None, mapa_colores=None, escala=None, zoom=6.6, nombre="municipio"):
    fig = _scatter_map(df, lat="lat", lon="lon", color=color, size=tamano,
                       hover_name=nombre if nombre in df else None, hover_data=hover, zoom=zoom,
                       center={"lat": float(df["lat"].mean()), "lon": float(df["lon"].mean())}, height=alto,
                       color_discrete_map=mapa_colores, color_continuous_scale=escala)
    return _layout_mapa(fig)


def tarjeta(titulo, texto, paso=None):
    num = f'<div class="paso-num">{paso}</div>' if paso else ""
    st.markdown(f'<div class="tarjeta">{num}<h4>{titulo}</h4><p>{texto}</p></div>', unsafe_allow_html=True)


def badge_fuente(fuente):
    txt = {"archivo": "Archivo de resultados", "modelo": "Modelo .pkl aplicado", "sustituto": "Sustituto automático"}[fuente]
    return f'<span class="badge b-{fuente}">{txt}</span>'


def aviso(texto, tipo="info"):
    st.markdown(f'<div class="aviso {tipo}">{texto}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ figuras
def fig_matriz(cm, clases, titulo="Matriz de confusión"):
    cm = np.asarray(cm)
    pct = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    texto = [[f"<b>{cm[i, j]}</b><br>{pct[i, j]:.0%}" for j in range(len(clases))] for i in range(len(clases))]
    fig = go.Figure(go.Heatmap(z=pct, x=clases, y=clases, colorscale="Oranges", zmin=0, zmax=1, text=texto,
                               texttemplate="%{text}", showscale=False, hovertemplate="Real %{y} · Predicho %{x}<extra></extra>"))
    fig.update_xaxes(title="Predicho", side="bottom"); fig.update_yaxes(title="Real", autorange="reversed")
    fig.update_layout(title=titulo)
    return estilo_fig(fig)


def fig_politicas(tabla, alto=360):
    t = tabla.sort_values("captura_riesgo")
    colores = [ACENTO if p.startswith("Agente") else "#5A6475" for p in t["politica"]]
    fig = go.Figure(go.Bar(x=t["captura_riesgo"], y=t["politica"], orientation="h", marker_color=colores,
                           error_x=dict(type="data", array=t["captura_std"], color="rgba(255,255,255,.35)", thickness=1.2)))
    for _, r in t.iterrows():
        fig.add_annotation(x=r["captura_riesgo"] + r["captura_std"] + .02, y=r["politica"], text=f"<b>{r['captura_riesgo']:.0%}</b>",
                           showarrow=False, xanchor="left", font=dict(color="#E6E9EF"))
    fig.update_xaxes(tickformat=".0%", range=[0, max(1.2, float((t["captura_riesgo"] + t["captura_std"]).max()) + .15)])
    fig.update_layout(title="Riesgo capturado en PRUEBA (100 % = oráculo que conoce el futuro)")
    return estilo_fig(fig, alto)


def fig_importancia(imp, top=15, alto=460):
    s = imp.head(top)[::-1]
    grupos = [VARIABLES[v][1] for v in s.index]
    fig = go.Figure()
    for g in dict.fromkeys(grupos):
        m = [x == g for x in grupos]
        fig.add_trace(go.Bar(x=s.values[m], y=s.index[m], orientation="h", name=g, marker_color=COLOR_GRUPO_VAR.get(g, "#8FA3BF")))
    fig.update_layout(title="¿Qué mira el agente para decidir? (gradiente × entrada)", barmode="stack",
                      yaxis=dict(categoryorder="array", categoryarray=list(s.index)))
    fig.update_xaxes(tickformat=".0%")
    return estilo_fig(fig, alto)


def fig_mapa_prioridad(tab, municipios, alto=520):
    df = tab.merge(municipios[["municipio", "lat", "lon", "provincia"]], on="municipio", how="left")
    df["conf"] = df["confianza"].map(ICONO_CONF)
    sel = df[df["priorizado"]]
    escala = (df["incertidumbre"] - df["incertidumbre"].min()) / (np.ptp(df["incertidumbre"]) + 1e-9)
    tr = go.Scattermap if hasattr(go, "Scattermap") else go.Scattermapbox
    fig = go.Figure()
    fig.add_trace(tr(lat=sel["lat"], lon=sel["lon"], mode="markers", name="Priorizado por el agente",
                     marker=dict(size=34, color=ACENTO, opacity=.9), hoverinfo="skip"))
    fig.add_trace(tr(lat=df["lat"], lon=df["lon"], mode="markers", name="Municipio",
                     marker=dict(size=12 + 16 * escala, color=df["valor_Q"], colorscale="Blues", showscale=True,
                                 colorbar=dict(title="Prioridad", x=1, thickness=10)),
                     text=df["municipio"], customdata=np.c_[df["ranking"], df["conf"], df["razones"]],
                     hovertemplate="<b>%{text}</b><br>Puesto %{customdata[0]} · Confianza %{customdata[1]}"
                                   "<br>%{customdata[2]}<extra></extra>"))
    sub = dict(center=dict(lat=float(df["lat"].mean()), lon=float(df["lon"].mean())), zoom=6.6, style="carto-darkmatter")
    if hasattr(go, "Scattermap"):
        fig.update_layout(map=sub)
    else:
        fig.update_layout(mapbox=sub)
    fig.update_layout(height=alto, showlegend=True)
    return _layout_mapa(fig)


def fig_q_incertidumbre(tab, n=15, alto=460):
    s = tab.head(n).iloc[::-1]
    fig = go.Figure(go.Bar(x=s["valor_Q"], y=s["municipio"], orientation="h",
                           marker_color=[ACENTO if p else "#5A6475" for p in s["priorizado"]],
                           error_x=dict(type="data", array=s["incertidumbre"], color="rgba(255,255,255,.55)", thickness=1.4)))
    lo = float((s["valor_Q"] - s["incertidumbre"]).min()); hi = float((s["valor_Q"] + s["incertidumbre"]).max())
    fig.update_xaxes(range=[lo - (hi - lo) * .15, hi + (hi - lo) * .05], title="Valor de prioridad (Q medio del comité)")
    fig.update_layout(title="Prioridad ± desacuerdo del comité de 5 redes (naranja = priorizado)")
    return estilo_fig(fig, alto)


def fig_curva(curva, alto=330):
    c = curva.copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=c["episodio"], y=c["retorno"], mode="lines", name="Por episodio",
                             line=dict(color="rgba(255,255,255,.18)", width=1)))
    fig.add_trace(go.Scatter(x=c["episodio"], y=c["retorno"].rolling(20, min_periods=1).mean(), mode="lines",
                             name="Media móvil (20)", line=dict(color=ACENTO, width=3)))
    fig.update_layout(title="Recompensa durante el entrenamiento", xaxis_title="Episodio", yaxis_title="Recompensa",
                      legend=dict(orientation="h", y=-0.25))
    return estilo_fig(fig, alto)


def fig_validacion(curva, alto=330):
    v = curva.dropna(subset=["captura_val"])
    fig = go.Figure(go.Scatter(x=v["episodio"], y=v["captura_val"], mode="lines+markers", name="Validación",
                               line=dict(color=AZUL, width=2.5), marker=dict(size=7)))
    if len(v):
        b = v.loc[v["captura_val"].idxmax()]
        fig.add_annotation(x=b["episodio"], y=b["captura_val"], text=f"Mejor: {b['captura_val']:.0%} (se guarda)",
                           showarrow=True, arrowcolor="#AEB6C5", font=dict(color="#E6E9EF"), ay=-30)
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(title="Riesgo capturado en validación (años no usados para entrenar)", xaxis_title="Episodio")
    return estilo_fig(fig, alto)
