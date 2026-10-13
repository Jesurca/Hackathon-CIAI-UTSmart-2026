"""
app.py — Plataforma de riesgo sísmico de Santander (Hackatón UTS)
=================================================================
Integra en una sola app:
  · los datos (catálogo SGC + fuente oficial) que el equipo cargue,
  · los resultados y modelos del reto Básico (supervisado) y Medio (no supervisado),
  · el agente por refuerzo del reto Avanzado, que usa todo lo anterior como estado.

Ejecutar (con el entorno virtual activado, dentro de esta carpeta):
    streamlit run app.py
"""
import io
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from agente_rl.utilidades import accuracy, f1_macro, kmeans, pca, prf, silueta

import estado as E
import ui
from agente_rl.columnas import CLASES, ESQUEMAS, NO_DISPONIBLE, aplicar_mapeo, detectar, faltantes_obligatorias, nivel_a_num, norm_municipio
from agente_rl.variables import FEATURES, VARIABLES, aplicar_modelo, tabla_variables

st.set_page_config(page_title="Riesgo sísmico · Santander", page_icon="🌋", layout="wide")
ui.aplicar_estilo()

try:
    E.iniciar()
except Exception as e:
    st.error(f"No se pudo iniciar con los datos de demostración: {e}. Ejecute `python generar_demo.py`.")
    st.stop()

ss = st.session_state
R = ss.R
D, panel, rep, MUN, EVT, CFG = R["D"], R["panel"], R["reporte"], R["municipios"], R["eventos"], R["cfg"]
MUN = MUN.rename(columns={"latitud": "lat", "longitud": "lon"})
EVT = EVT.rename(columns={"latitud": "lat", "longitud": "lon"})
ORIGEN = E.origen_datos()
HAY_AGENTE = "agente" in ss


def aviso_origen():
    if ORIGEN == "demo":
        ui.aviso("⚠️ Está viendo <b>datos de demostración</b>. Cargue los datasets del equipo en <b>📥 Cargar datos</b>.", "demo")
    elif ORIGEN == "mixto":
        ui.aviso("ℹ️ Mezcla de archivos del equipo y de demostración. Revise <b>📥 Cargar datos</b>.", "info")
    else:
        ui.aviso("✅ Datos cargados por el equipo.", "real")


def ultimo_periodo():
    return len(D["periodos"]) - 1


def fmt(n):
    return f"{n:,}".replace(",", ".")


# ============================================================================ navegación
PAGINAS = ["📥 Cargar datos", "🏠 Inicio", "📊 Datos e integración", "🎯 Básico · Supervisado",
           "🧭 Medio · No supervisado", "🤖 Avanzado · Agente RL", "🔮 Simulador ¿qué pasaría si?",
           "✅ Validación e incertidumbre", "🎤 Modo presentación", "📄 Reporte", "📌 Supuestos y alcance"]
with st.sidebar:
    st.markdown("## 🌋 Riesgo sísmico\n**Santander · Hackatón UTS**")
    st.divider()
    pagina = st.radio("Secciones", PAGINAS, index=1, label_visibility="collapsed", key="pagina")
    st.divider()
    o = {"demo": "🟠 Demostración", "equipo": "🟢 Datos del equipo", "mixto": "🔵 Mixto"}[ORIGEN]
    ag = ("🟢 " + ss.agente["info"]["origen"]) if HAY_AGENTE else "⚪ sin agente"
    st.markdown(f'<div class="estado"><b>Datos:</b> {o}<br><b>Municipios:</b> {len(D["municipios"])} · '
                f'<b>Periodos:</b> {len(D["periodos"])}<br><b>Básico:</b> {rep["supervisado"]["fuente"]} · '
                f'<b>Medio:</b> {rep["no_supervisado"]["fuente"]}<br><b>Agente:</b> {ag}</div>', unsafe_allow_html=True)
    st.caption("Fuentes: Servicio Geológico Colombiano · DANE · MinSalud · IGAC")


# ============================================================================ 📥 CARGAR DATOS
def pagina_cargar():
    st.title("📥 Cargar datos y modelos del equipo")
    if ss.get("msg_ok"):
        st.success(ss.pop("msg_ok"))
    c1, c2 = st.columns([3, 1])
    with c2:
        if st.button("↺ Volver a la demostración", width="stretch"):
            E.cargar_demo(); st.rerun()
    with c1:
        filas = []
        for t, r in ss.ranuras.items():
            filas.append({"Archivo": E.NOMBRES[t], "En uso": r["nombre"] if r else "— (se usa sustituto)",
                          "Origen": ("Demo" if r["origen"] == "demo" else "Equipo") if r else "—",
                          "Filas": fmt(len(r["df"])) if r else "—"})
        for t in ("supervisado", "clusters"):
            m = ss.modelos.get(t)
            filas.append({"Archivo": f"Modelo .pkl ({'Básico' if t == 'supervisado' else 'Medio'})",
                          "En uso": m["nombre"] if m else "—", "Origen": ("Demo" if m and m["origen"] == "demo" else "Equipo") if m else "—", "Filas": "—"})
        filas.append({"Archivo": "Agente RL (.npz)", "En uso": ss.pt["nombre"] if ss.get("pt") else "— (entrenar en la app)",
                      "Origen": ("Demo" if ss.get("pt") and ss.pt["origen"] == "demo" else "Equipo") if ss.get("pt") else "—", "Filas": "—"})
        st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")

    tabs = st.tabs(["1 · Catálogo SGC", "2 · Fuente oficial", "3 · Supervisado (Básico)", "4 · No supervisado (Medio)",
                    "5 · Agente (.npz)", "6 · Parámetros"])
    ayudas = {
        "catalogo": "CSV o Excel del catálogo del SGC. Necesita fecha, latitud, longitud, profundidad y magnitud. "
                    "Si trae municipio se usa; si no, cada sismo se asigna a la cabecera más cercana.",
        "oficial": "Una fila por municipio con su nombre, coordenadas y población. Lo demás (NBI, % rural, viviendas "
                   "precarias, IPS, área, altitud) es muy recomendable; si falta, se usa un valor fijo y se reporta.",
        "supervisado": "Predicciones del modelo del reto Básico. Ideal: una fila por municipio y trimestre con la clase "
                       "predicha (y la real para calcular la matriz de confusión).",
        "clusters": "Grupo asignado a cada municipio por el modelo del reto Medio. Opcional: nombre del grupo y PC1/PC2.",
    }
    for tab, tipo in zip(tabs[:4], ["catalogo", "oficial", "supervisado", "clusters"]):
        with tab:
            st.caption(ayudas[tipo])
            up = st.file_uploader(f"Subir {E.NOMBRES[tipo].lower()}", type=["csv", "xlsx", "xls"], key=f"up_{tipo}")
            if up is not None and ss.get(f"subido_{tipo}") != up.file_id:
                try:
                    ss.ranuras[tipo] = E.ranura(tipo, up.name, E.leer_tabla(up.name, up.getvalue()), "equipo")
                    ss[f"subido_{tipo}"] = up.file_id
                    st.toast(f"Archivo {up.name} cargado. Revise las columnas y presione Integrar.")
                except Exception as e:
                    st.error(f"No se pudo leer {up.name}: {e}")
            r = ss.ranuras.get(tipo)
            if tipo in ("supervisado", "clusters"):
                a, b = st.columns(2)
                with a:
                    mup = st.file_uploader("Modelo entrenado (.pkl / .joblib, opcional)", type=["pkl", "joblib"], key=f"mod_{tipo}",
                                           help="Permite predecir en vivo e incluir importancias. Solo cargue modelos de su equipo: "
                                                "un archivo .pkl puede ejecutar código al abrirse.")
                    if mup is not None and (not ss.modelos.get(tipo) or ss.modelos[tipo]["nombre"] != mup.name):
                        try:
                            ss.modelos[tipo] = {"nombre": mup.name, "obj": E.leer_modelo(mup.getvalue()), "origen": "equipo"}
                            st.toast(f"Modelo {mup.name} cargado")
                        except Exception as e:
                            st.error(f"No se pudo abrir el modelo: {e}. ¿Falta instalar la librería con que se entrenó (p. ej. xgboost)?")
                with b:
                    st.write("")
                    if r is not None and st.button("Quitar archivo de resultados", key=f"q_{tipo}"):
                        ss.ranuras[tipo] = None; st.rerun()
                    if ss.modelos.get(tipo) and st.button("Quitar modelo .pkl", key=f"qm_{tipo}"):
                        ss.modelos[tipo] = None; st.rerun()
                m = ss.modelos.get(tipo)
                if m:
                    nombres = getattr(m["obj"], "feature_names_in_", None)
                    st.caption(f"Modelo: **{type(m['obj']).__name__}** · "
                               + (f"{len(nombres)} columnas de entrada: {', '.join(map(str, nombres[:8]))}{'…' if len(nombres) > 8 else ''}"
                                  if nombres is not None else "no guarda los nombres de columnas"))
            if r is None:
                ui.aviso("Sin archivo: la app usará un sustituto automático y lo indicará en cada sección.", "info")
                continue
            _editor_mapeo(tipo, r)

    with tabs[4]:
        st.caption("Agente ya entrenado (archivo .npz que se descarga desde la app). Si no hay, o si cambiaron los datos, "
                   "entrénelo en **🤖 Avanzado · Agente RL → Entrenamiento** y descargue el .npz para la próxima vez.")
        pup = st.file_uploader("Subir agente (.npz)", type=["npz"], key="up_pt")
        if pup is not None and (not ss.get("pt") or ss.pt["nombre"] != pup.name):
            ss.pt = {"nombre": pup.name, "bytes": pup.getvalue(), "origen": "equipo"}
            st.toast("Agente cargado: presione Integrar y procesar")
        if ss.get("pt") and st.button("Quitar agente (entrenar desde la app)"):
            ss.pt = None; st.rerun()

    with tabs[5]:
        _parametros()

    st.divider()
    errs = E.errores_mapeo()
    for e in errs:
        st.error(e)
    if st.button("⚙️ Integrar y procesar", type="primary", disabled=bool(errs), width="stretch"):
        with st.spinner("Uniendo fuentes, construyendo las 25 variables y preparando el agente…"):
            try:
                E.procesar_todo()
                ss.msg_ok = "Listo: datos integrados. Revise el reporte de validación y navegue por las secciones."
                st.rerun()
            except Exception as e:
                st.error(f"No se pudo procesar: {e}. Revise que cada campo tenga la columna correcta "
                         "(o use «↺ Detectar de nuevo») y que las fechas de corte dejen periodos para validar y probar.")
    _reporte_validacion()


def _editor_mapeo(tipo, r):
    df = r["df"]
    st.markdown(f"**{r['nombre']}** · {fmt(len(df))} filas · {df.shape[1]} columnas")
    with st.expander("Vista previa", expanded=False):
        st.dataframe(df.head(50), width="stretch", height=240)
    c1, c2 = st.columns([3, 1])
    c1.markdown("**Asignación de columnas** (detectada automáticamente)")
    if c2.button("↺ Detectar de nuevo", key=f"redetectar_{tipo}_{r['uid']}", help="Vuelve a la asignación automática"):
        import uuid
        r["mapeo"] = detectar(df.columns, tipo); r["uid"] = uuid.uuid4().hex[:8]
        st.rerun()
    opciones = [NO_DISPONIBLE] + list(map(str, df.columns))
    cols = st.columns(3)
    for i, (canon, oblig, desc, _s) in enumerate(ESQUEMAS[tipo]):
        actual = r["mapeo"].get(canon, NO_DISPONIBLE)
        with cols[i % 3]:
            sel = st.selectbox(f"{canon}{' *' if oblig else ''}", opciones,
                               index=opciones.index(actual) if actual in opciones else 0, help=desc,
                               key=f"map_{tipo}_{canon}_{r['uid']}")
            r["mapeo"][canon] = sel
    falt = faltantes_obligatorias(r["mapeo"], tipo)
    from agente_rl.columnas import repetidas
    rep_ = repetidas(r["mapeo"])
    for orig, campos in rep_.items():
        st.warning(f"La columna «{orig}» está asignada a {', '.join(campos)}. Revise: cada campo debería usar una columna distinta.")
    if falt:
        st.error(f"Faltan columnas obligatorias: {', '.join(falt)}")
    else:
        try:
            _validar_rapido(tipo, aplicar_mapeo(df, r["mapeo"]))
        except Exception as e:
            st.warning(f"No se pudo revisar el archivo con esta asignación: {e}")


def _validar_rapido(tipo, d):
    msgs = []
    if tipo == "catalogo":
        f = pd.to_datetime(d["fecha"], errors="coerce")
        msgs.append(("ok" if f.notna().mean() > .95 else "warn", f"Fechas válidas: {f.notna().mean():.0%} · de {f.min():%Y-%m-%d} a {f.max():%Y-%m-%d}"))
        mag = pd.to_numeric(d["magnitud"], errors="coerce")
        msgs.append(("ok" if mag.between(-1, 9.5).mean() > .99 else "warn", f"Magnitudes entre {mag.min():.1f} y {mag.max():.1f}"))
        lat = pd.to_numeric(d["latitud"], errors="coerce"); lon = pd.to_numeric(d["longitud"], errors="coerce")
        dentro = (lat.between(5.5, 8.3) & lon.between(-74.6, -72.3)).mean()
        msgs.append(("ok" if dentro > .5 else "warn", f"{dentro:.0%} de los sismos caen en el recuadro de Santander"))
        if lon.median() > 0:
            msgs.append(("err", "Las longitudes son positivas: en Colombia deben ser negativas (oeste)."))
    elif tipo == "oficial":
        msgs.append(("ok" if d["municipio"].notna().all() else "warn", f"{d['municipio'].nunique()} municipios distintos"))
        lon = pd.to_numeric(d["longitud"], errors="coerce")
        if lon.median() > 0:
            msgs.append(("err", "Las longitudes son positivas: en Colombia deben ser negativas (oeste)."))
        opc = [c for c, o, _d, _s in ESQUEMAS["oficial"] if not o and c not in d.columns and c not in ("provincia", "codigo_dane")]
        if opc:
            msgs.append(("warn", f"Sin {', '.join(opc)}: se usará un valor fijo (supuesto documentado)."))
    elif tipo == "supervisado":
        v = d["nivel_predicho"].map(nivel_a_num)
        msgs.append(("ok" if v.notna().mean() > .95 else "err", f"Clases reconocidas: {v.notna().mean():.0%} (Bajo/Medio/Alto o 0/1/2)"))
        if "periodo" not in d.columns:
            msgs.append(("warn", "Sin periodo: la clase se usará igual en todos los trimestres."))
        if "nivel_real" not in d.columns:
            msgs.append(("warn", "Sin clase real: no se podrá calcular la matriz de confusión."))
    elif tipo == "clusters":
        msgs.append(("ok", f"{d['cluster'].nunique()} grupos · {d['municipio'].nunique()} municipios"))
    if tipo != "catalogo" and "oficial" != tipo and ss.ranuras.get("oficial"):
        ofi = set(aplicar_mapeo(ss.ranuras["oficial"]["df"], ss.ranuras["oficial"]["mapeo"])["municipio"].map(norm_municipio))
        mios = set(d["municipio"].map(norm_municipio))
        nc = sorted(mios - ofi)
        msgs.append(("ok" if not nc else "warn", f"Municipios que cruzan con la fuente oficial: {len(mios & ofi)}/{len(mios)}"
                     + (f" · no cruzan: {', '.join(nc[:8])}{'…' if len(nc) > 8 else ''}" if nc else "")))
    for nivel, m in msgs:
        {"ok": st.success, "warn": st.warning, "err": st.error}[nivel](m, icon={"ok": "✅", "warn": "⚠️", "err": "⛔"}[nivel])


def _parametros():
    cfg = ss.cfg
    st.caption("Estos parámetros definen cómo se arma el panel temporal y el entorno del agente.")
    a, b, c = st.columns(3)
    cfg["periodo"] = a.selectbox("Unidad temporal", [None, "Q", "M"], index=[None, "Q", "M"].index(cfg.get("periodo")),
                                 format_func=lambda x: {None: f"Automática (ahora: {'trimestre' if R['cfg']['periodo'] == 'Q' else 'mes'})",
                                                        "Q": "Trimestre", "M": "Mes"}[x],
                                 help="Automática: trimestres si hay 5 años o más de datos; si no, meses (más pasos para aprender).")
    cfg["magnitud_minima"] = b.number_input("Magnitud mínima (completitud)", 0.0, 4.0, float(cfg["magnitud_minima"]), 0.1)
    cfg["radio_asignacion_km"] = c.number_input("Radio para asignar sismos a un municipio (km)", 5, 150, int(cfg["radio_asignacion_km"]))
    st.markdown("**Cortes temporales** (validación que respeta el tiempo)")
    a, b = st.columns(2)
    actual_tr = pd.Timestamp(R["cfg"]["fin_entrenamiento"]).date(); actual_va = pd.Timestamp(R["cfg"]["fin_validacion"]).date()
    auto = a.toggle("Calcular cortes automáticamente (≈ 75 % / 12 % / 13 %)", value=cfg.get("fin_entrenamiento") is None)
    if auto:
        cfg["fin_entrenamiento"] = None; cfg["fin_validacion"] = None
        b.caption(f"Actual: entrenamiento hasta {actual_tr}, validación hasta {actual_va}, prueba después.")
    else:
        cfg["fin_entrenamiento"] = str(a.date_input("Fin de entrenamiento", actual_tr))
        cfg["fin_validacion"] = str(b.date_input("Fin de validación", actual_va))
    st.markdown("**Entorno del agente**")
    a, b, c = st.columns(3)
    ce = cfg["entorno"]
    auto_p = a.toggle("Presupuesto automático (≈ 1 de cada 8 municipios)", value=ce.get("presupuesto_por_periodo") is None)
    ce["presupuesto_por_periodo"] = None if auto_p else a.number_input(
        "Presupuesto por periodo", 10.0, 5000.0, float(R["cfg"]["entorno"]["presupuesto_por_periodo"]), 5.0)
    ce["peso_equidad"] = b.slider("Peso de equidad (premia municipios descuidados)", 0.0, 0.5, float(ce["peso_equidad"]), 0.05)
    ce["peso_costo"] = c.slider("Peso del costo", 0.0, 0.3, float(ce["peso_costo"]), 0.01)
    st.caption("Cambiar pesos o cortes requiere reentrenar el agente para que los aprenda.")


def _reporte_validacion():
    st.markdown("### Reporte de integración y validación")
    c = st.columns(5)
    c[0].metric("Sismos usados", fmt(rep["sismos_usados"]), f"de {fmt(rep['sismos_total'])}", delta_color="off")
    c[1].metric("Municipios que cruzan", f"{rep['cruzan']}/{rep['municipios_oficial']}")
    c[2].metric("Fuera del radio", fmt(rep["sismos_fuera_radio"]))
    c[3].metric("Periodos", rep["n_periodos"], f"{rep['periodo_inicio']} → {rep['periodo_fin']}", delta_color="off")
    c[4].metric("Filas del panel", fmt(rep["filas_panel"]))
    a, b = st.columns(2)
    with a:
        st.markdown(f"**Reto Básico** {ui.badge_fuente(rep['supervisado']['fuente'])}<br><span style='color:#AEB6C5'>{rep['supervisado']['modo']}</span>", unsafe_allow_html=True)
        if rep["supervisado"].get("aviso"):
            st.warning(rep["supervisado"]["aviso"])
        if rep["supervisado"].get("motivo_modelo"):
            st.warning(f"El modelo .pkl no se pudo aplicar: {rep['supervisado']['motivo_modelo']}")
    with b:
        st.markdown(f"**Reto Medio** {ui.badge_fuente(rep['no_supervisado']['fuente'])}<br><span style='color:#AEB6C5'>{rep['no_supervisado']['modo']}</span>", unsafe_allow_html=True)
        if rep["no_supervisado"].get("sin_grupo"):
            st.warning(f"Municipios sin grupo: {', '.join(rep['no_supervisado']['sin_grupo'][:10])}")
        if rep["no_supervisado"].get("motivo_modelo"):
            st.warning(f"El modelo .pkl no se pudo aplicar: {rep['no_supervisado']['motivo_modelo']}")
    st.markdown("**Agente**")
    if HAY_AGENTE:
        inf = ss.agente["info"]
        if inf.get("mismos_datos"):
            st.success(f"Agente {inf.get('nombre', '')} entrenado con estos mismos datos ✔")
        else:
            st.warning("El agente cargado se entrenó con otros datos o cortes. Puede usarse, pero lo correcto es "
                       "reentrenarlo en 🤖 Avanzado · Agente RL → Entrenamiento.")
    elif ss.get("agente_error"):
        st.error(f"No se pudo cargar el agente: {ss.agente_error}")
    else:
        st.info("No hay agente cargado: entrénelo en 🤖 Avanzado · Agente RL → Entrenamiento.")
    for s in rep["supuestos"]:
        st.warning(s, icon="📌")
    for s in rep["avisos"]:
        st.info(s)
    if rep["solo_catalogo"]:
        st.warning(f"Municipios del catálogo que no están en la fuente oficial (sus sismos se descartan): {', '.join(rep['solo_catalogo'][:12])}")
    if rep["solo_oficial"]:
        st.info(f"Municipios sin sismos asignados (se conservan con actividad 0): {', '.join(rep['solo_oficial'][:12])}")


# ============================================================================ 🏠 INICIO
def pagina_inicio():
    st.markdown("""<div class="hero"><h1>¿Dónde actuar primero ante el riesgo sísmico en Santander?</h1>
      <p>Unimos la sismicidad del Servicio Geológico Colombiano con información social y territorial oficial,
      clasificamos la actividad, agrupamos los municipios y entrenamos un agente por refuerzo que, con un
      presupuesto limitado, decide a quién priorizar y <b>dice qué tan seguro está</b>.</p>
      <span class="chip" style="color:#3FB68B">Básico · Clasificación</span>
      <span class="chip" style="color:#4C9AFF">Medio · Agrupación territorial</span>
      <span class="chip" style="color:#FF7A45">Avanzado · Aprendizaje por refuerzo</span></div>""", unsafe_allow_html=True)
    aviso_origen()
    c = st.columns(5)
    c[0].metric("Sismos analizados", fmt(rep["sismos_usados"]))
    c[1].metric("Municipios", len(D["municipios"]))
    c[2].metric("Variables", len(FEATURES))
    c[3].metric("Trimestres" if CFG["periodo"] == "Q" else "Meses", len(D["periodos"]))
    c[4].metric("Magnitud máxima", f"{EVT['magnitud'].max():.1f}")

    st.markdown("### Flujo integrado")
    pasos = [("Datos", f"Catálogo SGC + fuente oficial unidos por municipio ({rep['cruzan']} municipios).", "archivo"),
             ("Básico", "Modelo supervisado: nivel de actividad bajo, medio o alto por municipio y trimestre.", rep["supervisado"]["fuente"]),
             ("Medio", "Modelo no supervisado: grupos de municipios con comportamiento parecido.", rep["no_supervisado"]["fuente"]),
             ("Agente RL", "Usa las 25 variables (incluidas las salidas de Básico y Medio) para decidir a quién priorizar.",
              "archivo" if HAY_AGENTE else "sustituto"),
             ("Decisión", "Ranking con presupuesto, semáforo de confianza y razones de cada elección.", "archivo" if HAY_AGENTE else "sustituto")]
    cols = st.columns(5)
    for i, (col, (t, d, f)) in enumerate(zip(cols, pasos), 1):
        with col:
            ui.tarjeta(t, d + f"<br><br>{ui.badge_fuente(f) if i in (2, 3) else ''}", paso=f"PASO {i}")

    st.markdown("### Sismicidad registrada")
    izq, der = st.columns([3, 2])
    with izq:
        m = EVT.sample(min(2000, len(EVT)), random_state=1).assign(municipio="Sismo")
        fig = ui.mapa(m, color="profundidad", tamano=np.clip(m["magnitud"] - 0.5, .3, None),
                      hover={"magnitud": True, "profundidad": True, "fecha": True, "lat": False, "lon": False}, escala="Inferno", alto=470)
        fig.update_traces(marker=dict(opacity=.6))
        st.plotly_chart(fig, width="stretch")
    with der:
        ui.tarjeta("El Nido Sísmico de Bucaramanga", "Gran parte de los sismos se concentran bajo la Mesa de Los Santos, a "
                   "unos 140–160 km de profundidad. Por eso no basta con contar sismos: la profundidad, la exposición y "
                   "la vulnerabilidad cambian el riesgo real de cada municipio.")
        st.write("")
        if HAY_AGENTE:
            tab, _p = E.priorizacion(ultimo_periodo(), CFG["entorno"]["presupuesto_por_periodo"])
            top = tab[tab["priorizado"]].sort_values("orden")[["orden", "municipio", "confianza"]]
            top["confianza"] = top["confianza"].map(ui.ICONO_CONF)
            st.markdown(f"**Priorización del agente · {D['periodos'][-1]}**")
            st.dataframe(top, hide_index=True, width="stretch")
        else:
            st.info("Entrene o cargue el agente para ver la priorización.")


# ============================================================================ 📊 DATOS
def pagina_datos():
    st.title("📊 Datos e integración de fuentes")
    aviso_origen()
    t1, t2, t3, t4, t5 = st.tabs(["Catálogo sísmico", "Las 25 variables", "Cómo se unieron", "Matriz de correlación", "Panel integrado"])
    with t1:
        f1, f2 = st.columns(2)
        mlo, mhi = float(EVT["magnitud"].min()), float(EVT["magnitud"].max())
        rm = f1.slider("Magnitud", mlo, mhi, (mlo, mhi), 0.1)
        an = EVT["fecha"].dt.year
        ra = f2.slider("Años", int(an.min()), int(an.max()), (int(an.min()), int(an.max())))
        f = EVT[EVT["magnitud"].between(*rm) & an.between(*ra)]
        c = st.columns(4)
        c[0].metric("Eventos", fmt(len(f))); c[1].metric("Magnitud media", f"{f['magnitud'].mean():.2f}")
        c[2].metric("Profundidad media", f"{f['profundidad'].mean():.0f} km"); c[3].metric("Eventos M ≥ 4", int((f["magnitud"] >= 4).sum()))
        a, b = st.columns(2)
        with a:
            st.plotly_chart(ui.estilo_fig(px.histogram(f, x="magnitud", nbins=40, title="Distribución de magnitudes (Gutenberg-Richter)",
                                                       color_discrete_sequence=[ui.ACENTO], log_y=True)), width="stretch")
        with b:
            fig = px.scatter(f.sample(min(4000, len(f)), random_state=0), x="magnitud", y="profundidad", opacity=.35,
                             title="Magnitud vs. profundidad", color_discrete_sequence=[ui.AZUL])
            fig.update_yaxes(autorange="reversed", title="Profundidad (km)")
            st.plotly_chart(ui.estilo_fig(fig), width="stretch")
        pm = f.set_index("fecha").resample("ME").size().rename("eventos").reset_index()
        st.plotly_chart(ui.estilo_fig(px.area(pm, x="fecha", y="eventos", title="Eventos por mes", color_discrete_sequence=[ui.ACENTO]), 290), width="stretch")
    with t2:
        tv = tabla_variables()
        a, b = st.columns([1, 2])
        with a:
            cg = tv.groupby("grupo").size().reset_index(name="variables")
            fig = px.pie(cg, names="grupo", values="variables", hole=.6, title="Variables por grupo",
                         color="grupo", color_discrete_map=ui.COLOR_GRUPO_VAR)
            st.plotly_chart(ui.estilo_fig(fig, 360), width="stretch")
            st.dataframe(tv.groupby("fuente").size().reset_index(name="n"), hide_index=True, width="stretch")
        with b:
            st.dataframe(tv, hide_index=True, width="stretch", height=560,
                         column_config={"justificacion": st.column_config.TextColumn("¿Por qué se usa?", width="large")})
    with t3:
        c = st.columns(4)
        with c[0]: ui.tarjeta("1 · Catálogo SGC", f"{fmt(rep['sismos_total'])} sismos con fecha, coordenadas, profundidad y magnitud.", "ENTRADA")
        with c[1]: ui.tarjeta("2 · Asignación", rep["metodo_asignacion"] + f". Fuera del radio: {fmt(rep['sismos_fuera_radio'])}.", "ESPACIAL")
        with c[2]: ui.tarjeta("3 · Agregación", f"Por municipio y {'trimestre' if CFG['periodo'] == 'Q' else 'mes'}: conteos, energía, profundidad, valor b…", "PANEL")
        with c[3]: ui.tarjeta("4 · Unión oficial", f"Cruce por nombre normalizado: {rep['cruzan']} de {rep['municipios_oficial']} municipios.", "MERGE")
        st.write("")
        _reporte_validacion()
    with t4:
        corr = panel[FEATURES].corr().round(2)
        fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                        title="Correlación entre las 25 variables (rojo = se mueven juntas)")
        st.plotly_chart(ui.estilo_fig(fig, 640), width="stretch")
        alta = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack()
        alta = alta[alta.abs() > .85].sort_values(key=abs, ascending=False)
        if len(alta):
            st.caption("Pares muy correlacionados (|r| > 0,85): " + " · ".join(f"{a}–{b} ({v:+.2f})" for (a, b), v in alta.head(8).items()))
    with t5:
        vista = panel[["municipio", "periodo", "grupo"] + FEATURES + ["impacto_siguiente"]].assign(periodo=lambda d: d["periodo"].astype(str))
        st.dataframe(vista, hide_index=True, width="stretch", height=520)
        st.download_button("⬇️ Descargar panel integrado (CSV)", vista.to_csv(index=False).encode("utf-8"), "panel_integrado.csv", "text/csv")


# ============================================================================ 🎯 BÁSICO
def _datos_supervisado():
    """Devuelve (df con municipio, periodo?, real?, pred, probas?, conjunto?), origen."""
    sup = E.df_mapeado("supervisado")
    if sup is not None:
        return sup, "archivo"
    m = ss.modelos.get("supervisado")
    if m:
        pred, proba, motivo = aplicar_modelo(m["obj"], panel)
        if pred is not None:
            d = pd.DataFrame({"municipio": panel["municipio"].values, "periodo": panel["periodo"].astype(str).values,
                              "nivel_predicho": pred})
            if proba is not None and proba.shape[1] == 3:
                d[["prob_bajo", "prob_medio", "prob_alto"]] = proba
            return d, "modelo"
    d = pd.DataFrame({"municipio": panel["municipio"].values, "periodo": panel["periodo"].astype(str).values,
                      "nivel_predicho": panel["nivel_actividad"].values})
    return d, "sustituto"


def pagina_basico():
    st.title("🎯 Reto Básico · Nivel de actividad sísmica")
    sup, origen = _datos_supervisado()
    m = ss.modelos.get("supervisado")
    st.markdown(f"Origen de los resultados: {ui.badge_fuente(origen)}"
                + (f" · Modelo: <b>{type(m['obj']).__name__}</b>" if m else ""), unsafe_allow_html=True)
    aviso_origen()
    sup = sup.copy()
    sup["pred"] = sup["nivel_predicho"].map(nivel_a_num)
    tiene_real = "nivel_real" in sup.columns
    if tiene_real:
        sup["real"] = sup["nivel_real"].map(nivel_a_num)

    t1, t2, t3, t4 = st.tabs(["Desempeño", "Probar el modelo", "Mapa y evolución", "Datos"])
    with t1:
        if not tiene_real:
            st.info("El archivo no trae la clase real (`nivel_real`), así que no se pueden calcular métricas ni la "
                    "matriz de confusión. Pida a su compañero esa columna (idealmente con `conjunto` = train/val/test).")
        else:
            opciones = ["test", "val", "train", "todos"] if "conjunto" in sup.columns else ["todos"]
            conj = st.radio("Conjunto evaluado", opciones, horizontal=True,
                            format_func=lambda x: {"test": "Prueba (futuro)", "val": "Validación", "train": "Entrenamiento", "todos": "Todos"}[x])
            d = sup if conj == "todos" else sup[sup["conjunto"].astype(str).str.lower().str.startswith(conj[:2])]
            d = d.dropna(subset=["real", "pred"])
            if d.empty:
                st.warning("No hay filas en ese conjunto."); return
            y, yp = d["real"].astype(int), d["pred"].astype(int)
            c = st.columns(4)
            c[0].metric("Accuracy", f"{accuracy(y, yp):.2f}")
            c[1].metric("F1 macro", f"{f1_macro(y, yp):.2f}")
            pr, rc, f1, sp = prf(y, yp)
            c[2].metric("Precisión macro", f"{pr.mean():.2f}"); c[3].metric("Recall macro", f"{rc.mean():.2f}")
            a, b = st.columns(2)
            with a:
                cm = pd.crosstab(y, yp).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0).values
                st.plotly_chart(ui.fig_matriz(cm, CLASES, f"Matriz de confusión · {len(d):,} casos".replace(",", ".")), width="stretch")
            with b:
                st.markdown("**Reporte por clase**")
                st.dataframe(pd.DataFrame({"clase": CLASES, "precisión": pr.round(2), "recall": rc.round(2), "f1": f1.round(2), "soporte": sp}),
                             hide_index=True, width="stretch")
                if "conjunto" in sup.columns:
                    filas = []
                    for cj, g in sup.dropna(subset=["real", "pred"]).groupby(sup["conjunto"].astype(str)):
                        filas.append({"conjunto": cj, "accuracy": accuracy(g["real"].astype(int), g["pred"].astype(int)),
                                      "f1 macro": f1_macro(g["real"].astype(int), g["pred"].astype(int))})
                    st.markdown("**Desempeño por tramo temporal**")
                    st.dataframe(pd.DataFrame(filas).round(3), hide_index=True, width="stretch")
                    st.caption("Si cae mucho de entrenamiento a prueba, el modelo está sobreajustado.")
        if m:
            imp = None
            obj = m["obj"]; est = obj[-1] if hasattr(obj, "steps") else obj
            nombres = getattr(obj, "feature_names_in_", None)
            if hasattr(est, "feature_importances_") and nombres is not None:
                imp = pd.Series(est.feature_importances_, index=nombres)
            elif hasattr(est, "coef_") and nombres is not None:
                imp = pd.Series(np.abs(est.coef_).mean(0), index=nombres)
            if imp is not None:
                imp = (imp / imp.sum()).sort_values()
                fig = px.bar(imp.tail(15), orientation="h", title="Importancia de variables del modelo supervisado",
                             color_discrete_sequence=[ui.ACENTO])
                fig.update_layout(showlegend=False, yaxis_title=None, xaxis_title=None); fig.update_xaxes(tickformat=".0%")
                st.plotly_chart(ui.estilo_fig(fig, 420), width="stretch")

    with t2:
        if not m or getattr(m["obj"], "feature_names_in_", None) is None:
            st.info("Para probar el modelo en vivo, suba el archivo .pkl entrenado con un DataFrame (que guarde los "
                    "nombres de columnas) en 📥 Cargar datos → Supervisado.")
        else:
            nombres = list(m["obj"].feature_names_in_)
            st.markdown("Ingrese valores (por defecto, la mediana de Santander) y el modelo devuelve el nivel de actividad.")
            mun = st.selectbox("Partir de un municipio (último periodo)", ["— mediana —"] + D["municipios"])
            base = panel[panel["periodo"] == panel["periodo"].max()]
            ref = base.set_index("municipio").loc[mun] if mun != "— mediana —" else base.median(numeric_only=True)
            izq, der = st.columns([3, 2])
            vals = {}
            with izq:
                cols = st.columns(3)
                for i, n in enumerate(nombres):
                    v = float(ref[n]) if n in ref.index and pd.notna(ref[n]) else 0.0
                    vals[n] = cols[i % 3].number_input(n, value=round(v, 3), format="%.3f", key=f"in_{n}_{mun}")
            pred, proba, _ = aplicar_modelo(m["obj"], pd.DataFrame([vals]))
            with der:
                k = nivel_a_num(pred[0]); clase = CLASES[k] if k is not None else str(pred[0]); col = ui.COLOR_NIVEL.get(clase, ui.ACENTO)
                conf = f"{proba.max():.0%}" if proba is not None else "—"
                st.markdown(f'<div class="resultado" style="background:{col}22;border:1px solid {col}"><div style="color:#AEB6C5">Nivel de actividad estimado</div>'
                            f'<h2 style="color:{col}">{clase.upper()}</h2><div style="color:#AEB6C5">Confianza: {conf}</div></div>', unsafe_allow_html=True)
                if proba is not None and proba.shape[1] == 3:
                    fig = go.Figure(go.Bar(x=proba[0], y=CLASES, orientation="h", marker_color=[ui.COLOR_NIVEL[c] for c in CLASES],
                                           text=[f"{p:.0%}" for p in proba[0]], textposition="auto"))
                    fig.update_xaxes(range=[0, 1], tickformat=".0%")
                    st.plotly_chart(ui.estilo_fig(fig, 220), width="stretch")

    with t3:
        sup["clase"] = sup["pred"].map(lambda k: CLASES[int(k)] if pd.notna(k) else "Sin dato")
        if "periodo" in sup.columns:
            ev = sup.groupby(["periodo", "clase"]).size().reset_index(name="municipios")
            fig = px.area(ev, x="periodo", y="municipios", color="clase", color_discrete_map=ui.COLOR_NIVEL,
                          category_orders={"clase": CLASES}, title="Municipios por nivel de actividad en el tiempo")
            st.plotly_chart(ui.estilo_fig(fig, 330), width="stretch")
            per = st.select_slider("Periodo del mapa", sorted(sup["periodo"].astype(str).unique()), value=sorted(sup["periodo"].astype(str).unique())[-1])
            ult = sup[sup["periodo"].astype(str) == per]
        else:
            ult = sup
        ult = ult.assign(municipio=ult["municipio"].map(norm_municipio)).drop_duplicates("municipio").merge(MUN, on="municipio")
        fig = ui.mapa(ult, color="clase", tamano=np.log10(ult["poblacion"]) - 2.5, mapa_colores=ui.COLOR_NIVEL,
                      hover={"lat": False, "lon": False, "provincia": True})
        st.plotly_chart(fig, width="stretch")
    with t4:
        st.dataframe(sup.drop(columns=["pred"] + (["real"] if tiene_real else []) + (["clase"] if "clase" in sup else []), errors="ignore"),
                     hide_index=True, width="stretch", height=480)


# ============================================================================ 🧭 MEDIO
def pagina_medio():
    st.title("🧭 Reto Medio · Patrones territoriales")
    info = rep["no_supervisado"]
    m = ss.modelos.get("clusters")
    st.markdown(f"Origen de los grupos: {ui.badge_fuente(info['fuente'])}" + (f" · Modelo: <b>{type(m['obj']).__name__}</b>" if m else ""), unsafe_allow_html=True)
    aviso_origen()
    prom = panel.groupby("municipio")[FEATURES].mean()
    mun = MUN.set_index("municipio").join(prom, how="inner").reset_index()
    grupos = sorted(mun["grupo"].unique())
    colores = dict(zip(grupos, ui.PALETA))
    cols = st.columns(min(len(grupos), 6))
    for col, g in zip(cols, grupos):
        col.metric(g, (mun["grupo"] == g).sum(), help="Número de municipios del grupo")

    t1, t2, t3, t4 = st.tabs(["Mapa de grupos", "Perfil de cada grupo", "Espacio y calidad", "Comparar municipio"])
    with t1:
        st.plotly_chart(ui.mapa(mun, color="grupo", tamano=np.log10(mun["poblacion"]) - 2.5, mapa_colores=colores, alto=540,
                                hover={"provincia": True, "lat": False, "lon": False}), width="stretch")
    with t2:
        vp = st.multiselect("Variables del perfil", FEATURES, default=["n_sismos_4p", "energia_log_4p", "pct_superficiales", "dist_nido_km",
                                                                    "log_poblacion", "nbi", "viviendas_precarias", "hospitales_100k"])
        if vp:
            z = (mun[vp] - mun[vp].mean()) / mun[vp].std().replace(0, 1)
            perfil = z.assign(grupo=mun["grupo"]).groupby("grupo").mean()
            fig = px.imshow(perfil.round(2), text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", zmin=-1.5, zmax=1.5,
                            title="Promedio estandarizado por grupo (rojo = por encima del promedio de Santander)")
            st.plotly_chart(ui.estilo_fig(fig, 340), width="stretch")
            fig = go.Figure()
            for g in grupos:
                fig.add_trace(go.Scatterpolar(r=perfil.loc[g, vp].values, theta=vp, fill="toself", name=g.split(" · ")[0],
                                              line_color=colores[g], opacity=.7))
            fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(gridcolor="rgba(255,255,255,.1)")), title="Huella de cada grupo")
            st.plotly_chart(ui.estilo_fig(fig, 430), width="stretch")
    with t3:
        a, b = st.columns([3, 2])
        vars_z = ["energia_log_4p", "pct_superficiales", "dist_nido_km", "nbi", "densidad_pob", "viviendas_precarias", "n_sismos_4p", "hospitales_100k"]
        Z = ((mun[vars_z] - mun[vars_z].mean()) / mun[vars_z].std().replace(0, 1)).fillna(0).values
        with a:
            if {"PC1", "PC2"} <= set(mun.columns) and mun["PC1"].notna().all():
                pcs, titulo = mun[["PC1", "PC2"]].values, "Proyección PCA (del archivo del equipo)"
            else:
                pcs, ve = pca(Z, 2)
                titulo = f"Proyección PCA calculada por la app (PC1 {ve[0]:.0%} + PC2 {ve[1]:.0%})"
            dfp = mun.assign(PC1=pcs[:, 0], PC2=pcs[:, 1])
            fig = px.scatter(dfp, x="PC1", y="PC2", color="grupo", text="municipio", color_discrete_map=colores, title=titulo)
            fig.update_traces(textposition="top center", textfont_size=9, marker_size=11)
            fig.update_layout(legend=dict(orientation="h", y=-0.18, title=None))
            st.plotly_chart(ui.estilo_fig(fig, 560), width="stretch")
        with b:
            if mun["grupo"].nunique() > 1:
                sil = silueta(Z, mun["grupo"].values)
                st.metric("Silueta de los grupos actuales", f"{sil:.2f}", help="De −1 a 1: más alto = grupos más separados. Calculada sobre 8 variables estandarizadas.")
            ks = list(range(2, min(9, len(mun) - 1)))
            res = []
            for k in ks:
                lab, inercia = kmeans(Z, k)
                res.append((k, inercia, silueta(Z, lab)))
            codo = pd.DataFrame(res, columns=["k", "inercia", "silueta"])
            fig = go.Figure(go.Scatter(x=codo["k"], y=codo["silueta"], mode="lines+markers", line_color=ui.ACENTO))
            fig.add_vline(x=codo.loc[codo["silueta"].idxmax(), "k"], line_dash="dot", line_color="#AEB6C5")
            fig.update_layout(title="Referencia: silueta según k (K-Means)", xaxis_title="k")
            st.plotly_chart(ui.estilo_fig(fig, 230), width="stretch")
            st.plotly_chart(ui.estilo_fig(px.line(codo, x="k", y="inercia", markers=True, title="Método del codo", color_discrete_sequence=[ui.AZUL]), 230), width="stretch")
    with t4:
        nombre = st.selectbox("Municipio", sorted(mun["municipio"]))
        f = mun.set_index("municipio").loc[nombre]
        st.markdown(f"**{nombre.title()}** pertenece a **{f['grupo']}** · Provincia {f['provincia']}")
        cv = ["n_sismos_4p", "energia_log_4p", "pct_superficiales", "nbi", "densidad_pob", "hospitales_100k", "dist_nido_km"]
        tabla = pd.DataFrame({"Municipio": f[cv].astype(float), "Promedio de su grupo": mun[mun["grupo"] == f["grupo"]][cv].mean(),
                              "Promedio Santander": mun[cv].mean()}).round(2)
        st.dataframe(tabla, width="stretch")


# ============================================================================ 🤖 AGENTE
def pagina_agente():
    st.title("🤖 Reto Avanzado · Priorización con aprendizaje por refuerzo")
    st.caption("Double DQN con ensamble de 5 redes (comité de expertos) · implementado en NumPy, sin GPU")
    aviso_origen()
    t1, t2, t3, t4 = st.tabs(["Diseño del entorno", "Entrenamiento", "Priorización", "Evaluación"])
    ce = E.cfg_entorno()
    with t1:
        st.markdown("#### Sala de Crisis Santander: el problema como un juego de decisiones")
        st.markdown("Cada trimestre hay un **presupuesto limitado**. El agente elige municipios uno por uno (inspección, "
                    "simulacro, refuerzo) hasta agotarlo; luego avanza el tiempo y descubre qué pasó. Aprende a anticipar "
                    "dónde el riesgo será mayor sin abandonar a ningún territorio.")
        c = st.columns(4)
        with c[0]: ui.tarjeta("🧩 Estado", f"25 variables de cada uno de los {len(D['municipios'])} municipios (incluye la clase del reto Básico y el grupo del reto Medio) + descuido, ya elegido, alcanzable, presupuesto restante y avance del año.", "S")
        with c[1]: ui.tarjeta("🎮 Acción", "Elegir un municipio. Los ya elegidos y los que no caben en el presupuesto se enmascaran.", "A")
        with c[2]: ui.tarjeta("🏆 Recompensa", f"Acierto sobre el riesgo real del trimestre siguiente + {ce['peso_equidad']} × equidad − {ce['peso_costo']} × costo.", "R")
        with c[3]: ui.tarjeta("🏁 Episodio", f"{ce['periodos_por_episodio']} {'trimestres' if CFG['periodo'] == 'Q' else 'meses'} seguidos con presupuesto {ce['presupuesto_por_periodo']:.0f} por periodo.", "FIN")
        st.write("")
        a, b = st.columns([3, 2])
        with a:
            st.markdown("**Recompensa por cada municipio elegido**")
            st.latex(r"r = \frac{\text{impacto}_{t+1}(i)}{\text{impacto}_{t+1}(\text{oráculo})} + w_{eq}\,\frac{\text{descuido}_i}{8} - w_c\,\frac{c_i}{P}")
            st.latex(r"\text{impacto} = \underbrace{\log(1+\text{peligro})}_{\text{SGC: superficiales pesan más}}\times\underbrace{\text{exposición}}_{\text{población}}\times\underbrace{\text{vulnerabilidad}}_{\text{NBI, vivienda}}")
            st.caption("El agente nunca ve el futuro: el impacto del trimestre siguiente solo llega como recompensa después de actuar.")
        with b:
            if HAY_AGENTE:
                cfg_a = ss.agente["obj"].cfg
                hp = {"Algoritmo": "Bootstrapped Double DQN", "Expertos (cabezas)": cfg_a["cabezas_ensamble"],
                      "Red por municipio": f"{2 * (len(FEATURES) + 5)} → {' → '.join(map(str, cfg_a['capas_ocultas']))} → 1 (ReLU)",
                      "γ (descuento)": cfg_a["gamma"], "Tasa de aprendizaje": cfg_a["tasa_aprendizaje"],
                      "Memoria de repetición": cfg_a["memoria_replay"], "Implementación": "NumPy (Adam + Huber, a mano)"}
                st.dataframe(pd.DataFrame(hp.items(), columns=["Parámetro", "Valor"]).astype(str), hide_index=True, width="stretch")

    with t2:
        _entrenamiento(ce)
    if not HAY_AGENTE:
        for t in (t3, t4):
            with t:
                st.info("Primero cargue un agente .npz o entrénelo en la pestaña Entrenamiento.")
        return
    with t3:
        a, b = st.columns(2)
        t = a.select_slider("Periodo a priorizar", options=list(range(len(D["periodos"]))), value=ultimo_periodo(),
                            format_func=lambda i: D["periodos"][i])
        cmin = float(D["costo"][t].min()); pdef = float(ce["presupuesto_por_periodo"])
        pres = b.slider("Presupuesto del trimestre", round(cmin, 0), round(pdef * 3, 0), round(pdef, 0), 5.0,
                        help="Costo relativo; cada municipio cuesta según su área y ruralidad")
        tab, _ = E.priorizacion(t, pres)
        sel = tab[tab["priorizado"]].sort_values("orden")
        c = st.columns(4)
        c[0].metric("Municipios priorizados", len(sel)); c[1].metric("Presupuesto usado", f"{sel['costo'].sum():.0f} / {pres:.0f}")
        c[2].metric("Con confianza alta", int((sel["confianza"] == "Alta").sum()))
        c[3].metric("Con confianza baja", int((sel["confianza"] == "Baja").sum()), help="Revisar en campo antes de invertir")
        izq, der = st.columns([3, 2])
        with izq:
            st.plotly_chart(ui.fig_mapa_prioridad(tab, MUN), width="stretch")
        with der:
            v = sel[["orden", "municipio", "confianza", "costo", "razones"]].copy()
            v["confianza"] = v["confianza"].map(ui.ICONO_CONF)
            st.markdown(f"**Decisión del agente · {D['periodos'][t]}**")
            st.dataframe(v, hide_index=True, width="stretch",
                         column_config={"costo": st.column_config.NumberColumn(format="%.1f"),
                                        "razones": st.column_config.TextColumn("Variables que más pesaron", width="large")})
        st.plotly_chart(ui.fig_q_incertidumbre(tab), width="stretch")
        with st.expander("Ranking completo de municipios"):
            tv = tab.copy(); tv["confianza"] = tv["confianza"].map(ui.ICONO_CONF)
            st.dataframe(tv.drop(columns=["impacto_real_siguiente"]), hide_index=True, width="stretch")
        st.caption("🟢🟡🔴 Confianza = acuerdo entre las 5 redes del comité. Las flechas ↑↓ indican si la variable sube o baja la prioridad.")

    with t4:
        with st.spinner("Evaluando en el periodo de prueba…"):
            ev = E.evaluacion()
        tab = ev["tabla"]; ag = tab.set_index("politica").loc["Agente RL (ensamble)"]
        mejor_base = tab[~tab["politica"].str.startswith("Agente")].sort_values("captura_riesgo").iloc[-1]
        c = st.columns(4)
        c[0].metric("Riesgo capturado (prueba)", f"{ag['captura_riesgo']:.0%}", f"{ag['captura_riesgo'] - mejor_base['captura_riesgo']:+.0%} vs. mejor regla simple")
        c[1].metric("IC 95 % (bootstrap)", f"{ev['ic95'][0]:.0%} – {ev['ic95'][1]:.0%}")
        c[2].metric("Cobertura territorial", f"{ag['cobertura']:.0%}", f"{ag['cobertura'] - mejor_base['cobertura']:+.0%} vs. {mejor_base['politica']}", delta_color="normal")
        c[3].metric("Captura en validación", f"{ev['captura_val']:.0%}")
        a, b = st.columns(2)
        with a:
            st.plotly_chart(ui.fig_politicas(tab), width="stretch")
        with b:
            s = ev["series"]
            fig = px.line(s, x="periodo", y="captura", color="politica", markers=True, title="Riesgo capturado trimestre a trimestre (prueba)",
                          color_discrete_map={"Agente RL (ensamble)": ui.ACENTO, "Aleatoria": "#5A6475"},
                          color_discrete_sequence=[ui.AZUL, "#2EC4B6", "#A78BFA"])
            fig.update_traces(line=dict(width=1.5)); fig.update_traces(selector=dict(name="Agente RL (ensamble)"), line=dict(width=3.5))
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(ui.estilo_fig(fig, 360), width="stretch")
        st.dataframe(tab.rename(columns={"politica": "política", "captura_riesgo": "riesgo capturado", "captura_std": "desv.",
                                         "precision_top": "precisión top", "max_descuido": "máx. descuido"}).round(3),
                     hide_index=True, width="stretch")
        st.plotly_chart(ui.fig_importancia(ev["importancia"]), width="stretch")
        st.info("Lectura honesta: si el agente no supera a las reglas simples en riesgo capturado, su aporte debe estar en "
                "otra dimensión (equidad/cobertura, confianza). Esta comparación es la evidencia de lo que aprendió.")


def _entrenamiento(ce):
    if HAY_AGENTE:
        inf = ss.agente["info"]
        c = st.columns(4)
        c[0].metric("Agente activo", inf["origen"])
        c[1].metric("Mejor episodio", inf.get("episodio_mejor") or "—")
        c[2].metric("Captura en validación", f"{inf['captura_val']:.0%}" if inf.get("captura_val") is not None else "—")
        c[3].metric("¿Mismos datos?", "Sí" if inf.get("mismos_datos") or inf["origen"] == "entrenado en la app" else "No")
        if not (inf.get("mismos_datos") or inf["origen"] == "entrenado en la app"):
            st.warning("Este agente se entrenó con otros datos o cortes. Reentrénelo para que la priorización y la evaluación sean válidas.")
        if isinstance(inf.get("curva"), pd.DataFrame) and len(inf["curva"]):
            a, b = st.columns(2)
            a.plotly_chart(ui.fig_curva(inf["curva"]), width="stretch")
            b.plotly_chart(ui.fig_validacion(inf["curva"]), width="stretch")
        st.download_button("⬇️ Descargar agente (.npz)", E.servicio.exportar(ss.agente["obj"], inf), "agente_sismico.npz",
                           help="Guárdelo y súbalo en 📥 Cargar datos la próxima vez para no reentrenar.")
    st.markdown("#### Entrenar con los datos cargados")
    a, b, c, d = st.columns(4)
    ep = a.slider("Episodios", 50, 1500, 300, 50, help="Aproximadamente 0,5 a 1 s por episodio con 35 municipios.")
    k = b.slider("Expertos del comité", 3, 9, 5)
    weq = c.slider("Peso equidad", 0.0, 0.5, float(ce["peso_equidad"]), 0.05)
    wc = d.slider("Peso costo", 0.0, 0.3, float(ce["peso_costo"]), 0.01)
    if st.button("🚀 Entrenar agente", type="primary"):
        barra = st.progress(0.0, text="Preparando…")
        g1, g2 = st.columns(2); ph1, ph2 = g1.empty(), g2.empty()
        hist = []

        def cb(e, tot, fila, mejor, seg):
            hist.append(fila)
            barra.progress(e / tot, text=f"Episodio {e}/{tot} · mejor captura en validación {mejor:.0%} · {seg:.0f} s")
            if e % max(1, tot // 25) == 0 or e == tot:
                cdf = pd.DataFrame(hist)
                ph1.plotly_chart(ui.fig_curva(cdf, 300), width="stretch", key=f"c1_{e}")
                ph2.plotly_chart(ui.fig_validacion(cdf, 300), width="stretch", key=f"c2_{e}")
        cen = {**ce, "peso_equidad": weq, "peso_costo": wc}
        ag, info = E.servicio.entrenar(D, cen, {"episodios": ep, "cabezas_ensamble": k}, CFG.get("semilla", 42), cb, R["huella"])
        info["nombre"] = "entrenado en la app"
        ss.R["cfg"]["entorno"].update(peso_equidad=weq, peso_costo=wc)
        ss.agente = {"obj": ag, "info": info, "id": f"app:{pd.Timestamp.now().value}"}
        ss.cache = {}
        st.success(f"Agente entrenado en {info['segundos']:.0f} s. Mejor captura en validación: {info['captura_val']:.0%}.")
        st.rerun()


# ============================================================================ 🔮 SIMULADOR
def pagina_simulador():
    st.title("🔮 Simulador · ¿Qué pasaría si…?")
    st.markdown("Cambie una condición y vea cómo reacciona el agente. Útil para discutir decisiones: "
                "**¿y si el presupuesto baja?, ¿y si un municipio empeora su vivienda o aumenta su sismicidad?**")
    aviso_origen()
    if not HAY_AGENTE:
        st.info("Cargue o entrene el agente primero."); return
    ce = E.cfg_entorno(); pdef = float(ce["presupuesto_por_periodo"])
    t1, t2 = st.tabs(["Cambiar una variable", "Presupuesto vs. riesgo cubierto"])
    with t1:
        a, b, c = st.columns(3)
        t = a.select_slider("Periodo", list(range(len(D["periodos"]))), ultimo_periodo(), format_func=lambda i: D["periodos"][i], key="sim_t")
        mun = b.selectbox("Municipio", D["municipios"], index=D["municipios"].index("SAN GIL") if "SAN GIL" in D["municipios"] else 0)
        var = c.selectbox("Variable", [f for f in FEATURES if f not in ("riesgo_cluster",)],
                          index=FEATURES.index("viviendas_precarias"), format_func=lambda v: f"{v} · {VARIABLES[v][2]}")
        i, j = D["municipios"].index(mun), FEATURES.index(var)
        desv = D["normalizacion"]["desv"][var]; media = D["normalizacion"]["media"][var]
        actual = float(D["X"][t, i, j] * desv + media)
        a, b = st.columns([2, 1])
        nuevo = a.slider(f"Nuevo valor de {var} (actual {actual:.2f})", float(actual - 3 * desv), float(actual + 3 * desv), float(actual + 1.5 * desv))
        pres = b.slider("Presupuesto", round(pdef / 3), round(pdef * 2), round(pdef), 5, key="sim_p")
        base, _ = E.priorizacion(t, pres, explicar=False)
        X = D["X"].copy(); X[t, i, j] = (nuevo - media) / desv
        esc, _ = E.servicio.priorizar(ss.agente["obj"], D, ce, t, pres, X=X, explicar=False)
        r0, r1 = int(base.set_index("municipio").loc[mun, "ranking"]), int(esc.set_index("municipio").loc[mun, "ranking"])
        c = st.columns(3)
        c[0].metric(f"Puesto de {mun.title()}", r1, f"{r0 - r1:+d} puestos" if r0 != r1 else "sin cambio")
        c[1].metric("¿Queda priorizado?", "Sí" if esc.set_index("municipio").loc[mun, "priorizado"] else "No",
                    "antes: " + ("sí" if base.set_index("municipio").loc[mun, "priorizado"] else "no"), delta_color="off")
        s0, s1 = set(base[base["priorizado"]]["municipio"]), set(esc[esc["priorizado"]]["municipio"])
        c[2].metric("Municipios que entran / salen", f"{len(s1 - s0)} / {len(s0 - s1)}")
        if s1 - s0: st.success("Entran: " + ", ".join(sorted(s1 - s0)))
        if s0 - s1: st.warning("Salen: " + ", ".join(sorted(s0 - s1)))
        comp = base[["municipio", "valor_Q"]].merge(esc[["municipio", "valor_Q"]], on="municipio", suffixes=("_antes", "_despues"))
        comp = comp.sort_values("valor_Q_despues", ascending=False).head(15).iloc[::-1]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=comp["valor_Q_antes"], y=comp["municipio"], orientation="h", name="Antes", marker_color="#5A6475"))
        fig.add_trace(go.Bar(x=comp["valor_Q_despues"], y=comp["municipio"], orientation="h", name="Escenario", marker_color=ui.ACENTO))
        lo = float(comp[["valor_Q_antes", "valor_Q_despues"]].min().min()); hi = float(comp[["valor_Q_antes", "valor_Q_despues"]].max().max())
        fig.update_xaxes(range=[lo - (hi - lo) * .2, hi + (hi - lo) * .05], title="Valor de prioridad")
        fig.update_layout(barmode="group", title="Prioridad antes y después del escenario (top 15)")
        st.plotly_chart(ui.estilo_fig(fig, 480), width="stretch")
    with t2:
        t = st.select_slider("Periodo", list(range(len(D["periodos"]))), int(D["cortes"]["test"][-1]), format_func=lambda i: D["periodos"][i], key="sim_t2")
        st.caption("Con el impacto real del trimestre siguiente (conocido en el histórico), ¿cuánto riesgo habría cubierto el agente con cada presupuesto?")
        cm = float(np.median(D["costo"][t]))
        filas = []
        imp_tot = D["impacto"][t].sum()
        for k in range(1, min(16, len(D["municipios"]))):
            tb, p = E.priorizacion(t, cm * k, explicar=False)
            elegidos = tb[tb["priorizado"]]
            orden = np.argsort(-D["impacto"][t]); ideal = D["impacto"][t][orden[:len(elegidos)]].sum()
            filas.append({"intervenciones": len(elegidos), "presupuesto": p, "Agente": elegidos["impacto_real_siguiente"].sum() / imp_tot,
                          "Ideal (conociendo el futuro)": ideal / imp_tot})
        df = pd.DataFrame(filas).drop_duplicates("intervenciones").melt(id_vars=["intervenciones", "presupuesto"], var_name="serie", value_name="riesgo")
        fig = px.line(df, x="intervenciones", y="riesgo", color="serie", markers=True,
                      color_discrete_map={"Agente": ui.ACENTO, "Ideal (conociendo el futuro)": "#5A6475"},
                      title="% del riesgo total del departamento cubierto según el nº de municipios atendidos")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(ui.estilo_fig(fig, 420), width="stretch")


# ============================================================================ ✅ VALIDACIÓN
def pagina_validacion():
    st.title("✅ Validación e incertidumbre")
    aviso_origen()
    per = CFG["periodo"]
    tramos = pd.DataFrame([
        dict(conjunto="Entrenamiento", inicio=pd.Period(D["periodos"][D["cortes"]["train"][0]], per).start_time, fin=pd.Period(D["periodos"][D["cortes"]["train"][-1]], per).end_time, n=len(D["cortes"]["train"])),
        dict(conjunto="Validación", inicio=pd.Period(D["periodos"][D["cortes"]["val"][0]], per).start_time, fin=pd.Period(D["periodos"][D["cortes"]["val"][-1]], per).end_time, n=len(D["cortes"]["val"])),
        dict(conjunto="Prueba", inicio=pd.Period(D["periodos"][D["cortes"]["test"][0]], per).start_time, fin=pd.Period(D["periodos"][D["cortes"]["test"][-1]], per).end_time, n=len(D["cortes"]["test"]))])
    st.markdown("#### División temporal: el modelo nunca ve el futuro")
    fig = px.timeline(tramos, x_start="inicio", x_end="fin", y="conjunto", color="conjunto", text="n",
                      color_discrete_sequence=[ui.AZUL, "#F2A541", ui.ACENTO])
    fig.update_traces(texttemplate="%{text} periodos", textposition="inside")
    fig.update_yaxes(autorange="reversed", title=None); fig.update_layout(showlegend=False)
    st.plotly_chart(ui.estilo_fig(fig, 220), width="stretch")
    c = st.columns(3)
    with c[0]: ui.tarjeta("Básico", "Si el archivo trae la columna <b>conjunto</b>, las métricas se reportan por tramo temporal (ver la pestaña Desempeño).", "SUPERVISADO")
    with c[1]: ui.tarjeta("Medio", "Calidad con coeficiente de silueta y comparación contra K-Means de referencia para varios k.", "NO SUPERVISADO")
    with c[2]: ui.tarjeta("Agente", "Normalización y sustitutos calculados solo con entrenamiento; el mejor modelo se elige en validación y se califica en prueba contra reglas simples.", "REFUERZO")
    if HAY_AGENTE:
        st.markdown("#### Incertidumbre del agente")
        ev = E.evaluacion()
        tab, _ = E.priorizacion(ultimo_periodo(), CFG["entorno"]["presupuesto_por_periodo"])
        a, b = st.columns(2)
        with a:
            fig = px.histogram(tab, x="incertidumbre", color="confianza", nbins=20, color_discrete_map=ui.COLOR_CONF,
                               title="Desacuerdo del comité de expertos por municipio", category_orders={"confianza": ["Alta", "Media", "Baja"]})
            st.plotly_chart(ui.estilo_fig(fig, 330), width="stretch")
        with b:
            st.metric("IC 95 % del riesgo capturado en prueba", f"{ev['ic95'][0]:.0%} – {ev['ic95'][1]:.0%}")
            st.markdown("- **Comité de 5 redes:** si discrepan, la confianza baja (🔴) y conviene verificar en campo.\n"
                        "- **Bootstrap por trimestre:** el intervalo muestra cuánto varía el desempeño entre trimestres.\n"
                        "- **Explicación:** gradiente × entrada indica qué variables movieron cada decisión.")
    st.markdown("#### Fuentes de incertidumbre y cómo se manejan")
    st.dataframe(pd.DataFrame([
        ("Modelo del agente", "Ensamble de 5 redes: el desacuerdo se reporta como semáforo de confianza"),
        ("Pocos trimestres de prueba", "Intervalo de confianza 95 % por bootstrap"),
        ("Catálogo incompleto", "Filtro de magnitud mínima (completitud) configurable"),
        ("Asignación espacial", f"Radio máximo de {CFG['radio_asignacion_km']} km; sismos fuera se reportan"),
        ("Datos censales", "Valores faltantes rellenados con la mediana y reportados como supuesto"),
        ("Pesos de la recompensa", "Se pueden cambiar y reentrenar (análisis de sensibilidad)")], columns=["Fuente", "Cómo se maneja"]),
        hide_index=True, width="stretch")


# ============================================================================ 🎤 PRESENTACIÓN
def pagina_presentacion():
    n = 8
    ss.setdefault("slide", 0)
    a, b, c = st.columns([1, 6, 1])
    if a.button("◀ Anterior", width="stretch", disabled=ss.slide == 0):
        ss.slide -= 1; st.rerun()
    if c.button("Siguiente ▶", width="stretch", disabled=ss.slide == n - 1):
        ss.slide += 1; st.rerun()
    b.markdown(f'<div class="puntos">{"".join("●" if i == ss.slide else "○" for i in range(n))}</div>', unsafe_allow_html=True)
    s = ss.slide

    def cab(eyebrow, titulo, texto):
        st.markdown(f'<div class="slide"><div class="eyebrow">{eyebrow}</div><h1>{titulo}</h1><p>{texto}</p></div>', unsafe_allow_html=True)

    def grande(col, valor, etiqueta):
        col.markdown(f'<div class="tarjeta" style="min-height:0"><div class="big">{valor}</div><div class="big-l">{etiqueta}</div></div>', unsafe_allow_html=True)

    if s == 0:
        cab("EL PROBLEMA", "El dinero para prevenir no alcanza para todos",
            "Santander tiene el Nido Sísmico de Bucaramanga, uno de los más activos del mundo. ¿Dónde invertir primero?")
        c = st.columns(3)
        grande(c[0], fmt(rep["sismos_usados"]), "sismos analizados")
        grande(c[1], len(D["municipios"]), "municipios")
        grande(c[2], f"{len(D['periodos'])}", "trimestres de historia" if CFG["periodo"] == "Q" else "meses de historia")
    elif s == 1:
        cab("LOS DATOS", "Sismicidad + territorio, unidos por municipio",
            f"Catálogo del SGC y fuente oficial: {rep['cruzan']} municipios cruzan · 25 variables justificadas.")
        m = EVT.sample(min(2000, len(EVT)), random_state=1).assign(municipio="Sismo")
        fig = ui.mapa(m, color="profundidad", tamano=np.clip(m["magnitud"] - 0.5, .3, None), escala="Inferno", alto=430,
                      hover={"lat": False, "lon": False})
        st.plotly_chart(fig, width="stretch")
    elif s == 2:
        cab("RETO BÁSICO", "¿Qué tan activa está cada zona?", "Modelo supervisado: nivel de actividad bajo, medio o alto.")
        sup, _o = _datos_supervisado()
        if "nivel_real" in sup.columns:
            d = sup.assign(r=sup["nivel_real"].map(nivel_a_num), p=sup["nivel_predicho"].map(nivel_a_num)).dropna(subset=["r", "p"])
            if "conjunto" in d.columns:
                d = d[d["conjunto"].astype(str).str.lower().str.startswith("te")] if (d["conjunto"].astype(str).str.lower().str.startswith("te")).any() else d
            a, b = st.columns([1, 1])
            cm = pd.crosstab(d["r"].astype(int), d["p"].astype(int)).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0).values
            a.plotly_chart(ui.fig_matriz(cm, CLASES, "Matriz de confusión (prueba)"), width="stretch")
            grande(b, f"{accuracy(d['r'].astype(int), d['p'].astype(int)):.0%}", "accuracy en prueba")
            b.write("")
            grande(b, f"{f1_macro(d['r'].astype(int), d['p'].astype(int)):.2f}", "F1 macro")
        else:
            st.info("Sin clase real en el archivo del equipo.")
    elif s == 3:
        cab("RETO MEDIO", "Municipios que se parecen", "Modelo no supervisado: grupos con comportamiento sísmico y social similar.")
        grupos = sorted(MUN["grupo"].unique())
        st.plotly_chart(ui.mapa(MUN, color="grupo", tamano=np.log10(MUN["poblacion"]) - 2.5, mapa_colores=dict(zip(grupos, ui.PALETA)), alto=450,
                                hover={"lat": False, "lon": False}), width="stretch")
    elif s == 4:
        cab("RETO AVANZADO", "Un agente que aprende a decidir",
            "Estado: 25 variables + presupuesto · Acción: elegir municipio · Recompensa: acertar dónde dolió, con equidad y costo.")
        if HAY_AGENTE and isinstance(ss.agente["info"].get("curva"), pd.DataFrame):
            a, b = st.columns(2)
            a.plotly_chart(ui.fig_curva(ss.agente["info"]["curva"]), width="stretch")
            b.plotly_chart(ui.fig_validacion(ss.agente["info"]["curva"]), width="stretch")
        else:
            c = st.columns(3)
            with c[0]: ui.tarjeta("🧩 Estado", "25 variables por municipio + presupuesto y descuido.", "S")
            with c[1]: ui.tarjeta("🎮 Acción", "Elegir un municipio dentro del presupuesto.", "A")
            with c[2]: ui.tarjeta("🏆 Recompensa", "Riesgo real atendido + equidad − costo.", "R")
    elif s == 5:
        cab("¿FUNCIONA?", "Comparado contra reglas simples, en años que nunca vio", "100 % = un oráculo que conoce el futuro.")
        if HAY_AGENTE:
            st.plotly_chart(ui.fig_politicas(E.evaluacion()["tabla"], 400), width="stretch")
    elif s == 6:
        cab("LA DECISIÓN", f"Priorización · {D['periodos'][-1]}", "Con semáforo de confianza: 🟢 actuar · 🟡 revisar · 🔴 verificar en campo.")
        if HAY_AGENTE:
            tab, _ = E.priorizacion(ultimo_periodo(), CFG["entorno"]["presupuesto_por_periodo"])
            a, b = st.columns([3, 2])
            a.plotly_chart(ui.fig_mapa_prioridad(tab, MUN, 440), width="stretch")
            v = tab[tab["priorizado"]].sort_values("orden")[["orden", "municipio", "confianza"]]
            v["confianza"] = v["confianza"].map(ui.ICONO_CONF)
            b.dataframe(v, hide_index=True, width="stretch")
    elif s == 7:
        cab("ALCANCE", "Una herramienta para decidir, no un pronóstico",
            "Priorización exploratoria · supuestos explícitos · lista para crecer con datos oficiales de costos y suelos.")
        c = st.columns(3)
        with c[0]: ui.tarjeta("Supuestos", "Riesgo = peligro × exposición × vulnerabilidad. Costo según área y ruralidad.")
        with c[1]: ui.tarjeta("Limitaciones", "No predice terremotos. Resolución municipal. Pocos trimestres de prueba.")
        with c[2]: ui.tarjeta("Extensiones", "Costos reales (UNGRD), microzonificación, varios tipos de acción, agentes por provincia.")


# ============================================================================ 📄 REPORTE
def pagina_reporte():
    st.title("📄 Reporte descargable")
    st.markdown("Genera la evidencia completa para el jurado con los datos y el agente actuales.")
    aviso_origen()
    if not HAY_AGENTE:
        st.info("Cargue o entrene el agente para incluir la priorización."); return
    ev = E.evaluacion()
    tab, pres = E.priorizacion(ultimo_periodo(), CFG["entorno"]["presupuesto_por_periodo"])
    a, b = st.columns(2)
    with a:
        st.markdown("#### 📗 Excel")
        st.caption("Priorización, comparación de políticas, variables, reporte de unión, importancia y panel.")
        try:
            import openpyxl  # noqa: F401
            hay_excel = True
        except ImportError:
            hay_excel = False
            st.caption("No está instalado 'openpyxl': se descargará un .zip con las mismas tablas en CSV.")
        if st.button("Preparar Excel" if hay_excel else "Preparar tablas (.zip)"):
            tablas = {"Priorizacion": tab, "Politicas": ev["tabla"], "Captura_por_periodo": ev["series"],
                      "Importancia": ev["importancia"].rename_axis("variable").rename("importancia").reset_index(),
                      "Variables": tabla_variables(), "Municipios_grupos": MUN,
                      "Panel": panel[["municipio", "periodo"] + FEATURES].assign(periodo=lambda d: d["periodo"].astype(str))}
            if not hay_excel:
                import zipfile
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                    for n, t_ in tablas.items():
                        z.writestr(f"{n}.csv", t_.to_csv(index=False).encode("utf-8-sig"))
                st.download_button("⬇️ Descargar tablas (.zip)", buf.getvalue(), "reporte_priorizacion_sismica.zip", "application/zip")
                st.stop()
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                tab.to_excel(w, sheet_name="Priorizacion", index=False)
                ev["tabla"].to_excel(w, sheet_name="Politicas", index=False)
                ev["series"].to_excel(w, sheet_name="Captura_por_periodo", index=False)
                ev["importancia"].rename("importancia").to_frame().to_excel(w, sheet_name="Importancia")
                tabla_variables().to_excel(w, sheet_name="Variables", index=False)
                pd.DataFrame([(k, json.dumps(v, ensure_ascii=False, default=str) if isinstance(v, (list, dict)) else v)
                              for k, v in rep.items()], columns=["dato", "valor"]).to_excel(w, sheet_name="Union_fuentes", index=False)
                MUN.to_excel(w, sheet_name="Municipios_grupos", index=False)
                panel[["municipio", "periodo"] + FEATURES].assign(periodo=lambda d: d["periodo"].astype(str)).to_excel(w, sheet_name="Panel", index=False)
            st.download_button("⬇️ Descargar Excel", buf.getvalue(), "reporte_priorizacion_sismica.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with b:
        st.markdown("#### 🌐 HTML interactivo")
        st.caption("Una página con las gráficas interactivas y las tablas; se abre en cualquier navegador.")
        if st.button("Preparar HTML"):
            figs = [ui.fig_politicas(ev["tabla"]), ui.fig_importancia(ev["importancia"]), ui.fig_q_incertidumbre(tab)]
            if isinstance(ss.agente["info"].get("curva"), pd.DataFrame):
                figs += [ui.fig_curva(ss.agente["info"]["curva"]), ui.fig_validacion(ss.agente["info"]["curva"])]
            partes = [f.update_layout(template="plotly_white", paper_bgcolor="white", font_color="#111").to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False)
                      for i, f in enumerate(figs)]
            sel = tab[tab["priorizado"]].sort_values("orden")[["orden", "municipio", "confianza", "costo", "razones"]]
            html = f"""<!doctype html><html lang="es"><meta charset="utf-8"><title>Priorización sísmica Santander</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1050px;margin:24px auto;padding:0 16px;color:#111;background:#fff}}
table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border-bottom:1px solid #ddd;padding:5px 8px;text-align:left}}
.k{{display:inline-block;margin:6px 26px 6px 0}}.k b{{font-size:28px;display:block}}</style>
<h1>Priorización sísmica · Santander</h1><p>Datos: <b>{ORIGEN}</b> · Periodo: <b>{D['periodos'][-1]}</b> · Presupuesto: <b>{pres:.0f}</b> ·
Básico: {rep['supervisado']['fuente']} · Medio: {rep['no_supervisado']['fuente']} · Agente: {ss.agente['info']['origen']}</p>
<div class="k"><b>{ev['tabla'].iloc[0]['captura_riesgo']:.0%}</b>riesgo capturado en prueba</div>
<div class="k"><b>{ev['ic95'][0]:.0%} – {ev['ic95'][1]:.0%}</b>IC 95 %</div>
<div class="k"><b>{rep['sismos_usados']:,}</b>sismos · {len(D['municipios'])} municipios</div>
<h2>Municipios priorizados</h2>{sel.to_html(index=False)}
<h2>Evaluación</h2>{partes[0]}{ev['tabla'].round(3).to_html(index=False)}
<h2>Explicación e incertidumbre</h2>{''.join(partes[1:])}
<h2>Ranking completo</h2>{tab.drop(columns=['impacto_real_siguiente']).round(3).to_html(index=False)}</html>"""
            st.download_button("⬇️ Descargar HTML", html.encode("utf-8"), "reporte_priorizacion_sismica.html", "text/html")


# ============================================================================ 📌 SUPUESTOS
def pagina_supuestos():
    st.title("📌 Supuestos, limitaciones y extensiones")
    a, b, c = st.columns(3)
    with a:
        st.markdown("#### Supuestos")
        st.markdown("- Riesgo = peligro × exposición × vulnerabilidad.\n- Sismos superficiales (< 30 km) pesan 1,0; los "
                    "intermedios del Nido 0,3.\n- El costo de intervenir depende del área y la ruralidad.\n- La actividad "
                    "reciente (réplicas, enjambres) informa el trimestre siguiente.")
        for s in rep["supuestos"]:
            st.markdown(f"- {s}")
    with b:
        st.markdown("#### Limitaciones")
        st.markdown("- Priorización **exploratoria**: no pronostica terremotos ni reemplaza estudios oficiales.\n"
                    "- Resolución municipal.\n- Pocos trimestres de prueba → intervalos amplios.\n"
                    "- El agente depende del diseño de la recompensa.")
    with c:
        st.markdown("#### Extensiones")
        st.markdown("- Costos reales (Gobernación, UNGRD).\n- Remoción en masa y microzonificación.\n"
                    "- Varios tipos de acción con costos distintos.\n- PPO o un agente por provincia.")
    st.divider()
    st.markdown("#### ¿Cómo ayuda a decidir en Santander?")
    st.markdown("El Consejo Departamental de Gestión del Riesgo puede usar el ranking para decidir **dónde invertir primero** "
                "cada trimestre, ver **qué tan segura** es cada recomendación y **simular** cambios de presupuesto o de "
                "condiciones antes de decidir.")


{"📥 Cargar datos": pagina_cargar, "🏠 Inicio": pagina_inicio, "📊 Datos e integración": pagina_datos,
 "🎯 Básico · Supervisado": pagina_basico, "🧭 Medio · No supervisado": pagina_medio, "🤖 Avanzado · Agente RL": pagina_agente,
 "🔮 Simulador ¿qué pasaría si?": pagina_simulador, "✅ Validación e incertidumbre": pagina_validacion,
 "🎤 Modo presentación": pagina_presentacion, "📄 Reporte": pagina_reporte, "📌 Supuestos y alcance": pagina_supuestos}[pagina]()
