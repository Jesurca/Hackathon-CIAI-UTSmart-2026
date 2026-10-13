"""
INTEGRACIÓN DE FUENTES Y CONSTRUCCIÓN DE LAS 25 VARIABLES (a partir de DataFrames)
=================================================================================
Entradas (ya con nombres canónicos, ver columnas.py):
  cat  : catálogo de sismos SGC   -> fecha, latitud, longitud, profundidad, magnitud[, municipio]
  ofi  : fuente oficial territorial -> municipio, latitud, longitud, poblacion, ...
  sup  : resultados del modelo supervisado (reto Básico)        [opcional]
  clu  : resultados del modelo no supervisado (reto Medio)       [opcional]
  sup_modelo / clu_modelo : modelos .pkl de los compañeros       [opcional]

Salida: un PANEL con una fila por (municipio, periodo) y 25 variables, más un
reporte de la unión (evidencia para la rúbrica).

Regla anti-fuga: las variables del periodo t usan solo información hasta t; lo que
pasa en t+1 se usa únicamente como RECOMPENSA del agente.
"""
import numpy as np
import pandas as pd
from .utilidades import kmeans

from .columnas import CLASES, nivel_a_num, norm_municipio

NIDO = (6.80, -73.10)  # Nido sísmico de Bucaramanga (Mesa de Los Santos)

VARIABLES = {
    # --- Fuente 1: catálogo sísmico SGC ---
    "n_sismos":            ("SGC", "Sísmica", "Nº de sismos en el periodo", "Actividad reciente; base de cualquier índice de peligro."),
    "n_sismos_4p":         ("SGC", "Sísmica", "Nº de sismos últimos 4 periodos", "Suaviza el ruido trimestral; memoria de un año."),
    "mag_max":             ("SGC", "Sísmica", "Magnitud máxima del periodo", "El daño lo producen los eventos grandes, no el promedio."),
    "mag_media":           ("SGC", "Sísmica", "Magnitud media", "Distingue enjambres de microsismos de actividad moderada."),
    "energia_log":         ("SGC", "Sísmica", "log10 energía liberada (E = 10^(1.5M+4.8))", "Resume número y tamaño en una sola magnitud física."),
    "energia_log_4p":      ("SGC", "Sísmica", "log10 energía últimos 4 periodos", "Energía acumulada anual."),
    "prof_media":          ("SGC", "Sísmica", "Profundidad media (km)", "Los sismos superficiales atenúan menos y dañan más."),
    "pct_superficiales":   ("SGC", "Sísmica", "% sismos con profundidad < 30 km", "Separa el Nido (intermedio) de fallas corticales."),
    "n_m4_4p":             ("SGC", "Sísmica", "Nº sismos M≥4 últimos 4 periodos", "Eventos sentidos con potencial de daño."),
    "b_value":             ("SGC", "Sísmica", "Valor b de Gutenberg-Richter (Aki, 8 periodos)", "b bajo = proporción alta de sismos grandes."),
    "periodos_desde_m4":   ("SGC", "Sísmica", "Periodos desde el último M≥4 (tope 20)", "Captura réplicas y reactivación reciente."),
    "tendencia":           ("SGC", "Sísmica", "n_sismos − promedio de los 4 previos", "Detecta aceleración de la actividad."),
    "dist_nido_km":        ("SGC + oficial", "Sísmica", "Distancia al Nido de Bucaramanga (km)", "Rasgo tectónico dominante de Santander."),
    "tasa_sismica_km2":    ("SGC + oficial", "Sísmica", "Sismos últimos 4 periodos / área", "Compara municipios de distinto tamaño."),
    # --- Fuente 2: fuente oficial territorial ---
    "log_poblacion":       ("Oficial (DANE)", "Exposición", "log10 población", "Personas potencialmente afectadas."),
    "densidad_pob":        ("Oficial (DANE)", "Exposición", "log10 habitantes por km²", "La concentración urbana amplifica pérdidas."),
    "nbi":                 ("Oficial (DANE)", "Vulnerabilidad", "% Necesidades Básicas Insatisfechas", "Vulnerabilidad social y menor capacidad de recuperación."),
    "pct_rural":           ("Oficial (DANE)", "Vulnerabilidad", "% población rural", "Acceso difícil para atención y respuesta."),
    "altitud_m":           ("Oficial (IGAC)", "Territorial", "Altitud (m)", "Proxy de pendiente / remoción en masa inducida."),
    "viviendas_precarias": ("Oficial (DANE)", "Vulnerabilidad", "% viviendas con materiales precarios", "Vulnerabilidad física de las edificaciones."),
    "hospitales_100k":     ("Oficial (MinSalud)", "Capacidad", "IPS por 100.000 hab.", "Capacidad de respuesta (resta vulnerabilidad)."),
    "indice_exposicion":   ("Derivada", "Exposición", "log_poblacion × (1 + viviendas_precarias/100)", "Exposición física ponderada por vulnerabilidad."),
    # --- Integración con los modelos del equipo ---
    "nivel_actividad":     ("Reto Básico", "Modelos del equipo", "Clase bajo/medio/alto (0/1/2) del modelo supervisado", "Integra la clasificación del equipo como señal de estado."),
    "riesgo_cluster":      ("Reto Medio", "Modelos del equipo", "Energía media histórica del grupo territorial", "Integra los patrones no supervisados entre municipios."),
    "costo_intervencion":  ("Supuesto", "Decisión", "Costo relativo de intervenir (área y ruralidad)", "La priorización es realista: el presupuesto es limitado."),
}
FEATURES = list(VARIABLES)

VALORES_POR_DEFECTO = {"area_km2": 300.0, "nbi": 25.0, "pct_rural": 50.0, "altitud_m": 1000.0,
                       "viviendas_precarias": 20.0, "hospitales": 1.0}


def tabla_variables():
    return pd.DataFrame([(k, *v) for k, v in VARIABLES.items()],
                        columns=["variable", "fuente", "grupo", "descripcion", "justificacion"])


def _haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def _a_periodo(serie, freq):
    s = serie.astype(str).str.strip()
    try:
        return pd.PeriodIndex(s, freq=freq)
    except Exception:
        return pd.to_datetime(s, errors="coerce").dt.to_period(freq)


def aplicar_modelo(modelo, X):
    """Aplica un modelo sklearn (.pkl) si sus columnas existen en X. Devuelve (pred, proba, motivo)."""
    if modelo is None:
        return None, None, "sin modelo"
    nombres = getattr(modelo, "feature_names_in_", None)
    if nombres is None:
        return None, None, "el modelo no guarda los nombres de sus columnas (entrénelo con un DataFrame)"
    faltan = [c for c in nombres if c not in X.columns]
    if faltan:
        return None, None, f"faltan columnas que usa el modelo: {faltan[:6]}{'…' if len(faltan) > 6 else ''}"
    Xm = X[list(nombres)].astype(float).fillna(0)
    pred = modelo.predict(Xm)
    proba = modelo.predict_proba(Xm) if hasattr(modelo, "predict_proba") else None
    return pred, proba, "ok"


# ------------------------------------------------------------------ principal
def construir_panel(cat, ofi, cfg, sup=None, clu=None, sup_modelo=None, clu_modelo=None):
    per = cfg["periodo"]
    radio = cfg.get("radio_asignacion_km", 30)
    rep = {"supuestos": [], "avisos": []}

    # ---------- limpieza ----------
    cat = cat.copy(); ofi = ofi.copy()
    num = pd.to_numeric(cat["fecha"], errors="coerce")
    if num.notna().mean() > 0.9 and num.between(20000, 80000).mean() > 0.9:   # número de serie de Excel
        cat["fecha"] = pd.to_datetime(num, unit="D", origin="1899-12-30")
    else:
        cat["fecha"] = pd.to_datetime(cat["fecha"], errors="coerce", dayfirst=False)
    for c in ["latitud", "longitud", "profundidad", "magnitud"]:
        cat[c] = pd.to_numeric(cat[c], errors="coerce")
    n0 = len(cat)
    cat = cat.dropna(subset=["fecha", "latitud", "longitud", "magnitud"])
    cat["profundidad"] = cat["profundidad"].fillna(cat["profundidad"].median())
    rep["sismos_invalidos"] = n0 - len(cat)
    mmin = cfg.get("magnitud_minima", 0.0)
    rep["sismos_bajo_mmin"] = int((cat["magnitud"] < mmin).sum())
    cat = cat[cat["magnitud"] >= mmin]

    ofi["municipio"] = ofi["municipio"].map(norm_municipio)
    dup = ofi["municipio"].duplicated().sum()
    if dup:
        rep["avisos"].append(f"{dup} municipios repetidos en la fuente oficial; se conservó el primero.")
        ofi = ofi.drop_duplicates("municipio")
    for c in ["latitud", "longitud", "poblacion"]:
        ofi[c] = pd.to_numeric(ofi[c], errors="coerce")
    for c, v in VALORES_POR_DEFECTO.items():
        if c not in ofi.columns:
            ofi[c] = v
            rep["supuestos"].append(f"'{c}' no está en la fuente oficial: se usó el valor fijo {v} para todos.")
        else:
            ofi[c] = pd.to_numeric(ofi[c], errors="coerce")
            if ofi[c].isna().any():
                rep["avisos"].append(f"'{c}': {int(ofi[c].isna().sum())} vacíos rellenados con la mediana.")
                ofi[c] = ofi[c].fillna(ofi[c].median() if ofi[c].notna().any() else v)
    if "provincia" not in ofi.columns:
        ofi["provincia"] = "—"
    ofi["poblacion"] = ofi["poblacion"].fillna(ofi["poblacion"].median()).clip(lower=100)
    ofi["area_km2"] = ofi["area_km2"].clip(lower=1)

    # ---------- asignación de sismos a municipios ----------
    d = _haversine(cat["latitud"].values[:, None], cat["longitud"].values[:, None],
                   ofi["latitud"].values[None, :], ofi["longitud"].values[None, :])
    cercano = ofi["municipio"].values[d.argmin(1)]
    dist_min = d.min(1)
    en_ofi = set(ofi["municipio"])
    if "municipio" in cat.columns and cat["municipio"].notna().mean() > 0.5:
        # El SGC trae REGION como "Zapatoca - Santander, Colombia": se toma lo que va antes del " - "
        nombres = cat["municipio"].map(lambda x: norm_municipio(str(x).split(" - ")[0].split(",")[0]) if isinstance(x, str) else None)
        rep["nombres_sin_coincidencia_exacta"] = sorted({n for n in nombres.dropna().unique() if n not in en_ofi})[:30]

        def emparejar(n):   # exacto, o nombre corto único ("EL CARMEN" -> "EL CARMEN DE CHUCURI")
            if n is None or n in en_ofi:
                return n
            cand = [o for o in en_ofi if o.startswith(n + " ") or n.startswith(o + " ")]
            return cand[0] if len(cand) == 1 else None
        mapa = {n: emparejar(n) for n in nombres.dropna().unique()}
        cat["municipio"] = nombres.map(mapa)
        sin = cat["municipio"].isna().values
        rep["sismos_por_cercania"] = int(sin.sum())
        cat.loc[sin, "municipio"] = np.where(dist_min[sin] <= radio, cercano[sin], None)
        rep["metodo_asignacion"] = (f"Nombre del municipio en el catálogo; los que no coinciden con la fuente oficial "
                                    f"({rep['sismos_por_cercania']:,} sismos) se asignan a la cabecera más cercana (máx. {radio} km)").replace(",", ".")
    else:
        rep["metodo_asignacion"] = f"Cabecera municipal más cercana (máximo {radio} km)"
        cat["municipio"] = np.where(dist_min <= radio, cercano, None)

    en_cat = set(cat["municipio"].dropna()); en_ofi = set(ofi["municipio"])
    rep.update({
        "municipios_catalogo": len(en_cat), "municipios_oficial": len(en_ofi),
        "cruzan": len(en_cat & en_ofi),
        "solo_catalogo": sorted(en_cat - en_ofi), "solo_oficial": sorted(en_ofi - en_cat),
        "sismos_fuera_radio": int(cat["municipio"].isna().sum()),
        "sismos_total": int(n0),
    })
    cat = cat[cat["municipio"].isin(en_ofi)].copy()
    rep["sismos_usados"] = int(len(cat))
    if len(cat) < 50:
        raise ValueError("Menos de 50 sismos quedaron asignados a municipios de la fuente oficial. "
                         "Revise nombres de municipio, coordenadas o el radio de asignación.")

    # ---------- agregación por municipio-periodo ----------
    cat["periodo"] = cat["fecha"].dt.to_period(per)
    cat["energia"] = 10 ** (1.5 * cat["magnitud"] + 4.8)
    cat["sup"] = (cat["profundidad"] < 30).astype(float)
    cat["m4"] = (cat["magnitud"] >= 4).astype(float)
    cat["peligro"] = 10 ** (1.5 * cat["magnitud"]) * np.where(cat["sup"] == 1, 1.0, 0.3)
    g = cat.groupby(["municipio", "periodo"]).agg(
        n_sismos=("magnitud", "size"), mag_max=("magnitud", "max"), mag_media=("magnitud", "mean"),
        energia=("energia", "sum"), prof_media=("profundidad", "mean"), pct_superficiales=("sup", "mean"),
        n_m4=("m4", "sum"), peligro=("peligro", "sum"), suma_mag=("magnitud", "sum"))
    periodos = pd.period_range(cat["periodo"].min(), cat["periodo"].max(), freq=per)
    rep["periodo_inicio"], rep["periodo_fin"], rep["n_periodos"] = str(periodos[0]), str(periodos[-1]), len(periodos)
    idx = pd.MultiIndex.from_product([sorted(en_ofi), periodos], names=["municipio", "periodo"])
    p = g.reindex(idx).reset_index()
    for c in ["n_sismos", "energia", "n_m4", "peligro", "suma_mag", "mag_max", "mag_media", "pct_superficiales"]:
        p[c] = p[c].fillna(0)
    p = p.sort_values(["municipio", "periodo"])
    gm = p.groupby("municipio")
    roll = lambda c, w: gm[c].transform(lambda s: s.rolling(w, min_periods=1).sum())

    p["n_sismos_4p"] = roll("n_sismos", 4)
    p["energia_log"] = np.log10(1 + p["energia"])
    p["energia_log_4p"] = np.log10(1 + roll("energia", 4))
    p["prof_media"] = gm["prof_media"].transform(lambda s: s.ffill()).fillna(cat["profundidad"].median())
    p["n_m4_4p"] = roll("n_m4", 4)
    n8, s8 = roll("n_sismos", 8), roll("suma_mag", 8)
    b = np.log10(np.e) / ((s8 / n8.replace(0, np.nan)) - mmin + 0.05)
    p["b_value"] = np.where(n8 >= 20, b, 1.0).clip(0.3, 2.5)

    def _desde(s):
        out, c = [], 20
        for v in s:
            c = 0 if v > 0 else min(c + 1, 20); out.append(c)
        return out
    p["periodos_desde_m4"] = gm["n_m4"].transform(_desde)
    p["tendencia"] = p["n_sismos"] - gm["n_sismos"].transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean()).fillna(0)

    # ---------- unión con la fuente oficial ----------
    p = p.merge(ofi, on="municipio", how="left")
    p["dist_nido_km"] = _haversine(p["latitud"], p["longitud"], NIDO[0], NIDO[1])
    p["tasa_sismica_km2"] = p["n_sismos_4p"] / p["area_km2"]
    p["log_poblacion"] = np.log10(p["poblacion"])
    p["densidad_pob"] = np.log10(p["poblacion"] / p["area_km2"])
    p["hospitales_100k"] = p["hospitales"] / p["poblacion"] * 1e5
    p["indice_exposicion"] = p["log_poblacion"] * (1 + p["viviendas_precarias"] / 100)
    p["costo_intervencion"] = (10 + 20 * p["area_km2"] / p["area_km2"].max() + 15 * p["pct_rural"] / 100).round(2)

    # ---------- impacto real (SOLO recompensa / evaluación) ----------
    exposicion = (p["log_poblacion"] / p["log_poblacion"].max()) * (1 + p["viviendas_precarias"] / 100) * (1 + p["nbi"] / 100)
    p["impacto"] = np.log10(1 + p["peligro"]) * exposicion
    p["impacto_siguiente"] = p.groupby("municipio")["impacto"].shift(-1)
    p = p.dropna(subset=["impacto_siguiente"]).copy()
    rep["n_periodos"], rep["periodo_fin"] = int(p["periodo"].nunique()), str(p["periodo"].max())

    fin_train = pd.Period(cfg["fin_entrenamiento"], per)
    p, info_sup = _integrar_supervisado(p, sup, sup_modelo, per, fin_train)
    p, info_clu, municipios = _integrar_clusters(p, ofi, clu, clu_modelo, fin_train, cfg.get("semilla", 42))
    rep["supervisado"], rep["no_supervisado"] = info_sup, info_clu
    rep["filas_panel"] = len(p)
    return p, rep, municipios, cat


def _integrar_supervisado(p, sup, modelo, per, fin_train):
    info = {}
    if sup is not None and len(sup):
        s = sup.copy()
        s["municipio"] = s["municipio"].map(norm_municipio)
        s["nivel_num"] = s["nivel_predicho"].map(nivel_a_num)
        malos = s["nivel_num"].isna().sum()
        s = s.dropna(subset=["nivel_num"])
        no_cruzan = sorted(set(s["municipio"]) - set(p["municipio"]))
        if "periodo" in s.columns and s["periodo"].notna().any():
            s["periodo"] = _a_periodo(s["periodo"], per)
            s = s.dropna(subset=["periodo"]).drop_duplicates(["municipio", "periodo"], keep="last")
            p = p.merge(s[["municipio", "periodo", "nivel_num"]], on=["municipio", "periodo"], how="left")
            cubiertas = p["nivel_num"].notna().mean()
            p["nivel_num"] = p.groupby("municipio")["nivel_num"].transform(lambda x: x.ffill())
            info["modo"] = f"Archivo por municipio y periodo (cubre {cubiertas:.0%} de las filas del panel; el resto toma el último valor conocido)"
        else:
            s = s.drop_duplicates("municipio", keep="last")
            p = p.merge(s[["municipio", "nivel_num"]], on="municipio", how="left")
            info["modo"] = "Archivo con una clase por municipio (sin periodo): se usa igual en todos los trimestres"
            info["aviso"] = ("Sin columna de periodo, la clase puede contener información del futuro para los "
                             "primeros trimestres. Lo ideal es una predicción por municipio y trimestre.")
        p["nivel_actividad"] = p["nivel_num"].fillna(1).astype(float)
        p = p.drop(columns="nivel_num")
        info.update(fuente="archivo", filas_invalidas=int(malos), no_cruzan=no_cruzan)
        return p, info

    pred, _proba, motivo = aplicar_modelo(modelo, p)
    if pred is not None:
        p["nivel_actividad"] = [nivel_a_num(x) if nivel_a_num(x) is not None else 1 for x in pred]
        info.update(fuente="modelo", modo="Se aplicó el modelo .pkl del reto Básico a cada municipio y trimestre")
        return p, info

    q1, q2 = p.loc[p["periodo"] <= fin_train, "n_sismos_4p"].quantile([0.33, 0.66])
    p["nivel_actividad"] = np.select([p["n_sismos_4p"] <= q1, p["n_sismos_4p"] <= q2], [0, 1], 2).astype(float)
    info.update(fuente="sustituto", modo="Sustituto: terciles de sismos del último año (fijados con datos de entrenamiento)",
                motivo_modelo=motivo if modelo is not None else None)
    return p, info


def _integrar_clusters(p, ofi, clu, modelo, fin_train, semilla):
    info = {}
    tr = p[p["periodo"] <= fin_train]
    prom = tr.groupby("municipio")[[f for f in FEATURES if f in tr.columns]].mean()
    municipios = ofi.set_index("municipio")[["latitud", "longitud", "provincia", "poblacion"]].copy()
    if "codigo_dane" in ofi.columns:
        municipios["codigo_dane"] = ofi.set_index("municipio")["codigo_dane"]

    cl, nombres = None, {}
    if clu is not None and len(clu):
        c = clu.copy(); c["municipio"] = c["municipio"].map(norm_municipio)
        c = c.drop_duplicates("municipio", keep="last").set_index("municipio")
        cl = c["cluster"].astype(str)
        if "nombre_cluster" in c.columns:
            nombres = c.groupby("cluster")["nombre_cluster"].first().astype(str).to_dict()
            nombres = {str(k): v for k, v in nombres.items()}
        for pc in ["PC1", "PC2"]:
            if pc in c.columns:
                municipios[pc] = pd.to_numeric(c[pc], errors="coerce")
        info.update(fuente="archivo", modo="Archivo de grupos del reto Medio",
                    no_cruzan=sorted(set(c.index) - set(p["municipio"])),
                    sin_grupo=sorted(set(p["municipio"]) - set(c.index)))
    else:
        pred, _pr, motivo = aplicar_modelo(modelo, prom)
        if pred is not None:
            cl = pd.Series(pred, index=prom.index).astype(str)
            info.update(fuente="modelo", modo="Se aplicó el modelo .pkl del reto Medio al promedio de entrenamiento de cada municipio")
        else:
            cols = ["energia_log_4p", "pct_superficiales", "dist_nido_km", "nbi", "densidad_pob"]
            z = (prom[cols] - prom[cols].mean()) / prom[cols].std().replace(0, 1)
            lab, _ = kmeans(z.fillna(0).values, 4, semilla=semilla)
            cl = pd.Series(lab, index=prom.index).astype(str)
            info.update(fuente="sustituto", modo="Sustituto: K-Means (k=4) sobre promedios de entrenamiento",
                        motivo_modelo=motivo if modelo is not None else None)

    p["cluster"] = p["municipio"].map(cl).fillna("sin grupo")
    riesgo = tr.assign(cluster=tr["municipio"].map(cl)).groupby("cluster")["energia_log_4p"].mean()
    p["riesgo_cluster"] = p["cluster"].map(riesgo).fillna(riesgo.mean())

    # etiqueta legible: "G1 · nombre"
    orden = sorted(p["cluster"].unique(), key=lambda k: (-riesgo.get(k, 0)))
    etiqueta = {k: f"G{i + 1}" + (f" · {nombres[k]}" if k in nombres else "") for i, k in enumerate(orden)}
    p["grupo"] = p["cluster"].map(etiqueta)
    municipios["cluster"] = municipios.index.map(cl).fillna("sin grupo")
    municipios["grupo"] = municipios["cluster"].map(etiqueta).fillna("sin grupo")
    info["etiquetas"] = etiqueta
    return p, info, municipios.reset_index()
