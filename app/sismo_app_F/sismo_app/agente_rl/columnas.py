"""
Esquema de columnas que espera la app y DETECCIÓN AUTOMÁTICA de nombres.

Cada archivo que suba el equipo puede tener nombres distintos ("LATITUD (°)", "lat",
"Latitude"...). Aquí se define el nombre "canónico" que usa el agente y una lista de
sinónimos para reconocerlo solo. En la app, el usuario puede corregir cualquier
asignación con un selector.
"""
import re
import unicodedata

NO_DISPONIBLE = "— no disponible —"

# (canónico, obligatorio, descripción, sinónimos)
ESQUEMAS = {
    "catalogo": [
        ("fecha", True, "Fecha y hora del sismo", ["fecha hora utc", "fecha - hora utc", "fecha_hora", "fecha", "datetime", "date", "time", "fecha utc"]),
        ("latitud", True, "Latitud (grados)", ["latitud", "lat", "latitude"]),
        ("longitud", True, "Longitud (grados)", ["longitud", "lon", "long", "longitude", "lng"]),
        ("profundidad", True, "Profundidad (km)", ["profundidad", "prof km", "prof", "depth", "profundidad km", "profundidad_km"]),
        ("magnitud", True, "Magnitud", ["magnitud", "mag", "magnitude", "ml", "mw"]),
        ("municipio", False, "Municipio o REGION del SGC (ej. 'Zapatoca - Santander, Colombia'); si no coincide, se asigna por cercanía",
         ["municipio", "mpio", "nombre municipio", "municipality", "region", "región", "localizacion"]),
    ],
    "oficial": [
        ("municipio", True, "Nombre del municipio", ["municipio", "mpio", "nombre municipio", "nombre_municipio", "mpio_cnmbr"]),
        ("latitud", True, "Latitud del centroide / cabecera", ["latitud", "lat", "latitude", "y"]),
        ("longitud", True, "Longitud del centroide / cabecera", ["longitud", "lon", "long", "longitude", "x"]),
        ("poblacion", True, "Población total", ["poblacion", "poblacion total", "habitantes", "pob", "population"]),
        ("area_km2", False, "Área (km²)", ["area_km2", "area", "area km2", "superficie"]),
        ("nbi", False, "% Necesidades Básicas Insatisfechas", ["nbi", "nbi_pct", "necesidades basicas", "pct nbi"]),
        ("pct_rural", False, "% población rural", ["pct_rural", "rural", "poblacion rural pct", "porcentaje rural", "resto"]),
        ("altitud_m", False, "Altitud (m s.n.m.)", ["altitud", "altitud_m", "altura", "elevacion", "msnm"]),
        ("viviendas_precarias", False, "% viviendas en material precario", ["viviendas_precarias", "pct_viviendas_precarias", "viv_material_precario_pct", "vivienda precaria", "deficit cualitativo"]),
        ("hospitales", False, "Nº de IPS / hospitales", ["hospitales", "ips", "prestadores", "n_ips", "camas"]),
        ("provincia", False, "Provincia (opcional, para agrupar)", ["provincia", "subregion", "region"]),
        ("codigo_dane", False, "Código DANE / DIVIPOLA (opcional)", ["codigo_dane", "divipola", "cod_dane", "mpio_ccdgo", "codigo"]),
    ],
    "supervisado": [
        ("municipio", True, "Municipio", ["municipio", "mpio", "nombre municipio"]),
        ("periodo", False, "Periodo (p. ej. 2024Q1). Sin él, la clase se usa en todos los periodos", ["periodo", "trimestre", "period", "quarter", "fecha"]),
        ("nivel_predicho", True, "Clase predicha por el modelo (Bajo/Medio/Alto o 0/1/2)", ["nivel_predicho", "prediccion", "predicho", "y_pred", "pred", "nivel_actividad", "clase_predicha"]),
        ("nivel_real", False, "Clase real (para matriz de confusión y métricas)", ["nivel_real", "real", "y_true", "y_test", "etiqueta", "clase_real", "target"]),
        ("prob_bajo", False, "Probabilidad clase Bajo", ["prob_bajo", "p_bajo", "proba_0", "prob_0"]),
        ("prob_medio", False, "Probabilidad clase Medio", ["prob_medio", "p_medio", "proba_1", "prob_1"]),
        ("prob_alto", False, "Probabilidad clase Alto", ["prob_alto", "p_alto", "proba_2", "prob_2"]),
        ("conjunto", False, "Conjunto (train / val / test)", ["conjunto", "split", "set", "particion"]),
    ],
    "clusters": [
        ("municipio", True, "Municipio", ["municipio", "mpio", "nombre municipio"]),
        ("cluster", True, "Grupo asignado", ["cluster", "grupo", "label", "etiqueta", "kmeans"]),
        ("nombre_cluster", False, "Nombre descriptivo del grupo", ["nombre_cluster", "nombre grupo", "descripcion", "perfil"]),
        ("PC1", False, "Componente principal 1", ["pc1", "pca1", "componente1", "x_pca"]),
        ("PC2", False, "Componente principal 2", ["pc2", "pca2", "componente2", "y_pca"]),
    ],
}


def norm_texto(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def norm_municipio(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return " ".join(s.upper().replace("-", " ").split())


def detectar(columnas, tipo):
    """Devuelve {canónico: columna_del_archivo o NO_DISPONIBLE}."""
    cols = {c: norm_texto(c) for c in columnas}
    usadas, res = set(), {}
    for canon, _oblig, _desc, sinonimos in ESQUEMAS[tipo]:
        elegido = NO_DISPONIBLE
        sins = [norm_texto(x) for x in [canon] + sinonimos]
        for s in sins:                                     # 1) coincidencia exacta
            m = [c for c, n in cols.items() if n == s and c not in usadas]
            if m:
                elegido = m[0]; break
        if elegido == NO_DISPONIBLE:                       # 2) empieza por el sinónimo
            for s in sins:
                m = [c for c, n in cols.items() if (n.startswith(s + " ") or n.startswith(s)) and len(s) >= 3 and c not in usadas]
                if m:
                    elegido = m[0]; break
        if elegido != NO_DISPONIBLE:
            usadas.add(elegido)
        res[canon] = elegido
    return res


def aplicar_mapeo(df, mapeo):
    """Devuelve un DataFrame con los nombres canónicos (una misma columna puede usarse dos veces)."""
    import pandas as pd
    return pd.DataFrame({canon: df[orig].values for canon, orig in mapeo.items()
                         if orig != NO_DISPONIBLE and orig in df.columns}, index=df.index)


def repetidas(mapeo):
    """Columnas del archivo asignadas a más de un campo (casi siempre es un error)."""
    vistos = {}
    for canon, orig in mapeo.items():
        if orig != NO_DISPONIBLE:
            vistos.setdefault(orig, []).append(canon)
    return {o: c for o, c in vistos.items() if len(c) > 1}


def faltantes_obligatorias(mapeo, tipo):
    return [c for c, ob, _d, _s in ESQUEMAS[tipo] if ob and mapeo.get(c, NO_DISPONIBLE) == NO_DISPONIBLE]


# ------------------------------------------------------------------ niveles de actividad
CLASES = ["Bajo", "Medio", "Alto"]
_MAPA_NIVEL = {"bajo": 0, "low": 0, "0": 0, "0.0": 0, "medio": 1, "media": 1, "moderado": 1, "medium": 1,
               "1": 1, "1.0": 1, "alto": 2, "alta": 2, "high": 2, "2": 2, "2.0": 2}


def nivel_a_num(x):
    if x is None or (isinstance(x, float) and x != x):
        return None
    return _MAPA_NIVEL.get(norm_texto(x).replace(" ", ""), None)
