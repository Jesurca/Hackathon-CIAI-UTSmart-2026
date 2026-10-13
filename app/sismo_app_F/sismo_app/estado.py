"""
estado.py — manejo de los archivos cargados, el procesamiento y el agente
dentro de la sesión de Streamlit.

Conceptos:
  * "ranuras" (slots): catálogo, oficial, supervisado, clusters. Cada una guarda el
    archivo activo (DataFrame + nombre + mapeo de columnas + si es demo o del equipo).
  * modelos .pkl: supervisado y clusters (opcionales).
  * agente .npz (opcional): si no hay, se entrena desde la app.
  * R: resultado de procesar todo (panel de 25 variables, tensores, reporte de unión).
"""
import io
from pathlib import Path

import pickle

import pandas as pd
import streamlit as st

from agente_rl.columnas import ESQUEMAS, aplicar_mapeo, detectar, faltantes_obligatorias
from agente_rl.flujo import CFG_BASE, procesar

DEMO = Path(__file__).parent / "demo"
ARCH_DEMO = {"catalogo": "catalogo_sgc_demo.csv", "oficial": "fuente_oficial_demo.csv",
             "supervisado": "supervisado_resultados_demo.csv", "clusters": "clusters_resultados_demo.csv"}
MOD_DEMO = {"supervisado": "supervisado_modelo_demo.pkl", "clusters": "clusters_modelo_demo.pkl"}
NOMBRES = {"catalogo": "Catálogo sísmico (SGC)", "oficial": "Fuente oficial territorial",
           "supervisado": "Resultados del modelo supervisado", "clusters": "Resultados del modelo no supervisado"}

from agente_rl import servicio  # agente en NumPy puro: no necesita PyTorch ni GPU


# ------------------------------------------------------------------ lectura de archivos
def _fila_encabezado(matriz, max_filas=80):
    """Encuentra la fila de encabezados aunque arriba haya un bloque de 'parámetros de búsqueda'
    (como en los reportes del SGC): la fila con más celdas de texto, seguida de filas con números."""
    mejor, idx = -1, 0
    for i, fila in enumerate(matriz[:max_filas]):
        textos = sum(1 for x in fila if isinstance(x, str) and x.strip() and not _es_numero(x))
        sig = matriz[i + 1] if i + 1 < len(matriz) else []
        nums = sum(1 for x in sig if _es_numero(x))
        puntaje = textos + (0.5 * textos if nums >= 2 else 0)
        if textos >= 3 and puntaje > mejor:
            mejor, idx = puntaje, i
    return idx


def _es_numero(x):
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return x == x
    try:
        float(str(x).replace(",", "."))
        return True
    except (TypeError, ValueError):
        return False


def _a_dataframe(matriz):
    h = _fila_encabezado(matriz)
    ncol = max(len(f) for f in matriz)
    enc, vistos = [], {}
    for j in range(ncol):
        x = matriz[h][j] if j < len(matriz[h]) else None
        nombre = str(x).strip() if x is not None and str(x).strip() and str(x) != "nan" else f"columna_{j + 1}"
        if nombre in vistos:
            vistos[nombre] += 1; nombre = f"{nombre}_{vistos[nombre]}"
        else:
            vistos[nombre] = 0
        enc.append(nombre)
    filas = [list(f) + [None] * (ncol - len(f)) for f in matriz[h + 1:]]
    df = pd.DataFrame(filas, columns=enc).dropna(how="all")
    return df.loc[:, [c for c in df.columns if not (c.startswith("columna_") and df[c].isna().all())]]


def _leer_xlsx(datos: bytes) -> pd.DataFrame:
    """Lector de .xlsx sin librerías externas (un .xlsx es un ZIP con XML).
    Lee la primera hoja; la primera fila no vacía son los encabezados."""
    import re
    import zipfile
    import xml.etree.ElementTree as ET
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rel_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    z = zipfile.ZipFile(io.BytesIO(datos))
    compartidas = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns):
            compartidas.append("".join(t.text or "" for t in si.iter("{%s}t" % ns["m"])))
    # primera hoja según workbook.xml
    hoja = "xl/worksheets/sheet1.xml"
    try:
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        rid = wb.find("m:sheets/m:sheet", ns).get(rel_ns)
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        for r in rels:
            if r.get("Id") == rid:
                t = r.get("Target").lstrip("/")
                hoja = t if t.startswith("xl/") else "xl/" + t
    except Exception:
        pass
    # estilos con formato de fecha
    fechas_estilo = set()
    try:
        st_ = ET.fromstring(z.read("xl/styles.xml"))
        fmt_fecha = {14, 15, 16, 17, 18, 19, 20, 21, 22, 45, 46, 47}
        for nf in st_.findall("m:numFmts/m:numFmt", ns):
            if re.search(r"[dmyhs]", (nf.get("formatCode") or "").lower().replace("general", "")):
                fmt_fecha.add(int(nf.get("numFmtId")))
        for i, xf in enumerate(st_.findall("m:cellXfs/m:xf", ns)):
            if int(xf.get("numFmtId", 0)) in fmt_fecha:
                fechas_estilo.add(i)
    except Exception:
        pass

    def col_idx(ref):
        letras = re.match(r"[A-Z]+", ref).group(0)
        n = 0
        for ch in letras:
            n = n * 26 + ord(ch) - 64
        return n - 1

    filas, cols_fecha = [], set()
    for _ev, el in ET.iterparse(z.open(hoja)):
        if el.tag == "{%s}row" % ns["m"]:
            fila = {}
            for c in el.findall("m:c", ns):
                j = col_idx(c.get("r")); t = c.get("t"); v = c.find("m:v", ns)
                if t == "s" and v is not None:
                    val = compartidas[int(v.text)]
                elif t == "inlineStr":
                    val = "".join(x.text or "" for x in c.iter("{%s}t" % ns["m"]))
                elif v is None:
                    val = None
                elif t in ("str", "e"):
                    val = v.text
                elif t == "b":
                    val = v.text == "1"
                else:
                    try:
                        val = float(v.text)
                    except (TypeError, ValueError):
                        val = v.text
                    if c.get("s") and int(c.get("s")) in fechas_estilo:
                        cols_fecha.add(j)
                fila[j] = val
            if fila:
                filas.append(fila)
            el.clear()
    if not filas:
        raise ValueError("La hoja de Excel está vacía.")
    ncol = max(max(f) for f in filas) + 1
    matriz = [[f.get(j) for j in range(ncol)] for f in filas]
    df = _a_dataframe(matriz)
    enc = list(df.columns)
    for j in cols_fecha:  # números de serie de Excel -> fecha
        if j >= len(enc):
            continue
        c = enc[j]
        num = pd.to_numeric(df[c], errors="coerce")
        if num.notna().sum() == 0 or not num.dropna().between(1, 120000).all():
            continue
        df[c] = pd.to_datetime(num, unit="D", origin="1899-12-30").where(num.notna(), df[c])
    for c in df.columns:  # convierte a número las columnas que lo son
        if df[c].dtype == object:
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.notna().sum() >= 0.95 * df[c].notna().sum() and df[c].notna().any():
                df[c] = conv
    return df.dropna(how="all")


def leer_tabla(nombre, datos: bytes) -> pd.DataFrame:
    if nombre.lower().endswith((".xlsx", ".xlsm")):
        try:
            return _leer_xlsx(datos)        # lector propio: detecta encabezados y no necesita openpyxl
        except Exception:
            return pd.read_excel(io.BytesIO(datos))
    if nombre.lower().endswith(".xls"):
        try:
            return pd.read_excel(io.BytesIO(datos))
        except ImportError:
            raise ValueError("El formato .xls antiguo necesita 'xlrd'. Ábralo en Excel y guárdelo como .xlsx o CSV.")
    for enc in ("utf-8-sig", "latin-1"):
        try:
            texto = datos.decode(enc)
            import csv
            muestra = texto[:20000]
            sep = ";" if muestra.count(";") > muestra.count(",") else ","
            filas = list(csv.reader(io.StringIO(texto), delimiter=sep))
            filas = [f for f in filas if any(c.strip() for c in f)]
            df = _a_dataframe(filas)
            for c in df.columns:
                conv = pd.to_numeric(df[c].astype(str).str.replace(",", ".") if sep == ";" else df[c], errors="coerce")
                if conv.notna().sum() >= 0.95 * df[c].replace("", None).notna().sum() and conv.notna().any():
                    df[c] = conv
                else:
                    df[c] = df[c].where(df[c].astype(str).str.strip() != "", None)
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError("No se pudo leer el archivo (codificación desconocida).")


def leer_modelo(datos: bytes):
    """Abre un modelo .pkl/.joblib. Requiere que esté instalada la librería con que se entrenó (p. ej. scikit-learn)."""
    try:
        import joblib
        return joblib.load(io.BytesIO(datos))
    except ImportError:
        try:
            return pickle.loads(datos)
        except ModuleNotFoundError as e:
            raise ValueError(f"Para abrir este modelo hay que instalar la librería '{e.name}' (pip install {e.name}). "
                             "Mientras tanto, la app funciona con el archivo CSV de resultados.")


def ranura(tipo, nombre, df, origen):
    import uuid
    return {"nombre": nombre, "df": df, "mapeo": detectar(df.columns, tipo), "origen": origen, "uid": uuid.uuid4().hex[:8]}


# ------------------------------------------------------------------ inicialización
def iniciar():
    ss = st.session_state
    if "ranuras" in ss:
        return
    cargar_demo()


def cargar_demo():
    ss = st.session_state
    ss.ranuras = {t: ranura(t, f, leer_tabla(f, (DEMO / f).read_bytes()), "demo") for t, f in ARCH_DEMO.items()}
    ss.modelos = {}
    for t, f in MOD_DEMO.items():
        try:  # si la versión de scikit-learn es distinta, el .pkl demo puede no abrir: se sigue sin él
            ss.modelos[t] = {"nombre": f, "obj": leer_modelo((DEMO / f).read_bytes()), "origen": "demo"}
        except Exception:  # sin scikit-learn instalado: se usa solo el CSV de resultados
            ss.modelos[t] = None
    ss.pt = {"nombre": "agente_demo.npz", "bytes": (DEMO / "agente_demo.npz").read_bytes(), "origen": "demo"}
    ss.cfg = {**CFG_BASE, "entorno": dict(CFG_BASE["entorno"])}
    ss.pop("R", None); ss.pop("agente", None)
    ss.cache = {}
    procesar_todo()


def origen_datos():
    r = st.session_state.ranuras
    o = {v["origen"] for k, v in r.items() if v is not None}
    return "demo" if o == {"demo"} else ("equipo" if o == {"equipo"} else "mixto")


# ------------------------------------------------------------------ procesamiento
@st.cache_data(show_spinner=False, max_entries=4)
def _procesar_cache(cat, ofi, sup, clu, cfg_txt, clave_modelos, _sup_m, _clu_m):
    import json
    return procesar(cat, ofi, json.loads(cfg_txt), sup, clu, _sup_m, _clu_m)


def df_mapeado(tipo):
    r = st.session_state.ranuras.get(tipo)
    if r is None:
        return None
    return aplicar_mapeo(r["df"], r["mapeo"])


def errores_mapeo():
    errs = []
    for t, r in st.session_state.ranuras.items():
        if r is not None:
            f = faltantes_obligatorias(r["mapeo"], t)
            if f:
                errs.append(f"{NOMBRES[t]}: falta asignar {', '.join(f)}")
    return errs


def procesar_todo():
    import json
    ss = st.session_state
    errs = errores_mapeo()
    if errs:
        raise ValueError(" · ".join(errs))
    m = ss.modelos
    sup_m = m.get("supervisado", {}).get("obj") if m.get("supervisado") else None
    clu_m = m.get("clusters", {}).get("obj") if m.get("clusters") else None
    clave = "|".join(f"{k}:{v['nombre']}" for k, v in m.items() if v)
    R = _procesar_cache(df_mapeado("catalogo"), df_mapeado("oficial"), df_mapeado("supervisado"), df_mapeado("clusters"),
                        json.dumps(ss.cfg, default=str, sort_keys=True), clave, sup_m, clu_m)
    ss.R = R
    ss.cache = {}
    cargar_agente_activo()
    return R


def cargar_agente_activo():
    """Si hay un agente .npz cargado, lo usa. Si no, la sección del agente ofrece entrenar."""
    ss = st.session_state
    ss.pop("agente", None)
    ss.pop("agente_error", None)
    if not ss.get("pt"):
        return
    try:
        ag, info = servicio.cargar_agente(ss.pt["bytes"], ss.R["cfg"]["entorno"], ss.R["huella"])
        info["nombre"] = ss.pt["nombre"]
        ss.agente = {"obj": ag, "info": info, "id": f"pt:{ss.pt['nombre']}:{len(ss.pt['bytes'])}"}
    except Exception as e:
        ss.agente_error = str(e)


def cfg_entorno():
    """Configuración del entorno vigente (la del agente si existe, si no la de los datos)."""
    ss = st.session_state
    ce = dict(ss.R["cfg"]["entorno"])
    return ce


# ------------------------------------------------------------------ cálculos del agente (con caché en sesión)
def _cache(clave, fn):
    c = st.session_state.setdefault("cache", {})
    if clave not in c:
        c[clave] = fn()
    return c[clave]


def evaluacion():
    ss = st.session_state
    if "agente" not in ss:
        return None
    return _cache(("eval", ss.R["huella"], ss.agente["id"]),
                  lambda: servicio.evaluar(ss.agente["obj"], ss.R["D"], cfg_entorno()))


def priorizacion(t, presupuesto, explicar=True):
    ss = st.session_state
    return _cache(("prior", ss.R["huella"], ss.agente["id"], t, round(float(presupuesto), 2), explicar),
                  lambda: servicio.priorizar(ss.agente["obj"], ss.R["D"], cfg_entorno(), t, presupuesto, explicar=explicar))
