# Construye la fuente oficial municipal para la app: población censal DANE 2018 (servicio
# del geoportal DANE), área y provincia (anexo de municipios de Santander, con datos DANE
# citados) y centroides reales (promedio de los sismos del catálogo limpio del SGC).
# Uso: python notebooks/06_fuente_oficial_para_app.py
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

# Encuentra la raíz del repo (sube carpetas hasta hallar "datos").
RAIZ = Path.cwd()
while not (RAIZ / "datos").exists() and RAIZ.parent != RAIZ:
    RAIZ = RAIZ.parent

SALIDA = RAIZ / "datos/para_app/fuente_oficial.csv"
CRUDO_POBLACION = RAIZ / "datos/raw/dane_cnpv2018_municipios_santander.json"
CRUDO_AREAS = RAIZ / "datos/raw/municipios_santander_area_provincia_wikipedia.json"

# La fuente escribe algunos nombres distinto al catálogo (o al SGC): alias normalizados.
ALIAS = {
    "EL CARMEN": "EL CARMEN DE CHUCURI",
    "GIRON": "SAN JUAN DE GIRON",
    "SOCORRO": "EL SOCORRO",
    "JORDAN": "JORDAN SUBE",   # la DIVIPOLA lo llama "Jordán Sube"
}


def norm(s):
    """Normaliza nombres: sin tildes, mayúsculas, sin paréntesis ni guiones."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"\(.*?\)", "", s)
    return " ".join(s.upper().replace("-", " ").split())


# --- Población censal 2018 (DANE, CNPV): 87 municipios de Santander ---
dane = json.loads(CRUDO_POBLACION.read_text(encoding="utf-8"))
poblacion = {norm(f["attributes"]["MPIO_CNMBR"]): {"codigo_dane": f["attributes"]["MPIO_CCDGO"],
                                                    "poblacion": int(f["attributes"]["CL0_TT_PERSN"])}
             for f in dane["features"]}

# --- Área y provincia (anexo de Wikipedia de municipios de Santander, consultado hoy) ---
wiki = json.loads(CRUDO_AREAS.read_text(encoding="utf-8"))["municipios"]
wiki_norm = {norm(k): v for k, v in wiki.items()}

# --- Centroide real de cada municipio: promedio de sus sismos en el catálogo ---
eventos = pd.read_csv(RAIZ / "datos/processed/dataset_limpio.csv", usecols=["lat", "lon", "municipio"])
centro = eventos.groupby("municipio")[["lat", "lon"]].mean()

filas, sin_poblacion, sin_areas = [], [], []
for municipio in sorted(centro.index):
    clave = norm(municipio)
    p = poblacion.get(ALIAS.get(clave, clave)) or poblacion.get(clave)
    w = wiki_norm.get(ALIAS.get(clave, clave)) or wiki_norm.get(clave)
    if p is None:
        sin_poblacion.append(municipio)
    if w is None:
        sin_areas.append(municipio)
    filas.append({
        "codigo_dane": p["codigo_dane"] if p else "",
        # El anexo escribe "Provincia de X": se deja solo el nombre de la provincia.
        "provincia": w["provincia"].replace("Provincia de ", "").replace("Provincia ", "") if w else "—",
        "municipio": municipio,
        "latitud": round(float(centro.loc[municipio, "lat"]), 5),
        "longitud": round(float(centro.loc[municipio, "lon"]), 5),
        "poblacion": p["poblacion"] if p else None,
        "area_km2": w["area_km2"] if w else None,
    })

# Sin los cuatro campos obligatorios la app no acepta el archivo: se falla ruidosamente.
if sin_poblacion or sin_areas:
    raise SystemExit(f"Sin población: {sin_poblacion} | sin área o provincia: {sin_areas}")

tabla = pd.DataFrame(filas)
tabla.to_csv(SALIDA, index=False, encoding="utf-8")
print("fuente_oficial.csv:", len(tabla), "municipios")
print(tabla.head(3).to_string(index=False))
print("población total:", f"{int(tabla['poblacion'].sum()):,}", "| Bucaramanga:",
      f"{int(tabla.loc[tabla['municipio'] == 'BUCARAMANGA', 'poblacion'].iloc[0]):,}")
