# Exporta los insumos reales al formato que pide la app Streamlit (seccion "Cargar datos").
# Uso: python notebooks/04_exportar_para_app.py
import joblib
import pandas as pd
from pathlib import Path

# Encuentra la raíz del repo (sube carpetas hasta hallar "datos").
RAIZ = Path.cwd()
while not (RAIZ / "datos").exists() and RAIZ.parent != RAIZ:
    RAIZ = RAIZ.parent

SALIDA = RAIZ / "datos" / "para_app"
SALIDA.mkdir(parents=True, exist_ok=True)

# 1) Catalogo SGC: columnas que la app reconoce sola (fecha, latitud, longitud, profundidad, magnitud).
eventos = pd.read_csv(RAIZ / "datos/processed/dataset_limpio.csv",
                      usecols=["fecha", "lat", "lon", "prof_km", "mag", "municipio"])
eventos = eventos.rename(columns={"lat": "latitud", "lon": "longitud",
                                  "prof_km": "profundidad", "mag": "magnitud"})
eventos.to_csv(SALIDA / "catalogo_sgc.csv", index=False, encoding="utf-8")
print("catalogo_sgc.csv:", f"{len(eventos):,} filas")

# 2) Clusters: formato municipio, cluster, nombre_cluster, PC1, PC2 (el de la app).
paquete = joblib.load(RAIZ / "modelos/no_supervisado/modelo_no_supervisado.joblib")
F, Z = paquete["F"], paquete["Z"]
clusters = pd.DataFrame({
    "municipio": F.index,
    "cluster": F["grupo_num"],
    "nombre_cluster": F["grupo"],
    "PC1": Z[:, 0],
    "PC2": Z[:, 1],
}).reset_index(drop=True)
clusters.to_csv(SALIDA / "clusters_resultados.csv", index=False, encoding="utf-8")
print("clusters_resultados.csv:", len(clusters), "filas")
