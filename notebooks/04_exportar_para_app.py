# Exporta los insumos reales al formato que pide la app Streamlit (seccion "Cargar datos").
# Incluye el relleno de los municipios que no alcanzaron el mínimo de sismos del modelo no
# supervisado: se asignan al grupo del centroide más cercano con el pipeline guardado.
# Uso: python notebooks/04_exportar_para_app.py
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Encuentra la raíz del repo (sube carpetas hasta hallar "datos").
RAIZ = Path.cwd()
while not (RAIZ / "datos").exists() and RAIZ.parent != RAIZ:
    RAIZ = RAIZ.parent

SALIDA = RAIZ / "datos" / "para_app"
SALIDA.mkdir(parents=True, exist_ok=True)

# 1) Catalogo SGC: columnas que la app reconoce sola (fecha, latitud, longitud, profundidad, magnitud).
COLS_EVENTOS = ["fecha", "lat", "lon", "prof_km", "mag", "tipo_mag",
                "n_fases", "rms_seg", "gap_grados", "municipio"]
eventos = pd.read_csv(RAIZ / "datos/processed/dataset_limpio.csv", usecols=COLS_EVENTOS,
                      parse_dates=["fecha"])
catalogo = eventos[["fecha", "lat", "lon", "prof_km", "mag", "municipio"]].rename(
    columns={"lat": "latitud", "lon": "longitud", "prof_km": "profundidad", "mag": "magnitud"})
catalogo.to_csv(SALIDA / "catalogo_sgc.csv", index=False, encoding="utf-8")
print("catalogo_sgc.csv:", f"{len(catalogo):,} filas")

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

# ---------------------------------------------------------------------------------------
# 3) Municipios sin grupo: no alcanzaron el mínimo de sismos del modelo (MIN_EVENTOS=15),
# así que la app los marcaba como "sin grupo" (G5). Se calculan SUS variables con la misma
# receta del notebook (ventana 2006+, solo magnitud Ml) y se asignan al grupo del centroide
# más cercano usando el pipeline congelado: escalado -> PCA -> KMeans.
MIN_EVENTOS = 15
ANIO_INICIO = 2006


def haversine(lat1, lon1, lat2, lon2):
    """Distancia en km entre dos puntos del planeta."""
    R = 6371.0
    la1, lo1, la2, lo2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def variables_municipio(df, nombres, mediana):
    """Réplica fiel de las variables por municipio del notebook no supervisado."""
    g = df.groupby("municipio")
    T = pd.DataFrame({
        "n_sismos": g.size(),
        "sismos_x_anio": g.size() / df["fecha"].dt.year.nunique(),
        "mag_media": g["mag"].mean(),
        "mag_max": g["mag"].max(),
        "mag_std": g["mag"].std(),
        "mag_p90": g["mag"].quantile(0.90),
        "pct_m3_mas": g["es_m3"].mean(),
        "pct_m4_mas": g["es_m4"].mean(),
        "log_energia": np.log10(g["energia"].sum()),
        "prof_media": g["prof_km"].mean(),
        "prof_std": g["prof_km"].std(),
        "pct_superficial": g["sup"].mean(),
        "pct_intermedia": g["inter"].mean(),
        "pct_profunda": g["pro"].mean(),
        "rms_medio": g["rms_seg"].mean(),
        "gap_medio": g["gap_grados"].mean(),
        "fases_medias": g["n_fases"].mean(),
        "frac_reciente": g["reciente"].mean(),
        "lat_centro": g["lat"].mean(),
        "lon_centro": g["lon"].mean(),
    }).loc[nombres]
    # Distancia al nido sísmico de Bucaramanga (mismas coordenadas del notebook).
    T["dist_nido_km"] = haversine(T["lat_centro"], T["lon_centro"], 6.8, -73.1)
    # Los huecos se rellenan con la mediana del entrenamiento (F ya viene imputado en el joblib).
    T = T.fillna(mediana)
    # Distancia a la falla más cercana usando los mismos vértices guardados en el paquete.
    puntos = paquete["puntos"]
    T["dist_falla_km"] = [float(haversine(la, lo, puntos[:, 1], puntos[:, 0]).min())
                          for la, lo in zip(T["lat_centro"], T["lon_centro"])]
    return T


faltantes = sorted(set(eventos["municipio"]) - set(F.index))
if faltantes:
    # Misma ventana y magnitud del entrenamiento; sin ella las variables no serían comparables.
    ml = eventos[(eventos["fecha"].dt.year >= ANIO_INICIO) & (eventos["tipo_mag"] == "Ml")].copy()
    ml["es_m3"] = (ml["mag"] >= 3).astype(int)
    ml["es_m4"] = (ml["mag"] >= 4).astype(int)
    ml["energia"] = 10 ** (1.5 * ml["mag"] + 4.8)   # Gutenberg-Richter
    ml["reciente"] = (ml["fecha"] >= ml["fecha"].max() - pd.DateOffset(years=5)).astype(int)
    ml["sup"] = (ml["prof_km"] < 30).astype(int)
    ml["inter"] = ml["prof_km"].between(30, 150).astype(int)
    ml["pro"] = (ml["prof_km"] >= 150).astype(int)

    T = variables_municipio(ml, faltantes, mediana=F.median(numeric_only=True))
    # Aplica el pipeline congelado en el mismo orden de columnas con que se entrenó.
    X = T[paquete["columnas"]].copy()
    X[paquete["cols_log"]] = np.log1p(X[paquete["cols_log"]])
    Xs = np.clip(paquete["escalador"].transform(X), -5, 5)
    Zn = paquete["pca"].transform(Xs)
    grupo_num = paquete["modelo_final"].predict(Zn)

    # Número de grupo -> nombre legible ("Muy alto", "Bajo"...), tomado del propio F.
    nombre_grupo = dict(zip(F["grupo_num"], F["grupo"]))
    # Distancia de cada municipio nuevo a los 4 centroides (control de la asignación).
    centros = paquete["modelo_final"].cluster_centers_
    distancias = np.linalg.norm(Zn[:, None, :] - centros[None, :, :], axis=2)
    for i, m in enumerate(T.index):
        detalle = " | ".join(f"G{g}: {v:.2f}" for g, v in enumerate(distancias[i]))
        print(f"  {m}: {detalle} -> G{grupo_num[i]} {nombre_grupo[grupo_num[i]]}")
    nuevos = pd.DataFrame({
        "municipio": T.index,
        "cluster": grupo_num,
        "nombre_cluster": [nombre_grupo[g] for g in grupo_num],
        "PC1": Zn[:, 0],
        "PC2": Zn[:, 1],
    })
    clusters = pd.concat([clusters, nuevos], ignore_index=True)
    print("Asignados por centroide:", len(nuevos), "->", ", ".join(
        f"{m} ({n})" for m, n in zip(nuevos['municipio'], nuevos['nombre_cluster'])))

clusters.to_csv(SALIDA / "clusters_resultados.csv", index=False, encoding="utf-8")
print("clusters_resultados.csv:", len(clusters), "filas")
