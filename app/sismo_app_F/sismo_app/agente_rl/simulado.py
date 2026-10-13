"""
PASO 1 · DATOS SIMULADOS (solo mientras llegan los reales)
==========================================================
Genera dos archivos con LA MISMA ESTRUCTURA que tendrán los datos reales:

  1. catalogo_sgc.csv    -> sismos tipo catálogo del Servicio Geológico Colombiano
  2. dane_municipios.csv -> fuente oficial territorial (tipo DANE: población, NBI...)

Qué imita para que sea realista:
  * El Nido Sísmico de Bucaramanga (Mesa de los Santos): muchísimos sismos
    INTERMEDIOS (~140-160 km de profundidad), uno de los nidos más activos del mundo.
  * Sismicidad SUPERFICIAL a lo largo de la Falla Bucaramanga-Santa Marta.
  * Magnitudes con la ley de Gutenberg-Richter (b ≈ 1): muchos pequeños, pocos grandes.
  * Réplicas: después de un sismo M>=4 aumenta la actividad local (memoria temporal),
    lo que hace que el pasado sí ayude a anticipar el futuro.

IMPORTANTE: las coordenadas de municipios son APROXIMADAS y los datos son FICTICIOS.
Sirven para construir y probar el flujo; la presentación final usa los datos reales.

Uso: from agente_rl.simulado import generar ; cat, ofi = generar()
"""
from pathlib import Path
import numpy as np
import pandas as pd

# (municipio, lat, lon) aproximados de Santander. Reemplazar por DIVIPOLA en real.
MUNICIPIOS = [
    ("BUCARAMANGA", 7.119, -73.122), ("FLORIDABLANCA", 7.064, -73.090),
    ("GIRÓN", 7.070, -73.170), ("PIEDECUESTA", 6.990, -73.050),
    ("LOS SANTOS", 6.755, -73.100), ("BARRANCABERMEJA", 7.065, -73.850),
    ("SAN GIL", 6.555, -73.134), ("SOCORRO", 6.468, -73.260),
    ("BARBOSA", 5.930, -73.620), ("VÉLEZ", 6.010, -73.670),
    ("MÁLAGA", 6.700, -72.730), ("LEBRIJA", 7.110, -73.220),
    ("RIONEGRO", 7.270, -73.150), ("SABANA DE TORRES", 7.390, -73.500),
    ("PUERTO WILCHES", 7.350, -73.900), ("ZAPATOCA", 6.810, -73.270),
    ("ARATOCA", 6.690, -73.020), ("CEPITÁ", 6.750, -72.970),
    ("MOGOTES", 6.470, -72.970), ("CHARALÁ", 6.290, -73.150),
    ("OIBA", 6.260, -73.300), ("CIMITARRA", 6.310, -73.950),
    ("LANDÁZURI", 6.220, -73.810), ("MATANZA", 7.320, -73.020),
    ("TONA", 7.200, -73.020), ("CALIFORNIA", 7.350, -72.950),
    ("VETAS", 7.310, -72.870), ("CURITÍ", 6.610, -73.070),
    ("BETULIA", 6.900, -73.280), ("CONCEPCIÓN", 6.770, -72.690),
    ("SAN VICENTE DE CHUCURÍ", 6.880, -73.410), ("EL CARMEN DE CHUCURÍ", 6.700, -73.510),
    ("PUERTO PARRA", 6.650, -74.060), ("SIMACOTA", 6.440, -73.340),
    ("BARICHARA", 6.630, -73.220),
]
NIDO = (6.80, -73.10)  # centro aproximado del Nido de Bucaramanga
PROVINCIA = {**{m: "Metropolitana" for m in ["BUCARAMANGA", "FLORIDABLANCA", "GIRÓN", "PIEDECUESTA"]},
             **{m: "Soto" for m in ["LOS SANTOS", "LEBRIJA", "RIONEGRO", "MATANZA", "TONA", "CALIFORNIA", "VETAS"]},
             **{m: "Mares" for m in ["BARRANCABERMEJA", "SABANA DE TORRES", "PUERTO WILCHES", "BETULIA",
                                     "SAN VICENTE DE CHUCURÍ", "EL CARMEN DE CHUCURÍ", "PUERTO PARRA", "ZAPATOCA"]},
             **{m: "Guanentá" for m in ["SAN GIL", "ARATOCA", "CEPITÁ", "MOGOTES", "CHARALÁ", "CURITÍ", "BARICHARA"]},
             **{m: "Comunera" for m in ["SOCORRO", "OIBA", "SIMACOTA"]},
             **{m: "Vélez" for m in ["BARBOSA", "VÉLEZ", "CIMITARRA", "LANDÁZURI"]},
             **{m: "García Rovira" for m in ["MÁLAGA", "CONCEPCIÓN"]}}


def _gutenberg_richter(rng, n, mmin=1.5, b=1.0):
    """Magnitudes con distribución exponencial (ley de Gutenberg-Richter)."""
    beta = b * np.log(10)
    return mmin + rng.exponential(1 / beta, n)


def generar_catalogo(rng, inicio="2010-01-01", fin="2025-12-31"):
    dias = pd.date_range(inicio, fin, freq="D")
    eventos = []
    tasa_replica = np.zeros(len(MUNICIPIOS))  # actividad extra por réplicas
    lats = np.array([m[1] for m in MUNICIPIOS]); lons = np.array([m[2] for m in MUNICIPIOS])
    dist_nido = np.hypot(lats - NIDO[0], lons - NIDO[1])
    dist_falla = np.abs(lons - (-73.10 + 0.25 * (lats - 7.0)))  # traza simplificada

    for semana in range(0, len(dias), 7):
        fecha0 = dias[semana]
        # 1) Nido: ~10 sismos/semana, profundos, concentrados
        n = rng.poisson(10)
        for _ in range(n):
            lat = rng.normal(NIDO[0], 0.06); lon = rng.normal(NIDO[1], 0.06)
            eventos.append((fecha0 + pd.Timedelta(days=int(rng.integers(0, 7))),
                            lat, lon, rng.normal(148, 8), _gutenberg_richter(rng, 1)[0]))
        # 2) Corticales en la falla + fondo regional + réplicas
        tasa = 0.35 * np.exp(-dist_falla / 0.15) + 0.04 + tasa_replica
        conteos = rng.poisson(tasa)
        for i, c in enumerate(conteos):
            for _ in range(c):
                mag = _gutenberg_richter(rng, 1, b=0.9)[0]
                eventos.append((fecha0 + pd.Timedelta(days=int(rng.integers(0, 7))),
                                lats[i] + rng.normal(0, 0.05), lons[i] + rng.normal(0, 0.05),
                                abs(rng.normal(15, 8)), mag))
                if mag >= 4.0:  # dispara réplicas en las próximas semanas
                    tasa_replica[i] += 0.8 * (mag - 3.5)
        tasa_replica *= 0.85  # las réplicas se apagan con el tiempo (Omori simplificado)

    df = pd.DataFrame(eventos, columns=["fecha", "lat", "lon", "prof", "mag"])
    # Asignar municipio por cercanía (así lo haremos también si el real no lo trae)
    d = np.hypot(df["lat"].values[:, None] - lats, df["lon"].values[:, None] - lons)
    df["municipio"] = [MUNICIPIOS[j][0] for j in d.argmin(1)]
    df = df.sort_values("fecha")
    # Mismos nombres de columna que el catálogo del SGC (ver config.yaml)
    return pd.DataFrame({
        "FECHA - HORA UTC": df["fecha"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "LATITUD (°)": df["lat"].round(4), "LONGITUD (°)": df["lon"].round(4),
        "PROFUNDIDAD (Km)": df["prof"].round(1), "MAGNITUD": df["mag"].round(1),
        "MUNICIPIO": df["municipio"], "DEPARTAMENTO": "SANTANDER",
    })


def generar_fuente_oficial(rng):
    filas = []
    for nombre, lat, lon in MUNICIPIOS:
        urbano = nombre in {"BUCARAMANGA", "FLORIDABLANCA", "GIRÓN", "PIEDECUESTA", "BARRANCABERMEJA"}
        pob = int(rng.uniform(250_000, 600_000) if urbano else rng.lognormal(9.6, 0.7))
        filas.append({
            "codigo_dane": f"68{len(filas) * 7 + 1:03d}", "provincia": PROVINCIA.get(nombre, "—"),
            "municipio": nombre, "latitud": lat, "longitud": lon, "poblacion": pob,
            "area_km2": round(rng.uniform(80, 1200), 1),
            "nbi": round(rng.uniform(5, 15) if urbano else rng.uniform(12, 45), 1),
            "pct_rural": round(rng.uniform(3, 15) if urbano else rng.uniform(35, 85), 1),
            "altitud_m": int(rng.uniform(100, 2800)),
            "pct_viviendas_precarias": round(rng.uniform(3, 12) if urbano else rng.uniform(10, 40), 1),
            "hospitales": int(rng.integers(3, 15) if urbano else rng.integers(0, 2)),
        })
    return pd.DataFrame(filas)


def generar(semilla=42):
    rng = np.random.default_rng(semilla)
    return generar_catalogo(rng), generar_fuente_oficial(rng)
