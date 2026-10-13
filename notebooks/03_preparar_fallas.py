#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepara las fallas visibles del Atlas Geologico de Colombia 2020 (SGC, CC BY 4.0)
para los modelos del hackathon.

Uso:
    python notebooks/03_preparar_fallas.py

Lee el GeoJSON nacional de datos/raw/ (no versionado), descarta lineamientos y
fallas cubiertas o inferidas, recorta al area de Santander, mide longitudes en
metros y exporta datos/processed/fallas_santander.geojson.

Nota: el recorte es provisional por caja de coordenadas; cuando llegue el MGN del
DANE se reemplaza por un recorte con los poligonos municipales.
"""

import json
import math
import statistics
from pathlib import Path

from pyproj import Transformer
from shapely import make_valid
from shapely.geometry import MultiLineString, box, mapping, shape

RUTA_RAW = Path("datos/raw/atlas_fallas_2020_sgc.geojson")
RUTA_SALIDA = Path("datos/processed/fallas_santander.geojson")

# Caja aproximada de Santander (provisional, ver nota del encabezado).
LAT_MIN, LAT_MAX = 5.7, 8.2
LON_MIN, LON_MAX = -74.6, -72.4

A_METROS = Transformer.from_crs("EPSG:4326", "EPSG:3116", always_xy=True)


def es_falla_visible(tipo):
    """Devuelve True solo para trazas de falla con evidencia de campo."""
    t = (tipo or "").lower()
    # "cubierta" e "inferida" se excluyen: el equipo decidio usar solo trazos visibles.
    return t.startswith("falla") and "cubierta" not in t and "inferida" not in t


def clave_geometria(geom):
    """Clave estable para detectar duplicados exactos de una linea."""
    # 7 decimales ~ 1 cm: suficiente para considerar dos trazas identicas.
    if geom["type"] == "LineString":
        return tuple(round(v, 7) for punto in geom["coordinates"] for v in punto)
    return tuple(
        tuple(round(v, 7) for punto in parte for v in punto)
        for parte in geom["coordinates"]
    )


def partes_linea(geom):
    """Aplana cualquier geometria a la lista de sus partes LineString."""
    # Una interseccion vacia puede volver como LineString sin puntos.
    if geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return [geom]
    # Una interseccion tambien puede devolver MultiLineString o GeometryCollection.
    if geom.geom_type in ("MultiLineString", "GeometryCollection"):
        partes = []
        for hija in geom.geoms:
            partes.extend(partes_linea(hija))
        return partes
    return []


def largo_metros(parte):
    """Longitud de una linea en metros, proyectada con la CTM12 (EPSG:3116)."""
    xs = [p[0] for p in parte.coords]
    ys = [p[1] for p in parte.coords]
    px, py = A_METROS.transform(xs, ys)
    # Suma de distancias entre puntos consecutivos ya proyectados (metros).
    return sum(math.dist((px[i], py[i]), (px[i + 1], py[i + 1])) for i in range(len(px) - 1))


def main():
    datos = json.loads(RUTA_RAW.read_text(encoding="utf-8"))
    features = datos["features"]
    print(f"Registros originales: {len(features):,}")

    sin_geometria = no_visibles = duplicados = 0
    vistas = set()
    visibles = []
    for f in features:
        if not f.get("geometry"):
            sin_geometria += 1
            continue
        if not es_falla_visible(f["properties"].get("Tipo")):
            no_visibles += 1
            continue
        clave = clave_geometria(f["geometry"])
        if clave in vistas:
            duplicados += 1
            continue
        vistas.add(clave)
        f["_geom"] = shape(f["geometry"])
        visibles.append(f)
    print(
        f"Sin geometria: {sin_geometria} | no visibles (lineamiento/cubierta/inferida): "
        f"{no_visibles} | duplicados exactos: {duplicados}"
    )

    caja = box(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
    invalidas = 0
    dentro = []
    for f in visibles:
        geom = f["_geom"]
        if not geom.is_valid:
            invalidas += 1
            geom = make_valid(geom)
        recorte = geom.intersection(caja)
        # Descarto tramos con menos de dos puntos: no son lineas medibles.
        partes = [p for p in partes_linea(recorte) if len(p.coords) >= 2]
        if partes:
            dentro.append((f, partes))
    print(f"Geometrias invalidas reparadas: {invalidas} | en area de Santander: {len(dentro)}")

    salida = []
    diferencias = []
    for f, partes in dentro:
        props = f["properties"]
        # Medir en la proyeccion nacional (metros) es mas fiable que la columna de la fuente.
        longitud_m = sum(largo_metros(p) for p in partes)
        fuente = props.get("Shape__Length")
        if isinstance(fuente, (int, float)) and fuente:
            diferencias.append(abs(longitud_m - fuente) / fuente)
        # El GeoJSON de salida se queda en grados (WGS84); la longitud ya va en metros.
        geom_grados = MultiLineString(
            [[(round(x, 6), round(y, 6)) for x, y in p.coords] for p in partes]
        )
        salida.append(
            {
                "type": "Feature",
                "geometry": mapping(geom_grados),
                "properties": {
                    "FID": props.get("FID"),
                    "Tipo": props.get("Tipo"),
                    "NombreFall": (props.get("NombreFall") or "").strip(),
                    "longitud_m": round(longitud_m, 1),
                },
            }
        )

    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    coleccion = {
        "type": "FeatureCollection",
        "name": "fallas_santander_visible",
        "features": salida,
    }
    RUTA_SALIDA.write_text(json.dumps(coleccion, ensure_ascii=False), encoding="utf-8")

    largos = [f["properties"]["longitud_m"] for f in salida]
    print(f"Trazas exportadas: {len(salida)}")
    if largos:
        print(
            f"Longitud total: {sum(largos) / 1000:,.0f} km | mediana por traza: "
            f"{statistics.median(largos):,.0f} m | maxima: {max(largos):,.0f} m"
        )
    if diferencias:
        print(f"Diferencia de longitud vs Shape__Length (mediana): {statistics.median(diferencias):.1%}")
    print(f"Salida: {RUTA_SALIDA} ({RUTA_SALIDA.stat().st_size / 1024:,.0f} KB)")


if __name__ == "__main__":
    main()
