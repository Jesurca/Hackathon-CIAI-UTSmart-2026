#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unifica y limpia los catalogos sismicos del SGC (formatos viejo y nuevo) en un CSV.

Uso:
    python notebooks/01_limpieza_catalogos.py

Lee todos los .xlsx de datos/raw/, detecta el formato de cada archivo, estandariza
las columnas, elimina duplicados y escribe datos/processed/dataset_limpio.csv
(solo eventos, sin columnas de modelo).

Notas:
- El hueco 2014-2017 no esta descargado; el QC imprime la cobertura anual.
- La magnitud se guarda con su tipo (mag + tipo_mag); el filtro lo decide el modelo.
- Requiere pandas, openpyxl y python-calamine (los xlsx del SGC traen estilos invalidos).
"""

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

RUTA_RAW = Path("datos/raw")
RUTA_SALIDA = Path("datos/processed/dataset_limpio.csv")

# Contrato de salida: solo eventos, una fila por sismo observado.
COLUMNAS_SALIDA = [
    "fecha", "lat", "lon", "prof_km", "mag", "tipo_mag",
    "n_fases", "rms_seg", "gap_grados",
    "err_lat_km", "err_lon_km", "err_prof_km",
    "estado", "municipio", "departamento", "archivo_origen", "catalogo",
]


def norm(texto):
    """Minusculas sin tildes ni signos, para comparar encabezados."""
    s = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    # Primero minusculas: si no, el regex borraria las mayusculas como si fueran signos.
    s = s.lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def norm_municipio(texto):
    """Municipio canonico: mayusculas sin tildes, sin guiones bajos y con espacios simples."""
    if pd.isna(texto):
        return np.nan
    s = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    # El catalogo viejo trae "LOS_SANTOS" y el nuevo "Los Santos": se unifican aqui.
    return re.sub(r"[_\s]+", " ", s).strip().upper()


def norm_tipo_mag(valor):
    """Canonicaliza el tipo de magnitud del catalogo a Mw, Ml, Mb u Otro."""
    if pd.isna(valor):
        return np.nan
    t = norm(valor)
    # El formato nuevo trae variantes como "MLr_3", "MLv" o "Mw(mB)".
    if "mw" in t:
        return "Mw"
    if "ml" in t:
        return "Ml"
    if "mb" in t:
        return "Mb"
    return "Otro"


def destino_columna(col):
    """Mapea el nombre original de una columna al nombre canonico."""
    n = norm(col)
    # Las columnas de error van primero: contienen "latitud"/"longitud"/"prof".
    if "error" in n and "latitud" in n:
        return "err_lat_km"
    if "error" in n and "longitud" in n:
        return "err_lon_km"
    if "error" in n and "prof" in n:
        return "err_prof_km"
    # El formato nuevo junta fecha y hora en una sola columna.
    if "fecha" in n and "hora" in n:
        return "fecha_hora"
    if "fecha" in n:
        return "fecha"
    if "hora" in n:
        return "hora"
    if "latitud" in n:
        return "lat"
    if "longitud" in n:
        return "lon"
    if "prof" in n:
        return "prof_km"
    # "tipo magnitud" debe resolverse antes que "magnitud" a secas.
    if "tipo" in n and "mag" in n:
        return "tipo_mag"
    if "magnitud" in n and "mw" in n:
        return "mag_mw"
    if "magnitud" in n and "ml" in n:
        return "mag_ml"
    if "magnitud" in n:
        return "mag"
    if "fases" in n:
        return "n_fases"
    if n.startswith("rms"):
        return "rms_seg"
    if n.startswith("gap"):
        return "gap_grados"
    if "municipio" in n:
        return "municipio"
    if "departamento" in n:
        return "departamento"
    if "region" in n:
        return "region"
    if "estado" in n:
        return "estado"
    return None


def leer_excel(ruta, **kwargs):
    """Lee un xlsx tolerando archivos con estilos invalidos.

    calamine ignora los estilos (varios xlsx del SGC traen colores invalidos que
    openpyxl rechaza); si calamine no esta instalado, cae a openpyxl.
    """
    try:
        return pd.read_excel(ruta, engine="calamine", **kwargs)
    except Exception:
        return pd.read_excel(ruta, engine="openpyxl", **kwargs)


def localizar_encabezado(crudo):
    """Indice de la fila de encabezado: la que trae latitud, longitud y magnitud."""
    for i, fila in crudo.iterrows():
        texto = " ".join(norm(v) for v in fila.tolist())
        if "latitud" in texto and "longitud" in texto and "magnitud" in texto:
            return i
    return None


def leer_catalogo(ruta):
    """Lee un xlsx del SGC y devuelve un DataFrame estandarizado (o None si falla)."""
    crudo = leer_excel(ruta, header=None, nrows=40)
    fila_encabezado = localizar_encabezado(crudo)
    if fila_encabezado is None:
        print(f"  AVISO: no encontre encabezado en {ruta.name}; se omite")
        return None
    df = leer_excel(ruta, header=fila_encabezado)

    # Renombra solo las columnas reconocidas; las demas se descartan.
    mapa = {}
    for col in df.columns:
        destino = destino_columna(col)
        if destino and destino not in mapa.values():
            mapa[col] = destino
    df = df[list(mapa)].rename(columns=mapa)

    # Fecha: el formato viejo trae fecha y hora separadas; el nuevo, juntas.
    if "fecha_hora" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha_hora"].astype(str).str.strip(), errors="coerce")
    elif "fecha" in df.columns:
        base = df["fecha"].astype(str).str.strip()
        if "hora" in df.columns:
            # Hora vacia -> se conserva la fecha a medianoche en vez de perder el evento.
            hora = df["hora"].astype(str).str.strip().replace({"nan": "", "NaT": ""})
            base = (base + " " + hora).str.strip()
        df["fecha"] = pd.to_datetime(base, errors="coerce")

    for col in ["lat", "lon", "prof_km", "mag", "n_fases", "rms_seg", "gap_grados",
                "err_lat_km", "err_lon_km", "err_prof_km"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Magnitud unificada: el viejo trae Ml y Mw en columnas; el nuevo, valor + tipo.
    if "mag_mw" in df.columns:
        tiene_mw = df["mag_mw"].notna()
        df["mag"] = df["mag_mw"].where(tiene_mw, df.get("mag_ml"))
        df["tipo_mag"] = np.where(tiene_mw, "Mw", "Ml")
    elif "mag_ml" in df.columns:
        df["mag"] = df["mag_ml"]
        df["tipo_mag"] = "Ml"
    elif "mag" not in df.columns:
        print(f"  AVISO: sin columnas de magnitud en {ruta.name}; se omite")
        return None

    # Reduce las variantes del tipo de magnitud a su familia (Mw, Ml, Mb u Otro).
    if "tipo_mag" in df.columns:
        df["tipo_mag"] = df["tipo_mag"].map(norm_tipo_mag)

    # Municipio: el viejo lo trae por columnas; el nuevo, dentro de REGION.
    if "region" in df.columns:
        municipio = df["region"].astype(str).str.split(" - ").str[0]
        df["municipio"] = municipio.where(df["region"].notna())
        df["departamento"] = "SANTANDER"
    df["municipio"] = df["municipio"].map(norm_municipio) if "municipio" in df.columns else np.nan
    df["departamento"] = df["departamento"].map(norm_municipio) if "departamento" in df.columns else np.nan

    df["archivo_origen"] = ruta.name
    df["catalogo"] = "viejo" if ("mag_mw" in df.columns or "mag_ml" in df.columns) else "nuevo"

    # Completa el contrato: si falta una columna, queda vacia y se reporta en el QC.
    for col in COLUMNAS_SALIDA:
        if col not in df.columns:
            df[col] = np.nan
    return df[COLUMNAS_SALIDA]


def main():
    archivos = sorted(RUTA_RAW.glob("*.xlsx"))
    assert archivos, "No encontre archivos .xlsx en datos/raw"
    print(f"Archivos encontrados: {len(archivos)}")

    partes = []
    for ruta in archivos:
        df = leer_catalogo(ruta)
        if df is not None:
            print(f"  {ruta.name}: {len(df):,} filas | catalogo {df['catalogo'].iloc[0]}")
            partes.append(df)
    assert partes, "Ningun archivo se pudo procesar"
    total = pd.concat(partes, ignore_index=True)
    n0 = len(total)

    # Orden cronologico: deja el CSV legible y hace determinista el dedup.
    total = total.sort_values("fecha").reset_index(drop=True)

    # Duplicado exacto = mismo instante, lugar, profundidad y magnitud (redondeo a la precision del dato).
    total["_lat6"] = total["lat"].round(6)
    total["_lon6"] = total["lon"].round(6)
    total["_prof2"] = total["prof_km"].round(2)
    total["_mag1"] = total["mag"].round(1)
    total = total.drop_duplicates(
        subset=["fecha", "_lat6", "_lon6", "_prof2", "_mag1"], keep="first"
    ).drop(columns=["_lat6", "_lon6", "_prof2", "_mag1"]).reset_index(drop=True)
    duplicados = n0 - len(total)
    print(f"\nFilas: {n0:,} -> {len(total):,} (duplicados eliminados: {duplicados:,})")

    faltantes = total.isna().sum()
    print("Nulos por columna (solo > 0):")
    for col, n in faltantes[faltantes > 0].items():
        print(f"  {col}: {n:,} ({n / len(total):.1%})")

    print(f"Rango de fechas: {total.fecha.min()} -> {total.fecha.max()}")
    conteo_anios = total.fecha.dt.year.value_counts().sort_index()
    print("Eventos por anio:")
    for anio, n in conteo_anios.items():
        print(f"  {int(anio)}: {n:,}")
    anios = range(int(total.fecha.min().year), int(total.fecha.max().year) + 1)
    huecos = [a for a in anios if a not in conteo_anios.index]
    print(f"Anios sin datos: {huecos}")
    print(f"Municipios distintos: {total.municipio.nunique()}")
    print("Tipos de magnitud:")
    print(total.tipo_mag.value_counts(dropna=False).to_string())

    cuota_mw = (total.tipo_mag == "Mw").mean() if len(total) else 0.0
    if cuota_mw < 0.5:
        print(f"AVISO: Mw cubre {cuota_mw:.1%} de los eventos; revisa el filtro de magnitud antes de modelar.")

    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    total.to_csv(RUTA_SALIDA, index=False, encoding="utf-8")
    print(f"\nSalida: {RUTA_SALIDA} ({RUTA_SALIDA.stat().st_size / 1024:,.0f} KB)")


if __name__ == "__main__":
    main()
