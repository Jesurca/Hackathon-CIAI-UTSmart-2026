# datos/processed — dataset limpio compartido

Aquí van los datos ya limpios que usan los modelos.

| Archivo | Qué es | Quién lo genera |
|---|---|---|
| `dataset_limpio.csv` | 175.185 sismos de Santander (1993–2013 y 2018–2026). Columnas: fecha, lat, lon, prof_km, mag, tipo_mag, n_fases, rms_seg, gap_grados, err_lat_km, err_lon_km, err_prof_km, estado, municipio, departamento, archivo_origen, catalogo | `notebooks/01_limpieza_catalogos.py` |
| `fallas_santander.geojson` | Fallas visibles del Atlas 2020 en el área de Santander | `notebooks/03_preparar_fallas.py` |

Reglas:

- Cada fila del CSV es un sismo y cada columna una variable (§7 del brief).
- Solo el dueño de datos cambia el CSV; los modelos lo leen tal cual.
- Las columnas se explican en `Presentacion/diccionario_de_datos.md`.
