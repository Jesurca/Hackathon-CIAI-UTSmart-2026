# datos/processed — dataset limpio compartido

Contrato de integración de los 3 modelos: `dataset_limpio.csv`.

- Una sola tabla: cada fila es una observación y cada columna una variable (§7 del brief).
- Solo el dueño de datos modifica el CSV; los modelos lo consumen tal cual.
- Las columnas definitivas se documentan en `entregables/diccionario_de_datos.md`.
- Se genera con los cuadernos de `notebooks/` (descarga/unificación y limpieza/EDA).

## Archivos

| Archivo | Qué es | Quién lo genera |
|---|---|---|
| `dataset_limpio.csv` | Contrato compartido de los 3 modelos (pendiente) | Cuadernos 01 y 02 |
| `fallas_santander.geojson` | Fallas visibles (sin lineamientos, cubiertas ni inferidas) en el área de Santander; WGS84; propiedades `FID`, `Tipo`, `NombreFall` y `longitud_m` (metros, EPSG:3116) | `notebooks/03_preparar_fallas.py` |

Nota: `fallas_santander.geojson` usa un recorte provisional por caja de coordenadas; el recorte definitivo por polígonos municipales (MGN del DANE) queda pendiente.
