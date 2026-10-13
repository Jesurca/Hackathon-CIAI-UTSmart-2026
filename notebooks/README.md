# notebooks — cuadernos compartidos

Cuadernos y scripts del pipeline común (los de cada modelo viven en `modelos/<modelo>/`).

| # | Archivo | Para qué |
|---|---|---|
| 01 | descarga_unificacion.ipynb | Descarga del catálogo SGC (dos consultas: jun 1993–feb 2018 y mar 2018+) y unificación de columnas |
| 02 | limpieza_eda.ipynb | Duplicados, faltantes, unidades y fechas, atípicos, análisis exploratorio y features |
| 03 | 03_preparar_fallas.py | Filtra lineamientos, cubiertas e inferidas del Atlas de Fallas 2020 (SGC), recorta al área de Santander y calcula longitudes en metros |

Reglas: rutas relativas al repo; el resultado procesado va a `datos/processed/`.

Ejecutar el paso 03:

```bash
python notebooks/03_preparar_fallas.py
```

Requiere `shapely` y `pyproj` (los instala `requirements.txt` a través de `geopandas`).
