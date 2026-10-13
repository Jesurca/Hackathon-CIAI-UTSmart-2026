# notebooks — cuadernos compartidos

Cuadernos y scripts del pipeline común (el EDA y el entrenamiento de cada modelo viven en `modelos/<modelo>/`).

| # | Archivo | Para qué |
|---|---|---|
| 01 | 01_limpieza_catalogos.py | Unifica los catálogos del SGC (formato viejo y nuevo), elimina duplicados y escribe `datos/processed/dataset_limpio.csv` (solo eventos) |
| 03 | 03_preparar_fallas.py | Filtra lineamientos, cubiertas e inferidas del Atlas de Fallas 2020 (SGC), recorta al área de Santander y calcula longitudes en metros |

Reglas: rutas relativas al repo; el resultado procesado va a `datos/processed/`.

## Ejecutar

```bash
python notebooks/01_limpieza_catalogos.py
python notebooks/03_preparar_fallas.py
```

Dependencias: `pandas`, `openpyxl` y `python-calamine` para el 01 (varios xlsx del SGC traen estilos inválidos y calamine los lee sin problema); `shapely` y `pyproj` para el 03. Todas están en `requirements.txt`.
