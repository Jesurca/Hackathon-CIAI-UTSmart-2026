# notebooks — cuadernos compartidos

Cuadernos y scripts del pipeline común (el EDA y el entrenamiento de cada modelo viven en `modelos/<modelo>/`).

| # | Archivo | Para qué |
|---|---|---|
| 01 | 01_limpieza_catalogos.py | Unifica los catálogos del SGC (formato viejo y nuevo), elimina duplicados y escribe `datos/processed/dataset_limpio.csv` (solo eventos) |
| 03 | 03_preparar_fallas.py | Filtra lineamientos, cubiertas e inferidas del Atlas de Fallas 2020 (SGC), recorta al área de Santander y calcula longitudes en metros |
| 04 | 04_exportar_para_app.py | Exporta el catálogo limpio y los clusters a la app (`datos/para_app/`) y asigna por centroide los municipios con menos de 15 sismos |
| 05 | 05_supervisado_para_app.py | Variante supervisada municipio x trimestre para la app (anti-fuga): entrena y escribe `datos/para_app/supervisado_resultados.csv` |

Reglas: rutas relativas al repo; el resultado procesado va a `datos/processed/`.

## Ejecutar

```bash
python notebooks/01_limpieza_catalogos.py
python notebooks/03_preparar_fallas.py
python notebooks/04_exportar_para_app.py
python notebooks/05_supervisado_para_app.py
```

Dependencias: `pandas`, `openpyxl` y `python-calamine` para el 01 (varios xlsx del SGC traen estilos inválidos y calamine los lee sin problema); `shapely` y `pyproj` para el 03; `scikit-learn` para el 05. Todas están en `requirements.txt`.
