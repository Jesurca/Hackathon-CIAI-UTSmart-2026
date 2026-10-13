# Fuente oficial citada (entregable)

> Cada descarga queda registrada aquí y en `datos/raw/README.md`.

| Fecha | Fuente | URL | Filtros aplicados | Archivo obtenido |
|---|---|---|---|---|
| 2026-10-13 | Catálogo sísmico RSN — SGC (1993–2018) | https://bdrsnc.sgc.gov.co/paginas1/catalogo/Consulta_Experta/consultaexperta_2.php | Departamento Santander; bloques 1993–2000, 2001–2009 y 2010–2013; Excel | reporte1893.xlsx · reporte1476.xlsx · reporte1668.xlsx · reporte1977.xlsx |
| 2026-10-13 | Catálogo sísmico RSN — SGC (2018+) | https://bdrsnc.sgc.gov.co/paginas1/catalogo/Consulta_Experta_Seiscomp/consultaexperta.php | Región Santander; bloques 2018–2020, 2021–2023 y 2024–2026; Excel | reporte424.xlsx · 2021-2023.xlsx · reporte1336.xlsx |
| 2026-10-13 | Atlas Geológico de Colombia 2020: Fallas Geológicas — SGC (CC BY 4.0) | https://datos.sgc.gov.co/datasets/e03339c845d24e7baceb6d67397a23b3_0 | Nacional completo; el recorte a Santander lo hace `notebooks/03_preparar_fallas.py` | atlas_fallas_2020_sgc.geojson → `datos/processed/fallas_santander.geojson` |

Reglas (§6 del brief): citar la dirección web exacta, la fecha de descarga y los filtros; la fuente principal es el catálogo del SGC.
