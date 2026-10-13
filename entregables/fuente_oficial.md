# Fuente oficial citada (entregable)

> Registrar aquí cada descarga apenas se haga. El log operativo también queda en `datos/raw/README.md`.

| Fecha de descarga | Fuente | URL | Filtros aplicados | Archivo obtenido |
|---|---|---|---|---|
| pendiente | Catálogo sísmico RSN — SGC | http://bdrsnc.sgc.gov.co/paginas1/catalogo/ | pendiente (departamento, periodo, magnitud mínima) | pendiente |
| 2026-10-13 | Atlas Geológico de Colombia 2020: Fallas Geológicas — SGC (CC BY 4.0) | https://datos.sgc.gov.co/datasets/e03339c845d24e7baceb6d67397a23b3_0 | Ninguno (descarga nacional completa); recorte a Santander en `notebooks/03_preparar_fallas.py` | `atlas_fallas_2020_sgc.geojson` (crudo) → `datos/processed/fallas_santander.geojson` (limpio) |

Reglas (§6 del brief): citar la dirección web exacta, la fecha de descarga y los filtros aplicados; datos reales de fuente oficial; la fuente principal es el catálogo del SGC.
