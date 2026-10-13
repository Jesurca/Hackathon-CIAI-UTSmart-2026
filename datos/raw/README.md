# datos/raw — descargas originales (no versionado)

Aquí van los archivos originales descargados (Excel del SGC, GeoJSON, Shapefile, etc.). Esta carpeta está ignorada por git: solo se versiona este README.

## Registro de descargas (obligatorio para el entregable "fuente oficial citada")

| Fecha | Fuente | URL | Filtros aplicados | Archivo | Quién |
|---|---|---|---|---|---|
| pendiente | Catálogo sísmico RSN — SGC | http://bdrsnc.sgc.gov.co/paginas1/catalogo/ | pendiente (departamento, periodo, magnitud mínima) | pendiente | pendiente |
| 2026-10-13 | Atlas Geológico de Colombia 2020: Fallas Geológicas — SGC (licencia CC BY 4.0) | https://datos.sgc.gov.co/datasets/e03339c845d24e7baceb6d67397a23b3_0 | Ninguno: descarga nacional completa (el recorte a Santander lo hace `notebooks/03_preparar_fallas.py`) | `atlas_fallas_2020_sgc.geojson` (renombrado; el archivo original traía caracteres codificados en el nombre) | Diego |

Reglas: registrar dirección web, fecha de descarga y filtros apenas se descargue (§6 del brief). El documento consolidado del entregable va en `entregables/fuente_oficial.md`.
