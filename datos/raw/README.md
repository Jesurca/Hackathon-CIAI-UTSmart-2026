# datos/raw — descargas originales

Aquí están los archivos originales que descargamos (Excel y KML del SGC, GeoJSON, etc.).
Desde el 13 de octubre estos archivos se comparten por git. Regla: si un archivo pasa de 50 MB, no se sube; se deja el enlace.

## Registro de descargas (entregable "fuente oficial citada")

| Fecha | Fuente | URL | Filtros | Archivo | Quién |
|---|---|---|---|---|---|
| 2026-10-13 | Catálogo sísmico RSN — SGC (formato viejo, 1993–2018) | https://bdrsnc.sgc.gov.co/paginas1/catalogo/Consulta_Experta/consultaexperta_2.php | Departamento Santander; bloques 1993–2000, 2001–2009 y 2010–2013; Excel | reporte1893.xlsx, reporte1476.xlsx, reporte1668.xlsx, reporte1977.xlsx | equipo |
| 2026-10-13 | Catálogo sísmico RSN — SGC (formato nuevo, 2018+) | https://bdrsnc.sgc.gov.co/paginas1/catalogo/Consulta_Experta_Seiscomp/consultaexperta.php | Región Santander; bloques 2018–2020, 2021–2023 y 2024–2026; Excel | reporte424.xlsx, 2021-2023.xlsx, reporte1336.xlsx | equipo |
| 2026-10-13 | Atlas Geológico de Colombia 2020: Fallas Geológicas — SGC (CC BY 4.0) | https://datos.sgc.gov.co/datasets/e03339c845d24e7baceb6d67397a23b3_0 | Ninguno (descarga nacional completa) | atlas_fallas_2020_sgc.geojson | Diego |
| 2026-10-13 | Catálogo sísmico RSN — SGC (KML, no se usa) | https://bdrsnc.sgc.gov.co/paginas1/catalogo/ | Santander 2018+ | kml_sismos_1761.kml | equipo |
| 2026-10-13 | Unificación intermedia de un compañero (no es fuente oficial) | — | — | datos_sismicos_santander.xlsx | compañero |

Dos notas:

- `kml_sismos_1761.kml` no se usa: no trae tipo de magnitud, RMS, gap ni errores.
- `datos_sismicos_santander.xlsx` tiene los mismos sismos que nuestro dataset limpio (y 22 repetidos de más). Se guarda solo para comparar.

Reglas: registrar la dirección web, la fecha y los filtros apenas se descargue (§6 del brief). El documento del entregable va en `Presentacion/fuente_oficial.md`.
