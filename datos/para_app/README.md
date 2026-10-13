# datos/para_app — insumos listos para la app

Archivos en el formato que la app Streamlit reconoce (sección **📥 Cargar datos**). Se regeneran con:

```bash
python notebooks/04_exportar_para_app.py
```

| Archivo | Qué es | Columnas |
|---|---|---|
| `catalogo_sgc.csv` | Catálogo de sismos ya limpio (1993–2013 y 2018–2026) | fecha, latitud, longitud, profundidad, magnitud, municipio |
| `clusters_resultados.csv` | Resultados del modelo no supervisado (82 municipios) | municipio, cluster, nombre_cluster, PC1, PC2 |

Pendiente: `supervisado_resultados.csv` (municipio, periodo, nivel_predicho…) — se define con el agente que está adaptando la app, porque el notebook supervisado entrega probabilidad semanal y la app espera clasificación por municipio y trimestre.
