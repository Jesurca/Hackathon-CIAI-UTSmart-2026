# datos/para_app — insumos listos para la app

Archivos en el formato que la app Streamlit reconoce (sección **📥 Cargar datos**). Se regeneran con:

```bash
python notebooks/04_exportar_para_app.py      # catálogo y clusters
python notebooks/05_supervisado_para_app.py   # supervisado municipio x trimestre
```

| Archivo | Qué es | Columnas |
|---|---|---|
| `catalogo_sgc.csv` | Catálogo de sismos ya limpio (175.185 eventos, 1993–2026) | fecha, latitud, longitud, profundidad, magnitud, municipio |
| `clusters_resultados.csv` | Resultados del modelo no supervisado (87 municipios = todo el catálogo) | municipio, cluster, nombre_cluster, PC1, PC2 |
| `supervisado_resultados.csv` | Predicción supervisada municipio x trimestre (4.751 filas, 85 municipios) | municipio, periodo, nivel_predicho, nivel_real, prob_bajo, prob_medio, prob_alto, conjunto |

Notas para la demo:

- **Clusters (87):** el modelo entrenó con los 82 municipios de 15 o más sismos; los 5 restantes
  (CAPITANEJO, ENCISO, MACARAVITA, SAN JOSÉ DE MIRANDA, SAN MIGUEL) se asignaron al grupo del
  centroide más cercano con el pipeline guardado, para que la app no muestre el grupo "sin grupo" (G5).
- **Supervisado (85):** cada fila es un trimestre de corte; `nivel_predicho` clasifica el trimestre
  siguiente usando solo datos hasta el corte (anti-fuga). ENCISO y SAN MIGUEL no tienen trimestres
  consecutivos con actividad, así que no generan fila.
- **Fuente oficial:** la pestaña queda **vacía a propósito** — no hay población/vivienda reales en el
  repo; el archivo de `demo/` es de ejemplo y no debe presentarse como dato real.
