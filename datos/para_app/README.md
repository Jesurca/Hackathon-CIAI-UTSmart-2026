# datos/para_app — kit completo para la app

Todos los archivos que la app Streamlit consume (sección **📥 Cargar datos**), en su formato.
Se regeneran con:

```bash
python notebooks/04_exportar_para_app.py      # catálogo y clusters
python notebooks/05_supervisado_para_app.py   # supervisado municipio x trimestre
python notebooks/06_fuente_oficial_para_app.py
python notebooks/07_entrenar_agente_para_app.py
```

| Archivo | Qué es | Columnas |
|---|---|---|
| `catalogo_sgc.csv` | Catálogo de sismos ya limpio (175.185 eventos, 1993–2026) | fecha, latitud, longitud, profundidad, magnitud, municipio |
| `fuente_oficial.csv` | Tabla municipal (87): población censal DANE 2018, área y provincia | codigo_dane, provincia, municipio, latitud, longitud, poblacion, area_km2 |
| `clusters_resultados.csv` | Resultados del modelo no supervisado (87 municipios) | municipio, cluster, nombre_cluster, PC1, PC2 |
| `clusters_modelo.joblib` | Paquete completo del modelo no supervisado (escalador, PCA, KMeans, tabla) | — |
| `supervisado_resultados.csv` | Predicción supervisada municipio x trimestre (4.751 filas, 85 municipios) | municipio, periodo, nivel_predicho, nivel_real, prob_bajo, prob_medio, prob_alto, conjunto |
| `supervisado_modelo.pkl` | Modelo supervisado entrenado, con los nombres de columna del panel | — |
| `agente_sismico.npz` | Agente de refuerzo entrenado con estos mismos datos (300 episodios) | — |

Notas:

- **Clusters (87):** el modelo entrenó con los 82 municipios de 15 o más sismos; los 5 restantes
  (CAPITANEJO, ENCISO, MACARAVITA, SAN JOSÉ DE MIRANDA, SAN MIGUEL) se asignaron al grupo del
  centroide más cercano con el pipeline guardado (así la app no muestra "sin grupo").
- **Supervisado (85):** cada fila es un trimestre de corte; `nivel_predicho` clasifica el trimestre
  siguiente usando solo datos hasta el corte (anti-fuga). ENCISO y SAN MIGUEL no tienen trimestres
  consecutivos con actividad, así que no generan fila.
- **Fuente oficial:** población del CNPV 2018 (geoportal DANE, departamento 68); área y provincia del
  anexo de municipios de Santander; centroides = promedio de los sismos del catálogo. No incluye
  nbi, pct_rural, altitud ni viviendas precarias: la app usa ahí sus valores fijos y lo reporta.
- **Agente:** entrenado con este kit completo; al subir estos mismos archivos la app lo reconoce
  como "mismos datos" y no hay que reentrenar.
