# Card — Modelo supervisado (Nivel Básico)

- **Dueño:** equipo · **Estado:** en progreso (notebook listo para correr)
- **Notebook:** `modelo_supervisado.ipynb` — lee los `.xlsx` crudos de `datos/raw/` (encuentra la raíz del repo solo) y excluye `datos_sismicos_santander.xlsx` para no duplicar filas.
- **Objetivo:** clasificar el nivel de actividad sísmica de la zona (bajo / medio / alto) con métricas de desempeño.
- **Salidas:** `datos_sismicos_santander_limpio.csv` y `modelo_riesgo_sismico_v2.pkl` (se guardan en esta carpeta).
- **Métricas mínimas:** exactitud + matriz de confusión; si hay desbalance, precisión, recall y F1 (§4.1 del brief).
- **Métricas obtenidas:** pendiente de correr
- **Cómo correrlo:** abre el notebook y dale **Restart & Run All** (funciona desde cualquier carpeta).

Checklist: notebook + artefacto + métricas escritas aquí.
