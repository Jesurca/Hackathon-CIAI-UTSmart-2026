# Card — Modelo supervisado (Nivel Básico)

- **Dueño:** pendiente · **Estado:** pendiente
- **Objetivo:** clasificar el nivel de actividad sísmica de la zona (bajo / medio / alto) con métricas de desempeño.
- **Consume:** `datos/processed/dataset_limpio.csv` (features definidas en el diccionario).
- **Artefacto:** `modelo_supervisado.pkl` (lo genera `entrenar_supervisado.ipynb`).
- **Métricas mínimas:** exactitud + matriz de confusión; si hay desbalance, precisión, recall y F1 (§4.1 del brief).
- **Métricas obtenidas:** pendiente
- **Cargar:**

```python
import joblib
modelo = joblib.load("modelos/supervisado/modelo_supervisado.pkl")
```

Checklist: notebook de entrenamiento + artefacto + métricas escritas aquí.
