# Card — Modelo no supervisado (Nivel Medio)

- **Dueño:** pendiente · **Estado:** pendiente
- **Objetivo:** encontrar patrones o grupos y comparar unidades territoriales de Santander (sin usar etiquetas para entrenar).
- **Consume:** `datos/processed/dataset_limpio.csv` + variables territoriales (DANE/IGAC si aplica).
- **Artefacto:** `modelo_no_supervisado.pkl` (lo genera `entrenar_no_supervisado.ipynb`).
- **Evidencia mínima:** comparación de enfoques, interpretación de grupos e importancia de variables; visualización territorial reproducible (§4.2 del brief).
- **Resultados obtenidos:** pendiente
- **Cargar:**

```python
import joblib
modelo = joblib.load("modelos/no_supervisado/modelo_no_supervisado.pkl")
```

Checklist: notebook + artefacto + comparación de enfoques escrita aquí.
