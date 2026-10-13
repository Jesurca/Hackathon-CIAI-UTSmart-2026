# Modelo no supervisado — sismos de Santander

Agrupa los municipios de Santander según cómo se comportan los sismos. No usa etiquetas: solo busca parecidos.

## Cómo se corre

1. Primero corre la limpieza del catálogo (crea el dataset):
   `python notebooks/01_limpieza_catalogos.py`
2. Abre `modelo_no_supervisado.ipynb` y dale **Restart & Run All**.
3. Al terminar, mira los archivos que deja en esta carpeta.

## Qué usa

- `datos/processed/dataset_limpio.csv` (obligatorio): sismos limpios, 1993–2026.
- `datos/processed/fallas_santander.geojson` (opcional): distancia a fallas.

Detalles: ventana 2006–2026 · magnitud Ml (el 99 % de los sismos viene así) · 82 municipios con 15 o más sismos · no hay datos de 2014–2017.

## Qué produce

| Archivo | Qué es |
|---|---|
| `grupo_por_municipio.csv` | Cada municipio con su grupo |
| `perfil_grupos.csv` | Cómo es cada grupo (medianas) |
| `comparacion_modelos.csv` | Métricas de KMeans, Ward y GMM |
| `importancia_variables.csv` | Qué variables separan los grupos |
| `modelo_no_supervisado.joblib` | El modelo guardado |

## Los 4 grupos

- **Bajo**: pocos sismos y profundos.
- **Moderado**: actividad media.
- **Alto**: pocos sismos pero superficiales (más dañinos).
- **Muy alto**: mucha actividad (zona del nido sísmico).

Con los datos actuales: 23 Bajo · 37 Moderado · 8 Alto · 14 Muy alto.

## Límites (decirlos en el pitch)

- Falta 2014–2017.
- Falta población/vivienda por municipio (DANE) para hablar de riesgo real.
- La distancia a fallas es aproximada (del centro del municipio al punto más cercano).

## Cargar el modelo

```python
import joblib
art = joblib.load("modelos/no_supervisado/modelo_no_supervisado.joblib")
print(art["etiquetas"])
```
