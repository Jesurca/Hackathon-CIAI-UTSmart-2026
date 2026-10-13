# Modelo no supervisado — sismos de Santander

Agrupa los municipios de Santander según cómo se comportan los sismos. No usa etiquetas: solo busca parecidos.

## Cómo se corre

1. Primero corre la limpieza del catálogo (crea el dataset):
   `python notebooks/01_limpieza_catalogos.py`
2. Abre `modelo_no_supervisado.ipynb` y dale **Restart & Run All**.

No importa desde qué carpeta lo abras: el propio cuaderno encuentra la raíz del repo. En Colab, sube el CSV (y el de fallas) a `datos/processed/`.

## Caché (para no reentrenar siempre)

La primera corrida entrena y guarda `modelo_no_supervisado.joblib`. Las siguientes lo cargan (dice "Caché cargada") y terminan en segundos. Si cambia el dataset, las fallas o un parámetro, se reentrena sola. Para forzar reentreno: borra el `.joblib`.

## Qué usa

- `datos/processed/dataset_limpio.csv` (obligatorio): sismos limpios, 1993–2026.
- `datos/processed/fallas_santander.geojson` (opcional): distancia a fallas.

Detalles: ventana 2006–2026 · magnitud Ml (el 99 % de los sismos viene así) · 82 municipios con 15 o más sismos · no hay datos de 2014–2017.

## Qué produce

| Archivo | Qué es |
|---|---|
| `datosNoSup/grupo_por_municipio.csv` | Cada municipio con su grupo |
| `datosNoSup/perfil_grupos.csv` | Cómo es cada grupo (medianas) |
| `datosNoSup/comparacion_modelos.csv` | Métricas de KMeans, Ward y GMM |
| `datosNoSup/importancia_variables.csv` | Qué variables separan los grupos |
| `modelo_no_supervisado.joblib` | Caché + artefacto: preprocesado, modelo y etiquetas |

## Los 4 grupos

- **Bajo**: pocos sismos y profundos.
- **Moderado**: actividad media.
- **Alto**: pocos sismos pero superficiales (más dañinos).
- **Muy alto**: mucha actividad (zona del nido sísmico).

Con los datos actuales: 23 Bajo · 37 Moderado · 8 Alto · 14 Muy alto.

## Límites (decirlos en el pitch)

- **Falta 2014–2017**: ese bloque de datos no se descargó y no hay sismos en el archivo para esos 4 años. Las tasas por año se calculan solo con los años que sí existen (2006–2013 y 2018–2026).
- **Falta población/vivienda por municipio (DANE)**: el modelo mide qué tanto tiembla (amenaza), no cuánta gente o casas hay en el camino (riesgo real).
- **La distancia a fallas es aproximada**: se mide desde el centro del municipio al punto más cercano de la falla, no desde cada casa.

## Cargar el modelo

```python
import joblib
art = joblib.load("modelos/no_supervisado/modelo_no_supervisado.joblib")
print(art["etiquetas"])
```
