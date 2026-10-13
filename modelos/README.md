# modelos — los 3 modelos del reto

| Carpeta | Nivel | Modelo obligatorio | Artefacto esperado |
|---|---|---|---|
| `supervisado/` | Básico | Supervisado de clasificación (bajo / medio / alto) | `modelo_supervisado.pkl` |
| `no_supervisado/` | Medio | Únicamente no supervisado (patrones territoriales) | `modelo_no_supervisado.pkl` |
| `refuerzo/` | Avanzado | Aprendizaje por refuerzo (agente con entorno, estados, acciones y recompensa) | agente entrenado (`.zip` o equivalente) |

## Contrato de integración

- Todos consumen `datos/processed/dataset_limpio.csv` (mismas columnas, documentadas en el diccionario).
- Cada carpeta tiene un README-card con: dueño, estado, features que consume, salida, métricas y cómo cargar el artefacto.
- `app/app.py` carga los 3 artefactos para la demostración en vivo.
- No se aceptan sustitutos del modelo obligatorio (§5 del brief): una solución solo por reglas fijas no cuenta como modelo.
