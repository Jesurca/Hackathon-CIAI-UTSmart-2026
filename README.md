# Hackathon CIAI UTSmart 2026 — Riesgo sísmico en Santander

Repo del equipo (categoría **Avanzada**) para el reto "Santander Inteligente: IA para analizar y clasificar el riesgo sísmico". Trabajamos con datos reales del Servicio Geológico Colombiano (SGC) y construimos 3 modelos:

| Modelo | Nivel | Carpeta |
|---|---|---|
| Supervisado de clasificación | Básico | `modelos/supervisado/` |
| No supervisado (patrones territoriales) | Medio | `modelos/no_supervisado/` |
| Aprendizaje por refuerzo (priorización) | Avanzado | `modelos/refuerzo/` |

- **Brief completo y consultable:** [`Docs/RETO-HACKATHON-CIAI-UTSmart-2026.md`](Docs/RETO-HACKATHON-CIAI-UTSmart-2026.md). Consúltalo antes de preguntar (requisitos, rúbricas, agenda, fuentes y reglas).
- **Fechas clave:** entrega cerrada 13 oct 2026, 12:00 m. · presentaciones 14:00–16:45 · premiación 16 oct.

## Estructura

```
Hackathon-CIAI-UTSmart-2026/
├── Docs/                    # brief y documentación del reto
├── datos/
│   ├── raw/                 # descargas originales (ignoradas por git)
│   └── processed/           # dataset_limpio.csv (contrato compartido de los 3 modelos)
├── notebooks/               # cuadernos compartidos (descarga, limpieza, EDA)
├── modelos/
│   ├── supervisado/         # Nivel Básico
│   ├── no_supervisado/      # Nivel Medio
│   └── refuerzo/            # Nivel Avanzado (RL)
├── app/                     # prototipo Streamlit
├── entregables/             # checklist del jurado + pitch/
├── AGENTS.md                # contexto para asistentes de IA
├── requirements.txt
└── README.md
```

## Cómo colaborar (commits directos a `main`)

**Regla de oro:** cada quien toca SOLO su carpeta; nadie edita archivos compartidos (`datos/processed/`, README raíz) sin avisar al equipo.

1. `git pull` antes de empezar.
2. Trabaja en tu carpeta: `modelos/<tu-modelo>/`.
3. `git add .` y commit con formato Conventional Commits: `feat(modelo-supervisado): entrena random forest inicial`.
4. `git push`.

- No subir: `datos/raw/` (ya ignorado por git), `.env`, `.venv/`, checkpoints.
- Artefactos de más de 50 MB: no los subas; documenta el enlace externo en la card de tu modelo.
- El dataset compartido (`datos/processed/dataset_limpio.csv`) solo lo modifica el dueño de datos; los demás lo consumen.

## Qué va en cada carpeta

| Carpeta | Contenido | Dueño |
|---|---|---|
| `Docs/` | Brief y documentación del reto | equipo |
| `datos/raw/` | Descargas originales (no versionadas) + registro de fuente | dueño de datos |
| `datos/processed/` | `dataset_limpio.csv` (contrato de los 3 modelos) | dueño de datos |
| `notebooks/` | Cuadernos compartidos: descarga/unificación y limpieza/EDA | equipo |
| `modelos/supervisado/` | Notebook + `modelo_supervisado.pkl` + card | pendiente |
| `modelos/no_supervisado/` | Notebook + `modelo_no_supervisado.pkl` + card | pendiente |
| `modelos/refuerzo/` | Entorno + agente RL + card | pendiente |
| `app/` | Prototipo Streamlit con los 3 modelos | pendiente |
| `entregables/` | Checklist del jurado, diccionario, fuente citada y pitch | equipo |

## Entregables (12:00 m.)

Checklist con estado y ubicación: [`entregables/README.md`](entregables/README.md).
