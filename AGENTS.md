# AGENTS.md — Hackathon CIAI UTSmart 2026

## Proyecto

Reto de categoría Avanzada ("Santander Inteligente": riesgo sísmico en Santander). Modelos obligatorios: supervisado (Básico), no supervisado (Medio) y aprendizaje por refuerzo (Avanzado), con datos reales del SGC. Entrega cerrada: 13 oct 2026, 12:00 m.

## Fuente única de consulta

[`Docs/RETO-HACKATHON-CIAI-UTSmart-2026.md`](Docs/RETO-HACKATHON-CIAI-UTSmart-2026.md): requisitos, rúbricas, agenda, fuentes y reglas. Consúltalo antes de proponer o ejecutar trabajo; mantén la numeración § al editarlo.

## Estructura

- `datos/raw/` (ignorado en git) y `datos/processed/` (dataset limpio compartido).
- `notebooks/` (cuadernos compartidos).
- `modelos/{supervisado,no_supervisado,refuerzo}/` (un dueño por carpeta: notebook + artefacto + card).
- `app/` (prototipo Streamlit) y `entregables/` (documentos del jurado).
- `Docs/` documentación del reto.

## Stack (previsto, por confirmar)

Python 3 (pandas, scikit-learn, GeoPandas), Google Colab, Streamlit, gymnasium/Stable-Baselines3 para el agente RL.

## Convenciones

- Commits directos a `main`; cada quien toca solo su carpeta; `git pull` antes de push.
- Documentación en español en `Docs/`; el brief es la fuente única (no dupliques datos entre docs).
- No versionar `datos/raw/`, `.env` ni entornos; artefactos de más de 50 MB por enlace externo.
- Antes de tocar modelos o datos, revisa §3, §5 y §7 del brief.
