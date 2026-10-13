# app — prototipo Streamlit (equipo)

App única con las tres vistas del reto: **Básico (supervisado)**, **Medio (no supervisado)** y **Avanzado (agente RL)**.
Guía completa del autor de la app: [`sismo_app_F/sismo_app/LEEME.md`](sismo_app_F/sismo_app/LEEME.md).

## Cómo se corre

Desde la raíz del repo:

```bash
python -m streamlit run app/sismo_app_F/sismo_app/app.py
```

O doble clic en `run_app.bat` (raíz del repo).

- Dependencias: `streamlit`, `plotly`, `pandas`, `numpy` (ver `sismo_app_F/sismo_app/requirements.txt`). Ya quedaron instaladas en este PC.
- Arranca con datos de **demostración** (`demo/`). Los datos reales del equipo se cargan en la sección **📥 Cargar datos**.
- Los mapas necesitan internet para el fondo; el resto funciona sin conexión.

## Estado de integración con los modelos del equipo

| Modelo | Artefacto real que tenemos | Cómo entra a la app |
|---|---|---|
| No supervisado | `modelos/no_supervisado/modelo_no_supervisado.joblib` + CSV en `datosNoSup/` | Pendiente: convertir a `clusters_resultados.csv` (municipio, cluster, nombre_cluster, PC1, PC2) |
| Supervisado | `modelos/supervisado/modelo_supervisado.ipynb` (lee `datos/raw/`) | Pendiente: correr el notebook y generar `supervisado_resultados.csv` |
| Refuerzo | Por definir | Se revisa después del supervisado |
