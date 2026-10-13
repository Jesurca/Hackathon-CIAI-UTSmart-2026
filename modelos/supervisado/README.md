# Card — Modelo supervisado (Nivel Básico)

- **Dueño:** equipo · **Estado:** entrenado y validado (13 oct 2026)
- **Notebook:** `modelo_supervisado.ipynb` — corre con **Restart & Run All** desde cualquier carpeta: busca la raíz del repo solo, lee los Excel de `datos/raw/` (excluye `datos_sismicos_santander.xlsx` para no duplicar filas) y guarda sus salidas en esta carpeta.
- **Qué hace:** agrupa los sismos en ventanas de 7 días y estima la probabilidad de que la semana siguiente tenga actividad alta (magnitud ≥ 4,60).
- **Datos usados:** 205.916 sismos leídos → **175.056 tras limpieza** · periodo **1993-06-01 → 2026-10-13** · nada inventado (el notebook lo reporta paso a paso).
- **Modelo:** **Regresión logística** (elegida por AUC en validación cruzada entre varios modelos).
- **Métricas (prueba):** **AUC 0,51** frente a 0,46 de la persistencia; 348 ventanas de prueba, 52 de actividad alta. Capacidad moderada: se usa como apoyo, no como decisión única.
- **Última señal:** probabilidad estimada de actividad alta en la ventana siguiente: **57,7 %**.
- **Salidas:** `datos_sismicos_santander_limpio.csv` (dataset limpio del modelo) y `modelo_riesgo_sismico_v2.pkl` (modelo entrenado + metadatos: umbral, ventana y variables).
- **Nota:** existe una copia gemela `hackathonUTS_mejorado.ipynb` con la misma corrección de rutas; se mantiene porque el compañero la usa.

Checklist: notebook + artefacto + métricas escritas aquí. ✓
