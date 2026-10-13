# Plataforma de riesgo sísmico · Santander (Hackatón UTS)

Una sola app de Streamlit que integra los **tres retos**:

- **Básico (supervisado):** la app muestra métricas, matriz de confusión, importancias, un formulario de prueba en vivo y mapas.
- **Medio (no supervisado):** muestra el mapa de grupos, el perfil de cada grupo, el PCA, la silueta y el método del codo.
- **Avanzado (agente por refuerzo):** el agente **usa como estado las salidas de los dos modelos anteriores**. Se entrena o se carga dentro de la app, prioriza municipios con un presupuesto limitado e indica qué tan seguro está.

> La app arranca con **datos de demostración** (carpeta `demo/`) para que todo se vea desde el primer momento. Con los datos reales **no se toca el código**: se suben los archivos en la sección **📥 Cargar datos**.

---

## 1. Contenido de la carpeta

```
sismo_app/
├── app.py                 ← la interfaz (todas las secciones)
├── ui.py                  ← estilo visual y gráficas reutilizables
├── estado.py              ← carga de archivos, validación y procesamiento en la sesión
├── generar_demo.py        ← vuelve a crear la carpeta demo/ (opcional)
├── agente_rl/             ← EL AGENTE POR REFUERZO (paquete de Python)
│   ├── columnas.py        ← reconoce solo los nombres de columnas de cada archivo
│   ├── variables.py       ← une SGC + fuente oficial + Básico + Medio → 25 variables
│   ├── preparar.py        ← cortes temporales (entrenamiento / validación / prueba)
│   ├── entorno.py         ← entorno RL: estado, acción, recompensa
│   ├── agente.py          ← comité de 5 redes Q (NumPy: Double DQN, Adam y Huber programados a mano)
│   ├── servicio.py        ← entrenar, cargar el agente .npz, evaluar, priorizar
│   ├── utilidades.py      ← K-Means, PCA, silueta y métricas en NumPy
│   ├── metricas.py        ← métricas y políticas de comparación
│   ├── flujo.py           ← flujo completo archivos → agente
│   └── simulado.py        ← generador de datos de demostración
├── demo/                  ← datos y modelos de ejemplo = PLANTILLA del formato esperado
├── version_anterior/      ← copia de la app que tenían antes
├── requirements.txt
└── .streamlit/config.toml ← tema oscuro y límite de subida (500 MB)
```

`datos_demo.py` ya no se usa. Se puede borrar o dejar; la copia está en `version_anterior/`.

---

## 2. Instalación (Windows, PowerShell)

La app necesita **solo 4 librerías**: streamlit, plotly, pandas y numpy. **No usa PyTorch ni GPU**: el agente está programado en NumPy y entrena en pocos minutos en cualquier computador.

1. Abra PowerShell **dentro** de la carpeta `sismo_app`. Truco: en el Explorador escriba `powershell` en la barra de direcciones y presione Enter.
2. Active el entorno virtual:
   ```powershell
   .venv\Scripts\activate
   ```
3. Instale las librerías. Si ya tenía la versión anterior de la app, ya están instaladas y no descarga nada:
   ```powershell
   pip install -r requirements.txt
   ```
4. Ejecute la app:
   ```powershell
   streamlit run app.py
   ```
   Se abre en `http://localhost:8501`. Para detenerla, presione `Ctrl + C`.

**Opcionales**, solo si hay buen internet:
- `pip install openpyxl`: permite subir archivos Excel y descargar el reporte en .xlsx. Sin ella, use CSV; el reporte sale como .zip de CSV.
- `pip install scikit-learn`: permite abrir los modelos `.pkl` de los compañeros (formulario de prueba en vivo e importancias). Sin ella, la app usa sus **CSV de resultados**, y la matriz de confusión, las métricas, los grupos y el agente funcionan igual.

> Los mapas necesitan internet para el fondo. Sin conexión, el resto de la app funciona igual.

---

## 3. Qué debe entregar cada integrante (formato de archivos)

Los archivos de `demo/` son **ejemplos exactos** del formato. Los nombres de columna pueden ser distintos: la app los reconoce sola y, si no, se corrigen con un selector.

| Archivo | Columnas obligatorias | Opcionales (recomendadas) |
|---|---|---|
| **Catálogo SGC** (CSV/Excel) | fecha, latitud, longitud, profundidad, magnitud | municipio |
| **Fuente oficial** (una fila por municipio) | municipio, latitud, longitud, población | área_km2, nbi, pct_rural, altitud_m, viviendas_precarias, hospitales, provincia, codigo_dane |
| **Supervisado · resultados** | municipio, nivel_predicho (Bajo/Medio/Alto o 0/1/2) | **periodo** (ej. 2024Q1), **nivel_real**, prob_bajo/medio/alto, **conjunto** (train/val/test) |
| **Supervisado · modelo** (.pkl) | — | Entrenado con un DataFrame, para que guarde los nombres de columnas |
| **No supervisado · resultados** | municipio, cluster | nombre_cluster, PC1, PC2 |
| **No supervisado · modelo** (.pkl) | — | Por ejemplo `Pipeline(StandardScaler, KMeans)` |
| **Agente** (.npz) | — | Se descarga desde la app después de entrenar |

Recomendaciones para los compañeros:
- **Supervisado:** entreguen una predicción **por municipio y trimestre** (`periodo`), calculada sin usar datos posteriores a ese trimestre. Agreguen `nivel_real` y `conjunto` para que la app calcule la matriz de confusión y las métricas por tramo temporal.
- **Modelos .pkl:** guárdenlos con `joblib.dump(modelo, "modelo.pkl")`. Si las columnas del modelo existen entre las 25 variables del panel, la app **lo aplica sola**, aunque falte el CSV.
- **Versiones:** el .pkl debe abrirse con la **misma versión de scikit-learn** con que se entrenó. Si usaron xgboost u otra librería, instálenla en el entorno.
- **Seguridad:** carguen solo .pkl del propio equipo, porque un archivo .pkl puede ejecutar código al abrirse.

Si falta el archivo de un reto, la app usa un **sustituto automático** (terciles de actividad o K-Means) y lo indica con una etiqueta naranja en cada sección.

---

## 4. Cómo se usa (flujo recomendado el día de la hackatón)

1. **📥 Cargar datos:** suba cada archivo en su pestaña.
   - Revise la asignación de columnas y los mensajes de validación (✅ ⚠️ ⛔).
   - En **Parámetros** puede ajustar la unidad temporal, los cortes de validación, el presupuesto y los pesos de equidad y costo.
   - Presione **⚙️ Integrar y procesar** y lea el *Reporte de integración*: sismos usados, municipios que cruzan, supuestos.
2. **🤖 Avanzado → Entrenamiento:** entrene el agente con los datos cargados.
   - Unos 300 episodios tardan entre 2 y 5 minutos con 35 municipios.
   - Al terminar, **descargue el .npz**. La próxima vez súbalo en Cargar datos y no tendrá que reentrenar.
3. Recorra las secciones; cada una indica de dónde salen sus datos:
   - **🎯 Básico** y **🧭 Medio:** los resultados de los compañeros.
   - **🤖 Priorización:** mapa, semáforo de confianza y razones de cada decisión.
   - **🤖 Evaluación:** comparación con reglas simples en años no vistos.
   - **🔮 Simulador:** cambiar una variable o el presupuesto y ver qué hace el agente.
   - **✅ Validación:** cortes temporales e incertidumbre.
4. **🎤 Modo presentación:** 8 diapositivas con los datos actuales, en el orden problema → datos → Básico → Medio → agente → evaluación → decisión → alcance.
5. **📄 Reporte:** descargue el Excel y el HTML como evidencia para el jurado.

---

## 5. El agente en una frase

Cada trimestre, con un presupuesto limitado, el agente elige municipios hasta agotarlo.
- **Estado:** 25 variables, incluidas la clase del reto Básico y el grupo del reto Medio, más el presupuesto y el tiempo sin atención.
- **Recompensa:** premia acertar dónde ocurrió el riesgo real del trimestre siguiente (peligro × exposición × vulnerabilidad), suma un premio por equidad y resta el costo.
- **Confianza:** son 5 redes que funcionan como un comité de expertos. Cuando discrepan, el semáforo lo muestra: 🟢 Alta · 🟡 Media · 🔴 Baja.

---

## 6. Problemas frecuentes

| Síntoma | Solución |
|---|---|
| «Para leer Excel se necesita openpyxl» | Guarde el archivo como CSV o instale `openpyxl`. |
| «Faltan columnas obligatorias» | Asigne la columna correcta en el selector de esa pestaña. |
| Pocos municipios cruzan | Los nombres son distintos entre archivos. La app quita tildes y mayúsculas; revise las listas del reporte. |
| Muchos sismos «fuera del radio» | Suba el radio en Parámetros o revise el signo de las longitudes (deben ser negativas). |
| «El agente se entrenó con otros datos» | Reentrene en 🤖 Avanzado → Entrenamiento. |
| El .pkl no abre | Instale scikit-learn (la misma versión con que se entrenó) o use solo el CSV de resultados. |
| «El tramo de validación/prueba quedó vacío» | Ajuste los cortes en Parámetros o active los cortes automáticos. |
