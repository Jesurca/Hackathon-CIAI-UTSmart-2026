# Hackathon CIAI UTSmart 2026 — Riesgo sísmico en Santander

> **Estado:** vigente · **Última revisión:** 2026-10-13
> **Fuentes:** página oficial del evento (ciaiuts.com/eventos/hackaton-utsmart-2026, consultada el 2026-10-13) + brief compartido por el equipo.
> **Etiquetas de procedencia:** `[OFICIAL]` requisito del evento · `[GUÍA]` recomendación (no requisito) · `[EQUIPO]` indicación interna · `[POR CONFIRMAR]` falta verificar con mentores o comité.

**En una frase:** construir un flujo reproducible de Machine Learning que clasifique y priorice el riesgo sísmico en zonas de Santander, con modelos supervisado, no supervisado y de aprendizaje por refuerzo entrenados con datos reales del SGC, demostrado en vivo y entregado el **13 de octubre de 2026 antes de las 12:00 m.**

## 0. Cómo consultar este documento

| Pregunta | Sección |
|---|---|
| ¿De qué trata el reto y qué componente es obligatorio? | §2 |
| ¿Qué modelo exige mi categoría? | §3 |
| ¿Cómo nos califican (rúbricas)? | §4 |
| ¿Qué revisa el jurado antes de calificar? | §5 |
| ¿De dónde salen los datos y con qué URL? | §6 |
| ¿Cuántas variables y cómo se limpian? | §7 |
| ¿Qué se entrega y dónde vive cada cosa? | §8 |
| ¿Reglas del evento (equipos, IA, herramientas)? | §9 |
| ¿Agenda del 13 de octubre y turnos de presentación? | §10 |
| ¿Checklist mínimo del nivel Básico? | §11 |
| ¿Qué decisiones siguen abiertas? | §13 |
| ¿Qué significa RMS, gap, RL…? | §14 |

La numeración de secciones es estable: cita por número (§N) y no la cambies al editar.

## 1. Contexto del evento `[OFICIAL]`

- **Evento:** Hackathon CIAI UTSmart 2026 — "Santander Inteligente: IA para analizar y clasificar el riesgo sísmico".
- **Organiza:** CIAI (Centro de Innovación e Inteligencia Artificial) de la UTS. Edición: UTSmart 2026 · Hackathon interuniversitaria.
- **Fechas:** del martes 13 al viernes 16 de octubre de 2026. La competición técnica es el **13 de octubre** (07:00–18:00); la **premiación es el 16 de octubre** (hora por confirmar).
- **Sede:** Universitaria Tecnológica de Santander (UTS), Bucaramanga. Salones 302A, 302B, 315A y 307B.
- **Instituciones participantes:** UTS, UNAB y UDES.
- **Categorías:** Básico (20 equipos), Medio (10 equipos) y Avanzado (9 equipos): 39 equipos en total.
- **Equipos de Avanzado (nuestra categoría):** Dynamockapps, Grupo Niche, Syntax error II, CodeForce, DS3 Team, Monos, UDES Syntax Titans, Bit Conquistadores y LosPrint. Todos desarrollan en el salón **307B**.
- **Modalidad:** presencial, con acompañamiento del CIAI.

## 2. Descripción oficial del reto `[OFICIAL]`

- Santander es uno de los departamentos con mayor actividad sísmica de Colombia. Bajo la Mesa de Los Santos está el **Nido Sísmico de Bucaramanga**, una de las zonas de sismicidad intermedia más activas del mundo: cientos de sismos cada mes.
- El SGC registra y publica cada evento; el reto es convertir esos datos en modelos que **analicen y clasifiquen el riesgo sísmico** y, si el equipo lo decide, lo **anticipen**.
- **Temática:** sismos, con tema específico libre. Cada equipo elige primero su zona de estudio y después filtra los datos reales que le corresponden. Tema de referencia para quienes participan por primera vez: el comportamiento sísmico de Santander en los últimos 20 años.
- **Componente central obligatorio en los tres niveles:** implementar y demostrar un sistema de Machine Learning funcional, entrenado con datos reales de una fuente oficial.
- **No se acepta:** una presentación conceptual, un prototipo visual sin modelo, ni una solución basada solo en reglas fijas.

## 3. Las 3 categorías y sus modelos obligatorios `[OFICIAL]`

| Categoría | Modelo obligatorio | Variables mínimas | Reto | Producto | Salón de desarrollo |
|---|---|---|---|---|---|
| Básico | Modelo supervisado de clasificación | 10 | Limpiar y preparar los datos de la zona y clasificar su nivel de actividad sísmica como bajo, medio o alto, con métricas de desempeño | Cuaderno ejecutable o interfaz sencilla con una entrada y su resultado | 302A / 302B |
| Medio | Únicamente modelo no supervisado | 15 | Cumplir el reto Básico y comparar el comportamiento sísmico con otras unidades territoriales mediante patrones o grupos, con variables territoriales y visualización | Prototipo funcional con visualización territorial reproducible | 315A |
| Avanzado | **Obligatoriamente aprendizaje por refuerzo** | 20 | Cumplir los retos Básico y Medio, integrar la información sísmica con otra fuente oficial y construir una priorización exploratoria de zonas de Santander con un agente entrenado por refuerzo | Flujo funcional y verificable con demostración de la priorización (no se requiere API) | 307B |

> **Aclaración `[EQUIPO]` `[POR CONFIRMAR]`:** internamente se habló de "3 modelos supervisados". La web oficial indica que los 3 modelos que exige el recorrido Básico → Medio → Avanzado son: **supervisado** (Básico) + **no supervisado** (Medio) + **aprendizaje por refuerzo** (Avanzado). Confirmar con mentores.

## 4. Rúbricas oficiales (100 puntos por categoría) `[OFICIAL]`

Cada rúbrica suma 100 puntos; el puntaje del equipo es el promedio de sus jurados.

### 4.1. Básico

| Criterio | Puntos | Detalle |
|---|---|---|
| Datos: obtención y limpieza | 20 | Fuente oficial citada; conjunto de datos limpio y preparado; 10 variables justificadas |
| Presentación | 20 | Identificación del problema; exponen los integrantes; trabajo en equipo; manejo del tiempo y respuesta a preguntas |
| Modelo supervisado | 25 | Definición clara de cada nivel (bajo, medio, alto); entrenamiento, métricas básicas y comprensión de los resultados |
| Demostración | 10 | Cuaderno o interfaz en la que se observe una entrada y el resultado de la clasificación |
| Impacto e innovación | 15 | Impacto potencial para Santander y, cuando sea posible, un elemento innovador |
| Análisis de los datos | 10 | Exploración inicial y explicación clara de la metodología y los resultados |

### 4.2. Medio

| Criterio | Puntos | Detalle |
|---|---|---|
| Datos y preprocesamiento | 15 | 15 variables justificadas, incluidas territoriales; preparación y transformación de datos |
| Modelo no supervisado | 25 | Selección, entrenamiento y evaluación; comparación de enfoques; interpretación de grupos e importancia de variables |
| Arquitectura utilizada | 15 | Diagrama e integración de datos, procesamiento, modelo y visualización |
| Prototipo y visualización territorial | 10 | Prototipo funcional y visualización reproducible que compare unidades territoriales |
| Impacto e innovación | 15 | El análisis aporta una explicación útil para estimar o clasificar el riesgo, no solo grupos |
| Presentación | 20 | Identificación del problema; exponen los integrantes; trabajo en equipo |

### 4.3. Avanzado

| Criterio | Puntos | Detalle |
|---|---|---|
| Datos e integración de fuentes | 15 | 20 variables justificadas; evidencia de cómo se unió la información sísmica con otra fuente oficial |
| Aprendizaje por refuerzo | 20 | Agente entrenado por refuerzo (entorno, estados, acciones y recompensa); no se aceptan sustitutos |
| Validación e incertidumbre | 10 | Validación que respete el orden temporal; evaluación del agente; incertidumbre o explicación |
| Flujo integrado y trazabilidad | 15 | Flujo funcional, verificable y reproducible; solidez metodológica y trazabilidad |
| Impacto e innovación | 20 | Utilidad de la priorización para decidir en Santander; supuestos, limitaciones y extensiones |
| Presentación | 20 | Identificación del problema; exponen los integrantes; trabajo en equipo |

**Desempate:** primero el puntaje en impacto e innovación; después la solidez técnica del componente de Machine Learning. Un equipo ganador por categoría.

## 5. Requisitos habilitantes `[OFICIAL]`

Antes de calificar, el jurado verifica:

1. Un modelo de Machine Learning funcional, entrenado por el equipo y demostrado en vivo.
2. El tipo de modelo obligatorio de la categoría (supervisado en Básico, únicamente no supervisado en Medio y aprendizaje por refuerzo en Avanzado).
3. Datos reales de una fuente oficial citada.
4. El número mínimo de variables.

Un proyecto que no cumpla puede presentarse, pero **no es elegible para premiación**.

## 6. Fuentes de datos oficiales `[OFICIAL]`

| # | Fuente | URL | Contenido |
|---|---|---|---|
| 6.1 | SGC — Catálogo sísmico de la Red Sismológica Nacional (fuente principal) | http://bdrsnc.sgc.gov.co/paginas1/catalogo/ | Todos los sismos localizados en Colombia desde junio de 1993: fecha, coordenadas, profundidad, magnitud, fases, RMS, gap y municipio. Se descarga en Excel |
| 6.2 | SGC — Portal de Datos Abiertos | https://datos.sgc.gov.co/ | Zonas de amenaza sísmica, Mapa Geológico de Colombia e inventario de movimientos en masa (CSV, GeoJSON, Shapefile) |
| 6.3 | SGC — Sismicidad Histórica | https://sish.sgc.gov.co/ | Catálogo homogeneizado a Mw y fichas de sismos históricos con daños reportados |
| 6.4 | USGS — Earthquake Catalog | https://earthquake.usgs.gov/earthquakes/search/ | Catálogo mundial filtrable; sirve para contrastar magnitudes y localizaciones |
| 6.5 | Datos Abiertos Colombia | https://www.datos.gov.co/ | Incluye el reporte histórico de emergencias por desastres naturales de la Gobernación de Santander |
| 6.6 | DANE e IGAC | https://www.dane.gov.co/ | Población y vivienda por municipio, límites, cartografía, altitud y suelos |

**Reglas de citación `[OFICIAL]`:** citar por cada fuente la dirección web, la fecha de descarga y los filtros aplicados. Registrar en `Presentacion/fuente_oficial.md` y en el log de `datos/raw/README.md` al descargar.

## 7. Datos: origen, estructura y limpieza `[OFICIAL]`

- **Origen:** todos los equipos trabajan con datos reales de una fuente oficial; no se aceptan datos inventados ni sintéticos como fuente principal. La fuente principal es el catálogo sísmico de la RSN (SGC).
- **Estructura:** una sola tabla; cada fila es una observación y cada columna una variable.
- **Limpieza:** sin duplicados; faltantes tratados; unidades y fechas homogéneas; atípicos revisados.
- **Variables mínimas:** Básico 10, Medio 15, Avanzado 20. Las variables derivadas cuentan si se justifican.
- **Diccionario de datos:** nombre, tipo, unidad, fuente y por qué se incluye cada variable.
- **Variable objetivo:** bien definida y que **no se calcule directamente de las variables de entrada** (evitar fuga de datos).
- **Unificación de catálogos:** el catálogo del SGC tiene dos consultas con columnas distintas (junio de 1993 a febrero de 2018 y desde marzo de 2018) que el equipo debe unificar. Conviene consultar por meses o años.
- **Privacidad:** nada de datos personales sensibles sin autorización.

## 8. Entregables (cierre 12:00 m.) `[OFICIAL]`

| # | Entregable | Ubicación en el repo | Estado |
|---|---|---|---|
| 1 | Dataset limpio (CSV) | `datos/processed/dataset_limpio.csv` | listo (175.185 sismos; hueco 2014–2017 documentado) |
| 2 | Diccionario de datos con justificación por variable | `Presentacion/diccionario_de_datos.md` | pendiente |
| 3 | Fuente oficial citada (dirección web, fecha y filtros) | `Presentacion/fuente_oficial.md` | listo (catálogo SGC ×2 y Atlas de Fallas 2020) |
| 4 | Código del análisis y del entrenamiento (cuaderno o repo reproducible) | `notebooks/` y `modelos/*/` | listo (`notebooks/` 01–05 + cuadernos en `modelos/*/`) |
| 5 | El modelo obligatorio de la categoría con sus métricas y validación | cards en `modelos/*/README.md` | listo (cards de supervisado y no supervisado + LEEME del agente) |
| 6 | El prototipo o la demostración | `app/` | listo (`app/sismo_app_F/`; insumos reales en `datos/para_app/`) |
| 7 | La presentación para el pitch final | `Presentacion/pitch/` | pendiente |

> La web oficial agrupa 1 + 2 y enumera 6 entregables; aquí se separan para seguimiento. Checklist vivo en `Presentacion/README.md`.

## 9. Reglas de participación `[OFICIAL]`

- Equipos de 2 o 3 personas, de una misma institución o mixtos. Cada equipo compite en una sola categoría, inscrita antes de iniciar. Una persona solo puede estar en un equipo.
- El reto inicia a las 8:00 a. m. con 15 minutos de tolerancia; el tiempo perdido no se repone.
- Las herramientas son de cada participante: computador portátil con cargador, software y cuentas de nube o de IA (incluidas las de pago).
- Los modelos preentrenados, las APIs y la IA generativa se permiten como apoyo, pero **no reemplazan el modelo de Machine Learning que exige la categoría**.
- Se respeta la privacidad: nada de datos personales sensibles sin autorización.
- Modalidad presencial, con acompañamiento del CIAI.

## 10. Agenda oficial del 13 de octubre `[OFICIAL]`

| Hora | Bloque | Detalle |
|---|---|---|
| 07:00–08:00 | Apertura y ceremonia | Registro y validación de equipos, confirmación de categoría, presentación del CIAI y UTSmart, la temática, las reglas y las rúbricas |
| 08:00–12:00 | Desarrollo técnico | Zona y tema, datos, limpieza, entrenamiento, validación, prototipo y mentorías. **A las 12:00 m. cada equipo entrega su proyecto (entregas cerradas)** |
| 12:00–14:00 | Receso — entregas cerradas | El jurado revisa los proyectos y los requisitos habilitantes antes de las presentaciones |
| 14:00–16:45 | Presentaciones ante el jurado | En cuatro salones a la vez, con dos jurados por salón: primero Básico, luego Medio y al final Avanzado. 10 minutos de presentación y demostración, 3 de preguntas y 2 de calificación. Exponen los integrantes del equipo |
| 16:45–18:00 | Reevaluación (si hace falta) | Solo equipos con diferencias mayores a 20 puntos entre jurados, empates en el primer puesto o dudas sobre requisitos |
| 17:30–18:00 | Deliberación y suma de puntajes | Consolidación de puntajes, desempates, firma del acta y registro fotográfico |
| 16 de octubre | Premiación | Un equipo ganador por categoría (hora por confirmar) |

**Turnos de presentación — categoría Avanzado** (hora Colombia, sujetos a ajustes del día):

| Sala | Equipos (en orden) | Hora aproximada |
|---|---|---|
| 302A | Grupo Niche, CodeForce | 16:10, 16:25 |
| 302B | Dynamockapps, DS3 Team | 16:10, 16:25 |
| 315A | LosPrint, Syntax error II, Bit Conquistadores | 15:55, 16:10, 16:25 |
| 307B | Monos, UDES Syntax Titans | 15:55, 16:10 |

## 11. Guía oficial paso a paso — nivel Básico `[OFICIAL]`

1. **Elegir zona y pregunta:** un municipio, una provincia, el área metropolitana o un corredor vial, y la pregunta a responder.
2. **Obtener los datos:** catálogo del SGC (departamento, municipio o rectángulo de coordenadas, periodo y magnitud mínima) descargado en Excel. Registrar dirección, fecha de descarga y filtros.
3. **Limpiar y organizar:** quitar duplicados y filas sin coordenadas o sin magnitud; unificar fechas y las columnas de los dos periodos del catálogo; dejar mínimo 10 variables justificadas.
4. **Definir la variable objetivo:** por ejemplo, para cada municipio y año: bajo (sin sismos de magnitud 3,0 o más), medio (entre 3,0 y 3,9) o alto (al menos uno de 4,0 o más).
5. **Explorar:** al menos tres gráficas (sismos por año, histograma de magnitudes, mapa o dispersión latitud–longitud) y qué se observa en cada una.
6. **Entrenar un modelo supervisado:** separar entrenamiento y prueba (80/20) y entrenar un modelo sencillo de scikit-learn (árbol de decisión, regresión logística, KNN o random forest).
7. **Medir el modelo:** exactitud y matriz de confusión; si las clases están desbalanceadas, precisión, recall y F1. Explicar en qué casos se equivoca.
8. **Construir la demostración:** cuaderno ejecutable o interfaz sencilla (Streamlit o Gradio) con una entrada y su resultado.
9. **Preparar el pitch:** problema y zona, fuente de datos, variables, modelo y métricas, demostración, impacto y limitaciones.

Herramientas sugeridas: Python con pandas, scikit-learn y matplotlib en Google Colab o Jupyter.

## 12. Estrategia y stack sugeridos `[GUÍA]`

Recomendaciones del brief del equipo (no son requisitos oficiales):

- **Features sugeridas:** clase de profundidad (superficial, intermedia, profunda); frecuencia de sismos por zona (hotspots) con clustering K-Means; distancia del epicentro a fallas geológicas conocidas y a cascos urbanos.
- **Evaluación:** no confiar solo en la exactitud; usar matriz de confusión, precisión, recall y F1. En riesgo, es preferible un falso positivo a un falso negativo: maximizar el recall de la clase "Alto Riesgo".
- **Balanceo:** SMOTE o ajuste de pesos para que el modelo no se sesgue hacia la clase mayoritaria.
- **Dashboard:** Streamlit con mapa interactivo (Folium o Plotly); consulta por región o coordenadas y respuesta inmediata de la clasificación.
- **Stack:** Google Colab + Git/GitHub; Python con pandas y GeoPandas; scikit-learn; Streamlit.
- **Pitch:** abrir con el mapa de calor; explicar cómo un comité de gestión del riesgo de Santander tomaría decisiones preventivas; responder a la premisa "Santander Inteligente".

## 13. Discrepancias y decisiones pendientes `[POR CONFIRMAR]`

1. **"3 modelos supervisados" vs oficial:** el equipo habló de 3 supervisados; la web oficial describe supervisado (Básico) + no supervisado (Medio) + RL (Avanzado). Confirmar con mentores (§3).
2. **Cronograma del brief vs agenda oficial:** el brief proponía trabajar hasta las 18:00 con modelado a las 11:30–14:00; la agenda oficial cierra entregas a las **12:00 m.** y las presentaciones empiezan a las 14:00 (§10). Planificar con la agenda oficial.
3. **Variable objetivo:** el requisito pide que no se calcule directamente de las variables de entrada, pero el ejemplo oficial del Básico usa umbrales de magnitud por municipio y año (derivables de los datos). Definir el target, justificarlo y dejarlo documentado (§7).
4. **Alcance para Avanzado:** "cumplir los retos Básico y Medio" sugiere entregar también los modelos supervisado y no supervisado; confirmar si se exigen formalmente o basta con evidenciar sus componentes en el flujo.
5. **Registro de fuentes:** hecho — catálogo del SGC (×2) y Atlas de Fallas 2020 con URL, fecha y filtros en `Presentacion/fuente_oficial.md` y `datos/raw/README.md` (§6).
6. **Artefactos grandes:** GitHub rechaza archivos de más de 100 MB; si un modelo o dataset pesa demasiado, subirlo por enlace externo y documentarlo en la card.

## 14. Glosario

- **RMS (error cuadrático medio):** error del ajuste de la localización del sismo, en segundos; más bajo = mejor localización.
- **Gap azimutal:** mayor hueco angular sin estaciones alrededor del epicentro, en grados; más bajo = mejor cobertura.
- **Fases sísmicas:** lecturas de arribo de ondas (por ejemplo P y S) usadas para localizar el evento.
- **Mw (magnitud momento):** escala usada para homogeneizar el catálogo histórico.
- **Nido sísmico:** zona con aglomeración de sismos a profundidad intermedia; el caso local es el Nido Sísmico de Bucaramanga, bajo la Mesa de Los Santos.
- **Hotspot:** zona de alta frecuencia sísmica detectada, por ejemplo, con K-Means.
- **SMOTE:** técnica de balanceo que sintetiza ejemplos de la clase minoritaria; solo para entrenar, nunca como fuente de datos.
- **Matriz de confusión, precisión, recall y F1:** métricas de clasificación; recall = fracción de casos reales que el modelo detecta.
- **Aprendizaje por refuerzo (RL):** paradigma donde un agente aprende una política de acciones (entorno, estados, acciones y recompensa) maximizando recompensa acumulada; aquí se usa para priorizar zonas.
- **Requisitos habilitantes:** verificaciones del jurado previas a calificar (§5).

## 15. Próximos pasos (hoy)

- [ ] Descargar el catálogo del SGC por periodos y registrar URL, fecha y filtros (§6).
- [ ] Producir `datos/processed/dataset_limpio.csv` y el diccionario de datos (entregables 1–2).
- [ ] Entrenar los modelos: supervisado (Básico), no supervisado (Medio) y agente RL (Avanzado).
- [ ] Construir el prototipo en `app/` con los 3 modelos.
- [ ] Preparar el pitch en `Presentacion/pitch/`.
- [ ] Confirmar con mentores la composición de los 3 modelos y la variable objetivo (§13).

## 16. Registro de cambios del documento

| Fecha | Cambio |
|---|---|
| 2026-10-13 | Creación: brief del equipo + datos oficiales extraídos de la página del evento (rúbricas, agenda, reglas, fuentes y turnos) |
