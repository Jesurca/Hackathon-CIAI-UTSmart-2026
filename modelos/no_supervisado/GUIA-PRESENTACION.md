# Modelo no supervisado — guía para aprenderlo y presentarlo

> **Para qué sirve:** explica en lenguaje sencillo qué hace el modelo no supervisado, cómo funciona por dentro y cómo contarlo en diapositivas. Está alineado con la versión actual de `modelo_no_supervisado.ipynb` (conectado a `datos/processed/dataset_limpio.csv`, k=4 fijo).
> **Cuándo leerla:** antes de armar el pitch y antes de la demostración. Léela con el notebook abierto: las secciones coinciden.
> **Estado de los números:** los resultados de la sección 4 (perfil, grupos, importancia) son reales y reproducibles: se obtienen ejecutando el notebook con `fallas_santander.geojson` presente. Coinciden con la card del modelo.

---

## 1. La idea en 30 segundos

- **Pregunta:** ¿qué municipios de Santander se comportan símicamente parecido y a cuáles les parece el Nido Sísmico de Bucaramanga?
- **Sin etiquetas previas:** nadie escribió "este municipio es de riesgo alto". El algoritmo agrupa por similitud y después ponemos nombres (Bajo / Moderado / Alto / Muy alto) solo para explicarlo.
- **Analogía:** ordenar una caja de fotos de familias distintas sin que nadie te dé los apellidos: primero agrupas por parecido, después les pones nombre.
- **El hallazgo que engancha al jurado:** no hay un solo tipo de peligro. El modelo descubre **dos**: municipios con *mucha actividad profunda* (el Nido) y municipios con *pocos sismos pero muy superficiales* (más dañinos porque la atenuación es menor).
- **El flujo completo en una línea:**

```
dataset_limpio.csv (175.185 sismos) -> filtro Ml + 2006-2026 -> perfil por municipio (20 vars + fallas)
-> escalar + PCA -> KMeans k=4 -> índice de actividad -> 4 grupos -> mapa
```

- **Dónde suma puntos** (rúbrica Medio §4.2 del brief): datos y preprocesamiento 15 · modelo no supervisado 25 · arquitectura 15 · prototipo/visualización 10 · impacto 15 · presentación 20.

---

## 2. Los datos reales con los que trabaja

Verificado el 2026-10-13. El notebook lee `datos/processed/dataset_limpio.csv` y se queda solo con magnitud **Ml** (el 99,2 % de los sismos viene así) y la ventana **2006–2026** que eligió el equipo.

| Dato | Valor | Por qué importa |
|---|---|---|
| Eventos (Ml, 2006+) | 144.456 | del catálogo del SGC, fuente oficial |
| Años con datos | 17 | hay hueco 2014–2017; las tasas se calculan sobre los años que sí existen (2006–2013 y 2018–2026) |
| Municipios (≥15 sismos) | **82** | los que entran al modelo; los demás tienen estadística poco confiable |
| Municipio con más eventos | LOS SANTOS (~53 %) | el Nido Sísmico domina; por eso hay que escalar bien |
| Magnitud máxima | 6,2 | evento extremo del catálogo |
| Profundidad | hasta 224 km | el Nido es intermedio-profundo; los superficiales (<30 km) son más dañinos |
| Variables de modelo | 20 | supera el mínimo de Medio (15) |

**La jugada clave del modelo:** cambiar la unidad de análisis.

```
ANTES: 1 fila = 1 sismo        (144.456 filas)
DESPUÉS: 1 fila = 1 municipio  (82 filas, 20 variables-resumen)
```

---

## 3. El pipeline paso a paso

Cada paso coincide con una sección del notebook. Léelos juntos.

### Paso 0 — Configuración

| Parámetro | Valor | Qué controla |
|---|---|---|
| `RUTA_EVENTOS` | `datos/processed/dataset_limpio.csv` | lee el dataset ya limpio (no toca crudos) |
| `ANIO_INICIO` | 2006 | ventana de análisis |
| `MIN_EVENTOS` | 15 | municipios con menos sismos se descartan |
| `K_FINAL` | 4 | Bajo, Moderado, Alto, Muy alto (decisión del equipo) |
| `tipo_mag` | solo Ml | el 99,2 % de los eventos; homogeniza magnitudes |

### Paso 1 — Cargar y filtrar

Lee el CSV ya limpio, filtra año ≥ 2006 y magnitud Ml. **Ojo al jurado:** el notebook documenta que 2014–2017 no tiene datos porque ese bloque del catálogo no se descargó. Las tasas por año se calculan solo con los 17 años que sí tienen datos (no 21).

### Paso 2 — Variables por municipio (la parte central)

De cada municipio se calculan **20 variables** en 6 familias:

| Familia | Variables | Qué capturan |
|---|---|---|
| Frecuencia | `n_sismos`, `sismos_x_anio` | qué tanto tiembla |
| Magnitud | `mag_media`, `mag_max`, `mag_std`, `mag_p90`, `pct_m3_mas`, `pct_m4_mas` | tamaño típico y eventos sentidos/dañinos |
| Energía | `log_energia` | energía liberada acumulada (10^(1,5·M+4,8) julios, Gutenberg-Richter). Una M6 suelta ~1.000× la energía de una M4 |
| Profundidad | `prof_media`, `prof_std`, `pct_superficial`, `pct_intermedia`, `pct_profunda` | los superficiales (<30 km) son más dañinos; el Nido es intermedio-profundo (>150 km) |
| Calidad/Tendencia | `rms_medio`, `gap_medio`, `fases_medias`, `frac_reciente` | confianza de localización y actividad reciente (últimos 5 años) |
| Territorial | `dist_nido_km`, `dist_falla_km` | cercanía al Nido y a fallas geológicas visibles |

`frac_reciente` = fracción de la sismicidad del municipio en los últimos 5 años.

### Paso 3 — Fallas geológicas (opcional pero recomendado)

Si existe `datos/processed/fallas_santander.geojson`, agrega `dist_falla_km`: distancia del centro del municipio a la traza de falla visible más cercana (haversine). Es una aproximación — se declara como limitación.

### Paso 4 — Escalar y reducir

1. `log1p` a las variables de escala (conteos, energía, distancias): las colas son larguísimas porque LOS SANTOS tiene la mitad de los eventos.
2. `RobustScaler` (mediana / rango intercuartil): un puñado de municipios extremos no aplasta al resto.
3. Recorte a `[-5, 5]`: elimina atípicos extremos.
4. PCA: se queda con las componentes que explican el ≥90 % de la varianza (en la corrida real: **6 componentes = 90,5 %**). Menos ruido y menos colinealidad. Las coordenadas no entran al modelo (sirven para el mapa).

### Paso 5 — Evidencia: comparar 3 algoritmos × k = 2..8

Como pide la rúbrica (comparar enfoques), se prueban:

| Algoritmo | Intuición |
|---|---|
| KMeans | grupos "redondos" alrededor de centros |
| Ward (jerárquico) | va fusionando parecidos, como un árbol genealógico |
| GMM (mezcla gaussiana) | grupos probabilísticos con forma de elipse |

Cada combinación se mide con 4 métricas:

| Métrica | Dirección | Significado simple |
|---|---|---|
| Silhouette ↑ | más alto mejor | ¿tan juntos los de dentro vs. lejos los de otros grupos? |
| Davies-Bouldin ↓ | más bajo mejor | ¿qué tan mezclados quedan? |
| Calinski-Harabasz ↑ | más alto mejor | separación neta entre grupos |
| Estabilidad (ARI) ↑ | más alto mejor | si quito el 20 % de los municipios al azar, ¿los grupos se repiten? (10 repeticiones) |

**Importante:** esta sección queda como *evidencia* para el jurado. El modelo final no se elige por torneo: el equipo acordó **k = 4 fijo** (Bajo / Moderado / Alto / Muy alto).

### Paso 6 — Armar los grupos

```python
modelo_final = KMeans(n_clusters=4, random_state=42)
F["grupo_num"] = modelo_final.fit_predict(Z)
```

Después se calcula un **índice de actividad** = promedio de z-scores de `sismos_x_anio`, `log_energia`, `mag_max`, `pct_m4_mas`, `pct_superficial`. Este índice **solo ordena y nombra** los grupos. El clustering no lo usa → sigue siendo 100 % no supervisado.

### Paso 7 — ¿Qué variables separan los grupos?

Sin modelos supervisados: **Kruskal-Wallis** por variable entre grupos, con tamaño de efecto η² = H/(n−1). Así se obtiene el ranking de qué variable distingue más un grupo de otro.

### Paso 8 — Gráficos

- Municipios en el plano PCA (PC1 vs PC2) coloreados por grupo.
- Mapa de epicentros + centros de municipio por grupo + trazas de fallas + el Nido (★).
- Barras de importancia de variables (η²).

### Paso 9 — Guardar resultados

Todo queda en `modelos/no_supervisado/`:

| Archivo | Contenido |
|---|---|
| `grupo_por_municipio.csv` | municipio → grupo + índice de actividad |
| `perfil_grupos.csv` | mediana de cada variable por grupo |
| `comparacion_modelos.csv` | métricas del torneo KMeans/Ward/GMM |
| `importancia_variables.csv` | ranking η² |
| `modelo_no_supervisado.joblib` | escalador + PCA + KMeans + columnas + etiquetas (para el RL y la app) |

### Paso 10 — Para el pitch

> "Encontramos dos tipos de peligro: **Muy alto** = mucha actividad (zona del nido sísmico, sismos profundos); **Alto** = pocos sismos pero superficiales (más dañinos)."

---

## 4. Resultados reales (reproducidos con fallas)

Ejecutar el notebook con `fallas_santander.geojson` presente da exactamente estos números.

### Los 4 grupos

| Grupo | Municipios | Perfil resumido |
|---|---|---|
| **Bajo** | 23 | pocos sismos, profundos (mediana 128 km), lejos de fallas (~2 km) |
| **Moderado** | 37 | actividad media, sismos intermedios (~111 km), la mayoría |
| **Alto** | 8 | pocos sismos pero **muy superficiales** (mediana 53 km), los más dañinos |
| **Muy alto** | 14 | mucha actividad, sismos profundos (~136 km), cerca del Nido (~24 km) |

### Municipios por grupo

- **Bajo (23):** MATANZA, CHARTA, GAMBITA, SURATA, FLORIDABLANCA, CHARALA, CURITI, SAN GIL, SUAITA, MOLAGAVITA, OCAMONTE, SOCORRO, GUAPOTA, PINCHOTE, VALLE DE SAN JOSE, OIBA, CONCEPCION, PARAMO, CONFINES, CERRITO, MALAGA, PALMAS DEL SOCORRO, CALIFORNIA
- **Moderado (37):** CIMITARRA, VILLANUEVA, RIONEGRO, EL PLAYON, EL CARMEN, EL PENON, SANTA HELENA DEL OPON, SABANA DE TORRES, LANDAZURI, JORDAN, BARRANCABERMEJA, VELEZ, SUCRE, BOLIVAR, SIMACOTA, SAN VICENTE DE CHUCURI, LEBRIJA, HATO, CHIPATA, PUENTE NACIONAL, CHIMA, LA PAZ, CONTRATACION, BARICHARA, GALAN, ALBANIA, BARBOSA, EL GUACAMAYO, FLORIAN, SAN BENITO, GUAVATA, JESUS MARIA, GUEPSA, GUADALUPE, AGUADA, CABRERA, PALMAR
- **Alto (8):** LA BELLEZA, ENCINO, PUERTO PARRA, ONZAGA, COROMORO, CARCASI, SAN JOAQUIN, VETAS
- **Muy alto (14):** LOS SANTOS, MOGOTES, ZAPATOCA, CEPITA, SANTA BARBARA, SAN ANDRES, PUERTO WILCHES, TONA, PIEDECUESTA, BUCARAMANGA, BETULIA, GIRON, GUACA, ARATOCA

### Variables que más separan los grupos (η²)

| Variable | η² | Interpretación |
|---|---|---|
| `pct_m4_mas` | 0,69 | fracción de eventos grandes |
| `pct_intermedia` | 0,64 | fracción de sismos intermedios |
| `pct_profunda` | 0,61 | fracción de sismos profundos |
| `mag_media` | 0,54 | magnitud promedio |
| `gap_medio` | 0,53 | calidad de localización |

### Perfil numérico (medianas por grupo)

| Variable | Bajo | Moderado | Alto | Muy alto |
|---|---|---|---|---|
| n_sismos | 57 | 430 | 34 | 804 |
| sismos_x_anio | 3,4 | 25,3 | 2,0 | 47,3 |
| mag_max | 3,3 | 3,6 | 3,0 | 4,5 |
| pct_superficial | 0,11 | 0,04 | **0,53** | 0,03 |
| pct_profunda | 0,36 | 0,01 | 0,06 | 0,21 |
| log_energia | 10,1 | 10,8 | 9,6 | 11,8 |
| prof_media (km) | 127,6 | 110,9 | **53,2** | 136,1 |
| dist_nido_km | 49,7 | 79,2 | 63,7 | **24,0** |

---

## 5. Conceptos en palabras simples

| Término | En una frase | En este modelo |
|---|---|---|
| No supervisado | aprender sin etiquetas previas | nadie dice qué municipio es "alto"; el algoritmo lo descubre |
| Clustering | agrupar por similitud | municipios con perfil sísmico parecido caen juntos |
| Unidad de análisis | qué es "una fila" | municipio (no el sismo individual) |
| Estandarizar | poner variables en la misma escala | RobustScaler para que kilómetros y grados sean comparables |
| PCA | resumir variables en componentes | 90,5 % de la información en 6 componentes, sin ruido |
| Silhouette | nota de calidad del agrupamiento (−1 a 1) | elige qué tan "limpios" quedan los grupos |
| ARI | parecido entre dos agrupamientos | mide estabilidad al quitar el 20 % de municipios |
| η² | fuerza con que una variable separa grupos | ranking de importancia sin usar supervisado |
| Amenaza vs riesgo | lo que el suelo puede hacer vs. lo que puede dañar | medimos amenaza; el riesgo suma población y vivienda (DANE) |
| Mw / Ml | tipos de magnitud | el modelo usa solo Ml (el 99,2 %), más estable para comparar |

---

## 6. Guion de diapositivas (10 slides, ~10 minutos)

**Slide 1 — Portada.** "¿Qué municipios de Santander tiemblan parecido?" · equipo, categoría Avanzada, componente Medio. *Muestra:* mapa de calor de epicentros. *Dices:* la pregunta en una frase.

**Slide 2 — Los datos.** 175.185 sismos del SGC (1993–2026), magnitud Ml, ventana 2006–2026, 82 municipios. *Muestra:* el filtro y el mapa de densidad. *Dices:* fuente oficial; menciona el hueco 2014–2017 tú primero.

**Slide 3 — De sismos a perfiles.** 1 fila por municipio, 20 variables en 6 familias. *Muestra:* la tabla de familias de variables. *Dices:* "un municipio no es un sismo; es un comportamiento".

**Slide 4 — Sin etiquetas.** Cómo agrupa el modelo y por qué los nombres van después. *Muestra:* el flujo en una línea (sección 1). *Dices:* la analogía de las fotos de familia.

**Slide 5 — Escalar y PCA.** Por qué LOS SANTOS (la mitad de los eventos) obliga a `log1p` + `RobustScaler` + PCA 90 %. *Muestra:* el antes/después. *Dices:* "sin escalar, un municipio aplastaría a los otros 81".

**Slide 6 — La evidencia (torneo).** 3 algoritmos × k=2..8 con 4 métricas. *Muestra:* las curvas de métricas vs k. *Dices:* la rúbrica pide comparar enfoques; elegimos k=4 por decisión del equipo.

**Slide 7 — El resultado.** Los 4 grupos en el mapa. *Muestra:* mapa de municipios coloreado + Nido. *Dices:* "el grupo de mayor actividad concentra 14 municipios, entre ellos LOS SANTOS, BUCARAMANGA, PIEDECUESTA…".

**Slide 8 — Dos tipos de peligro.** La idea fuerza: Muy alto = mucha actividad profunda; Alto = pocos pero superficiales. *Muestra:* la tabla comparativa de perfiles. *Dices:* "no hay un solo patrón de riesgo".

**Slide 9 — Qué separa los grupos.** Top 5 variables por η². *Muestra:* barras de importancia. *Dices:* "sin modelos supervisados, las proporciones de profundidad y magnitud son lo que más distingue".

**Slide 10 — Limitaciones honestas + cierre.** Falta 2014–2017; falta población/vivienda (amenaza ≠ riesgo); distancia a fallas aproximada. *Muestra:* nada, habla tú. *Dices:* decirlo tú primero te pone en ventaja. Cierra con "esto alimenta el agente de priorización (RL)".

---

## 7. Preguntas difíciles del jurado (respuestas listas)

1. **"¿Por qué es no supervisado si ustedes ponen Bajo/Moderado/Alto/Muy alto?"** — Los grupos los descubre KMeans sin etiquetas. El índice de actividad se calcula después y solo nombra y ordena; ninguna etiqueta entra al entrenamiento.
2. **"¿Por qué k=4 y no otro?"** — Decisión del equipo respaldada por evidencia: se compararon KMeans/Ward/GMM para k=2..8 con 4 métricas (silhouette, Davies-Bouldin, Calinski-Harabasz, estabilidad ARI). k=4 da la historia más clara al jurado.
3. **"¿Cómo sé que los grupos no son ruido?"** — Estabilidad ARI: se repite el agrupamiento 10 veces quitando el 20 % de los municipios al azar y los grupos se mantienen.
4. **"¿Eso es riesgo sísmico?"** — Es amenaza. El riesgo suma vulnerabilidad (población, vivienda). Por eso el notebook deja lista la unión con DANE y lo decimos como mejora.
5. **"¿Por qué LOS SANTOS domina todo?"** — Tiene ~la mitad de los eventos. Por eso `log1p` + `RobustScaler` + recorte: para que los otros 81 municipios también se vean.
6. **"¿Y el hueco 2014–2017?"** — Falta descargar ese periodo. Las tasas por año usan solo los 17 años con datos, no 21; queda documentado.
7. **"¿Por qué solo 82 municipios?"** — `MIN_EVENTOS = 15` descarta unidades con estadística poco confiable.
8. **"¿Cómo se conecta con el aprendizaje por refuerzo?"** — El grupo y el índice de actividad por municipio son parte del estado del agente RL (`modelo_no_supervisado.joblib`).
9. **"¿Por qué solo magnitud Ml?"** — Es el 99,2 % del catálogo y es la escala más estable del SGC para comparar municipios. Mezclar Ml/Mw introduciría un sesgo de escala.

---

## 8. Checklist antes de presentar

- [ ] Notebook ejecutado con `fallas_santander.geojson` presente (para los 20 grupos reproducibles)
- [ ] Artifacts generados: `grupo_por_municipio.csv`, `perfil_grupos.csv`, `comparacion_modelos.csv`, `importancia_variables.csv`, `modelo_no_supervisado.joblib`
- [ ] Card del modelo (`README.md`) con estado "listo", dueño y métricas
- [ ] Diccionario de datos completado con las 20 variables (`entregables/diccionario_de_datos.md`)
- [ ] Mapas y gráficos exportados a imágenes para el deck
- [ ] Ensayo del pitch: 10 minutos con demostración en vivo
- [ ] Abrir el pitch con el hallazgo de los "dos tipos de peligro"

---

*Guía escrita el 2026-10-13 a partir de `modelos/no_supervisado/modelo_no_supervisado.ipynb`. Los números de resultados son reproducibles (corridos con fallas incluidas, coinciden con la card). No reemplaza la card: el estado y las métricas oficiales viven en `README.md`.*
