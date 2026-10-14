# Exporta una predicción supervisada municipio x trimestre al formato que pide la app.
# El modelo del reto Básico es semanal y global; la app necesita una fila por municipio y
# trimestre, así que aquí se entrena la variante municipal con la misma receta anti-fuga:
# la fila del trimestre t usa SOLO datos hasta el final de t, y la etiqueta es el nivel
# del trimestre t+1 (terciles de energía calculados únicamente con el tramo de train).
# Uso: python notebooks/05_supervisado_para_app.py
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Encuentra la raíz del repo (sube carpetas hasta hallar "datos").
RAIZ = Path.cwd()
while not (RAIZ / "datos").exists() and RAIZ.parent != RAIZ:
    RAIZ = RAIZ.parent

SALIDA = RAIZ / "datos" / "para_app" / "supervisado_resultados.csv"

# --- Parámetros del experimento (quedan a la vista del jurado) ---
CLASES = ["Bajo", "Medio", "Alto"]         # terciles de energía del trimestre siguiente
# Los nombres coinciden con las columnas del panel de la app, para que el .pkl también aplique.
FEATURES = ["n_sismos", "mag_media", "mag_max", "energia_log", "prof_media", "pct_superficiales"]
CORTE_TRAIN = pd.Period("2021Q4", freq="Q")  # todo lo anterior, train
CORTE_VAL = pd.Period("2023Q4", freq="Q")    # hasta aquí val; el resto, test
SEMILLA = 42
# Si el mejor modelo no supera este F1, se avisa para no vender humo en el pitch.
F1_MINIMO = 0.4

eventos = pd.read_csv(RAIZ / "datos/processed/dataset_limpio.csv",
                      usecols=["fecha", "mag", "prof_km", "municipio"], parse_dates=["fecha"])
FECHA_MAX = eventos["fecha"].max()

# Energía liberada por evento (regla de Gutenberg-Richter, igual que en el reto Básico).
eventos["energia"] = 10 ** (1.5 * eventos["mag"] + 4.8)
eventos["superficial"] = (eventos["prof_km"] < 30).astype(int)
eventos["trimestre"] = eventos["fecha"].dt.to_period("Q")

# Una fila por municipio y trimestre: solo mira lo ocurrido dentro de ese trimestre.
panel = (eventos.groupby(["municipio", "trimestre"])
         .agg(n_sismos=("mag", "size"),
              mag_media=("mag", "mean"),
              mag_max=("mag", "max"),
              energia_log=("energia", lambda s: np.log10(s.sum())),
              prof_media=("prof_km", "mean"),
              pct_superficiales=("superficial", "mean"))
         .reset_index())

# La etiqueta es el nivel del trimestre inmediatamente siguiente; se exige que sea el
# trimestre calendario contiguo para no cruzar el hueco 2014-2017 ni un año sin datos.
panel["energia_siguiente"] = panel.groupby("municipio")["energia_log"].shift(-1)
panel["trimestre_siguiente"] = panel.groupby("municipio")["trimestre"].shift(-1)
panel = panel[panel["trimestre_siguiente"] == panel["trimestre"] + 1].copy()
# El último trimestre del archivo está incompleto (termina hoy): su energía no es comparable.
panel = panel[panel["trimestre_siguiente"].dt.end_time <= FECHA_MAX].copy()

# Cortes de terciles calculados SOLO con el tramo de train (evita fuga de información).
energia_train = panel.loc[panel["trimestre"] <= CORTE_TRAIN, "energia_siguiente"]
t1, t2 = energia_train.quantile([1 / 3, 2 / 3])
panel["nivel_real"] = pd.cut(panel["energia_siguiente"], bins=[-np.inf, t1, t2, np.inf], labels=CLASES)

# El "conjunto" va por trimestre de corte: el modelo solo puede ver datos hasta ese trimestre.
panel["conjunto"] = np.select([panel["trimestre"] <= CORTE_TRAIN, panel["trimestre"] <= CORTE_VAL],
                              ["train", "val"], default="test")

datos = panel.dropna(subset=FEATURES + ["nivel_real"]).copy()
train = datos[datos["conjunto"] == "train"]
val = datos[datos["conjunto"] == "val"]

# Dos candidatos: bosque aleatorio (suele ganar con tablas) y regresión logística
# (la familia del modelo semanal del equipo). Se elige por F1 macro en validación.
candidatos = {
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=SEMILLA, n_jobs=-1),
    "LogisticRegression": Pipeline([("escalado", StandardScaler()),
                                    ("modelo", LogisticRegression(max_iter=2000, random_state=SEMILLA))]),
}
mejor_nombre, mejor_f1, mejor_modelo = None, -1.0, None
for nombre, modelo in candidatos.items():
    modelo.fit(train[FEATURES], train["nivel_real"].astype(str))
    f1 = f1_score(val["nivel_real"].astype(str), modelo.predict(val[FEATURES]), average="macro")
    print(f"{nombre}: F1 macro (val) = {f1:.3f}")
    if f1 > mejor_f1:
        mejor_nombre, mejor_f1, mejor_modelo = nombre, f1, modelo

# El modelo entrenado va al kit de la app (por si algún día falta el CSV): el .pkl guarda
# los nombres de columna del panel, así que la app puede aplicarlo solo.
joblib.dump(mejor_modelo, RAIZ / "datos" / "para_app" / "supervisado_modelo.pkl")

# Predicciones y probabilidades para TODAS las filas (las de train son dentro de muestra,
# solo sirven para completar el archivo; las métricas que importan son val y test).
prob = mejor_modelo.predict_proba(datos[FEATURES])
datos["nivel_predicho"] = mejor_modelo.predict(datos[FEATURES])

salida = pd.DataFrame({
    "municipio": datos["municipio"],
    # El periodo es el trimestre de corte: la predicción se hace al terminarlo.
    "periodo": datos["trimestre"].astype(str),
    "nivel_predicho": datos["nivel_predicho"],
    "nivel_real": datos["nivel_real"].astype(str),
    "prob_bajo": prob[:, list(mejor_modelo.classes_).index("Bajo")].round(4),
    "prob_medio": prob[:, list(mejor_modelo.classes_).index("Medio")].round(4),
    "prob_alto": prob[:, list(mejor_modelo.classes_).index("Alto")].round(4),
    "conjunto": datos["conjunto"],
})
salida.to_csv(SALIDA, index=False, encoding="utf-8")

# --- Resumen para consola (evidencia para la memoria) ---
print(f"\nModelo elegido: {mejor_nombre}")
print(f"Cortes de terciles (energía log del trimestre siguiente): {t1:.2f} y {t2:.2f}")
print(f"Filas: {len(salida):,} | Municipios: {salida['municipio'].nunique()}")
print(salida.groupby("conjunto").size().to_string())
print(salida["nivel_real"].value_counts().to_string())
for conjunto in ["val", "test"]:
    bloque = datos[datos["conjunto"] == conjunto]
    exactitud = accuracy_score(bloque["nivel_real"].astype(str), bloque["nivel_predicho"])
    f1 = f1_score(bloque["nivel_real"].astype(str), bloque["nivel_predicho"], average="macro")
    print(f"{conjunto}: exactitud = {exactitud:.3f} | F1 macro = {f1:.3f} ({len(bloque):,} filas)")
if mejor_f1 < F1_MINIMO:
    print(f"\nAviso: F1 macro de validación ({mejor_f1:.3f}) por debajo de {F1_MINIMO};")
    print("conviene decirlo con honestidad en el pitch (es un complemento municipal, no el modelo Básico).")
print(f"\nsupervisado_resultados.csv -> {SALIDA}")
