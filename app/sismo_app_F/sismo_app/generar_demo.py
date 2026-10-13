"""
GENERA LA DEMOSTRACIÓN COMPLETA (carpeta demo/) con EXACTAMENTE el formato que
deben entregar los compañeros. Sirve de plantilla y para que la app funcione desde ya.

    python generar_demo.py            (tarda unos minutos: entrena el agente)

Archivos que produce:
  demo/catalogo_sgc_demo.csv               catálogo de sismos (formato SGC)
  demo/fuente_oficial_demo.csv             tabla municipal (tipo DANE)
  demo/supervisado_resultados_demo.csv     municipio, periodo, nivel_real, nivel_predicho, prob_*, conjunto
  demo/supervisado_modelo_demo.pkl         RandomForest entrenado (joblib)
  demo/clusters_resultados_demo.csv        municipio, cluster, nombre_cluster, PC1, PC2
  demo/clusters_modelo_demo.pkl            Pipeline(StandardScaler + KMeans) (joblib)
  demo/agente_demo.npz                     agente RL entrenado (NumPy)
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from agente_rl import servicio
from agente_rl.columnas import CLASES, aplicar_mapeo, detectar
from agente_rl.flujo import CFG_BASE, procesar
from agente_rl.simulado import generar

DEMO = Path(__file__).parent / "demo"
VARS_SUP = ["mag_max", "mag_media", "energia_log", "prof_media", "pct_superficiales", "n_m4_4p",
            "b_value", "dist_nido_km", "tendencia", "periodos_desde_m4"]
VARS_CLU = ["energia_log_4p", "pct_superficiales", "dist_nido_km", "nbi", "densidad_pob", "viviendas_precarias"]


def leer(nombre, tipo):
    df = pd.read_csv(DEMO / nombre)
    return aplicar_mapeo(df, detectar(df.columns, tipo))


def main(episodios=300):
    DEMO.mkdir(exist_ok=True)
    cat, ofi = generar(42)
    cat.to_csv(DEMO / "catalogo_sgc_demo.csv", index=False)
    ofi.to_csv(DEMO / "fuente_oficial_demo.csv", index=False)
    cat, ofi = leer("catalogo_sgc_demo.csv", "catalogo"), leer("fuente_oficial_demo.csv", "oficial")

    base = procesar(cat, ofi, CFG_BASE)              # panel con sustitutos para entrenar los modelos demo
    p, cfg = base["panel"], base["cfg"]
    fin_tr, fin_va = pd.Period(cfg["fin_entrenamiento"], "Q"), pd.Period(cfg["fin_validacion"], "Q")
    conj = np.where(p["periodo"] <= fin_tr, "train", np.where(p["periodo"] <= fin_va, "val", "test"))

    # ---------- Reto Básico (supervisado): nivel de actividad del trimestre ----------
    q1, q2 = p.loc[conj == "train", "n_sismos"].quantile([0.5, 0.85])
    y = np.select([p["n_sismos"] <= q1, p["n_sismos"] <= q2], [0, 1], 2)
    rf = RandomForestClassifier(300, max_depth=8, random_state=42, class_weight="balanced")
    rf.fit(p.loc[conj == "train", VARS_SUP], y[conj == "train"])
    proba = rf.predict_proba(p[VARS_SUP])
    sup = pd.DataFrame({"municipio": p["municipio"], "periodo": p["periodo"].astype(str),
                        "nivel_real": [CLASES[i] for i in y], "nivel_predicho": [CLASES[i] for i in proba.argmax(1)],
                        "prob_bajo": proba[:, 0].round(3), "prob_medio": proba[:, 1].round(3),
                        "prob_alto": proba[:, 2].round(3), "conjunto": conj})
    sup.to_csv(DEMO / "supervisado_resultados_demo.csv", index=False)
    joblib.dump(rf, DEMO / "supervisado_modelo_demo.pkl", compress=3)

    # ---------- Reto Medio (no supervisado): grupos territoriales ----------
    prom = p[conj == "train"].groupby("municipio")[VARS_CLU].mean()
    km = make_pipeline(StandardScaler(), KMeans(4, n_init=20, random_state=42)).fit(prom)
    g = km.predict(prom)
    z = StandardScaler().fit_transform(prom)
    pcs = PCA(2, random_state=42).fit_transform(z)
    perfil = pd.DataFrame(z, index=prom.index, columns=VARS_CLU).groupby(g).mean()
    nombres = {}
    for k, f in perfil.iterrows():
        act = "Alta actividad" if f["energia_log_4p"] > 0.3 else ("Baja actividad" if f["energia_log_4p"] < -0.3 else "Actividad media")
        vul = "alta vulnerabilidad" if f["nbi"] > 0.2 else "menor vulnerabilidad"
        nombres[k] = f"{act}, {vul}"
    clu = pd.DataFrame({"municipio": prom.index, "cluster": g, "nombre_cluster": [nombres[k] for k in g],
                        "PC1": pcs[:, 0].round(3), "PC2": pcs[:, 1].round(3)})
    clu.to_csv(DEMO / "clusters_resultados_demo.csv", index=False)
    joblib.dump(km, DEMO / "clusters_modelo_demo.pkl")

    # ---------- Reto Avanzado: agente RL con todo integrado ----------
    sup_l, clu_l = leer("supervisado_resultados_demo.csv", "supervisado"), leer("clusters_resultados_demo.csv", "clusters")
    R = procesar(cat, ofi, CFG_BASE, sup_l, clu_l)
    print(f"[panel] {R['panel'].shape} · cortes {R['cfg']['fin_entrenamiento']} / {R['cfg']['fin_validacion']} · "
          f"presupuesto {R['cfg']['entorno']['presupuesto_por_periodo']}")

    def cb(ep, tot, fila, mejor, seg):
        if ep % 25 == 0:
            print(f"  ep {ep}/{tot} · captura val {fila['captura_val']:.3f} · mejor {mejor:.3f} · {seg:.0f}s")
    agente, info = servicio.entrenar(R["D"], R["cfg"]["entorno"], {"episodios": episodios}, callback=cb, huella=R["huella"])
    (DEMO / "agente_demo.npz").write_bytes(servicio.exportar(agente, info))
    ev = servicio.evaluar(agente, R["D"], R["cfg"]["entorno"])
    print(ev["tabla"].round(3).to_string(index=False))
    print("Demo lista en", DEMO)


if __name__ == "__main__":
    main()
