"""
API de alto nivel del agente, pensada para la app de Streamlit (y para scripts).

    entrenar(D, cfg, callback)      -> agente, curva, info
    cargar_agente(bytes, D, cfg)    -> agente, info
    evaluar(agente, D, cfg)         -> métricas en PRUEBA, líneas base, IC 95 %, importancia
    priorizar(agente, D, cfg, t)    -> tabla de priorización con confianza y razones

Todo en NumPy: no requiere PyTorch ni GPU.

Formulación del problema (MDP), ver entorno.py:
  estado  = 25 variables por municipio + descuido, ya_elegido, alcanzable, presupuesto, avance
  acción  = elegir un municipio (hasta agotar el presupuesto del trimestre)
  recompensa = impacto real (t+1) / impacto del oráculo + equidad − costo
"""
import random
import time

import numpy as np
import pandas as pd

from .agente import AgenteEnsamble
from .entorno import N_DINAMICAS, EntornoPriorizacionSismica
from .metricas import ejecutar, politica_agente, politica_aleatoria, politica_variable
from .variables import FEATURES

CFG_AGENTE = {"cabezas_ensamble": 5, "capas_ocultas": [64, 32], "tasa_aprendizaje": 1e-3,
              "gamma": 0.9, "tam_lote": 32, "memoria_replay": 50000, "episodios": 300, "epsilon_inicial": 1.0,
              "epsilon_final": 0.05, "decaimiento_epsilon": 0.985, "actualizar_objetivo_cada": 200}
D_IN = len(FEATURES) + N_DINAMICAS


def fijar_semillas(s):
    random.seed(s); np.random.seed(s)


def presupuesto_sugerido(D, intervenciones=None):
    """Presupuesto ≈ costo mediano × nº de intervenciones (≈ 1 de cada 8 municipios, entre 3 y 12)."""
    n = len(D["municipios"])
    k = intervenciones or int(np.clip(round(n / 8), 3, 12))
    return float(np.round(np.median(D["costo"]) * k, 1))


def crear_entornos(D, cfg_entorno, semilla=0):
    kw = dict(X=D["X"], impacto=D["impacto"], costo=D["costo"], cfg_entorno=cfg_entorno, municipios=D["municipios"])
    return (EntornoPriorizacionSismica(periodos_validos=D["cortes"]["train"], semilla=semilla, **kw),
            EntornoPriorizacionSismica(periodos_validos=D["cortes"]["val"], episodio_completo=True, **kw),
            EntornoPriorizacionSismica(periodos_validos=D["cortes"]["test"], episodio_completo=True, **kw))


def _ajustar_presupuesto(D, cfg_entorno):
    """Garantiza que todo municipio sea alcanzable con el presupuesto (evita acciones imposibles)."""
    ce = dict(cfg_entorno)
    ce["presupuesto_por_periodo"] = max(ce["presupuesto_por_periodo"], float(D["costo"].max()) + 1e-3)
    return ce


# ------------------------------------------------------------------ entrenamiento
def entrenar(D, cfg_entorno, cfg_agente=None, semilla=42, callback=None, huella=None):
    cfg_a = {**CFG_AGENTE, **(cfg_agente or {})}
    ce = _ajustar_presupuesto(D, cfg_entorno)
    fijar_semillas(semilla)
    env, env_val, _ = crear_entornos(D, ce, semilla)
    agente = AgenteEnsamble(D_IN, cfg_a, semilla)
    curva, mejor, mejor_pesos, mejor_ep = [], -np.inf, None, 0
    cada = max(5, cfg_a["episodios"] // 40)
    t0 = time.time()
    for ep in range(1, cfg_a["episodios"] + 1):
        obs, info = env.reset(); agente.nuevo_episodio()
        hecho, R, perdidas = False, 0.0, []
        while not hecho:
            a = agente.actuar(obs, info["mascara"])
            obs2, r, hecho, _, info = env.step(a)
            agente.mem.agregar(obs, a, r, obs2, info["mascara"], float(hecho))
            l = agente.aprender()
            if l is not None:
                perdidas.append(l)
            obs, R = obs2, R + r
        fila = {"episodio": ep, "retorno": R, "epsilon": agente.eps,
                "perdida": float(np.mean(perdidas)) if perdidas else np.nan, "captura_val": np.nan}
        if ep % cada == 0 or ep == cfg_a["episodios"]:
            v = ejecutar(env_val, politica_agente(agente))["captura_riesgo"]
            fila["captura_val"] = v
            if v > mejor:
                mejor, mejor_ep, mejor_pesos = v, ep, agente.pesos()
        curva.append(fila)
        if callback:
            callback(ep, cfg_a["episodios"], fila, mejor, time.time() - t0)
    if mejor_pesos is not None:
        agente.poner_pesos(mejor_pesos)
    info = {"origen": "entrenado en la app", "episodio_mejor": mejor_ep, "captura_val": float(mejor),
            "dispositivo": AgenteEnsamble.dev, "segundos": round(time.time() - t0, 1), "cfg_entorno": ce,
            "huella_datos": huella, "curva": pd.DataFrame(curva)}
    return agente, info


def exportar(agente, info, D=None):
    """Bytes del archivo .npz del agente (pesos + información del entrenamiento)."""
    curva = info.get("curva")
    meta = {k: info.get(k) for k in ("episodio_mejor", "captura_val", "cfg_entorno", "huella_datos")}
    meta.update({"features": FEATURES, "curva": curva.to_dict("list") if isinstance(curva, pd.DataFrame) else None})
    return agente.exportar(meta)


def cargar_agente(contenido, cfg_entorno=None, huella=None):
    """Carga un agente .npz (bytes o ruta) y verifica que use las mismas 25 variables."""
    try:
        agente, meta = AgenteEnsamble.importar(contenido)
    except Exception as e:
        raise ValueError(f"El archivo no es un agente .npz válido de esta app ({e}).")
    if meta.get("features") and list(meta["features"]) != FEATURES:
        raise ValueError("El agente fue entrenado con otro conjunto de variables. Reentrénelo con los datos cargados.")
    if meta.get("d_in") != D_IN:
        raise ValueError("El agente no corresponde al formato de esta app. Reentrénelo con los datos cargados.")
    curva = pd.DataFrame(meta["curva"]) if meta.get("curva") else None
    info = {"origen": "archivo .npz", "episodio_mejor": meta.get("episodio_mejor"), "captura_val": meta.get("captura_val"),
            "dispositivo": AgenteEnsamble.dev, "cfg_entorno": meta.get("cfg_entorno") or cfg_entorno,
            "huella_datos": meta.get("huella_datos"), "curva": curva,
            "mismos_datos": (huella is not None and meta.get("huella_datos") == huella)}
    return agente, info


# ------------------------------------------------------------------ evaluación
def evaluar(agente, D, cfg_entorno, semillas_azar=20):
    ce = _ajustar_presupuesto(D, cfg_entorno)
    fijar_semillas(0)
    _, env_val, env_test = crear_entornos(D, ce)
    pols = {"Agente RL (ensamble)": politica_agente(agente),
            "Más energía sísmica (año)": politica_variable("energia_log_4p"),
            "Más exposición": politica_variable("indice_exposicion"),
            "Clase del modelo supervisado": politica_variable("nivel_actividad")}
    res = {n: ejecutar(env_test, p) for n, p in pols.items()}
    azar = [ejecutar(env_test, politica_aleatoria(np.random.default_rng(s))) for s in range(semillas_azar)]
    res["Aleatoria"] = {k: float(np.mean([a[k] for a in azar])) for k in ["captura_riesgo", "precision_top", "cobertura", "max_descuido"]}
    res["Aleatoria"].update(captura_std=float(np.std([a["captura_riesgo"] for a in azar])),
                            serie_captura=list(np.mean([a["serie_captura"] for a in azar], 0)),
                            serie_periodos=azar[0]["serie_periodos"])
    tabla = pd.DataFrame({n: {m: r[m] for m in ["captura_riesgo", "captura_std", "precision_top", "cobertura", "max_descuido"]}
                          for n, r in res.items()}).T.reset_index(names="politica")
    series = pd.concat([pd.DataFrame({"politica": n, "periodo": [D["periodos"][t] for t in r["serie_periodos"]],
                                      "captura": r["serie_captura"]}) for n, r in res.items()])
    s = np.array(res["Agente RL (ensamble)"]["serie_captura"])
    rng = np.random.default_rng(0)
    boot = [rng.choice(s, len(s)).mean() for _ in range(2000)]
    val = ejecutar(env_val, politica_agente(agente))
    return {"tabla": tabla, "series": series, "ic95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            "captura_val": val["captura_riesgo"], "importancia": importancia(agente, D, ce)}


def _grad_x_entrada(agente, obs, i):
    return agente.grad_x_entrada(obs, i, len(FEATURES))


def importancia(agente, D, cfg_entorno, top=5):
    imp = np.zeros(len(FEATURES))
    for t in D["cortes"]["test"]:
        env = EntornoPriorizacionSismica(D["X"], D["impacto"], D["costo"], [t], {**cfg_entorno, "periodos_por_episodio": 1},
                                         D["municipios"], episodio_completo=True)
        obs, _ = env.reset()
        q, _ = agente.valores(obs)
        for i in np.argsort(-q)[:top]:
            imp += np.abs(_grad_x_entrada(agente, obs, i))
    return pd.Series(imp / max(imp.sum(), 1e-9), index=FEATURES).sort_values(ascending=False)


def priorizar(agente, D, cfg_entorno, t, presupuesto=None, X=None, explicar=True):
    """Decisión del agente en el periodo t. X permite simular escenarios '¿qué pasaría si?'."""
    ce = {**_ajustar_presupuesto(D, cfg_entorno), "periodos_por_episodio": 1}
    if presupuesto:
        ce["presupuesto_por_periodo"] = max(float(presupuesto), float(D["costo"][t].min()) + 1e-3)
    Xu = D["X"] if X is None else X
    env = EntornoPriorizacionSismica(Xu, D["impacto"], D["costo"], [t], ce, D["municipios"], episodio_completo=True)
    obs, info = env.reset()
    q_med, q_std = agente.valores(obs)
    orden, hecho = [], False
    while not hecho:
        a = agente.actuar(obs, info["mascara"], explorar=False)
        orden.append(a)
        obs, _, hecho, _, info = env.step(a)
    obs0, _ = env.reset()
    filas = []
    for i, m in enumerate(D["municipios"]):
        g = _grad_x_entrada(agente, obs0, i) if explicar else np.zeros(len(FEATURES))
        top = np.argsort(-np.abs(g))[:3]
        filas.append({"municipio": m, "valor_Q": float(q_med[i]), "incertidumbre": float(q_std[i]),
                      "costo": float(D["costo"][t, i]), "priorizado": i in orden,
                      "orden": orden.index(i) + 1 if i in orden else None,
                      "razones": " · ".join(f"{FEATURES[j]} {'↑' if g[j] > 0 else '↓'}" for j in top) if explicar else "",
                      "impacto_real_siguiente": float(D["impacto"][t, i])})
    tab = pd.DataFrame(filas).sort_values("valor_Q", ascending=False)
    u1, u2 = np.quantile(q_std, [0.33, 0.66])
    tab["confianza"] = np.where(tab["incertidumbre"] <= u1, "Alta", np.where(tab["incertidumbre"] <= u2, "Media", "Baja"))
    tab.insert(0, "ranking", range(1, len(tab) + 1))
    return tab, ce["presupuesto_por_periodo"]
