"""
FLUJO COMPLETO: archivos del equipo -> panel de 25 variables -> tensores del agente.
Lo usan tanto la app (app.py) como el generador de la demostración (generar_demo.py),
así el resultado es idéntico por ambos caminos (trazabilidad).
"""
import hashlib

import pandas as pd

from .preparar import a_tensores, sugerir_cortes
from .variables import construir_panel

CFG_BASE = {
    "periodo": None, "magnitud_minima": 1.5, "radio_asignacion_km": 30, "semilla": 42,
    "fin_entrenamiento": None, "fin_validacion": None,          # None -> se sugieren solos
    "entorno": {"presupuesto_por_periodo": None, "periodos_por_episodio": 4, "peso_equidad": 0.15, "peso_costo": 0.05},
}


def huella(*dfs, extra=""):
    """Identificador corto de los datos usados (para saber si un agente .npz corresponde a ellos)."""
    h = hashlib.sha1(extra.encode())
    for d in dfs:
        if d is not None:
            h.update(pd.util.hash_pandas_object(d.reset_index(drop=True), index=False).values.tobytes())
    return h.hexdigest()[:12]


def completar_cortes(cat, cfg):
    cfg = {**cfg, "entorno": dict(cfg["entorno"])}
    if not cfg.get("periodo"):   # automático: trimestres si hay ≥ 5 años de datos; si no, meses
        f = pd.to_datetime(cat["fecha"], errors="coerce").dropna()
        cfg["periodo"] = "Q" if (f.max() - f.min()).days >= 5 * 365 else "M"
    if not cfg.get("fin_entrenamiento") or not cfg.get("fin_validacion"):
        f = pd.to_datetime(cat["fecha"], errors="coerce").dropna()
        a, b = sugerir_cortes(f.min().to_period(cfg["periodo"]), f.max().to_period(cfg["periodo"]), cfg["periodo"])
        cfg["fin_entrenamiento"] = cfg.get("fin_entrenamiento") or str(a)
        cfg["fin_validacion"] = cfg.get("fin_validacion") or str(b)
    return cfg


def procesar(cat, ofi, cfg, sup=None, clu=None, sup_modelo=None, clu_modelo=None):
    cfg = completar_cortes(cat, cfg)
    panel, rep, municipios, eventos = construir_panel(cat, ofi, cfg, sup, clu, sup_modelo, clu_modelo)
    D = a_tensores(panel, cfg)
    if not cfg["entorno"].get("presupuesto_por_periodo"):
        from .servicio import presupuesto_sugerido
        cfg["entorno"]["presupuesto_por_periodo"] = presupuesto_sugerido(D)
    h = huella(cat, ofi, sup, clu, extra=f"{cfg['fin_entrenamiento']}|{cfg['fin_validacion']}|{cfg['periodo']}")
    return {"panel": panel, "reporte": rep, "municipios": municipios, "eventos": eventos, "D": D, "cfg": cfg, "huella": h}
