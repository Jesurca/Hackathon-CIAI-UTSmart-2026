"""
Del panel (municipio × periodo) a tensores (T, N, F) para el entorno, con cortes
temporales de VALIDACIÓN (walk-forward):

  |-------- ENTRENAMIENTO --------|-- VALIDACIÓN --|-- PRUEBA --|
  El agente aprende con el pasado, se elige con validación y se juzga en un futuro
  que nunca vio. La normalización se calcula SOLO con entrenamiento.
"""
import numpy as np
import pandas as pd

from .variables import FEATURES


def sugerir_cortes(periodo_inicio, periodo_fin, freq="Q", frac=(0.75, 0.87)):
    """Propone fechas de corte (≈75 % / 12 % / 13 %) según el rango de los datos."""
    ps = pd.period_range(periodo_inicio, periodo_fin, freq=freq)
    i1 = max(1, int(len(ps) * frac[0]) - 1)
    i2 = max(i1 + 1, int(len(ps) * frac[1]) - 1)
    i2 = min(i2, len(ps) - 3)
    return ps[i1].end_time.date(), ps[i2].end_time.date()


def a_tensores(panel, cfg):
    per = cfg["periodo"]
    panel = panel.copy()
    municipios = sorted(panel["municipio"].unique())
    periodos = sorted(panel["periodo"].unique())
    fin_tr = pd.Period(cfg["fin_entrenamiento"], per)
    fin_va = pd.Period(cfg["fin_validacion"], per)

    tr = panel[panel["periodo"] <= fin_tr]
    if tr.empty:
        raise ValueError("No hay periodos de entrenamiento: revise la fecha de fin de entrenamiento.")
    media, desv = tr[FEATURES].mean(), tr[FEATURES].std().replace(0, 1).fillna(1)

    pv = panel.set_index(["periodo", "municipio"]).sort_index()
    T, N = len(periodos), len(municipios)
    X = ((pv[FEATURES] - media) / desv).to_numpy(np.float32).reshape(T, N, len(FEATURES))
    imp = pv["impacto_siguiente"].to_numpy(np.float32).reshape(T, N)
    costo = pv["costo_intervencion"].to_numpy(np.float32).reshape(T, N)

    idx = np.arange(T); p = np.array(periodos)
    cortes = {"train": idx[p <= fin_tr], "val": idx[(p > fin_tr) & (p <= fin_va)], "test": idx[p > fin_va]}
    for k in ("val", "test"):
        if len(cortes[k]) == 0:
            raise ValueError(f"El tramo de {'validación' if k == 'val' else 'prueba'} quedó vacío: ajuste las fechas de corte.")
    return dict(X=np.nan_to_num(X), impacto=imp, costo=costo, municipios=municipios,
                periodos=[str(x) for x in periodos], cortes=cortes, panel=pv,
                normalizacion={"media": media.to_dict(), "desv": desv.to_dict()})
