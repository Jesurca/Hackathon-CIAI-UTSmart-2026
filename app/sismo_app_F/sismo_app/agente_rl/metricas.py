"""
Evaluación del agente y de las LÍNEAS BASE (para demostrar que el RL aporta).

Políticas comparadas:
  * aleatoria        -> piso: elegir al azar dentro del presupuesto
  * mas_sismos       -> lo que haría una persona sin modelo: "donde más tembló"
  * mas_expuesto     -> "donde vive más gente vulnerable", ignora la sismicidad
  * agente_RL        -> nuestro ensamble Q
  * oráculo          -> techo teórico (conoce el futuro); define el 100 %

Métricas (por trimestre, promediadas en el periodo evaluado):
  * captura_riesgo   = impacto capturado / impacto del oráculo   (0-1, más es mejor)
  * precision_top    = % de municipios elegidos que el oráculo también elegía
  * cobertura        = % de municipios atendidos al menos una vez (equidad)
  * max_descuido     = el mayor nº de trimestres que un municipio quedó sin atención
"""
import numpy as np

from .variables import FEATURES


def politica_aleatoria(rng):
    return lambda env, obs, m: int(rng.choice(np.flatnonzero(m)))


def politica_variable(nombre):
    j = FEATURES.index(nombre)
    def pol(env, obs, m):
        v = obs[:, j].copy(); v[~m] = -np.inf
        return int(np.argmax(v))
    return pol


def politica_agente(agente):
    return lambda env, obs, m: agente.actuar(obs, m, explorar=False)


def ejecutar(env, politica):
    obs, info = env.reset()
    hecho, oraculo_sets = False, {}
    while not hecho:
        a = politica(env, obs, info["mascara"])
        obs, r, hecho, _, info = env.step(a)
    # métricas por periodo
    por_t = {}
    for t, a, ra in env.historial:
        por_t.setdefault(t, []).append((a, ra))
    captura, precision, atendidos, desc = [], [], set(), np.zeros(env.N)
    max_desc = 0
    for t, picks in por_t.items():
        captura.append(sum(ra for _, ra in picks))
        orden = np.argsort(-env.imp[t]); rest, ideal = env.cfg["presupuesto_por_periodo"], set()
        for i in orden:
            if env.costo[t, i] <= rest:
                rest -= env.costo[t, i]; ideal.add(int(i))
        elegidos = {a for a, _ in picks}
        precision.append(len(elegidos & ideal) / max(1, len(elegidos)))
        atendidos |= elegidos
        desc = np.where(np.isin(np.arange(env.N), list(elegidos)), 0, desc + 1)
        max_desc = max(max_desc, desc.max())
    return {"captura_riesgo": float(np.mean(captura)), "captura_std": float(np.std(captura)),
            "precision_top": float(np.mean(precision)), "cobertura": len(atendidos) / env.N,
            "max_descuido": int(max_desc), "serie_captura": captura,
            "serie_periodos": list(por_t.keys())}
