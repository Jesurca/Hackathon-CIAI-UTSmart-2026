"""
PASO 3 · EL ENTORNO DE APRENDIZAJE POR REFUERZO  ("Sala de Crisis Santander")
=============================================================================
Formalización como Proceso de Decisión de Markov (MDP) — lo que pide la rúbrica:

  AGENTE   : un "coordinador departamental de gestión del riesgo" artificial.
  ESTADO   : para cada municipio, sus 25 variables del periodo t (normalizadas)
             + 3 variables dinámicas que el propio agente modifica:
                 ya_elegido        (1 si ya lo priorizó en este trimestre)
                 descuido          (trimestres desde la última vez que lo atendió)
                 alcanzable        (1 si su costo cabe en el presupuesto que queda)
             + 2 variables globales: presupuesto restante y avance del episodio.
  ACCIÓN   : elegir UN municipio para priorizar (inspección, simulacro, refuerzo).
             Se repite hasta agotar el presupuesto del trimestre; luego el tiempo
             avanza al siguiente trimestre. Acciones inválidas se enmascaran.
  RECOMPENSA (por cada municipio elegido):
        r =  impacto_real(t+1) / impacto_oráculo(t+1)      <- ¿acertó dónde dolió?
           + peso_equidad * descuido normalizado           <- no abandonar territorios
           - peso_costo   * costo / presupuesto            <- usar bien el dinero
     El "impacto" = peligro sísmico (energía, superficiales pesan más)
                    x exposición (población) x vulnerabilidad (NBI, viviendas).
     El agente NUNCA ve el futuro: solo recibe la recompensa después de actuar.
  EPISODIO : `periodos_por_episodio` trimestres consecutivos (por defecto un año).

Es secuencial de verdad: lo que elige hoy cambia el "descuido" de mañana,
por eso no es un simple ranking y justifica usar RL.
"""
import numpy as np


class _Box:  # mismo concepto que gymnasium.spaces.Box (sin instalar gymnasium)
    def __init__(self, shape):
        self.shape = shape


class _Discrete:  # mismo concepto que gymnasium.spaces.Discrete
    def __init__(self, n):
        self.n = n


N_DINAMICAS = 5  # ya_elegido, descuido, alcanzable, presupuesto_restante, avance


class EntornoPriorizacionSismica:
    """Interfaz estilo Gymnasium: reset() y step(acción) -> (obs, recompensa, terminado, truncado, info)."""
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, X, impacto, costo, periodos_validos, cfg_entorno, municipios,
                 episodio_completo=False, semilla=0):
        """
        X        : array (T, N, F) variables normalizadas por periodo y municipio
        impacto  : array (T, N) impacto REAL del periodo siguiente (solo recompensa)
        costo    : array (T, N) costo de intervenir cada municipio
        periodos_validos : índices t que este entorno puede usar (train / val / test)
        episodio_completo : True -> recorre TODOS los periodos válidos (evaluación)
        """
        self.X, self.imp, self.costo = X, impacto, costo
        self.T, self.N, self.F = X.shape
        self.validos = np.array(periodos_validos)
        self.cfg = cfg_entorno
        self.municipios = municipios
        self.completo = episodio_completo
        self.rng = np.random.default_rng(semilla)
        self.observation_space = _Box((self.N, self.F + N_DINAMICAS))
        self.action_space = _Discrete(self.N)

    # ------------------------------------------------------------------ utilidades
    def _oraculo(self, t):
        """Mejor valor posible con información perfecta (cota superior, voraz por impacto)."""
        orden = np.argsort(-self.imp[t]); restante, total = self.cfg["presupuesto_por_periodo"], 0.0
        for i in orden:
            if self.costo[t, i] <= restante:
                restante -= self.costo[t, i]; total += self.imp[t, i]
        return max(total, 1e-6)

    def mascara(self):
        return (~self.elegidos) & (self.costo[self.t] <= self.restante)

    def _obs(self):
        P = self.cfg["presupuesto_por_periodo"]
        din = np.stack([
            self.elegidos.astype(np.float32),
            np.minimum(self.descuido, 8) / 8.0,
            self.mascara().astype(np.float32),
            np.full(self.N, self.restante / P),
            np.full(self.N, self.paso_periodo / max(1, len(self.secuencia) - 1)),
        ], axis=1)
        return np.concatenate([self.X[self.t], din], axis=1).astype(np.float32)

    # ------------------------------------------------------------------ API gymnasium
    def reset(self, seed=None, options=None):
        L = self.cfg["periodos_por_episodio"]
        if self.completo:
            self.secuencia = list(self.validos)
        else:
            inicio = self.rng.integers(0, max(1, len(self.validos) - L + 1))
            self.secuencia = list(self.validos[inicio:inicio + L])
        self.paso_periodo = 0
        self.t = self.secuencia[0]
        self.descuido = np.zeros(self.N) if self.completo else self.rng.integers(0, 5, self.N).astype(float)
        self._nuevo_periodo()
        self.historial = []
        return self._obs(), {"mascara": self.mascara()}

    def _nuevo_periodo(self):
        self.elegidos = np.zeros(self.N, dtype=bool)
        self.restante = self.cfg["presupuesto_por_periodo"]
        self.z = self._oraculo(self.t)

    def step(self, a):
        assert self.mascara()[a], "acción inválida (ya elegido o fuera de presupuesto)"
        P = self.cfg["presupuesto_por_periodo"]
        r_acierto = self.imp[self.t, a] / self.z
        r_equidad = self.cfg["peso_equidad"] * min(self.descuido[a], 8) / 8.0
        r_costo = self.cfg["peso_costo"] * self.costo[self.t, a] / P
        r = r_acierto + r_equidad - r_costo
        self.elegidos[a] = True
        self.restante -= self.costo[self.t, a]
        self.historial.append((int(self.t), int(a), float(r_acierto)))

        terminado = False
        if not self.mascara().any():  # se acabó el presupuesto -> avanza el tiempo
            self.descuido = np.where(self.elegidos, 0, self.descuido + 1)
            self.paso_periodo += 1
            if self.paso_periodo >= len(self.secuencia):
                terminado = True
            else:
                self.t = self.secuencia[self.paso_periodo]
                self._nuevo_periodo()
        info = {"mascara": self.mascara() if not terminado else np.zeros(self.N, bool),
                "r_acierto": r_acierto, "r_equidad": r_equidad, "r_costo": r_costo}
        return self._obs(), float(r), terminado, False, info

    def render(self):
        t, a, ra = self.historial[-1]
        return f"periodo {t}: priorizó {self.municipios[a]} (acierto {ra:.2f})"
