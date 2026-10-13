"""
EL AGENTE: ENSAMBLE DE REDES Q (Bootstrapped Double DQN) escrito en NumPy puro
=============================================================================
No necesita PyTorch ni GPU: la red es pequeña y entrena en segundos/minutos en CPU.

Idea clave (el toque diferenciador):
  En vez de UNA red, se entrenan K redes Q independientes (un "comité de expertos").
  Cada experiencia la ve solo una parte del comité (bootstrap). Entonces:
     * Q medio      -> decisión del comité
     * Q desviación -> INCERTIDUMBRE (semáforo de confianza)

Arquitectura "por municipio": la misma red puntúa a cada municipio usando
(sus variables + el promedio del departamento). Sirve para 35 u 87 municipios.

    entrada (2·d) -> Dense(64) ReLU -> Dense(32) ReLU -> Q

Aprendizaje: Double DQN (la red online elige, la red objetivo evalúa), pérdida
Huber, optimizador Adam y recorte de gradiente, todo implementado a mano.
"""
import io
import json
import random
from collections import deque

import numpy as np


# ------------------------------------------------------------------ una red Q (MLP)
class RedQ:
    def __init__(self, d_in, ocultas, rng):
        dims = [2 * d_in] + list(ocultas) + [1]
        self.W, self.b = [], []
        for i in range(len(dims) - 1):
            escala = np.sqrt(2.0 / dims[i]) if i < len(dims) - 2 else 0.01
            self.W.append(rng.normal(0, escala, (dims[i + 1], dims[i])))
            self.b.append(np.zeros(dims[i + 1]))

    @staticmethod
    def entrada(obs):                      # obs (B, N, d) -> (B, N, 2d)
        ctx = np.broadcast_to(obs.mean(1, keepdims=True), obs.shape)
        return np.concatenate([obs, ctx], -1)

    def forward(self, obs, guardar=False):
        x = self.entrada(obs)
        B, N, D = x.shape
        h = x.reshape(B * N, D)
        cache = [h]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W.T + b
            if i < len(self.W) - 1:
                h = np.maximum(h, 0)
            cache.append(h)
        q = h.reshape(B, N)
        return (q, cache) if guardar else q

    def backward(self, cache, dq):
        """dq (B, N) -> gradientes de W y b."""
        g = dq.reshape(-1, 1)
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            a_prev = cache[i]
            gW[i] = g.T @ a_prev
            gb[i] = g.sum(0)
            if i > 0:
                g = (g @ self.W[i]) * (cache[i] > 0)
        return gW, gb

    def copiar_de(self, otra):
        self.W = [w.copy() for w in otra.W]; self.b = [b.copy() for b in otra.b]


class Adam:
    def __init__(self, red, lr, b1=0.9, b2=0.999, eps=1e-8, wd=1e-4):
        self.lr, self.b1, self.b2, self.eps, self.wd, self.t = lr, b1, b2, eps, wd, 0
        self.m = [np.zeros_like(p) for p in red.W + red.b]
        self.v = [np.zeros_like(p) for p in red.W + red.b]

    def paso(self, red, gW, gb, clip=5.0):
        grads = gW + gb
        norma = np.sqrt(sum((g ** 2).sum() for g in grads))
        if norma > clip:
            grads = [g * clip / norma for g in grads]
        self.t += 1
        params = red.W + red.b
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t); vh = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * (mh / (np.sqrt(vh) + self.eps) + (self.wd * p if p.ndim == 2 else 0))


class Memoria:
    """Replay buffer: guarda experiencias para aprender de ellas muchas veces."""
    def __init__(self, cap, k, p_boot=0.8):
        self.buf, self.k, self.p = deque(maxlen=cap), k, p_boot

    def agregar(self, s, a, r, s2, m2, done):
        self.buf.append((s, a, r, s2, m2, done, (np.random.rand(self.k) < self.p).astype(np.float64)))

    def muestra(self, n):
        lote = random.sample(self.buf, n)
        s, a, r, s2, m2, d, b = map(np.array, zip(*lote))
        return s.astype(np.float64), a, r, s2.astype(np.float64), m2.astype(bool), d, b

    def __len__(self):
        return len(self.buf)


# ------------------------------------------------------------------ el comité
class AgenteEnsamble:
    dev = "CPU · NumPy"

    def __init__(self, d_in, cfg, semilla=0):
        self.cfg, self.k, self.d_in = cfg, cfg["cabezas_ensamble"], d_in
        rng = np.random.default_rng(semilla)
        self.redes = [RedQ(d_in, cfg["capas_ocultas"], rng) for _ in range(self.k)]
        self.objetivo = [RedQ(d_in, cfg["capas_ocultas"], rng) for _ in range(self.k)]
        for o, r in zip(self.objetivo, self.redes):
            o.copiar_de(r)
        self.opt = [Adam(r, cfg["tasa_aprendizaje"]) for r in self.redes]
        self.mem = Memoria(cfg["memoria_replay"], self.k)
        self.eps = cfg["epsilon_inicial"]
        self.pasos = 0
        self.cabeza_activa = 0

    # ---------------------------------------------------------------- decidir
    def _q_todas(self, obs):                     # obs (N, d) -> (K, N)
        x = np.asarray(obs, dtype=np.float64)[None]
        return np.stack([r.forward(x)[0] for r in self.redes])

    def valores(self, obs):
        """Q medio (decisión) y desviación entre expertos (incertidumbre) por municipio."""
        q = self._q_todas(obs)
        return q.mean(0), q.std(0, ddof=1)

    def actuar(self, obs, mascara, explorar=True):
        validos = np.flatnonzero(mascara)
        if explorar and random.random() < self.eps:
            return int(np.random.choice(validos))
        if explorar:   # exploración tipo Thompson: decide un experto distinto por episodio
            q = self.redes[self.cabeza_activa].forward(np.asarray(obs, dtype=np.float64)[None])[0]
        else:
            q = self._q_todas(obs).mean(0)
        q = q.copy(); q[~mascara] = -np.inf
        return int(np.argmax(q))

    def nuevo_episodio(self):
        self.cabeza_activa = random.randrange(self.k)
        self.eps = max(self.cfg["epsilon_final"], self.eps * self.cfg["decaimiento_epsilon"])

    def grad_x_entrada(self, obs, i, n_var, h=1e-3):
        """Explicación: gradiente × entrada de cada variable para el municipio i."""
        obs = np.asarray(obs, dtype=np.float64)
        g = np.zeros(n_var)
        for j in range(n_var):
            a, b = obs.copy(), obs.copy()
            a[i, j] += h; b[i, j] -= h
            g[j] = (self._q_todas(a)[:, i].mean() - self._q_todas(b)[:, i].mean()) / (2 * h)
        return g * obs[i, :n_var]

    # ---------------------------------------------------------------- aprender
    def aprender(self):
        n = self.cfg["tam_lote"]
        if len(self.mem) < n * 5:
            return None
        s, a, r, s2, m2, d, boot = self.mem.muestra(n)
        hay_sig = m2.any(1)
        perdidas = []
        for k, (red, obj, opt) in enumerate(zip(self.redes, self.objetivo, self.opt)):
            q, cache = red.forward(s, guardar=True)
            q2_on = red.forward(s2); q2_on[~m2] = -1e9
            a2 = q2_on.argmax(1)                                   # Double DQN: la online elige…
            q2 = obj.forward(s2)[np.arange(n), a2]                 # …la objetivo evalúa
            q2 = np.where(hay_sig, q2, 0.0)
            y = r + self.cfg["gamma"] * (1 - d) * q2
            delta = q[np.arange(n), a] - y
            w = boot[:, k]; wsum = max(w.sum(), 1.0)
            perdidas.append(float((np.where(np.abs(delta) < 1, 0.5 * delta ** 2, np.abs(delta) - 0.5) * w).sum() / wsum))
            dq = np.zeros_like(q)
            dq[np.arange(n), a] = np.clip(delta, -1, 1) * w / wsum   # derivada de Huber
            gW, gb = red.backward(cache, dq)
            opt.paso(red, gW, gb)
        self.pasos += 1
        if self.pasos % self.cfg["actualizar_objetivo_cada"] == 0:
            for o, rr in zip(self.objetivo, self.redes):
                o.copiar_de(rr)
        return float(np.mean(perdidas))

    # ---------------------------------------------------------------- pesos
    def pesos(self):
        return [([w.copy() for w in r.W], [b.copy() for b in r.b]) for r in self.redes]

    def poner_pesos(self, pesos):
        for r, o, (W, b) in zip(self.redes, self.objetivo, pesos):
            r.W = [w.copy() for w in W]; r.b = [x.copy() for x in b]
            o.copiar_de(r)

    def exportar(self, meta):
        """Bytes de un archivo .npz con los pesos y la información del entrenamiento."""
        arr = {}
        for k, r in enumerate(self.redes):
            for i, (W, b) in enumerate(zip(r.W, r.b)):
                arr[f"r{k}_W{i}"] = W.astype(np.float32); arr[f"r{k}_b{i}"] = b.astype(np.float32)
        buf = io.BytesIO()
        np.savez_compressed(buf, __meta__=np.array(json.dumps({**meta, "cfg": self.cfg, "d_in": self.d_in}, default=str)), **arr)
        return buf.getvalue()

    @classmethod
    def importar(cls, contenido):
        f = np.load(io.BytesIO(contenido) if isinstance(contenido, (bytes, bytearray)) else contenido, allow_pickle=False)
        meta = json.loads(str(f["__meta__"]))
        cfg = meta["cfg"]
        ag = cls(meta["d_in"], cfg)
        pesos = []
        for k in range(cfg["cabezas_ensamble"]):
            n = len(cfg["capas_ocultas"]) + 1
            pesos.append(([f[f"r{k}_W{i}"].astype(np.float64) for i in range(n)], [f[f"r{k}_b{i}"].astype(np.float64) for i in range(n)]))
        ag.poner_pesos(pesos)
        return ag, meta
