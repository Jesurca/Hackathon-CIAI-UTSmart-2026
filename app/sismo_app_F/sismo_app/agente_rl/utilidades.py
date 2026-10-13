"""
Reemplazos en NumPy de lo poco que se usaba de scikit-learn
(K-Means, PCA, silueta y métricas de clasificación), para que la app
funcione solo con streamlit + pandas + numpy + plotly.
"""
import numpy as np


def kmeans(X, k, n_init=10, iters=100, semilla=0):
    """Devuelve (etiquetas, inercia)."""
    X = np.asarray(X, float); rng = np.random.default_rng(semilla)
    mejor = (np.inf, None)
    for _ in range(n_init):
        C = X[rng.choice(len(X), k, replace=False)]
        for _ in range(iters):
            lab = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
            nC = np.array([X[lab == c].mean(0) if (lab == c).any() else C[c] for c in range(k)])
            if np.allclose(nC, C):
                break
            C = nC
        inercia = float(((X - C[lab]) ** 2).sum())
        if inercia < mejor[0]:
            mejor = (inercia, lab)
    return mejor[1], mejor[0]


def pca(X, n=2):
    """Devuelve (componentes, varianza explicada)."""
    Xc = np.asarray(X, float); Xc = Xc - Xc.mean(0)
    _U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:n].T, (S ** 2 / max((S ** 2).sum(), 1e-12))[:n]


def silueta(X, etiquetas):
    X = np.asarray(X, float); lab = np.asarray(etiquetas)
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    s = []
    for i in range(len(X)):
        mismo = lab == lab[i]; mismo[i] = False
        if not mismo.any():
            s.append(0.0); continue
        a = D[i, mismo].mean()
        b = min(D[i, lab == o].mean() for o in set(lab) if o != lab[i])
        s.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(s))


def prf(y, yp, clases=(0, 1, 2)):
    """Precisión, recall, F1 y soporte por clase."""
    y, yp = np.asarray(y), np.asarray(yp)
    p, r, f, sop = [], [], [], []
    for c in clases:
        tp = ((yp == c) & (y == c)).sum(); fp = ((yp == c) & (y != c)).sum(); fn = ((yp != c) & (y == c)).sum()
        pi = tp / (tp + fp) if tp + fp else 0.0; ri = tp / (tp + fn) if tp + fn else 0.0
        p.append(pi); r.append(ri); f.append(2 * pi * ri / (pi + ri) if pi + ri else 0.0); sop.append(int((y == c).sum()))
    return np.array(p), np.array(r), np.array(f), np.array(sop)


def accuracy(y, yp):
    return float((np.asarray(y) == np.asarray(yp)).mean())


def f1_macro(y, yp):
    return float(prf(y, yp)[2].mean())
