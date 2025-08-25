from __future__ import annotations
import numpy as np

def _select_k_eigenvectors(lmbda: np.ndarray, k: int, rng: np.random.Generator):
    n = len(lmbda)
    E = np.zeros((k + 1, n), dtype=float)
    E[0, :] = 1.0
    for i in range(n):
        li = lmbda[i]
        jmax = min(i + 1, k)
        for j in range(jmax, 0, -1):
            prev = E[j, i - 1] if i > 0 else 0.0
            base = E[j - 1, i - 1] if i > 0 else (1.0 if j == 1 else 0.0)
            E[j, i] = prev + li * base

    S = []
    j = k
    for i in range(n - 1, -1, -1):
        if j == 0:
            break
        denom = E[j, i]
        if denom <= 0:
            continue
        base = E[j - 1, i - 1] if i > 0 else (1.0 if j == 1 else 0.0)
        prob = (lmbda[i] * base) / denom
        if np.random.default_rng().random() < prob:
            S.append(i)
            j -= 1
    S.reverse()
    return S

def k_dpp_sample(L: np.ndarray, k: int, eps: float = 1e-12, seed: int | None = None):
    n = L.shape[0]
    if k <= 0:
        return []
    if k >= n:
        return list(range(n))

    vals, vecs = np.linalg.eigh(L)
    vals = np.maximum(vals, 0.0)
    rng = np.random.default_rng(seed)

    S_idx = _select_k_eigenvectors(vals, k, rng)
    if len(S_idx) < k:
        S_idx = list(np.argsort(vals)[-k:])

    V = vecs[:, S_idx]  # (n x r) with r == k

    Y = []
    for _ in range(k):
        P_rows = np.sum(V * V, axis=1)
        sP = P_rows.sum()
        if sP <= eps:
            break
        i = int(rng.choice(n, p=P_rows / sP))
        Y.append(i)

        col_weights = V[i, :] ** 2
        sC = col_weights.sum()
        if sC <= eps:
            break
        j = int(rng.choice(V.shape[1], p=col_weights / sC))

        denom = V[i, j]
        if abs(denom) <= eps:
            break
        v = V[:, j] / denom  # shape (n,)

        # V <- V - v * V[i, :]
        V = V - np.outer(v, V[i, :])

        if V.shape[1] <= 1:
            break
        V = np.delete(V, j, axis=1)
        Q, _ = np.linalg.qr(V)
        V = Q

    return list(dict.fromkeys(Y))[:k]
