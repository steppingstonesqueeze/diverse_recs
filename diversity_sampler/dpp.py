from __future__ import annotations
import numpy as np

def k_dpp_sample(L: np.ndarray, k: int, eps: float=1e-12, seed: int=None):
    """Sample a size-k subset from a DPP specified by PSD kernel L (L-ensemble, fixed-size k-DPP).
    Uses eigen-decomposition and the standard k-DPP algorithm (Kulesza & Taskar).
    For large n, prefer running this on a reduced candidate pool.
    """
    n = L.shape[0]
    if k <= 0:
        return []
    if k >= n:
        return list(range(n))
    # Eigendecomposition
    vals, vecs = np.linalg.eigh(L)
    vals = np.maximum(vals, 0.0)  # clip negatives from numerical error
    # Select exactly k eigenvalues using elementary symmetric polynomials
    E = np.zeros((k+1, len(vals)+1), dtype=float)
    E[0, :] = 1.0
    for i, l in enumerate(vals, start=1):
        E[1:min(i,k)+1, i] = E[1:min(i,k)+1, i-1] + l * E[0:min(i-1,k), i-1]
        E[0, i] = 1.0
        for j in range(2, min(i, k)+1):
            E[j, i] = E[j, i-1] + l * E[j-1, i-1]
    # Select a subset of eigenvectors
    rng = np.random.default_rng(seed)
    S = []
    i = len(vals)
    j = k
    while j > 0:
        if i == 0:
            break
        i -= 1
        if E[j, i] == 0:
            continue
        prob = vals[i] * E[j-1, i] / E[j, i]
        if rng.random() < prob:
            S.append(i)
            j -= 1
    V = vecs[:, S]  # selected eigenvectors
    # Sample items sequentially
    Y = []
    for _ in range(k):
        # compute probabilities proportional to row norms of V^2
        P = np.sum(V**2, axis=1)
        if P.max() < eps:
            break
        i = int(rng.choice(len(P), p=P/np.sum(P)))
        Y.append(i)
        # orthogonalize V against e_i
        if V.ndim == 1 or V.shape[1] == 1:
            break
        vi = V[i, :].copy()
        vi_norm2 = np.dot(vi, vi)
        if vi_norm2 <= eps:
            break
        V = V - np.outer(V[:, :] @ vi, vi) / vi_norm2
        # drop the component aligned with vi (reduce dimensionality)
        # Re-orthogonalize via QR for stability
        Q, R = np.linalg.qr(V)
        V = Q[:, :max(0, V.shape[1]-1)]
        if V.size == 0:
            break
    # Deduplicate and trim
    return list(dict.fromkeys(Y))[:k]
