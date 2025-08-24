from __future__ import annotations
import numpy as np

def l2_normalize(X: np.ndarray, eps: float=1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return X / n

def cosine_similarity(X: np.ndarray) -> np.ndarray:
    Xn = l2_normalize(X)
    return Xn @ Xn.T

def rbf_kernel(X: np.ndarray, gamma: float=None) -> np.ndarray:
    # gamma default = 1 / median(pairwise squared distance)
    n = X.shape[0]
    norms = np.einsum('ij,ij->i', X, X)
    D2 = norms[:, None] + norms[None, :] - 2*(X @ X.T)
    if gamma is None:
        # median of upper triangle distances (avoid zeros)
        iu = np.triu_indices(n, k=1)
        med = np.median(D2[iu]) if iu[0].size > 0 else 1.0
        if med <= 0:
            med = 1.0
        gamma = 1.0 / med
    K = np.exp(-gamma * np.maximum(D2, 0.0))
    return K

def build_kernel(X: np.ndarray, quality: np.ndarray=None, kind: str="cosine", alpha: float=1.0, jitter: float=1e-6) -> np.ndarray:
    if kind == "cosine":
        S = cosine_similarity(X)
    elif kind == "rbf":
        S = rbf_kernel(X, gamma=None)
    else:
        raise ValueError("Unknown kernel kind")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0,1]")
    S = alpha * S + (1.0 - alpha) * np.eye(S.shape[0])
    if quality is not None:
        q = np.array(quality).astype(float).ravel()
        q = np.maximum(q, 0.0)
        D = np.diag(q)
        L = D @ S @ D
    else:
        L = S
    if jitter and jitter > 0:
        L = L + jitter * np.eye(L.shape[0])
    return L
