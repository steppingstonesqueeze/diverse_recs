from __future__ import annotations
import numpy as np

def diversity_metrics(X: np.ndarray, idxs):
    idxs = list(idxs)
    if len(idxs) == 0:
        return {"k": 0}
    Xs = X[idxs]
    Xn = Xs / np.maximum(np.linalg.norm(Xs, axis=1, keepdims=True), 1e-12)
    S = Xn @ Xn.T
    n = len(idxs)
    iu = np.triu_indices(n, k=1)
    mean_pairwise_cos = float(S[iu].mean()) if iu[0].size>0 else 1.0
    np.fill_diagonal(S, -np.inf)
    nn_sim = S.max(axis=1)
    mean_nn_cos_dist = float((1.0 - nn_sim).mean())
    return {
        "k": n,
        "mean_pairwise_cosine": mean_pairwise_cos,
        "mean_nn_cosine_distance": mean_nn_cos_dist,
    }
