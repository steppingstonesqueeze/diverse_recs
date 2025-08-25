from __future__ import annotations
import numpy as np
from enum import Enum
from typing import List, Optional

class SelectionObjective(Enum):
    FARTHEST_FIRST = "farthest_first"           # max'96min spread
    FACILITY_LOCATION = "facility_location"     # coverage
    SUM_DIVERSITY = "sum_diversity"             # minimize pairwise inner products

def _l2_normalize(X: np.ndarray, eps: float=1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return X / n

def farthest_first(X: np.ndarray, k: int, metric: str="euclidean", seed: Optional[int]=None) -> List[int]:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n == 0 or k <= 0: return []
    if k >= n: return list(range(n))
    if metric == "cosine":
        Xn = _l2_normalize(X)
        d = np.full(n, np.inf)
        s = int(rng.integers(0, n))
        sel = [s]
        d = np.minimum(d, 1.0 - (Xn @ Xn[s]))
        for _ in range(1, k):
            i = int(np.argmax(d)); sel.append(i)
            d = np.minimum(d, 1.0 - (Xn @ Xn[i]))
        return sel
    else:
        d = np.full(n, np.inf)
        s = int(rng.integers(0, n))
        sel = [s]
        diff = X - X[s]; d = np.minimum(d, np.einsum("ij,ij->i", diff, diff))
        for _ in range(1, k):
            i = int(np.argmax(d)); sel.append(i)
            diff = X - X[i]; d = np.minimum(d, np.einsum("ij,ij->i", diff, diff))
        return sel

def facility_location_greedy(X: np.ndarray, k: int, metric: str="cosine") -> List[int]:
    n = X.shape[0]
    if n == 0 or k <= 0: return []
    if k >= n: return list(range(n))
    if metric == "cosine":
        Xn = _l2_normalize(X)
        best = np.full(n, -np.inf)
        sel = []
        for _ in range(k):
            S = Xn @ Xn.T
            gains = np.maximum(0.0, S - best[:, None])
            j = int(np.argmax(gains.sum(axis=0)))
            sel.append(j)
            best = np.maximum(best, S[:, j])
        return list(dict.fromkeys(sel))[:k]
    else:
        norms = np.einsum("ij,ij->i", X, X)
        best = np.full(n, -np.inf)
        sel = []
        for _ in range(k):
            dots = X @ X.T
            S = -(norms[:, None] + norms[None, :] - 2 * dots)
            gains = np.maximum(0.0, S - best[:, None])
            j = int(np.argmax(gains.sum(axis=0)))
            sel.append(j)
            best = np.maximum(best, S[:, j])
        return list(dict.fromkeys(sel))[:k]

def sum_diversity_greedy(X: np.ndarray, k: int, metric: str="cosine") -> List[int]:
    n = X.shape[0]
    if n == 0 or k <= 0: return []
    if k >= n: return list(range(n))
    if metric == "cosine":
        Xn = _l2_normalize(X)
        mu = Xn.mean(axis=0, keepdims=True)
        scores = (Xn @ mu.T).ravel()
        sel = [int(np.argmin(scores))]
        s = Xn[sel[0]].copy()
        for _ in range(1, k):
            dots = Xn @ s
            dots[sel] = np.inf
            j = int(np.argmin(dots))
            sel.append(j); s += Xn[j]
        return sel
    else:
        return farthest_first(X, k, metric="euclidean")

def select_diverse(X: np.ndarray, k: int, objective: SelectionObjective=SelectionObjective.FARTHEST_FIRST,
                   metric: str="cosine", seed: Optional[int]=None) -> List[int]:
    if objective == SelectionObjective.FARTHEST_FIRST:
        return farthest_first(X, k, metric=metric, seed=seed)
    if objective == SelectionObjective.FACILITY_LOCATION:
        return facility_location_greedy(X, k, metric=metric)
    if objective == SelectionObjective.SUM_DIVERSITY:
        return sum_diversity_greedy(X, k, metric=metric)
    raise ValueError(f"Unknown objective: {objective}")
