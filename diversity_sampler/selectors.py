from __future__ import annotations
import numpy as np
from enum import Enum
from typing import Tuple, Optional, List, Callable

class SelectionObjective(Enum):
    FARTHEST_FIRST = "farthest_first"           # max-min (Gonzalez)
    FACILITY_LOCATION = "facility_location"     # coverage via similarity
    SUM_DIVERSITY = "sum_diversity"             # minimize pairwise inner products

def l2_normalize(X: np.ndarray, eps: float=1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return X / n

def farthest_first(X: np.ndarray, k: int, metric: str = "euclidean", seed: Optional[int]=None) -> List[int]:
    """Gonzalez farthest-first traversal. Returns indices of selected points.
    metric: 'euclidean' or 'cosine' (assumes X normalized if cosine).
    Complexity: O(n k d).
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n == 0 or k <= 0:
        return []
    if k >= n:
        return list(range(n))

    if metric == "cosine":
        Xn = l2_normalize(X)
        # cosine distance = 1 - dot
        dists = np.full(n, np.inf)
        start = int(rng.integers(0, n))
        selected = [start]
        dists = np.minimum(dists, 1.0 - (Xn @ Xn[start]))
        for _ in range(1, k):
            i = int(np.argmax(dists))
            selected.append(i)
            dists = np.minimum(dists, 1.0 - (Xn @ Xn[i]))
        return selected
    else:
        dists = np.full(n, np.inf)
        start = int(rng.integers(0, n))
        selected = [start]
        diff = X - X[start]
        dists = np.minimum(dists, np.einsum('ij,ij->i', diff, diff))
        for _ in range(1, k):
            i = int(np.argmax(dists))
            selected.append(i)
            diff = X - X[i]
            dists = np.minimum(dists, np.einsum('ij,ij->i', diff, diff))
        return selected

def facility_location_greedy(X: np.ndarray, k: int, metric: str="cosine") -> List[int]:
    """Greedy maximize sum_i max_{s in S} sim(i, s).
    For 'cosine', uses dot products on l2-normalized X.
    Complexity: O(n k d).
    """
    n = X.shape[0]
    if n == 0 or k <= 0:
        return []
    if k >= n:
        return list(range(n))
    if metric == "cosine":
        Xn = l2_normalize(X)
        best = np.full(n, -np.inf)
        selected = []
        for _ in range(k):
            # compute gain of adding each candidate j: sum_i max(0, Xn[i]·Xn[j] - best[i])
            sims = Xn @ Xn.T   # we can avoid full matrix by caching; for clarity keep this. For n ~ up to 10k it's OK.
            gains = np.maximum(0.0, sims - best[:, None])
            # Sum gains per j
            gsum = gains.sum(axis=0)
            j = int(np.argmax(gsum))
            selected.append(j)
            best = np.maximum(best, sims[:, j])
            # mask selected column to avoid reselecting
            Xn[j] = Xn[j]  # no-op; rely on best to prevent duplicates
        # Deduplicate indices (in case numerical ties)
        return list(dict.fromkeys(selected))[:k]
    else:
        # Euclidean similarity: use negative squared distance as similarity for facility-location
        best = np.full(n, -np.inf)
        selected = []
        for _ in range(k):
            # compute similarities -||x - x_j||^2
            # sim_ij = -||x_i||^2 - ||x_j||^2 + 2 x_i·x_j
            norms = np.einsum('ij,ij->i', X, X)
            dots = X @ X.T
            sims = - (norms[:, None] + norms[None, :] - 2*dots)
            gains = np.maximum(0.0, sims - best[:, None])
            gsum = gains.sum(axis=0)
            j = int(np.argmax(gsum))
            selected.append(j)
            best = np.maximum(best, sims[:, j])
        return list(dict.fromkeys(selected))[:k]

def sum_diversity_greedy(X: np.ndarray, k: int, metric: str="cosine") -> List[int]:
    """Greedy minimize sum of pairwise inner products (or maximize negative of that).
    For unit-normalized X, minimizing sum dot equals minimizing ||sum(X_S)||^2.
    Strategy: start with the vector of smallest correlation to the mean, then iteratively pick argmin x·s where s=sum of selected vectors.
    """
    n = X.shape[0]
    if n == 0 or k <= 0:
        return []
    if k >= n:
        return list(range(n))
    if metric == "cosine":
        Xn = l2_normalize(X)
        # start with the point most opposite to dataset mean
        mu = Xn.mean(axis=0, keepdims=True)
        scores = (Xn @ mu.T).ravel()
        selected = [int(np.argmin(scores))]
        s = Xn[selected[0]].copy()
        for _ in range(1, k):
            dots = Xn @ s
            # choose minimal dot (most orthogonal/opposite to sum)
            # avoid duplicates by setting already selected to +inf
            dots[selected] = np.inf
            j = int(np.argmin(dots))
            selected.append(j)
            s += Xn[j]
        return selected
    else:
        # For euclidean, we can use cosine on normalized; distance-only version falls back to farthest-first
        return farthest_first(X, k, metric="euclidean")

def select_diverse(X: np.ndarray, k: int, objective: SelectionObjective=SelectionObjective.FARTHEST_FIRST, metric: str="cosine", seed: Optional[int]=None) -> List[int]:
    if objective == SelectionObjective.FARTHEST_FIRST:
        return farthest_first(X, k, metric=metric, seed=seed)
    elif objective == SelectionObjective.FACILITY_LOCATION:
        return facility_location_greedy(X, k, metric=metric)
    elif objective == SelectionObjective.SUM_DIVERSITY:
        return sum_diversity_greedy(X, k, metric=metric)
    else:
        raise ValueError(f"Unknown objective: {objective}")
