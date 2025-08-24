from __future__ import annotations
import heapq
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from typing import Any, Callable
from enum import Enum
import numpy as np


def _hash64_to_unit(x: int) -> float:
    # Map a 64-bit integer to (0,1)
    return (x & ((1<<64)-1)) / float(1<<64)

def _stable_hash_bytes(b: bytes) -> int:
    # 64-bit stable hash (murmur-like not included; use Python's sha256 then take 64 bits)
    import hashlib
    h = hashlib.sha256(b).digest()
    return int.from_bytes(h[:8], "little", signed=False)

def stable_hash_any(obj) -> int:
    if isinstance(obj, (bytes, bytearray)):
        return _stable_hash_bytes(bytes(obj))
    if isinstance(obj, str):
        return _stable_hash_bytes(obj.encode('utf-8'))
    if isinstance(obj, int):
        return obj & ((1<<64)-1)
    # Fallback: JSON-serialize
    import json
    return _stable_hash_bytes(json.dumps(obj, sort_keys=True, default=str).encode('utf-8'))

class PoolMode:
    BOTTOMK = "bottomk"      # uniform without replacement via k smallest hash
    PRIORITY = "priority"    # weighted PPSWOR via priority sampling (keep largest priority)

@dataclass(order=True)
class _HeapItem:
    # For heap ordering; 'key' first determines comparison
    key: float
    item_id: object = field(compare=False)
    idx: int = field(compare=False)

class CandidatePool:
    """Streaming candidate pool with mergeability.

    Modes:
      - BOTTOMK: exact uniform sample without replacement over item_ids (by 64-bit hash).
                 Keeps the k SMALLEST hash values. Merge by taking global k smallest.
      - PRIORITY: priority sampling for weighted items. For item with weight w>0,
                  draw u~U(0,1], priority p = u**(1/w), keep k LARGEST p.
                  Merge by taking global k largest priorities.

    Stores features per item (last-seen wins). Deduplicates by item_id.
    """
    def __init__(self, capacity: int, mode: str = PoolMode.BOTTOMK, seed: Optional[int]=None):
        assert capacity > 0
        assert mode in (PoolMode.BOTTOMK, PoolMode.PRIORITY)
        self.capacity = capacity
        self.mode = mode
        self._heap: List[_HeapItem] = []
        self._items: Dict[object, Tuple[np.ndarray, float, Optional[float], Optional[dict], float]] = {}
        # item_id -> (features, weight, timestamp, metadata, priority_key (hash or priority p))
        self._rng = np.random.default_rng(seed)
        self._next_idx = 0

    def __len__(self):
        return len(self._items)

    def _heap_key(self, priority: float) -> float:
        # BOTTOMK: we keep smallest -> use -priority for max-heap behavior via min-heap? No: we'll keep as max-heap by key.
        # We'll implement explicit logic:
        return priority

    def _maybe_insert(self, item_id, key_value, features, weight, ts, metadata):
        # For BOTTOMK: keep k smallest key_value
        # For PRIORITY: keep k largest key_value
        if self.mode == PoolMode.BOTTOMK:
            # If not full, add. Else compare if key_value < current worst (max)
            if item_id in self._items:
                # If already present and we got a smaller key (shouldn't happen for stable hash), keep smallest
                prev = self._items[item_id][-1]
                if key_value < prev:
                    self._items[item_id] = (features, weight, ts, metadata, key_value)
                else:
                    # Update features/weight/ts/metadata even if key same
                    self._items[item_id] = (features, weight, ts, metadata, prev)
                return

            if len(self._heap) < self.capacity:
                heapq.heappush(self._heap, _HeapItem(key_value, item_id, self._next_idx)); self._next_idx += 1
                self._items[item_id] = (features, weight, ts, metadata, key_value)
            else:
                worst = self._heap[-1] if False else None  # not used
                # In a min-heap of keys, the largest key is not easily accessible, so we maintain a max-key at root if we invert sign,
                # but to keep it simple: we can look at current worst by peeking at max of heap (O(k)) rarely.
                # We'll optimize by keeping the current worst as max(self._heap, key=lambda x: x.key).key (O(k)).
                # For capacity up to tens of thousands this is fine.
                worst_key = max(self._heap, key=lambda x: x.key).key
                if key_value < worst_key:
                    # Remove the current worst (max key) once
                    # Find index of worst
                    idx = max(range(len(self._heap)), key=lambda i: self._heap[i].key)
                    removed = self._heap[idx]
                    self._heap[idx] = self._heap[-1]
                    self._heap.pop()
                    if idx < len(self._heap):
                        heapq.heapify(self._heap)
                    # Delete from items
                    if removed.item_id in self._items:
                        del self._items[removed.item_id]
                    # Insert new
                    heapq.heappush(self._heap, _HeapItem(key_value, item_id, self._next_idx)); self._next_idx += 1
                    self._items[item_id] = (features, weight, ts, metadata, key_value)
                else:
                    # skip
                    pass
        else:
            # PRIORITY: keep k largest key_value
            if item_id in self._items:
                prev = self._items[item_id][-1]
                if key_value > prev:
                    self._items[item_id] = (features, weight, ts, metadata, key_value)
                else:
                    self._items[item_id] = (features, weight, ts, metadata, prev)
                return

            if len(self._heap) < self.capacity:
                heapq.heappush(self._heap, _HeapItem(-key_value, item_id, self._next_idx)); self._next_idx += 1  # store negative so smallest is largest priority
                self._items[item_id] = (features, weight, ts, metadata, key_value)
            else:
                # Worst is smallest priority => largest negative key
                smallest_priority = -min(self._heap, key=lambda x: x.key).key
                if key_value > smallest_priority:
                    idx = min(range(len(self._heap)), key=lambda i: self._heap[i].key)  # most negative -> smallest priority
                    removed = self._heap[idx]
                    self._heap[idx] = self._heap[-1]
                    self._heap.pop()
                    if idx < len(self._heap):
                        heapq.heapify(self._heap)
                    if removed.item_id in self._items:
                        del self._items[removed.item_id]
                    heapq.heappush(self._heap, _HeapItem(-key_value, item_id, self._next_idx)); self._next_idx += 1
                    self._items[item_id] = (features, weight, ts, metadata, key_value)
                else:
                    pass

    def add(self, item_id, features: np.ndarray, weight: float = 1.0, ts: Optional[float]=None, metadata: Optional[dict]=None):
        """Add an item to the streaming pool.

        features: 1D array-like
        weight: positive weight for PRIORITY mode; ignored for BOTTOMK (but stored).
        ts: optional timestamp (defaults to now)
        metadata: optional dict
        """
        if ts is None:
            ts = time.time()
        x = np.asarray(features, dtype=float).ravel()
        if self.mode == PoolMode.BOTTOMK:
            h = stable_hash_any(item_id)
            key = _hash64_to_unit(h)
        else:
            if weight <= 0:
                weight = 1e-9
            u = float(self._rng.random())
            if u <= 0.0:
                u = np.nextafter(0.0, 1.0)
            key = u ** (1.0 / float(weight))  # keep k largest
        self._maybe_insert(item_id, key, x, float(weight), ts, metadata)

    def merge(self, other: 'CandidatePool'):
        assert self.mode == other.mode
        # Just add other's items through _maybe_insert with their stored key (priority/hash)
        for item_id, (feat, w, ts, meta, key) in other._items.items():
            self._maybe_insert(item_id, key, feat, w, ts, meta)

    def candidates(self):
        """Return (ids, X, weights, timestamps, metadata) with consistent order."""
        ids = list(self._items.keys())
        X = np.vstack([self._items[i][0] for i in ids]) if ids else np.zeros((0,0))
        W = np.array([self._items[i][1] for i in ids], dtype=float)
        TS = np.array([self._items[i][2] for i in ids], dtype=float)
        META = [self._items[i][3] for i in ids]
        return ids, X, W, TS, META

# diversity_sampler/dpp.py
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
        if rng.random() < prob:
            S.append(i)
            j -= 1
    S.reverse()
    return S

def k_dpp_sample(L: np.ndarray, k: int, eps: float = 1e-12, seed: int | None = None):
    """
    Sample a size-k subset from an L-ensemble DPP with PSD kernel L (n x n).
    """
    n = L.shape[0]
    if k <= 0:
        return []
    if k >= n:
        return list(range(n))

    vals, vecs = np.linalg.eigh(L)
    vals = np.maximum(vals, 0.0)
    rng = np.random.default_rng(seed)

    # 1) pick exactly k eigenvectors
    S_idx = _select_k_eigenvectors(vals, k, rng)
    if len(S_idx) < k:
        S_idx = list(np.argsort(vals)[-k:])

    V = vecs[:, S_idx]  # (n x r) with r == k

    # 2) sequentially pick items
    Y = []
    for _ in range(k):
        # row norms squared -> selection probabilities over items
        P_rows = np.sum(V * V, axis=1)
        sP = P_rows.sum()
        if sP <= eps:
            break
        i = int(rng.choice(n, p=P_rows / sP))
        Y.append(i)

        # pick a column j with prob uc0u8733  V[i, j]^2
        col_weights = V[i, :] ** 2
        sC = col_weights.sum()
        if sC <= eps:
            break
        j = int(rng.choice(V.shape[1], p=col_weights / sC))

        # v = V[:, j] / V[i, j]
        denom = V[i, j]
        if abs(denom) <= eps:
            break
        v = V[:, j] / denom  # shape (n,)

        # Update: V <- V - v * V[i, :]
        V = V - np.outer(v, V[i, :])  # (n x r) - (n x 1) @ (1 x r)

        # Drop column j and re-orthonormalize
        if V.shape[1] <= 1:
            break
        V = np.delete(V, j, axis=1)
        Q, _ = np.linalg.qr(V)  # (n x (r-1))
        V = Q

    # Deduplicate in case of numerical repeats
    return list(dict.fromkeys(Y))[:k]

def diversity_metrics(X: np.ndarray, idxs):
    idxs = list(idxs)
    if len(idxs) == 0:
        return {"k": 0}
    Xs = X[idxs]
    # mean pairwise cosine similarity
    Xn = Xs / np.maximum(np.linalg.norm(Xs, axis=1, keepdims=True), 1e-12)
    S = Xn @ Xn.T
    n = len(idxs)
    iu = np.triu_indices(n, k=1)
    mean_pairwise_cos = float(S[iu].mean()) if iu[0].size>0 else 1.0
    # nearest-neighbor cosine distance
    np.fill_diagonal(S, -np.inf)
    nn_sim = S.max(axis=1)
    mean_nn_cos_dist = float((1.0 - nn_sim).mean())
    return {
        "k": n,
        "mean_pairwise_cosine": mean_pairwise_cos,
        "mean_nn_cosine_distance": mean_nn_cos_dist,
    }

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
