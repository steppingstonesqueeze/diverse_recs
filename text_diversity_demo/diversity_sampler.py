from __future__ import annotations
import heapq
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Tuple
import numpy as np

def _hash64_to_unit(x: int) -> float:
    return (x & ((1<<64)-1)) / float(1<<64)

def _stable_hash_bytes(b: bytes) -> int:
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
    import json
    return _stable_hash_bytes(json.dumps(obj, sort_keys=True, default=str).encode('utf-8'))

class PoolMode:
    BOTTOMK = "bottomk"
    PRIORITY = "priority"

@dataclass(order=True)
class _HeapItem:
    key: float
    item_id: object = field(compare=False)
    idx: int = field(compare=False)

class CandidatePool:
    """Streaming candidate pool with mergeability."""
    def __init__(self, capacity: int, mode: str = PoolMode.BOTTOMK, seed: Optional[int]=None):
        assert capacity > 0
        assert mode in (PoolMode.BOTTOMK, PoolMode.PRIORITY)
        self.capacity = capacity
        self.mode = mode
        self._heap: List[_HeapItem] = []
        self._items: Dict[object, Tuple[np.ndarray, float, Optional[float], Optional[dict], float]] = {}
        self._rng = np.random.default_rng(seed)
        self._next_idx = 0

    def __len__(self):
        return len(self._items)

    def _maybe_insert(self, item_id, key_value, features, weight, ts, metadata):
        if self.mode == PoolMode.BOTTOMK:
            if item_id in self._items:
                prev = self._items[item_id][-1]
                if key_value < prev:
                    self._items[item_id] = (features, weight, ts, metadata, key_value)
                else:
                    self._items[item_id] = (features, weight, ts, metadata, prev)
                return
            if len(self._heap) < self.capacity:
                heapq.heappush(self._heap, _HeapItem(key_value, item_id, self._next_idx)); self._next_idx += 1
                self._items[item_id] = (features, weight, ts, metadata, key_value)
            else:
                worst_key = max(self._heap, key=lambda x: x.key).key
                if key_value < worst_key:
                    idx = max(range(len(self._heap)), key=lambda i: self._heap[i].key)
                    removed = self._heap[idx]
                    self._heap[idx] = self._heap[-1]
                    self._heap.pop()
                    if idx < len(self._heap):
                        heapq.heapify(self._heap)
                    if removed.item_id in self._items:
                        del self._items[removed.item_id]
                    heapq.heappush(self._heap, _HeapItem(key_value, item_id, self._next_idx)); self._next_idx += 1
                    self._items[item_id] = (features, weight, ts, metadata, key_value)
        else:
            if item_id in self._items:
                prev = self._items[item_id][-1]
                if key_value > prev:
                    self._items[item_id] = (features, weight, ts, metadata, key_value)
                else:
                    self._items[item_id] = (features, weight, ts, metadata, prev)
                return
            if len(self._heap) < self.capacity:
                heapq.heappush(self._heap, _HeapItem(-key_value, item_id, self._next_idx)); self._next_idx += 1
                self._items[item_id] = (features, weight, ts, metadata, key_value)
            else:
                smallest_priority = -min(self._heap, key=lambda x: x.key).key
                if key_value > smallest_priority:
                    idx = min(range(len(self._heap)), key=lambda i: self._heap[i].key)
                    removed = self._heap[idx]
                    self._heap[idx] = self._heap[-1]
                    self._heap.pop()
                    if idx < len(self._heap):
                        heapq.heapify(self._heap)
                    if removed.item_id in self._items:
                        del self._items[removed.item_id]
                    heapq.heappush(self._heap, _HeapItem(-key_value, item_id, self._next_idx)); self._next_idx += 1
                    self._items[item_id] = (features, weight, ts, metadata, key_value)

    def add(self, item_id, features: np.ndarray, weight: float = 1.0, ts: Optional[float]=None, metadata: Optional[dict]=None):
        if ts is None:
            import time
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
            key = u ** (1.0 / float(weight))
        self._maybe_insert(item_id, key, x, float(weight), ts, metadata)

    def merge(self, other: 'CandidatePool'):
        assert self.mode == other.mode
        for item_id, (feat, w, ts, meta, key) in other._items.items():
            self._maybe_insert(item_id, key, feat, w, ts, meta)

    def candidates(self):
        ids = list(self._items.keys())
        X = np.vstack([self._items[i][0] for i in ids]) if ids else np.zeros((0,0))
        W = np.array([self._items[i][1] for i in ids], dtype=float)
        TS = np.array([self._items[i][2] for i in ids], dtype=float)
        META = [self._items[i][3] for i in ids]
        return ids, X, W, TS, META

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

def l2_normalize(X: np.ndarray, eps: float=1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return X / n

def cosine_similarity(X: np.ndarray) -> np.ndarray:
    Xn = l2_normalize(X)
    return Xn @ Xn.T

def rbf_kernel(X: np.ndarray, gamma: float=None) -> np.ndarray:
    n = X.shape[0]
    norms = np.einsum('ij,ij->i', X, X)
    D2 = norms[:, None] + norms[None, :] - 2*(X @ X.T)
    D2 = np.maximum(D2, 0.0)
    if gamma is None:
        iu = np.triu_indices(n, k=1)
        med = np.median(D2[iu]) if iu[0].size > 0 else 1.0
        if not np.isfinite(med) or med <= 0:
            med = 1.0
        gamma = 1.0 / med
    K = np.exp(-gamma * D2)
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

def embed(X: np.ndarray, method: str = "pca", random_state: int = 0, n_components: int = 2, **kwargs):
    method = method.lower()
    if method == "pca":
        try:
            from sklearn.decomposition import PCA
        except Exception as e:
            raise ImportError("scikit-learn is required for PCA (pip install scikit-learn)") from e
        pca = PCA(n_components=n_components, random_state=random_state)
        Y = pca.fit_transform(X)
    elif method == "tsne":
        try:
            from sklearn.manifold import TSNE
        except Exception as e:
            raise ImportError("scikit-learn is required for t-SNE (pip install scikit-learn)") from e
        tsne = TSNE(n_components=n_components, random_state=random_state, init='pca', perplexity=30, learning_rate='auto')
        Y = tsne.fit_transform(X)
    elif method == "umap":
        try:
            import umap
        except Exception as e:
            raise ImportError("umap-learn is required for UMAP (pip install umap-learn)") from e
        reducer = umap.UMAP(n_components=n_components, random_state=random_state)
        Y = reducer.fit_transform(X)
    else:
        raise ValueError(f"Unknown method: {method}")
    if Y.shape[1] != 2:
        Y = Y[:, :2]
    return Y

def plot_embedding(Y: np.ndarray, selected_idxs=None, title: str = "", save_path: str | None = None):
    import matplotlib.pyplot as plt
    selected_set = set(selected_idxs) if selected_idxs is not None else set()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(Y[:, 0], Y[:, 1], s=10, alpha=0.25, linewidths=0)
    if selected_idxs is not None and len(selected_set) > 0:
        sel = np.array(sorted(selected_set), dtype=int)
        ax.scatter(Y[sel, 0], Y[sel, 1], s=40, alpha=0.95, edgecolors="black", linewidths=0.5)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("dim-1"); ax.set_ylabel("dim-2")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160)
        plt.close(fig)
    else:
        return fig, ax

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
