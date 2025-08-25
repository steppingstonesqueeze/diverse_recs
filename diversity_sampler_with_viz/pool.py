from __future__ import annotations
import heapq
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
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
