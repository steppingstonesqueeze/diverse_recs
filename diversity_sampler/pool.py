from __future__ import annotations
import heapq
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
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
