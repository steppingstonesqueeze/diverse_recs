import numpy as np
from diversity_sampler import (
    CandidatePool, PoolMode,
    select_diverse, SelectionObjective,
    build_kernel, k_dpp_sample, diversity_metrics
)
import csv, os

def _save_tsv(path, ids, X):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter=",")
        w.writerow(["id"] + [f"f{i}" for i in range(X.shape[1])])
        for _id, row in zip(ids, X):
            w.writerow([_id] + [f"{v:.8g}" for v in row])

def demo():
    rng = np.random.default_rng(0)
    n, d = 2000, 64
    X = rng.normal(size=(n, d))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)  # normalize for cosine

    # 1) Candidate pool (uniform bottom-k by hash) -> 1000 points
    pool = CandidatePool(capacity=1000, mode=PoolMode.BOTTOMK, seed=42)
    for i in range(n):
        pool.add(item_id=i, features=X[i], weight=1.0)
    ids, Xcand, W, TS, META = pool.candidates()
    print(f"Candidates: {len(ids)}")

    # Save the 1000 candidate points
    #_save_tsv("candidates_1000.tsv", ids, Xcand)
    _save_tsv("candidates_1000.csv", ids, Xcand)

    # 2A) Greedy facility-location (k=50)
    k = 50
    greedy_idx = select_diverse(
        Xcand, k=k,
        objective=SelectionObjective.FACILITY_LOCATION,
        metric="cosine", seed=0
    )
    print("Greedy facility-location metrics:", diversity_metrics(Xcand, greedy_idx))
    greedy_ids = [ids[i] for i in greedy_idx]
    #_save_tsv("greedy50.tsv", greedy_ids, Xcand[greedy_idx])
    _save_tsv("greedy50.csv", greedy_ids, Xcand[greedy_idx])

    # 2B) k-DPP (k=50) on a better-conditioned kernel + fallback fill
    # Tip: include (1-alpha)*I and a touch more jitter to stabilize rank.
    L = build_kernel(
        Xcand,
        quality=None,
        kind="cosine",   # try "rbf" if you prefer
        alpha=0.9,
        jitter=1e-5
    )
    dpp_idx = k_dpp_sample(L, k=k, seed=0)

    # Fallback: if numerics return < k, fill with farthest-first on the remainder
    if len(dpp_idx) < k:
        remaining = sorted(set(range(len(ids))) - set(dpp_idx))
        fill_idx_local = select_diverse(
            Xcand[remaining], k=k - len(dpp_idx),
            objective=SelectionObjective.FARTHEST_FIRST,
            metric="cosine", seed=1
        )
        dpp_idx += [remaining[j] for j in fill_idx_local]

    print("k-DPP metrics:", diversity_metrics(Xcand, dpp_idx))
    dpp_ids = [ids[i] for i in dpp_idx]
    #_save_tsv("kdpp50.tsv", dpp_ids, Xcand[dpp_idx])
    _save_tsv("kdpp50.csv", dpp_ids, Xcand[dpp_idx])

if __name__ == "__main__":
    demo()
