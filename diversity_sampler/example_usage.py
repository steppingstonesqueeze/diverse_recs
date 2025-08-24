import numpy as np
from diversity_sampler import CandidatePool, PoolMode, select_diverse, SelectionObjective, build_kernel, k_dpp_sample, diversity_metrics

def demo():
    # Simulate a stream
    rng = np.random.default_rng(0)
    n = 2000
    d = 64
    X = rng.normal(size=(n, d))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)  # normalize for cosine
    # quality scores (optional)
    q = np.clip(rng.lognormal(mean=0.0, sigma=1.0, size=n), 0, 10)

    # 1) Build candidate pool (uniform bottom-k)
    pool = CandidatePool(capacity=1000, mode=PoolMode.BOTTOMK, seed=42)
    for i in range(n):
        pool.add(item_id=i, features=X[i], weight=1.0)
    ids, Xcand, W, TS, META = pool.candidates()
    print(f"Candidates: {len(ids)}")

    # 2A) Greedy selection (facility location)
    k = 50
    sel_idxs = select_diverse(Xcand, k=k, objective=SelectionObjective.FACILITY_LOCATION, metric="cosine", seed=0)
    print("Greedy facility-location metrics:", diversity_metrics(Xcand, sel_idxs))

    # 2B) k-DPP sampling on same candidates
    k = 10
    L = build_kernel(Xcand, quality=None, kind="cosine", alpha=1.0, jitter=1e-6)
    Y = k_dpp_sample(L, k=k, seed=0)
    print("k-DPP metrics:", diversity_metrics(Xcand, Y))

if __name__ == "__main__":
    demo()
