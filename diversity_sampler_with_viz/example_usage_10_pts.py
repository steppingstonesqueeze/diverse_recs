import numpy as np
from diversity_sampler import CandidatePool, PoolMode, select_diverse, SelectionObjective, build_kernel, k_dpp_sample, diversity_metrics, embed, plot_embedding
import os, csv

def _save_tsv(path, ids, X):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id"] + [f"f{i}" for i in range(X.shape[1])])
        for _id, row in zip(ids, X):
            w.writerow([_id] + [f"{v:.8g}" for v in row])

def demo():
    rng = np.random.default_rng(0)
    n, d = 2000, 64
    X = rng.normal(size=(n, d))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    pool = CandidatePool(capacity=1000, mode=PoolMode.BOTTOMK, seed=42)
    for i in range(n):
        pool.add(item_id=i, features=X[i], weight=1.0)
    ids, Xcand, W, TS, META = pool.candidates()
    print(f"Candidates: {len(ids)}")
    _save_tsv("candidates_1000.tsv", ids, Xcand)

    k = 10
    greedy_idx = select_diverse(Xcand, k=k, objective=SelectionObjective.FACILITY_LOCATION, metric="cosine", seed=0)
    print("Greedy facility-location metrics:", diversity_metrics(Xcand, greedy_idx))
    greedy_ids = [ids[i] for i in greedy_idx]
    _save_tsv("greedy10.tsv", greedy_ids, Xcand[greedy_idx])

    L = build_kernel(Xcand, quality=None, kind="cosine", alpha=0.9, jitter=1e-5)
    dpp_idx = k_dpp_sample(L, k=k, seed=0)
    if len(dpp_idx) < k:
        remaining = sorted(set(range(len(ids))) - set(dpp_idx))
        fill_idx_local = select_diverse(Xcand[remaining], k=k - len(dpp_idx), objective=SelectionObjective.FARTHEST_FIRST, metric="cosine", seed=1)
        dpp_idx += [remaining[j] for j in fill_idx_local]
    print("k-DPP metrics:", diversity_metrics(Xcand, dpp_idx))
    dpp_ids = [ids[i] for i in dpp_idx]
    _save_tsv("kdpp10.tsv", dpp_ids, Xcand[dpp_idx])

    os.makedirs("embeds", exist_ok=True)
    for method in ["pca", "tsne", "umap"]:
        try:
            Y = embed(Xcand, method=method, random_state=0, n_components=2)
        except ImportError as e:
            print(f"[viz] Skipping {method.upper()} (dependency missing): {e}")
            continue
        plot_embedding(Y, selected_idxs=None, title=f"Candidates ({method.upper()})", save_path=f"embeds/candidates_{method}.png")
        plot_embedding(Y, selected_idxs=greedy_idx, title=f"Greedy-10 ({method.upper()})", save_path=f"embeds/greedy10_{method}.png")
        plot_embedding(Y, selected_idxs=dpp_idx, title=f"k-DPP-10 ({method.upper()})", save_path=f"embeds/kdpp10_{method}.png")
        print(f"[viz] Saved {method.upper()} plots to embeds/*.png")

if __name__ == "__main__":
    demo()
