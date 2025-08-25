#!/usr/bin/env python3
"""
End-to-end text diversity demo in a single file.

Features:
- Generate N short paragraphs (<100 words) OR read your TSV with a 'text' column
- Embed using Sentence-BERT (sentence-transformers)
- Select k items via:
    * Greedy facility-location (coverage)
    * k-DPP (repulsive sampling) with numeric fallback to fill to k
- Save outputs: texts.tsv, embeddings.npy, greedy_k.tsv, kdpp_k.tsv

Install:
    python -m pip install numpy sentence-transformers

Usage:
    python run_demo.py --n 1000 --k 50 --outdir outputs
    python run_demo.py --input my_texts.tsv --k 50 --outdir outputs
"""

from __future__ import annotations
import argparse, csv, os, sys
from typing import List, Optional
import numpy as np

# ------------------------- Text generation -------------------------

def generate_paragraphs(n=1000, seed=0, max_words=100) -> List[str]:
    rng = np.random.default_rng(seed)
    topics = [
        "machine learning", "finance", "healthcare", "sports", "music", "travel",
        "technology", "education", "food", "movies", "climate", "astronomy",
        "productivity", "psychology", "art", "photography", "biology", "economics",
        "politics", "history", "mathematics", "physics", "chemistry", "robotics"
    ]
    verbs = [
        "explores", "analyzes", "describes", "summarizes", "compares",
        "highlights", "questions", "reviews", "predicts", "addresses"
    ]
    modifiers = [
        "practical", "fundamental", "modern", "classical", "scalable", "robust",
        "simple", "probabilistic", "efficient", "interpretable", "diverse", "random"
    ]
    extras = [
        "in practice", "at scale", "for beginners", "with examples", "using data",
        "under uncertainty", "in real time", "on devices", "in production",
        "with constraints", "in the wild", "and beyond"
    ]

    paras: List[str] = []
    for _ in range(n):
        words = 0
        sentences: List[str] = []
        # 3'965 short sentences
        for _ in range(int(rng.integers(3, 6))):
            t = rng.choice(topics)
            v = rng.choice(verbs)
            a = rng.choice(modifiers)
            e = rng.choice(extras)
            s = f"This paragraph {v} {a} aspects of {t} {e}."
            wcount = len(s.split())
            if words + wcount > max_words:
                break
            sentences.append(s)
            words += wcount
        if not sentences:
            sentences = ["This paragraph summarizes practical aspects of technology in practice."]
        paras.append(" ".join(sentences))
    return paras

# ------------------------- Embeddings (SBERT) -------------------------

def embed_texts(
    texts: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    device: Optional[str] = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Embed texts using Sentence-BERT (downloads model on first run).
    Returns (n, d) numpy array. If normalize=True, rows are L2-normalized.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import torch
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: sentence-transformers. "
            "Install with: python -m pip install sentence-transformers"
        ) from e

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=False,
    )
    if normalize:
        nrm = np.linalg.norm(emb, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        emb = emb / nrm
    return emb

# ------------------------- Selection algorithms -------------------------

def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return X / n

def facility_location_greedy(X: np.ndarray, k: int, metric: str = "cosine") -> List[int]:
    """
    Greedy maximize sum_i max_{s in S} sim(i, s).
    For cosine: assumes X is normalized (we'll normalize here for safety).
    Complexity O(n^2 + n*k) via precomputed similarity.
    """
    n = X.shape[0]
    if k <= 0: return []
    if k >= n: return list(range(n))
    if metric == "cosine":
        Xn = _l2_normalize(X)
        S = Xn @ Xn.T  # (n, n)
    else:
        norms = np.einsum("ij,ij->i", X, X)
        S = -(norms[:, None] + norms[None, :] - 2 * (X @ X.T))  # -||x - y||^2

    best = np.full(n, -np.inf)
    selected: List[int] = []
    for _ in range(k):
        gains = np.maximum(0.0, S - best[:, None])
        j = int(np.argmax(gains.sum(axis=0)))
        selected.append(j)
        best = np.maximum(best, S[:, j])
    # de-dup & trim in case of numeric ties
    return list(dict.fromkeys(selected))[:k]

def farthest_first(X: np.ndarray, k: int, metric: str = "cosine", seed: int = 0) -> List[int]:
    """
    Gonzalez farthest-first traversal (max-min spread). Used as a simple fallback.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if k <= 0: return []
    if k >= n: return list(range(n))

    if metric == "cosine":
        Xn = _l2_normalize(X)
        d = np.full(n, np.inf)
        start = int(rng.integers(0, n))
        sel = [start]
        d = np.minimum(d, 1.0 - (Xn @ Xn[start]))
        for _ in range(1, k):
            i = int(np.argmax(d)); sel.append(i)
            d = np.minimum(d, 1.0 - (Xn @ Xn[i]))
        return sel
    else:
        d = np.full(n, np.inf)
        start = int(rng.integers(0, n))
        sel = [start]
        diff = X - X[start]; d = np.minimum(d, np.einsum("ij,ij->i", diff, diff))
        for _ in range(1, k):
            i = int(np.argmax(d)); sel.append(i)
            diff = X - X[i]; d = np.minimum(d, np.einsum("ij,ij->i", diff, diff))
        return sel

def build_kernel(X: np.ndarray, kind: str = "cosine", alpha: float = 0.9, jitter: float = 1e-5) -> np.ndarray:
    """
    Build a well-conditioned L-ensemble kernel for DPP:
        L = alpha * S + (1 - alpha) * I  (+ jitter * I)
    where S is cosine similarity or RBF. We default to cosine (with L2-normalized X).
    """
    if kind == "cosine":
        Xn = _l2_normalize(X)
        S = Xn @ Xn.T
    elif kind == "rbf":
        norms = np.einsum("ij,ij->i", X, X)
        D2 = norms[:, None] + norms[None, :] - 2 * (X @ X.T)
        iu = np.triu_indices(len(X), k=1)
        med = np.median(D2[iu]) if iu[0].size else 1.0
        med = 1.0 if med <= 0 else med
        gamma = 1.0 / med
        S = np.exp(-gamma * np.maximum(D2, 0.0))
    else:
        raise ValueError("Unknown kernel kind")

    L = alpha * S + (1.0 - alpha) * np.eye(S.shape[0])
    if jitter and jitter > 0:
        L = L + jitter * np.eye(L.shape[0])
    return L

def _select_k_eigenvectors(lmbda: np.ndarray, k: int, rng: np.random.Generator) -> List[int]:
    """
    Standard DP over elementary symmetric polynomials to pick exactly k eigenvalues.
    """
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
        if denom <= 0:  # numerical
            continue
        base = E[j - 1, i - 1] if i > 0 else (1.0 if j == 1 else 0.0)
        prob = (lmbda[i] * base) / denom
        if rng.random() < prob:
            S.append(i)
            j -= 1
    S.reverse()
    return S

def k_dpp_sample(L: np.ndarray, k: int, seed: Optional[int] = None) -> List[int]:
    """
    Sample a size-k subset from an L-ensemble DPP with PSD kernel L (n x n).
    Uses eigendecomposition + standard k-DPP algorithm.
    """
    n = L.shape[0]
    if k <= 0: return []
    if k >= n: return list(range(n))

    vals, vecs = np.linalg.eigh(L)  # symmetric
    vals = np.maximum(vals, 0.0)
    rng = np.random.default_rng(seed)

    # pick exactly k eigenvectors (or fall back to top-k eigenvalues)
    S_idx = _select_k_eigenvectors(vals, k, rng)
    if len(S_idx) < k:
        S_idx = list(np.argsort(vals)[-k:])

    V = vecs[:, S_idx]  # (n x r), ideally r == k
    Y: List[int] = []
    eps = 1e-12

    for _ in range(k):
        if V.size == 0:
            break
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
        v = V[:, j] / denom  # (n,)

        # Update: V <- V - v * V[i, :]
        V = V - np.outer(v, V[i, :])

        # Drop the column j, then re-orthonormalize columns
        if V.shape[1] <= 1:
            V = np.zeros((n, 0))
            continue
        V = np.delete(V, j, axis=1)
        Q, _ = np.linalg.qr(V)
        V = Q

    # Dedup & trim (just in case)
    return list(dict.fromkeys(Y))[:k]

# ------------------------- Utilities -------------------------

def diversity_metrics(X: np.ndarray, idxs: List[int]) -> dict:
    idxs = list(idxs)
    if not idxs: return {"k": 0}
    Xs = X[idxs]
    Xn = _l2_normalize(Xs)
    S = Xn @ Xn.T
    n = len(idxs)
    iu = np.triu_indices(n, k=1)
    mean_pairwise_cos = float(S[iu].mean()) if iu[0].size > 0 else 1.0
    np.fill_diagonal(S, -np.inf)
    nn_sim = S.max(axis=1)
    mean_nn_cos_dist = float((1.0 - nn_sim).mean())
    return {
        "k": n,
        "mean_pairwise_cosine": mean_pairwise_cos,
        "mean_nn_cosine_distance": mean_nn_cos_dist,
    }

def save_tsv(path: str, ids: List[int], texts: List[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "text"])
        for _id, s in zip(ids, texts):
            w.writerow([_id, s])

def read_texts_tsv(path: str) -> List[str]:
    """
    Reads a TSV. If there is a 'text' column, uses it. Otherwise treats each non-empty line (after header) as text.
    """
    texts: List[str] = []
    with open(path, "r", newline="") as f:
        sniff = f.readline()
        f.seek(0)
        has_tab = "t" in sniff
        reader = csv.DictReader(f, delimiter="t" if has_tab else ",")
        if reader.fieldnames and ("text" in reader.fieldnames):
            for row in reader:
                s = (row.get("text") or "").strip()
                if s:
                    texts.append(s)
        else:
            # fallback: read all non-empty rows as a single column of text
            f.seek(0)
            for line in f:
                s = line.strip()
                if s and not s.lower().startswith("text"):
                    texts.append(s)
    if not texts:
        raise ValueError(f"No texts found in {path}. Expect a 'text' column or plain one-text-per-line.")
    return texts

# ------------------------- Main -------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate or load texts, embed with SBERT, select k via greedy FL and k-DPP.")
    ap.add_argument("--input", type=str, default=None, help="Optional TSV with a 'text' column (tab-separated). If omitted, synthetic texts are generated.")
    ap.add_argument("--n", type=int, default=1000, help="Number of synthetic paragraphs (ignored if --input is provided).")
    ap.add_argument("--k", type=int, default=50, help="Subset size.")
    ap.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Sentence-Transformers model name.")
    ap.add_argument("--outdir", type=str, default="outputs", help="Output directory.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed.")
    ap.add_argument("--kernel", type=str, default="cosine", choices=["cosine", "rbf"], help="Kernel for DPP.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Get texts
    if args.input:
        texts = read_texts_tsv(args.input)
        print(f"Loaded {len(texts)} texts from {args.input}")
    else:
        texts = generate_paragraphs(n=args.n, seed=args.seed, max_words=100)
        print(f"Generated {len(texts)} synthetic paragraphs.")

    ids = list(range(len(texts)))
    save_tsv(os.path.join(args.outdir, "texts.tsv"), ids, texts)

    # 2) Embed with SBERT
    X = embed_texts(texts, model_name=args.model, batch_size=64, normalize=True)
    np.save(os.path.join(args.outdir, "embeddings.npy"), X)
    print(f"Embeddings saved: {X.shape} -> {os.path.join(args.outdir, 'embeddings.npy')}")

    # 3A) Greedy facility-location
    k = min(args.k, len(ids))
    greedy_idx = facility_location_greedy(X, k=k, metric="cosine")
    greedy_ids = [ids[i] for i in greedy_idx]
    save_tsv(os.path.join(args.outdir, "greedy_k.tsv"), greedy_ids, [texts[i] for i in greedy_ids])
    print("Greedy facility-location metrics:", diversity_metrics(X, greedy_idx))

    # 3B) k-DPP + fallback fill
    L = build_kernel(X, kind=args.kernel, alpha=0.9, jitter=1e-5)
    dpp_idx = k_dpp_sample(L, k=k, seed=args.seed)
    if len(dpp_idx) < k:
        remaining = sorted(set(range(len(ids))) - set(dpp_idx))
        fill_local = farthest_first(X[remaining], k=k - len(dpp_idx), metric="cosine", seed=args.seed + 1)
        dpp_idx += [remaining[j] for j in fill_local]
    dpp_ids = [ids[i] for i in dpp_idx]
    save_tsv(os.path.join(args.outdir, "kdpp_k.tsv"), dpp_ids, [texts[i] for i in dpp_ids])
    print("k-DPP metrics:", diversity_metrics(X, dpp_idx))

    print(f"Done. Outputs in: {args.outdir}")
    print("Files written:")
    print(" - texts.tsv")
    print(" - embeddings.npy")
    print(" - greedy_k.tsv")
    print(" - kdpp_k.tsv")

if __name__ == "__main__":
    main()
