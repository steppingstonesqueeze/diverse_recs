from __future__ import annotations
import numpy as np

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
