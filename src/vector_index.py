# src/vector_index.py

import numpy as np

try:
    import faiss
    _HAS_FAISS = True
except Exception:
    faiss = None
    _HAS_FAISS = False

from sklearn.neighbors import NearestNeighbors


class VectorIndex:
    """
    Wrapper that uses FAISS if available; otherwise falls back to sklearn NearestNeighbors.
    """
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings.astype("float32")

        if _HAS_FAISS:
            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(self.embeddings)
            self.backend = "faiss"
        else:
            self.index = NearestNeighbors(
                n_neighbors=5,
                metric="euclidean"
            )
            self.index.fit(self.embeddings)
            self.backend = "sklearn"

    def search(self, query_emb: np.ndarray, k: int = 5):
        """Return (indices, distances)"""
        if self.backend == "faiss":
            distances, indices = self.index.search(query_emb.astype("float32"), k)
            return indices.flatten(), distances.flatten()
        else:
            distances, indices = self.index.kneighbors(query_emb, n_neighbors=k)
            return indices.flatten(), distances.flatten()
