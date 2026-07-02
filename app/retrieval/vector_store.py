"""
FAISS vector store for SHL Assessment Recommender.

Manages the FAISS index for fast approximate nearest neighbor search
over catalog item embeddings.

Design Decision: FAISS with inner product (cosine on normalized vectors).
Improves: Performance (sub-millisecond search), Recall@10 (wide retrieval net).
Alternatives considered: Chroma (heavier), pgvector (needs DB).
Trade-offs: In-memory only. Fine for 377 items (~0.5MB).
"""

import numpy as np

from app.utils.logger import setup_logger

logger = setup_logger(__name__)

# Try importing faiss
try:
    import faiss
except ImportError:
    faiss = None
    logger.warning("FAISS not available, falling back to numpy-based search")


class VectorStore:
    """FAISS-based vector store for catalog embeddings.

    Uses Inner Product index on L2-normalized vectors (equivalent to cosine similarity).

    Attributes:
        index: FAISS index instance.
        dimension: Embedding dimension.
        count: Number of vectors stored.
    """

    def __init__(self) -> None:
        self.index = None
        self.dimension: int = 0
        self.count: int = 0
        self._embeddings: np.ndarray | None = None  # Fallback for non-FAISS

    def save(self, filepath: str) -> None:
        """Save FAISS index or numpy array to disk."""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if faiss is not None and self.index is not None:
            faiss.write_index(self.index, filepath)
            logger.info("FAISS index saved to %s", filepath)
        else:
            np.save(filepath + ".npy", self._embeddings)
            logger.info("Numpy embeddings saved to %s.npy", filepath)

    def load(self, filepath: str) -> bool:
        """Load FAISS index or numpy array from disk.

        Returns:
            True if load was successful, False otherwise.
        """
        import os
        if faiss is not None:
            if os.path.exists(filepath):
                try:
                    self.index = faiss.read_index(filepath)
                    self.dimension = self.index.d
                    self.count = self.index.ntotal
                    logger.info("Loaded FAISS index from %s (%d vectors, dim=%d)", filepath, self.count, self.dimension)
                    return True
                except Exception as e:
                    logger.error("Failed to load FAISS index from %s: %s", filepath, e)
        else:
            npy_path = filepath + ".npy"
            if os.path.exists(npy_path):
                try:
                    self._embeddings = np.load(npy_path)
                    self.dimension = self._embeddings.shape[1]
                    self.count = self._embeddings.shape[0]
                    logger.info("Loaded Numpy embeddings from %s (%d vectors, dim=%d)", npy_path, self.count, self.dimension)
                    return True
                except Exception as e:
                    logger.error("Failed to load Numpy embeddings from %s: %s", npy_path, e)
        return False

    def build(self, embeddings: np.ndarray) -> None:
        """Build FAISS index from embeddings.

        Args:
            embeddings: NumPy array of shape (n_items, embedding_dim).
                        Must be L2-normalized for cosine similarity.
        """
        self.dimension = embeddings.shape[1]
        self.count = embeddings.shape[0]

        if faiss is not None:
            # Use Inner Product on normalized vectors = cosine similarity
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(embeddings)
            logger.info(
                "FAISS index built: %d vectors, dim=%d",
                self.count,
                self.dimension,
            )
        else:
            # Numpy fallback
            self._embeddings = embeddings
            logger.info(
                "NumPy fallback index built: %d vectors, dim=%d",
                self.count,
                self.dimension,
            )

    def search(self, query_embedding: np.ndarray, top_k: int = 30) -> list[tuple[int, float]]:
        """Search for nearest neighbors.

        Args:
            query_embedding: Query vector of shape (1, embedding_dim).
            top_k: Number of results to return.

        Returns:
            List of (index, score) tuples, sorted by descending similarity.
        """
        top_k = min(top_k, self.count)

        if faiss is not None and self.index is not None:
            scores, indices = self.index.search(query_embedding, top_k)
            results = [
                (int(idx), float(score))
                for idx, score in zip(indices[0], scores[0])
                if idx >= 0
            ]
        else:
            # Numpy fallback: compute dot product
            scores = np.dot(self._embeddings, query_embedding.T).flatten()
            top_indices = np.argsort(scores)[::-1][:top_k]
            results = [
                (int(idx), float(scores[idx]))
                for idx in top_indices
            ]

        logger.debug(
            "Vector search: top_k=%d, best_score=%.4f",
            top_k,
            results[0][1] if results else 0.0,
        )
        return results
