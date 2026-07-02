"""
Embedding generation for SHL Assessment Recommender.

Uses sentence-transformers (all-MiniLM-L6-v2) for local embedding generation.
No API key needed — runs entirely on CPU.

Design Decision: Local model avoids API costs and latency for embeddings.
Improves: Performance (no network call), Cost (free), Recall@10 (consistent embeddings).
Alternatives considered: OpenAI embeddings (cost), Cohere (API dependency).
Trade-offs: ~80MB model size, ~100ms per query. Acceptable for 377 items.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

# Global model instance (lazy-loaded)
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Get or lazily load the sentence transformer model.

    Returns:
        Loaded SentenceTransformer model.

    Note:
        Lazy loading improves cold start — model loaded on first use.
    """
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded (dim=%d)", _model.get_sentence_embedding_dimension())
    return _model


def generate_embeddings(texts: list[str]) -> np.ndarray:
    """Generate embeddings for a list of texts.

    Args:
        texts: List of text strings to embed.

    Returns:
        NumPy array of shape (len(texts), embedding_dim).
    """
    model = get_model()
    logger.info("Generating embeddings for %d texts", len(texts))
    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        normalize_embeddings=True,  # L2 normalize for cosine similarity
        batch_size=64,
    )
    logger.info("Embeddings generated: shape %s", embeddings.shape)
    return np.array(embeddings, dtype=np.float32)


def generate_query_embedding(query: str) -> np.ndarray:
    """Generate embedding for a single query.

    Args:
        query: Query text.

    Returns:
        NumPy array of shape (1, embedding_dim).
    """
    model = get_model()
    embedding = model.encode(
        [query],
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.array(embedding, dtype=np.float32)
