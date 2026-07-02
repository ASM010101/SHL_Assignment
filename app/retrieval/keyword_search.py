"""
TF-IDF keyword search for SHL Assessment Recommender.

Provides BM25-style keyword matching as complement to semantic search.
Catches exact keyword matches that semantic search may miss.

Design Decision: TF-IDF over BM25 for simplicity with scikit-learn.
Improves: Recall@10 (catches exact matches like "Java 8", "OPQ32r").
Alternatives considered: BM25 via rank_bm25 (marginal improvement, extra dependency).
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class KeywordSearchEngine:
    """TF-IDF based keyword search engine.

    Complements semantic search by catching exact term matches.
    For example: "Java 8" matches the product named "Java 8 (New)" even if
    semantic embedding doesn't rank it highest.

    Attributes:
        vectorizer: Fitted TF-IDF vectorizer.
        tfidf_matrix: TF-IDF matrix of all catalog texts.
    """

    def __init__(self) -> None:
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix = None
        self.count: int = 0

    def build(self, texts: list[str]) -> None:
        """Build TF-IDF index from catalog texts.

        Args:
            texts: List of catalog retrieval texts.
        """
        self.count = len(texts)
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words="english",
            ngram_range=(1, 2),  # Unigrams + bigrams for phrase matching
            sublinear_tf=True,   # Log-scaled term frequency (like BM25)
            min_df=1,
            max_df=0.95,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        logger.info(
            "TF-IDF index built: %d documents, %d features",
            self.count,
            self.tfidf_matrix.shape[1],
        )

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """Search for matching documents using TF-IDF cosine similarity.

        Args:
            query: Search query text.
            top_k: Number of results to return.

        Returns:
            List of (index, score) tuples, sorted by descending similarity.
        """
        if self.vectorizer is None or self.tfidf_matrix is None:
            return []

        top_k = min(top_k, self.count)
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [
            (int(idx), float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0.0
        ]

        logger.debug(
            "Keyword search: query='%s', hits=%d, best_score=%.4f",
            query[:50],
            len(results),
            results[0][1] if results else 0.0,
        )
        return results
