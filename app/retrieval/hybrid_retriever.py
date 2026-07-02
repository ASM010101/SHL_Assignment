"""
Hybrid retriever for SHL Assessment Recommender.

Combines semantic (FAISS) and keyword (TF-IDF) search with rule-based boosting
and metadata filtering for maximum Recall@10.

Design Decision: Hybrid retrieval with Reciprocal Rank Fusion (RRF).
Improves: Recall@10 (combines strengths of both retrieval methods).

Why RRF? It's robust to score scale differences between FAISS (cosine 0-1)
and TF-IDF (different scale). RRF normalizes via rank position.
"""

from dataclasses import dataclass, field
from typing import Optional

from app.catalog.loader import CatalogItem, CatalogStore
from app.retrieval.embeddings import generate_query_embedding
from app.retrieval.keyword_search import KeywordSearchEngine
from app.retrieval.vector_store import VectorStore
from app.utils.helpers import rewrite_query
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieval result with scores from all stages.

    Attributes:
        item: The catalog item.
        semantic_score: FAISS cosine similarity score.
        keyword_score: TF-IDF similarity score.
        boost_score: Rule-based boosting score.
        final_score: Combined score after fusion.
        semantic_rank: Rank in semantic results.
        keyword_rank: Rank in keyword results.
    """

    item: CatalogItem
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    boost_score: float = 0.0
    final_score: float = 0.0
    semantic_rank: int = 999
    keyword_rank: int = 999


@dataclass
class HiringProfile:
    """Structured representation of accumulated hiring requirements.

    Extracted from conversation history by the ConversationAnalyzer.
    Used to drive retrieval, filtering, and ranking decisions.

    Design Decision: Explicit structured profile vs. raw text.
    Improves: Recall@10 (targeted retrieval), Coherence (consistent state).
    """

    role: Optional[str] = None
    seniority: Optional[str] = None
    skills: list[str] = field(default_factory=list)
    soft_skills: list[str] = field(default_factory=list)
    domain: Optional[str] = None
    industry: Optional[str] = None
    languages: list[str] = field(default_factory=list)
    purpose: Optional[str] = None
    assessment_types: list[str] = field(default_factory=list)
    job_level: Optional[str] = None
    constraints: list[str] = field(default_factory=list)
    raw_jd: Optional[str] = None
    confidence: float = 0.0
    # Track what the user has explicitly added/removed
    add_requests: list[str] = field(default_factory=list)
    remove_requests: list[str] = field(default_factory=list)

    def has_minimum_info(self) -> bool:
        """Check if we have enough info to make recommendations."""
        return self.confidence >= 0.5

    def to_query(self) -> str:
        """Convert profile to a search query string."""
        parts: list[str] = []
        if self.role:
            parts.append(self.role)
        if self.seniority:
            parts.append(self.seniority)
        parts.extend(self.skills)
        parts.extend(self.soft_skills)
        if self.domain:
            parts.append(self.domain)
        if self.industry:
            parts.append(self.industry)
        if self.purpose:
            parts.append(self.purpose)
        parts.extend(self.assessment_types)
        if self.raw_jd:
            parts.append(self.raw_jd)
        return " ".join(parts)

    def compute_confidence(self) -> float:
        """Compute confidence score based on filled fields.

        Weights:
            role_or_purpose: 0.4 (REQUIRED)
            seniority: 0.2 (HIGH VALUE)
            skills: 0.2 (HIGH VALUE)
            domain: 0.05
            industry: 0.05
            language: 0.05
            assessment_type: 0.05
        """
        score = 0.0
        if self.role or self.purpose or self.raw_jd:
            score += 0.4
        if self.seniority:
            score += 0.2
        if self.skills:
            score += 0.2
        if self.domain:
            score += 0.05
        if self.industry:
            score += 0.05
        if self.languages:
            score += 0.05
        if self.assessment_types:
            score += 0.05
        self.confidence = min(score, 1.0)
        return self.confidence


# ─── Seniority Mapping ───────────────────────────────────────────────────────
# Maps user-described seniority to catalog job_levels for filtering.

SENIORITY_TO_JOB_LEVELS: dict[str, list[str]] = {
    "entry": ["Entry-Level", "Graduate", "General Population"],
    "junior": ["Entry-Level", "Graduate"],
    "graduate": ["Graduate", "Entry-Level"],
    "mid": ["Mid-Professional", "Professional Individual Contributor"],
    "senior": ["Mid-Professional", "Professional Individual Contributor", "Manager"],
    "lead": ["Manager", "Front Line Manager", "Mid-Professional"],
    "manager": ["Manager", "Front Line Manager", "Supervisor"],
    "director": ["Director", "Executive", "Manager"],
    "executive": ["Executive", "Director"],
    "vp": ["Executive", "Director"],
    "cxo": ["Executive", "Director"],
    "c-suite": ["Executive", "Director"],
}


class HybridRetriever:
    """Hybrid retrieval combining semantic + keyword + rule-based boosting.

    Pipeline:
        1. Query rewriting (expand user terms)
        2. Semantic search (FAISS, top-30)
        3. Keyword search (TF-IDF, top-20)
        4. Reciprocal Rank Fusion (combine results)
        5. Rule-based boosting (metadata matching)
        6. Return top-k candidates

    Purpose: Maximize recall by combining complementary retrieval methods.
    Alternatives: Semantic-only (misses exact matches), keyword-only (misses intent).
    Complexity: O(n log n) for sorting. Negligible for 377 items.
    Scalability: FAISS handles millions. TF-IDF may need Elasticsearch at scale.
    """

    def __init__(
        self,
        catalog: CatalogStore,
        vector_store: VectorStore,
        keyword_engine: KeywordSearchEngine,
    ) -> None:
        self.catalog = catalog
        self.vector_store = vector_store
        self.keyword_engine = keyword_engine

    def retrieve(
        self,
        profile: HiringProfile,
        top_k: int = 20,
        semantic_k: int = 30,
        keyword_k: int = 20,
    ) -> list[RetrievalResult]:
        """Retrieve candidate assessments using hybrid search.

        Args:
            profile: Structured hiring profile with constraints.
            top_k: Final number of candidates to return.
            semantic_k: Number of semantic search results.
            keyword_k: Number of keyword search results.

        Returns:
            List of RetrievalResult sorted by final_score descending.
        """
        # Step 1: Build query from profile
        raw_query = profile.to_query()
        if not raw_query.strip():
            logger.warning("Empty query from profile, returning empty results")
            return []

        # Step 2: Rewrite query for better retrieval
        expanded_query = rewrite_query(raw_query)
        logger.debug("Retrieval query: '%s'", expanded_query[:100])

        # Step 3: Semantic search
        query_embedding = generate_query_embedding(expanded_query)
        semantic_results = self.vector_store.search(query_embedding, top_k=semantic_k)

        # Step 4: Keyword search
        keyword_results = self.keyword_engine.search(expanded_query, top_k=keyword_k)

        # Step 5: Merge with Reciprocal Rank Fusion
        candidates = self._reciprocal_rank_fusion(semantic_results, keyword_results)

        # Step 6: Apply rule-based boosting
        self._apply_boosts(candidates, profile)

        # Step 7: Sort by final score
        candidates.sort(key=lambda x: x.final_score, reverse=True)

        # Step 8: Return top-k
        results = candidates[:top_k]

        logger.info(
            "Hybrid retrieval: query_len=%d, semantic=%d, keyword=%d, merged=%d, returning=%d",
            len(raw_query),
            len(semantic_results),
            len(keyword_results),
            len(candidates),
            len(results),
        )

        return results

    def _reciprocal_rank_fusion(
        self,
        semantic_results: list[tuple[int, float]],
        keyword_results: list[tuple[int, float]],
        k: int = 60,
    ) -> list[RetrievalResult]:
        """Combine results using Reciprocal Rank Fusion (RRF).

        RRF score = sum(1 / (k + rank_i)) for each result list.
        This is robust to score scale differences between retrieval methods.

        Args:
            semantic_results: (index, score) from FAISS.
            keyword_results: (index, score) from TF-IDF.
            k: RRF constant (default 60, standard value).

        Returns:
            Merged list of RetrievalResults.
        """
        result_map: dict[int, RetrievalResult] = {}

        # Process semantic results
        for rank, (idx, score) in enumerate(semantic_results):
            if idx < 0 or idx >= len(self.catalog.items):
                continue
            item = self.catalog.items[idx]
            result = RetrievalResult(
                item=item,
                semantic_score=score,
                semantic_rank=rank + 1,
            )
            result.final_score = 1.0 / (k + rank + 1)
            result_map[idx] = result

        # Process keyword results
        for rank, (idx, score) in enumerate(keyword_results):
            if idx < 0 or idx >= len(self.catalog.items):
                continue
            if idx in result_map:
                result_map[idx].keyword_score = score
                result_map[idx].keyword_rank = rank + 1
                result_map[idx].final_score += 1.0 / (k + rank + 1)
            else:
                item = self.catalog.items[idx]
                result = RetrievalResult(
                    item=item,
                    keyword_score=score,
                    keyword_rank=rank + 1,
                )
                result.final_score = 1.0 / (k + rank + 1)
                result_map[idx] = result

        return list(result_map.values())

    def _apply_boosts(
        self,
        candidates: list[RetrievalResult],
        profile: HiringProfile,
    ) -> None:
        """Apply rule-based score boosts based on profile matching.

        Boosts improve ranking for items that match structured profile fields.
        This is deterministic and interpretable.

        Args:
            candidates: List of candidates to boost in-place.
            profile: Hiring profile with constraints.
        """
        for candidate in candidates:
            boost = 0.0
            item = candidate.item

            # Boost 1: Skill name mentioned in assessment name (+0.3)
            name_lower = item.name.lower()
            for skill in profile.skills:
                if skill.lower() in name_lower:
                    boost += 0.3
                    break

            # Boost 2: Job level matches seniority (+0.2)
            if profile.seniority:
                seniority_key = profile.seniority.lower().split()[0] if profile.seniority else ""
                matching_levels = SENIORITY_TO_JOB_LEVELS.get(seniority_key, [])
                if matching_levels:
                    if any(level in item.job_levels for level in matching_levels):
                        boost += 0.2
                    elif item.job_levels:
                        boost -= 0.1  # Slight penalty for mismatch

            # Boost 3: Assessment type matches (+0.2)
            if profile.assessment_types:
                for atype in profile.assessment_types:
                    atype_lower = atype.lower()
                    for key in item.keys:
                        if atype_lower in key.lower():
                            boost += 0.2
                            break

            # Boost 4: Language matches (+0.1)
            if profile.languages:
                for lang in profile.languages:
                    if any(lang.lower() in l.lower() for l in item.languages):
                        boost += 0.1
                        break

            # Boost 5: Domain match in description (+0.15)
            if profile.domain:
                if profile.domain.lower() in item.description.lower():
                    boost += 0.15

            # Boost 6: Explicitly requested additions (+0.5)
            for add_req in profile.add_requests:
                if add_req.lower() in name_lower or add_req.lower() in item.description.lower():
                    boost += 0.5
                    break

            candidate.boost_score = boost
            candidate.final_score += boost
