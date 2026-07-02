"""
Multi-stage ranker for SHL Assessment Recommender.

Takes hybrid retrieval candidates and produces final ranked recommendations
through LLM-based reranking with catalog grounding.

Pipeline:
    Stage 1: Vector + Keyword Retrieval (HybridRetriever)  → top-20
    Stage 2: Rule-based scoring (HybridRetriever boosts)   → scored
    Stage 3: LLM Reranking (this module)                   → top-10
    Stage 4: Output Validation (OutputValidator)            → verified

Design Decision: LLM reranks but doesn't select from scratch.
Improves: Recall@10 (contextual understanding), Hallucination reduction (grounded).
"""

from app.retrieval.hybrid_retriever import RetrievalResult, HiringProfile
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class Ranker:
    """Multi-stage ranking engine.

    Produces the final ordered list of recommendations from retrieval candidates.

    Purpose: Bridge between retrieval (recall-optimized) and output (precision-optimized).
    Alternatives considered: Pure LLM selection (hallucination risk), pure rule-based (no context).
    Trade-offs: LLM reranking adds ~2s latency but significantly improves relevance.
    """

    def rank(
        self,
        candidates: list[RetrievalResult],
        profile: HiringProfile,
        max_results: int = 10,
    ) -> list[RetrievalResult]:
        """Rank candidates and return top results.

        For now, uses the combined score from hybrid retrieval.
        The LLM reranking happens in the recommender agent which has LLM access.

        Args:
            candidates: Pre-scored candidates from hybrid retrieval.
            profile: Hiring profile for context.
            max_results: Maximum results to return.

        Returns:
            Top-ranked candidates.
        """
        # Already sorted by final_score from hybrid retrieval
        # Apply additional heuristics

        ranked = list(candidates)

        # Penalize items the user explicitly asked to remove
        if profile.remove_requests:
            for candidate in ranked:
                for remove_term in profile.remove_requests:
                    if remove_term.lower() in candidate.item.name.lower():
                        candidate.final_score -= 10.0  # Effectively removes it
                        logger.debug("Removed '%s' per user request", candidate.item.name)

        # Re-sort after adjustments
        ranked.sort(key=lambda x: x.final_score, reverse=True)

        # Filter out negative scores (removed items)
        ranked = [r for r in ranked if r.final_score > 0]

        # Cap at max_results
        results = ranked[:max_results]

        logger.info(
            "Ranking complete: %d candidates -> %d results",
            len(candidates),
            len(results),
        )

        return results

    def format_candidates_for_llm(
        self,
        candidates: list[RetrievalResult],
        limit: int = 15,
    ) -> str:
        """Format candidates as text for LLM reranking prompt.

        Args:
            candidates: Retrieved candidates.
            limit: Max candidates to include in prompt.

        Returns:
            Formatted text block with candidate details.
        """
        lines: list[str] = []
        for i, candidate in enumerate(candidates[:limit], 1):
            item = candidate.item
            lines.append(
                f"{i}. {item.name}\n"
                f"   URL: {item.link}\n"
                f"   Type: {item.test_type_code} ({', '.join(item.keys)})\n"
                f"   Description: {item.description[:200]}\n"
                f"   Duration: {item.duration or 'N/A'}\n"
                f"   Levels: {', '.join(item.job_levels)}\n"
                f"   Score: {candidate.final_score:.4f}"
            )
        return "\n\n".join(lines)
