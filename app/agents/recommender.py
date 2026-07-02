"""
Recommender agent for SHL Assessment Recommender.

Generates assessment recommendations using:
1. Hybrid retrieval (semantic + keyword + boosting)
2. LLM reranking with context
3. Output validation

Design Decision: Retrieval → Rank → LLM select → Validate pipeline.
Improves: Recall@10 (multi-signal retrieval), Hallucination reduction (grounded).
"""

import json
from typing import Optional

from app.catalog.loader import CatalogStore
from app.prompts.templates import RECOMMENDATION_PROMPT, REFINEMENT_PROMPT
from app.retrieval.hybrid_retriever import HiringProfile, HybridRetriever, RetrievalResult
from app.retrieval.ranker import Ranker
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class Recommender:
    """Generates assessment recommendations.

    Pipeline:
        1. Retrieve candidates via HybridRetriever
        2. Rank candidates via Ranker
        3. Format candidates for LLM reranking
        4. Parse LLM selection
        5. Validate against catalog

    Purpose: Bridge between retrieval (recall) and output (precision).
    Alternatives: Direct LLM generation (hallucination risk), pure retrieval (no context).
    Trade-offs: Extra LLM call for reranking adds ~2s. Worth it for relevance.
    """

    def __init__(
        self,
        catalog: CatalogStore,
        retriever: HybridRetriever,
        ranker: Ranker,
    ) -> None:
        self.catalog = catalog
        self.retriever = retriever
        self.ranker = ranker

    def get_recommendation_prompt(
        self,
        conversation: str,
        profile: HiringProfile,
        candidates: list[RetrievalResult],
    ) -> str:
        """Generate the recommendation prompt for the LLM.

        Args:
            conversation: Formatted conversation text.
            profile: Hiring profile.
            candidates: Retrieved and ranked candidates.

        Returns:
            Formatted recommendation prompt.
        """
        candidates_text = self.ranker.format_candidates_for_llm(candidates, limit=15)

        return RECOMMENDATION_PROMPT.format(
            conversation=conversation,
            role=profile.role or "Not specified",
            seniority=profile.seniority or "Not specified",
            skills=", ".join(profile.skills) if profile.skills else "Not specified",
            soft_skills=", ".join(profile.soft_skills) if profile.soft_skills else "Not specified",
            domain=profile.domain or "Not specified",
            purpose=profile.purpose or "Not specified",
            candidates=candidates_text,
        )

    def retrieve_candidates(
        self,
        profile: HiringProfile,
        top_k: int = 20,
    ) -> list[RetrievalResult]:
        """Retrieve and rank candidate assessments.

        Args:
            profile: Hiring profile with constraints.
            top_k: Number of candidates to retrieve.

        Returns:
            Ranked list of retrieval results.
        """
        # Step 1: Hybrid retrieval
        candidates = self.retriever.retrieve(profile, top_k=top_k)

        # Step 2: Apply ranking
        ranked = self.ranker.rank(candidates, profile, max_results=top_k)

        logger.info("Retrieved %d candidates for profile", len(ranked))
        return ranked

    def parse_llm_recommendations(
        self,
        llm_response: str,
        candidates: list[RetrievalResult],
    ) -> tuple[str, list[dict[str, str]]]:
        """Parse LLM response into reply + recommendations.

        Args:
            llm_response: Raw LLM response text.
            candidates: Available candidates to select from.

        Returns:
            Tuple of (reply_text, recommendations_list).
        """
        try:
            # Clean response
            cleaned = llm_response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()

            data = json.loads(cleaned)
            reply = data.get("reply", "Here are my recommended assessments.")
            selected_indices = data.get("selected_indices", [])

            recommendations: list[dict[str, str]] = []
            for idx in selected_indices:
                # Convert 1-indexed to 0-indexed
                actual_idx = idx - 1
                if 0 <= actual_idx < len(candidates):
                    item = candidates[actual_idx].item
                    recommendations.append({
                        "name": item.name,
                        "url": item.link,
                        "test_type": item.test_type_code,
                    })

            # Ensure at least 1 recommendation if candidates exist
            if not recommendations and candidates:
                # Fallback: use top-ranked candidates
                for candidate in candidates[:5]:
                    recommendations.append({
                        "name": candidate.item.name,
                        "url": candidate.item.link,
                        "test_type": candidate.item.test_type_code,
                    })
                reply = "Based on the requirements, here are the assessments I recommend."

            return reply, recommendations

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Failed to parse LLM recommendation: %s", e)
            # Fallback: return top candidates
            recommendations = []
            for candidate in candidates[:5]:
                recommendations.append({
                    "name": candidate.item.name,
                    "url": candidate.item.link,
                    "test_type": candidate.item.test_type_code,
                })
            return "Based on the requirements, here are the assessments I recommend.", recommendations

    def get_refinement_prompt(
        self,
        conversation: str,
        current_recs: list[dict[str, str]],
        change_request: str,
        new_candidates: list[RetrievalResult],
    ) -> str:
        """Generate the refinement prompt for the LLM.

        Args:
            conversation: Formatted conversation text.
            current_recs: Current recommendation list.
            change_request: User's change request.
            new_candidates: New candidates retrieved for additions.

        Returns:
            Formatted refinement prompt.
        """
        current_text = "\n".join(
            f"- {r['name']} ({r['test_type']}): {r['url']}"
            for r in current_recs
        )

        new_candidates_text = self.ranker.format_candidates_for_llm(new_candidates, limit=10)

        return REFINEMENT_PROMPT.format(
            conversation=conversation,
            current_recommendations=current_text,
            change_request=change_request,
            new_candidates=new_candidates_text,
        )

    def parse_refinement_response(
        self,
        llm_response: str,
        new_candidates: list[RetrievalResult],
        current_recs: list[dict[str, str]],
    ) -> tuple[str, list[dict[str, str]]]:
        """Parse LLM refinement response.

        Args:
            llm_response: Raw LLM response.
            new_candidates: New candidates list.
            current_recs: Current recommendations.

        Returns:
            Tuple of (reply_text, updated_recommendations).
        """
        try:
            cleaned = llm_response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()

            data = json.loads(cleaned)
            reply = data.get("reply", "I've updated the recommendations.")

            # Build updated list
            recommendations: list[dict[str, str]] = []

            # Keep specified items from previous
            kept_names = set(data.get("kept_from_previous", []))
            for rec in current_recs:
                if rec["name"] in kept_names:
                    recommendations.append(rec)

            # Add new selections
            selected_indices = data.get("selected_indices", [])
            for idx in selected_indices:
                actual_idx = idx - 1
                if 0 <= actual_idx < len(new_candidates):
                    item = new_candidates[actual_idx].item
                    # Don't add duplicates
                    if not any(r["url"] == item.link for r in recommendations):
                        recommendations.append({
                            "name": item.name,
                            "url": item.link,
                            "test_type": item.test_type_code,
                        })

            if not recommendations:
                recommendations = current_recs
                reply = "I've kept the current recommendations."

            return reply, recommendations

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Failed to parse refinement response: %s", e)
            return "I've kept the current recommendations.", current_recs
