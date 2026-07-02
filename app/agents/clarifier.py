"""
Clarification agent for SHL Assessment Recommender.

Generates targeted clarification questions based on missing profile fields.
Follows the Clarification Policy: required fields first, high-value next,
never more than MAX_CLARIFICATION_ROUNDS.

Design Decision: Prioritized clarification with deterministic field selection.
Improves: Behavior probes (asks before recommending), Recall@10 (better queries).
"""

from app.retrieval.hybrid_retriever import HiringProfile
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

# ─── Clarification Templates ─────────────────────────────────────────────────
# Pre-written questions for each missing field. Avoids LLM overhead for common cases.

CLARIFICATION_QUESTIONS: dict[str, str] = {
    "role": "What role or position are you hiring for?",
    "seniority": "What seniority level is this role? (Entry-level, mid-career, senior, executive?)",
    "skills": "What are the key skills or competencies required for this role?",
    "purpose": "Is this assessment for hiring/selection, development, or screening?",
    "domain": "What domain or functional area does this role fall under?",
    "industry": "What industry is your organization in?",
    "language": "Do you need assessments in a specific language?",
    "assessment_type": "Are you looking for a specific type of assessment? (Technical knowledge, personality, cognitive ability, etc.)",
}


class Clarifier:
    """Generates clarification questions based on missing profile information.

    Purpose: Ensure we have enough context before recommending.
    Alternatives: Always recommend immediately (fails behavior probes).
    Trade-offs: Extra turn for clarification vs. potentially irrelevant recommendations.
    Failure modes: Over-questioning (mitigated by MAX_CLARIFICATION_ROUNDS).
    """

    def get_clarification_question(
        self,
        profile: HiringProfile,
        missing_fields: list[str],
        turn_count: int,
    ) -> str:
        """Get the highest-priority clarification question.

        Priority order:
        1. Role/JD (REQUIRED, weight 0.4)
        2. Seniority (HIGH VALUE, weight 0.2)
        3. Skills (HIGH VALUE, weight 0.2)
        4. Purpose (OPTIONAL, weight 0.05)

        Args:
            profile: Current hiring profile.
            missing_fields: List of missing field descriptions.
            turn_count: Current turn count.

        Returns:
            Clarification question string.
        """
        # Priority 1: Role/JD
        if not profile.role and not profile.raw_jd:
            return CLARIFICATION_QUESTIONS["role"]

        # Priority 2: Seniority
        if not profile.seniority:
            return CLARIFICATION_QUESTIONS["seniority"]

        # Priority 3: Skills
        if not profile.skills and not profile.domain:
            return CLARIFICATION_QUESTIONS["skills"]

        # Priority 4: Purpose (only if we have time)
        if not profile.purpose and turn_count < 4:
            return CLARIFICATION_QUESTIONS["purpose"]

        # Fallback: generic question
        return "Could you provide more details about the role requirements to help me narrow down the best assessments?"

    def format_greeting_response(self) -> str:
        """Generate a greeting response with initial clarification.

        Returns:
            Greeting + clarification question.
        """
        return (
            "Hello! I'm the SHL Assessment Recommender. I can help you find "
            "the right assessments for your hiring needs. To get started, "
            "could you tell me about the role you're looking to fill?"
        )
