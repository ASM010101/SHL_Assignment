"""
Conversation planner for SHL Assessment Recommender.

Deterministic routing engine that decides what the agent should do next.
Uses rules first, not the LLM. This ensures predictable behavior.

Design Decision: Deterministic routing with rule-based priority.
Improves: Behavior probes (no vague-turn-1 recommendations), Coherence,
          Determinism (same input → same routing decision).
"""

from enum import Enum

from app.agents.intent_detector import Intent
from app.config import CONFIDENCE_THRESHOLD, MAX_CLARIFICATION_ROUNDS, MAX_TURNS
from app.retrieval.hybrid_retriever import HiringProfile
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class Action(str, Enum):
    """Agent action to take, decided by the planner."""

    GREET_AND_CLARIFY = "GREET_AND_CLARIFY"
    CLARIFY = "CLARIFY"
    RECOMMEND = "RECOMMEND"
    REFINE = "REFINE"
    COMPARE = "COMPARE"
    REFUSE = "REFUSE"
    END_CONVERSATION = "END_CONVERSATION"


class ConversationPlanner:
    """Deterministic conversation planner.

    Decides the next agent action based on:
    - Detected intent
    - Hiring profile confidence
    - Turn count
    - Conversation state

    Purpose: Prevent non-deterministic LLM routing decisions.
    Alternatives: LLM-based routing (unpredictable, may recommend on turn 1).
    Trade-offs: Rigid rules may miss edge cases. LLM handles nuance.
    Failure modes: Wrong intent → wrong action. Mitigated by intent detector accuracy.

    Rule Priority (highest first):
    1. Injection/Off-topic → REFUSE (safety)
    2. Turn cap approaching + enough info → RECOMMEND (don't run out of turns)
    3. Greeting on turn 1 → GREET_AND_CLARIFY
    4. Comparison → COMPARE
    5. Refinement → REFINE
    6. Confirmation/Goodbye with recommendations → END
    7. Enough info → RECOMMEND
    8. Not enough info → CLARIFY
    """

    def plan(
        self,
        intent: Intent,
        profile: HiringProfile,
        turn_count: int,
        has_active_recommendations: bool,
        clarification_count: int = 0,
    ) -> Action:
        """Decide the next agent action.

        Args:
            intent: Classified intent of the latest user message.
            profile: Current hiring profile with confidence score.
            turn_count: Total messages in conversation.
            has_active_recommendations: Whether recommendations were previously given.
            clarification_count: How many clarification rounds have been used.

        Returns:
            Action to take next.
        """
        # Rule 1: Safety first — injection and off-topic always refused
        if intent == Intent.INJECTION:
            logger.info("Planner: REFUSE (injection detected)")
            return Action.REFUSE

        if intent == Intent.OFF_TOPIC:
            logger.info("Planner: REFUSE (off-topic)")
            return Action.REFUSE

        # Rule 2: Turn cap management — force recommendation if running out of turns
        # If we've used 6+ turns and have minimum info, must recommend now
        if turn_count >= MAX_TURNS - 2 and profile.has_minimum_info():
            if not has_active_recommendations:
                logger.info("Planner: RECOMMEND (turn cap approaching, enough info)")
                return Action.RECOMMEND

        # Rule 3: If at max turns, end conversation
        if turn_count >= MAX_TURNS:
            if has_active_recommendations:
                logger.info("Planner: END_CONVERSATION (max turns reached)")
                return Action.END_CONVERSATION
            elif profile.has_minimum_info():
                logger.info("Planner: RECOMMEND (max turns, forcing recommendation)")
                return Action.RECOMMEND
            else:
                # Even with insufficient info, try our best
                logger.info("Planner: RECOMMEND (max turns, best effort)")
                return Action.RECOMMEND

        # Rule 4: Greeting on first message
        if intent == Intent.GREETING and turn_count <= 1:
            logger.info("Planner: GREET_AND_CLARIFY")
            return Action.GREET_AND_CLARIFY

        # Rule 5: Comparison request
        if intent == Intent.COMPARISON:
            logger.info("Planner: COMPARE")
            return Action.COMPARE

        # Rule 6: Refinement request (only if we have active recommendations)
        if intent == Intent.REFINEMENT:
            if has_active_recommendations:
                logger.info("Planner: REFINE")
                return Action.REFINE
            else:
                # Treat as new search info
                logger.info("Planner: CLARIFY (refinement but no active recs)")
                # Update profile and try to recommend
                if profile.has_minimum_info():
                    return Action.RECOMMEND
                return Action.CLARIFY

        # Rule 7: Confirmation or goodbye with active recommendations → end
        if intent in (Intent.CONFIRMATION, Intent.GOODBYE):
            if has_active_recommendations:
                logger.info("Planner: END_CONVERSATION (user confirmed)")
                return Action.END_CONVERSATION
            elif profile.has_minimum_info():
                # User said "yes" but we haven't recommended yet — recommend now
                logger.info("Planner: RECOMMEND (user confirmed, providing recommendations)")
                return Action.RECOMMEND

        # Rule 8: Enough info accumulated → recommend
        if profile.has_minimum_info():
            if intent in (Intent.SEARCH, Intent.CLARIFICATION_RESPONSE) or not has_active_recommendations:
                logger.info("Planner: RECOMMEND (confidence=%.2f >= %.2f)", profile.confidence, CONFIDENCE_THRESHOLD)
                return Action.RECOMMEND

        # Rule 9: Exhausted clarification rounds → recommend with what we have
        if clarification_count >= MAX_CLARIFICATION_ROUNDS:
            if profile.role or profile.skills or profile.raw_jd:
                logger.info("Planner: RECOMMEND (max clarifications reached)")
                return Action.RECOMMEND

        # Rule 10: Default — ask for more info
        logger.info("Planner: CLARIFY (confidence=%.2f < %.2f)", profile.confidence, CONFIDENCE_THRESHOLD)
        return Action.CLARIFY
