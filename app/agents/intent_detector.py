"""
Intent detector for SHL Assessment Recommender.

Classifies the intent of the latest user message using a combination
of deterministic rules and LLM fallback.

Design Decision: Rules first, LLM fallback. Reduces non-determinism.
Improves: Behavior probes (deterministic refusal), Performance (no LLM for simple cases).
"""

import re
from enum import Enum
from typing import Optional

from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class Intent(str, Enum):
    """Classified intent of a user message.

    Each intent maps to a specific agent action in the ConversationPlanner.
    """

    GREETING = "GREETING"
    SEARCH = "SEARCH"
    CLARIFICATION_RESPONSE = "CLARIFICATION_RESPONSE"
    REFINEMENT = "REFINEMENT"
    COMPARISON = "COMPARISON"
    CONFIRMATION = "CONFIRMATION"
    OFF_TOPIC = "OFF_TOPIC"
    INJECTION = "INJECTION"
    GOODBYE = "GOODBYE"


# ─── Deterministic Patterns ──────────────────────────────────────────────────
# These patterns catch clear-cut intents without needing an LLM call.
# Improves: Determinism, Performance, Behavior probes.

GREETING_PATTERNS = [
    r"^(hi|hello|hey|greetings|good\s*(morning|afternoon|evening)|howdy)\b",
    r"^(what can you|how can you|what do you)\s+(do|help)",
]

GOODBYE_PATTERNS = [
    r"\b(bye|goodbye|thank\s*you|thanks|that'?s?\s*(all|it|enough|perfect|great|good))\b",
    r"\b(no\s*more|we'?re?\s*done|locking\s*it\s*in|confirmed?|perfect|looks?\s*good)\b",
    r"\b(that'?s?\s*what\s*we\s*need)\b",
]

COMPARISON_PATTERNS = [
    r"\b(compare|comparison|difference|differ|versus|vs\.?)\b",
    r"\bwhat.{0,20}(difference|differ)\b",
    r"\bhow.{0,10}(differ|compare)\b",
    r"\b(between .+ and .+)\b",
]

REFINEMENT_PATTERNS = [
    r"\b(add|remove|drop|include|exclude|replace|swap|change|update|instead|actually)\b",
    r"\b(also\s+(add|include|need))\b",
    r"\b(don'?t\s*(need|want|include))\b",
    r"\b(keep|keep\s*it|locking|lock\s*it)\b",
]

OFF_TOPIC_PATTERNS = [
    r"\b(weather|sports|news|politics|recipe|joke|story|movie|music|game)\b",
    r"\b(who\s*(are\s*you|made\s*you|created\s*you|built\s*you))\b",
    r"\b(what\s*is\s*the\s*(meaning|purpose)\s*of\s*life)\b",
    r"\b(write\s*(me\s*)?(a\s*)?(poem|essay|story|code|script))\b",
    r"\b(legal\s*(advice|question|issue)|lawsuit|sue|attorney|lawyer)\b",
    r"\b(salary|compensation|pay|benefits|negotiate|offer)\b",
]

INJECTION_PATTERNS = [
    r"\b(ignore\s*(previous|all|your|above)\s*(instructions?|rules?|prompts?|directives?))\b",
    r"\b(you\s*are\s*now|act\s*as|pretend\s*to\s*be|role\s*play)\b",
    r"\b(system\s*prompt|reveal\s*(your|the)\s*(instructions?|prompt|rules))\b",
    r"\b(jailbreak|bypass|override|hack)\b",
    r"\b(DAN|do\s*anything\s*now)\b",
    r"```.*system.*```",
]

CONFIRMATION_PATTERNS = [
    r"^(yes|yeah|yep|sure|ok|okay|correct|right|exactly|confirmed?)\b",
    r"\b(that'?s?\s*(right|correct|perfect|great|good|what\s*we\s*need))\b",
    r"\b(looks?\s*good|sounds?\s*good|perfect|we'?ll\s*go\s*with)\b",
    r"\b(keep\s*(it|them|verify|the))\b",
]


def _match_patterns(text: str, patterns: list[str]) -> bool:
    """Check if text matches any pattern in the list."""
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def detect_intent_deterministic(
    message: str,
    conversation_history: list[dict[str, str]],
) -> Optional[Intent]:
    """Detect intent using deterministic pattern matching.

    Tries rules first. Returns None if no clear match (needs LLM).

    Args:
        message: Latest user message text.
        conversation_history: Full conversation history.

    Returns:
        Detected Intent or None if uncertain.
    """
    text = message.strip()

    # Check injection FIRST (highest priority)
    if _match_patterns(text, INJECTION_PATTERNS):
        logger.info("Deterministic intent: INJECTION")
        return Intent.INJECTION

    # Check off-topic
    if _match_patterns(text, OFF_TOPIC_PATTERNS):
        # Only if no assessment-related content
        assessment_keywords = [
            "assess", "test", "hire", "hiring", "candidate", "role",
            "developer", "engineer", "manager", "skill", "shl", "opq",
            "verify", "personality", "cognitive", "ability",
        ]
        has_assessment_context = any(kw in text.lower() for kw in assessment_keywords)
        if not has_assessment_context:
            logger.info("Deterministic intent: OFF_TOPIC")
            return Intent.OFF_TOPIC

    # Check if this is the first message (greeting or initial search)
    is_first_message = len(conversation_history) <= 1

    if is_first_message:
        if _match_patterns(text, GREETING_PATTERNS) and len(text.split()) < 10:
            logger.info("Deterministic intent: GREETING")
            return Intent.GREETING
        else:
            # First substantial message is always a search
            logger.info("Deterministic intent: SEARCH (first message)")
            return Intent.SEARCH

    # Check comparison (before refinement, as "compare" is more specific)
    if _match_patterns(text, COMPARISON_PATTERNS):
        logger.info("Deterministic intent: COMPARISON")
        return Intent.COMPARISON

    # Check refinement
    if _match_patterns(text, REFINEMENT_PATTERNS):
        # Distinguish between refinement and confirmation
        # "keep Verify G+" is confirmation, "add AWS" is refinement
        if _match_patterns(text, CONFIRMATION_PATTERNS) and not any(
            kw in text.lower() for kw in ["add", "remove", "drop", "instead", "also", "change", "update"]
        ):
            logger.info("Deterministic intent: CONFIRMATION")
            return Intent.CONFIRMATION
        logger.info("Deterministic intent: REFINEMENT")
        return Intent.REFINEMENT

    # Check goodbye/confirmation
    if _match_patterns(text, GOODBYE_PATTERNS):
        logger.info("Deterministic intent: GOODBYE")
        return Intent.GOODBYE

    if _match_patterns(text, CONFIRMATION_PATTERNS):
        logger.info("Deterministic intent: CONFIRMATION")
        return Intent.CONFIRMATION

    # No clear match — return None for LLM fallback
    return None


class IntentDetector:
    """Hybrid intent detector: deterministic rules + LLM fallback.

    Purpose: Classify user intent for routing by ConversationPlanner.
    Alternatives: Pure LLM (non-deterministic), pure rules (misses nuance).
    Trade-offs: Rules handle 80% of cases instantly; LLM handles edge cases.
    Failure modes: Misclassification → wrong agent action. Mitigated by
    ConversationPlanner's additional checks.
    """

    def detect(
        self,
        message: str,
        conversation_history: list[dict[str, str]],
        llm_intent: Optional[str] = None,
    ) -> Intent:
        """Detect intent of the latest user message.

        Args:
            message: Latest user message.
            conversation_history: Full conversation history.
            llm_intent: Optional LLM-classified intent as fallback.

        Returns:
            Classified Intent enum value.
        """
        # Try deterministic detection first
        intent = detect_intent_deterministic(message, conversation_history)

        if intent is not None:
            return intent

        # Use LLM classification if provided
        if llm_intent:
            try:
                intent = Intent(llm_intent.strip().upper())
                logger.info("LLM intent: %s", intent.value)
                return intent
            except ValueError:
                logger.warning("Invalid LLM intent: %s", llm_intent)

        # Default: treat as clarification response (user answering our question)
        # This is the safest default for mid-conversation messages
        if len(conversation_history) > 1:
            logger.info("Default intent: CLARIFICATION_RESPONSE")
            return Intent.CLARIFICATION_RESPONSE

        logger.info("Default intent: SEARCH")
        return Intent.SEARCH
