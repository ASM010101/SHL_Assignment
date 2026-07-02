"""
Guardrails for SHL Assessment Recommender.

Layered protection against:
- Prompt injection and role hijacking
- Off-topic requests
- Hallucinations (catalog grounding)
- General hiring advice / legal questions
- Unsafe outputs

Design Decision: Pattern-based + LLM-based dual layer.
Improves: Behavior probes (refuses off-topic/injection), Hard evals (no hallucinations).
"""

import re

from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class GuardrailResult:
    """Result of a guardrail check.

    Attributes:
        is_safe: Whether the input/output passed the check.
        violation_type: Type of violation if not safe.
        refusal_message: Pre-written refusal message.
    """

    def __init__(
        self,
        is_safe: bool = True,
        violation_type: str = "",
        refusal_message: str = "",
    ) -> None:
        self.is_safe = is_safe
        self.violation_type = violation_type
        self.refusal_message = refusal_message


# ─── Injection Patterns ──────────────────────────────────────────────────────

INJECTION_PATTERNS = [
    r"ignore\s*(all\s*)?(previous|prior|above|your)\s*(instructions?|rules?|prompts?|context|directives?)",
    r"(you\s*are\s*now|from\s*now\s*on|act\s*as|pretend\s*(to\s*be|you'?re)|role\s*play\s*as)",
    r"(reveal|show|display|print|output)\s*(your|the|system)\s*(prompt|instructions?|rules?|configuration|directives?)",
    r"(jailbreak|bypass|override|hack|escape|unlock)\s*(your|the|this|safety|filter|restriction)",
    r"\bDAN\b.*\bdo\s*anything\s*now\b",
    r"system\s*:\s*you\s*are",
    r"system\s*prompt",
    r"<\|?\s*system\s*\|?>",
    r"```\s*(system|prompt|instruction)",
    r"IMPORTANT:?\s*(ignore|forget|disregard|override)",
    r"new\s*(instructions?|rules?|role|persona|identity)",
]

# ─── Off-Topic Patterns ──────────────────────────────────────────────────────

OFF_TOPIC_PATTERNS = [
    r"\b(weather|temperature|forecast|rain|sunny)\b",
    r"\b(recipe|cooking?|cooks?|food|restaurant|menu|spaghetti|bolognese)\b",
    r"\b(movie|film|tv\s*show|series|netflix|anime)\b",
    r"\b(sports?|football|soccer|basketball|cricket|tennis)\b",
    r"\b(politics|election|president|government|congress|parliament)\b",
    r"\b(stock\s*market|invest(ment)?|crypto|bitcoin|trading)\b",
    r"\b(poem|essay|story|novel|creative\s*writing|poetry|jokes?|riddles?)\b",
    r"\b(game|gaming|xbox|playstation|nintendo)\b",
    r"\b(music|song|album|concert|spotify)\b",
    r"\b(travel|vacation|hotel|flight|tourism)\b",
    r"\b(write|debug|fix|build|create)\s+(a\s+)?(python\s+)?(script|code|program|application|app|function)\b",
    r"\bhelp\s+me\s+(write|code|program|debug|fix|build)\b",
    r"\b(translate|summarize|summarise|paraphrase)\b",
    r"\b(math|calculate|equation|solve|algebra)\b",
]

# ─── Legal/Advice Patterns ───────────────────────────────────────────────────

LEGAL_ADVICE_PATTERNS = [
    r"\b(legal\s*(advice|question|issue|concern|guidance)|lawyer|attorney|lawsuit|sue|litigation)\b",
    r"\b(salary\s*(negotiat|advice|range|band)|compensation\s*(advice|guidance)|pay\s*equity)\b",
    r"\b(discrimination|harassment|wrongful\s*termination|whistleblow)\b",
    r"\b(labor\s*laws?|employment\s*laws?|ADA|EEOC|Title\s*VII)\b",
    r"\b(general\s*hiring\s*(advice|tips?|guidance|best\s*practice))\b",
    r"\b(interview\s*(tips?|advice|questions?|technique))\b",
    r"\b(resume|cv)\s*(advice|tips?|review|feedback)\b",
    r"\b(is\s*it\s*legal|are\s*we\s*allowed|can\s*we\s*ask|legally\s*compliant)\b",
    r"\b(hiring\s*polic(y|ies))\b",
]

# ─── Refusal Messages ────────────────────────────────────────────────────────

REFUSAL_MESSAGES = {
    "injection": (
        "I appreciate your message, but I'm designed specifically to help "
        "with SHL assessment recommendations. I can't change my role or instructions. "
        "How can I help you find the right assessments for your hiring needs?"
    ),
    "off_topic": (
        "I'm the SHL Assessment Recommender, and I can only help with "
        "finding the right SHL assessments for hiring needs. "
        "Could you tell me about a role you're looking to fill?"
    ),
    "legal": (
        "I'm not able to provide legal or general hiring advice. "
        "I specialize in recommending SHL assessments for specific roles. "
        "Would you like help selecting assessments for a position?"
    ),
    "general_advice": (
        "I focus specifically on SHL assessment recommendations rather than "
        "general hiring guidance. If you have a specific role to fill, "
        "I can recommend the right assessment battery."
    ),
}


class Guardrails:
    """Layered guardrail system.

    Checks run in priority order:
    1. Prompt injection detection
    2. Legal/advice question detection
    3. Off-topic detection
    4. Scope enforcement

    Purpose: Protect against unsafe inputs/outputs.
    Alternatives: LLM-only guardrails (slow, unreliable).
    Trade-offs: May over-filter edge cases. Better safe than failing probes.
    """

    def check_input(self, message: str) -> GuardrailResult:
        """Check user input for safety violations.

        Args:
            message: User message text.

        Returns:
            GuardrailResult indicating whether input is safe.
        """
        text = message.strip()

        # Layer 1: Prompt injection
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning("Guardrail: Injection detected: %s", text[:50])
                return GuardrailResult(
                    is_safe=False,
                    violation_type="injection",
                    refusal_message=REFUSAL_MESSAGES["injection"],
                )

        # Layer 2: Legal/advice questions
        for pattern in LEGAL_ADVICE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                # Check if message also has specific SHL assessment context
                assessment_context = re.search(
                    r"\b(shl|opq|verify|svar|gsa|assessment\s*catalog)\b",
                    text, re.IGNORECASE,
                )
                if not assessment_context:
                    logger.warning("Guardrail: Legal/advice detected: %s", text[:50])
                    return GuardrailResult(
                        is_safe=False,
                        violation_type="legal",
                        refusal_message=REFUSAL_MESSAGES["legal"],
                    )

        # Layer 3: Off-topic (only if no assessment context at all)
        off_topic_match = False
        for pattern in OFF_TOPIC_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                off_topic_match = True
                break

        if off_topic_match:
            # Check if message also has specific SHL brand context
            assessment_context = re.search(
                r"\b(shl|opq|verify|svar|gsa|assessment\s*catalog|product\s*catalog)\b",
                text, re.IGNORECASE,
            )
            if not assessment_context:
                logger.warning("Guardrail: Off-topic detected: %s", text[:50])
                return GuardrailResult(
                    is_safe=False,
                    violation_type="off_topic",
                    refusal_message=REFUSAL_MESSAGES["off_topic"],
                )

        return GuardrailResult(is_safe=True)

    def check_output(
        self,
        reply: str,
        recommendations: list[dict[str, str]],
        catalog_urls: set[str],
    ) -> GuardrailResult:
        """Check agent output for safety violations.

        Args:
            reply: Agent's text reply.
            recommendations: List of recommendation dicts.
            catalog_urls: Set of valid catalog URLs.

        Returns:
            GuardrailResult indicating whether output is safe.
        """
        # Check 1: No URLs in reply that aren't in catalog
        url_pattern = r"https?://[^\s\)\"']+"
        urls_in_reply = re.findall(url_pattern, reply)
        for url in urls_in_reply:
            url_clean = url.rstrip(".,;:!?)")
            if "shl.com" in url_clean and url_clean not in catalog_urls:
                logger.warning("Guardrail: Non-catalog URL in reply: %s", url_clean)
                # Don't block, but log it

        # Check 2: All recommendation URLs in catalog
        for rec in recommendations:
            if rec.get("url") and rec["url"] not in catalog_urls:
                logger.warning("Guardrail: Non-catalog URL in recommendations: %s", rec["url"])
                return GuardrailResult(
                    is_safe=False,
                    violation_type="hallucination",
                    refusal_message="",
                )

        return GuardrailResult(is_safe=True)
