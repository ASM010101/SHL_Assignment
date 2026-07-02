"""
Utility helpers for SHL Assessment Recommender.

Includes: test type mapping, query rewriting, text normalization.

Design Decision: Query rewriting bridges user language to catalog language.
Improves: Recall@10 (better retrieval matches).
"""

import re
from app.config import TEST_TYPE_MAP
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

# ─── Test Type Mapping ───────────────────────────────────────────────────────


def get_test_type_code(keys: list[str]) -> str:
    """Convert catalog 'keys' list to short test type code string.

    Examples:
        ["Knowledge & Skills"] -> "K"
        ["Personality & Behavior", "Competencies"] -> "P, C"
        ["Ability & Aptitude"] -> "A"

    Args:
        keys: List of catalog key categories.

    Returns:
        Comma-separated test type codes.
    """
    codes = []
    for key in keys:
        code = TEST_TYPE_MAP.get(key, "")
        if code and code not in codes:
            codes.append(code)
    return ", ".join(codes) if codes else "K"


# ─── Query Rewriting ─────────────────────────────────────────────────────────

# Maps common user language to catalog-friendly retrieval terms.
# This improves Recall@10 by bridging vocabulary mismatch.
EXPANSION_MAP: dict[str, list[str]] = {
    # Role-related
    "developer": ["programming", "software", "engineering", "coding"],
    "manager": ["leadership", "management", "supervisory"],
    "executive": ["leadership", "strategic", "director", "C-suite"],
    "analyst": ["analytical", "data", "reasoning", "numerical"],
    "sales": ["selling", "commercial", "customer", "persuasion", "negotiation"],
    "customer service": ["contact center", "service", "support", "communication"],
    "hr": ["human resources", "people", "talent", "recruitment"],
    "finance": ["financial", "accounting", "numerical", "banking"],

    # Skill-related
    "java": ["Java", "JVM", "Spring", "backend"],
    "python": ["Python", "scripting", "programming"],
    "javascript": ["JavaScript", "frontend", "web", "Angular", "React", "Node"],
    ".net": [".NET", "C#", "ASP.NET", "WPF", "WCF"],
    "sql": ["SQL", "database", "relational", "queries"],
    "cloud": ["AWS", "Azure", "cloud computing", "deployment"],
    "devops": ["Docker", "CI/CD", "Kubernetes", "deployment", "infrastructure"],
    "data science": ["machine learning", "data analysis", "statistics", "Python", "R"],

    # Assessment type hints
    "personality": ["OPQ", "personality", "behavioral", "workplace behavior"],
    "cognitive": ["reasoning", "ability", "aptitude", "Verify", "G+"],
    "aptitude": ["reasoning", "ability", "cognitive", "Verify"],
    "reasoning": ["cognitive", "ability", "aptitude", "logical", "Verify"],
    "coding test": ["programming", "technical", "knowledge", "skills"],
    "soft skills": ["personality", "behavior", "communication", "teamwork", "leadership"],
    "leadership": ["management", "leadership", "OPQ", "competency", "executive"],
    "communication": ["verbal", "written", "stakeholder", "interpersonal"],
    "teamwork": ["collaboration", "team", "interpersonal", "behavioral"],

    # Purpose hints
    "screening": ["high-volume", "entry-level", "selection", "graduate"],
    "development": ["feedback", "360", "coaching", "growth"],
    "selection": ["hiring", "recruitment", "candidate", "assessment"],
    "promotion": ["leadership", "readiness", "potential", "development"],
}


def rewrite_query(query: str) -> str:
    """Expand a user query with retrieval-friendly terms.

    Adds domain-specific synonyms and related concepts to improve
    semantic and keyword retrieval coverage.

    Args:
        query: Raw user query text.

    Returns:
        Expanded query string with additional retrieval terms.

    Example:
        >>> rewrite_query("hiring a Java developer")
        'hiring a Java developer programming software engineering coding JVM Spring backend'
    """
    query_lower = query.lower()
    expansions: list[str] = []

    for trigger, terms in EXPANSION_MAP.items():
        if trigger in query_lower:
            expansions.extend(terms)

    if expansions:
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_expansions: list[str] = []
        for term in expansions:
            if term.lower() not in seen and term.lower() not in query_lower:
                seen.add(term.lower())
                unique_expansions.append(term)

        expanded = f"{query} {' '.join(unique_expansions)}"
        logger.debug("Query rewritten: '%s' -> '%s'", query, expanded)
        return expanded

    return query


# ─── Text Normalization ──────────────────────────────────────────────────────


def normalize_text(text: str) -> str:
    """Normalize text for matching: lowercase, strip extra whitespace.

    Args:
        text: Input text.

    Returns:
        Normalized text.
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_assessment_names(text: str, catalog_names: list[str]) -> list[str]:
    """Extract assessment names mentioned in text by matching against catalog.

    Used for comparison requests like "compare OPQ and GSA".

    Args:
        text: User message text.
        catalog_names: List of all catalog assessment names.

    Returns:
        List of matched catalog assessment names.
    """
    text_lower = text.lower()
    matched: list[str] = []

    # Sort by length descending to match longer names first
    sorted_names = sorted(catalog_names, key=len, reverse=True)

    for name in sorted_names:
        # Check for exact or partial match
        name_lower = name.lower()
        # Also check common abbreviations
        if name_lower in text_lower:
            matched.append(name)
            continue

        # Check each word of the name (for abbreviations like OPQ, GSA, SVAR)
        words = name.split()
        for word in words:
            if len(word) >= 3 and word.lower() in text_lower:
                if name not in matched:
                    matched.append(name)
                    break

    return matched[:5]  # Cap at 5 to prevent noise


def count_turns(messages: list[dict[str, str]]) -> int:
    """Count the number of conversation turns (user + assistant pairs).

    Args:
        messages: List of message dicts with 'role' and 'content'.

    Returns:
        Number of turns.
    """
    return len(messages)
