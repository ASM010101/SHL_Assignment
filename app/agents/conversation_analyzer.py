"""
Conversation analyzer for SHL Assessment Recommender.

Extracts a structured HiringProfile from conversation history using LLM.
The profile accumulates constraints across turns (stateless reconstruction).

Design Decision: LLM-based extraction into structured object.
Improves: Recall@10 (targeted retrieval), Coherence (consistent state across turns).
"""

import json
from typing import Optional

from app.prompts.templates import CONVERSATION_ANALYSIS_PROMPT
from app.retrieval.hybrid_retriever import HiringProfile
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


def format_conversation(messages: list[dict[str, str]]) -> str:
    """Format conversation history as readable text.

    Args:
        messages: List of message dicts.

    Returns:
        Formatted conversation string.
    """
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


class ConversationAnalyzer:
    """Extracts structured hiring profile from conversation history.

    Purpose: Transform unstructured conversation into structured HiringProfile.
    Alternatives: Rule-based extraction (misses context), raw text (no structure).
    Trade-offs: LLM call adds ~1s latency but gives accurate structured data.
    Failure modes: LLM returns invalid JSON → fallback to basic extraction.
    """

    def analyze(
        self,
        messages: list[dict[str, str]],
        llm_response: Optional[str] = None,
    ) -> HiringProfile:
        """Extract HiringProfile from conversation history.

        Args:
            messages: Full conversation history.
            llm_response: Pre-computed LLM analysis response (if available).

        Returns:
            Structured HiringProfile with computed confidence.
        """
        profile = HiringProfile()

        # If we have an LLM response, parse it
        if llm_response:
            profile = self._parse_llm_response(llm_response, profile)

        # Also do rule-based extraction as supplement
        profile = self._rule_based_extraction(messages, profile)

        # Compute confidence
        profile.compute_confidence()

        logger.info(
            "Profile extracted: role=%s, seniority=%s, skills=%s, confidence=%.2f",
            profile.role,
            profile.seniority,
            profile.skills[:3],
            profile.confidence,
        )

        return profile

    def _parse_llm_response(self, response: str, profile: HiringProfile) -> HiringProfile:
        """Parse LLM JSON response into HiringProfile.

        Args:
            response: LLM response text (expected JSON).
            profile: Profile to update.

        Returns:
            Updated HiringProfile.
        """
        try:
            # Clean response: strip markdown code blocks if present
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()

            data = json.loads(cleaned)

            if data.get("role"):
                profile.role = data["role"]
            if data.get("seniority"):
                profile.seniority = data["seniority"]
            if data.get("skills"):
                profile.skills = [s for s in data["skills"] if s]
            if data.get("soft_skills"):
                profile.soft_skills = [s for s in data["soft_skills"] if s]
            if data.get("domain"):
                profile.domain = data["domain"]
            if data.get("industry"):
                profile.industry = data["industry"]
            if data.get("languages"):
                profile.languages = [l for l in data["languages"] if l]
            if data.get("purpose"):
                profile.purpose = data["purpose"]
            if data.get("assessment_types"):
                profile.assessment_types = [a for a in data["assessment_types"] if a]
            if data.get("job_level"):
                profile.job_level = data["job_level"]
            if data.get("constraints"):
                profile.constraints = [c for c in data["constraints"] if c]
            if data.get("raw_jd"):
                profile.raw_jd = data["raw_jd"]
            if data.get("add_requests"):
                profile.add_requests = [a for a in data["add_requests"] if a]
            if data.get("remove_requests"):
                profile.remove_requests = [r for r in data["remove_requests"] if r]

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Failed to parse LLM profile response: %s", e)

        return profile

    def _rule_based_extraction(
        self,
        messages: list[dict[str, str]],
        profile: HiringProfile,
    ) -> HiringProfile:
        """Supplement LLM extraction with rule-based pattern matching.

        Catches simple cases that the LLM might miss.

        Args:
            messages: Conversation history.
            profile: Current profile to supplement.

        Returns:
            Updated HiringProfile.
        """
        all_text = " ".join(msg.get("content", "") for msg in messages).lower()

        # Extract seniority if not already set
        if not profile.seniority:
            seniority_patterns = {
                "entry": ["entry.?level", "junior", "fresh", "graduate", "intern"],
                "mid": ["mid.?level", "mid.?professional", r"\b3.?5\s*years", r"\b4.?6\s*years"],
                "senior": ["senior", r"\b5\+?\s*years", r"\b7\+?\s*years", r"\b10\+?\s*years", "experienced"],
                "executive": ["executive", "director", "cxo", "c.?suite", "vp", r"\b15\+?\s*years", "leadership.?bench"],
            }
            import re
            for level, patterns in seniority_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, all_text, re.IGNORECASE):
                        profile.seniority = level
                        break
                if profile.seniority:
                    break

        # Extract common tech skills if not already set
        if not profile.skills:
            tech_skills = [
                "java", "python", "javascript", "c#", ".net", "sql", "react",
                "angular", "node", "docker", "aws", "azure", "kubernetes",
                "spring", "ruby", "php", "swift", "kotlin", "scala", "go",
                "html", "css", "typescript", "vue", "django", "flask",
                "mongodb", "postgresql", "mysql", "redis", "kafka",
                "jenkins", "git", "linux", "terraform", "ansible",
                "salesforce", "sap", "excel", "photoshop", "autocad",
            ]
            found_skills = [skill for skill in tech_skills if skill in all_text]
            if found_skills:
                profile.skills = found_skills

        return profile

    def get_analysis_prompt(self, messages: list[dict[str, str]]) -> str:
        """Generate the conversation analysis prompt.

        Args:
            messages: Conversation history.

        Returns:
            Formatted prompt string.
        """
        conversation = format_conversation(messages)
        return CONVERSATION_ANALYSIS_PROMPT.format(conversation=conversation)

    def get_missing_fields(self, profile: HiringProfile) -> list[str]:
        """Identify which important fields are still missing.

        Used by the clarifier to determine what to ask about.

        Args:
            profile: Current hiring profile.

        Returns:
            List of missing field descriptions.
        """
        missing: list[str] = []

        if not profile.role and not profile.raw_jd:
            missing.append("role or job description (REQUIRED - what role are you hiring for?)")
        if not profile.seniority:
            missing.append("seniority level (HIGH VALUE - entry-level, mid, senior, executive?)")
        if not profile.skills and not profile.domain:
            missing.append("key skills or domain area (HIGH VALUE - what should they be good at?)")
        if not profile.purpose:
            missing.append("assessment purpose (OPTIONAL - selection, development, screening?)")

        return missing
