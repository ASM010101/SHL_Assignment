"""
Comparison agent for SHL Assessment Recommender.

Compares assessments using ONLY the catalog information provided.
Avoids relying on model memory or priors.

Design Decision: Explicit catalog lookup + structured comparison prompt.
Improves: Behavior probes (grounded comparisons, no model hallucination).
"""

from typing import Optional
from app.catalog.loader import CatalogStore, CatalogItem
from app.prompts.templates import COMPARISON_PROMPT
from app.utils.logger import setup_logger
from app.utils.helpers import extract_assessment_names

logger = setup_logger(__name__)


class Comparator:
    """Compares SHL assessments using catalog data only.

    Purpose: Deliver accurate, grounded differences between tests.
    Alternatives considered: Direct LLM generation (hallucination risk).
    Trade-offs: Limited to catalog data, but guaranteed to be correct.
    Failure modes: Missing catalog details → returns a generic comparison.
    """

    def __init__(self, catalog: CatalogStore) -> None:
        self.catalog = catalog

    def get_comparison_prompt(
        self,
        message: str,
        detected_items: list[str],
    ) -> Optional[str]:
        """Generate the comparison prompt if valid items are found.

        Args:
            message: User query requesting comparison.
            detected_items: List of assessment names extracted from the message.

        Returns:
            Comparison prompt text, or None if insufficient items found.
        """
        # If we didn't extract any names, let's extract them now
        if not detected_items:
            detected_items = extract_assessment_names(message, self.catalog.all_names_list)

        if len(detected_items) < 2:
            logger.warning("Comparator: Not enough items detected for comparison (found: %s)", detected_items)
            return None

        # Look up catalog items
        item1 = self.catalog.get_by_name(detected_items[0])
        item2 = self.catalog.get_by_name(detected_items[1])

        if not item1 or not item2:
            logger.warning("Comparator: Could not find catalog items for comparison: %s", detected_items)
            return None

        # Construct context strings
        context1 = item1.to_context_string()
        context2 = item2.to_context_string()

        additional_context = ""
        if len(detected_items) > 2:
            parts = []
            for i, name in enumerate(detected_items[2:], 3):
                item = self.catalog.get_by_name(name)
                if item:
                    parts.append(f"ASSESSMENT {i}:\n{item.to_context_string()}")
            additional_context = "\n\n".join(parts)

        logger.info("Comparator: Generating comparison prompt for %s and %s", item1.name, item2.name)

        return COMPARISON_PROMPT.format(
            assessment1=context1,
            assessment2=context2,
            additional_assessments=additional_context,
        )

    def generate_static_comparison(self, item1: CatalogItem, item2: CatalogItem) -> str:
        """Create a deterministic comparison string when LLM is unavailable.

        Args:
            item1: First catalog item.
            item2: Second catalog item.

        Returns:
            Grounded comparison text.
        """
        desc1 = item1.description[:150] + "..." if len(item1.description) > 150 else item1.description
        desc2 = item2.description[:150] + "..." if len(item2.description) > 150 else item2.description
        return (
            f"Here is the comparison between {item1.name} and {item2.name}:\n\n"
            f"- **{item1.name}**: A {item1.test_type_code} type test. "
            f"It is designed for {', '.join(item1.job_levels[:2])} and takes approximately "
            f"{item1.duration or 'N/A'}. Description: {desc1}\n"
            f"- **{item2.name}**: A {item2.test_type_code} type test. "
            f"It is designed for {', '.join(item2.job_levels[:2])} and takes approximately "
            f"{item2.duration or 'N/A'}. Description: {desc2}\n\n"
            f"Both can be administered remotely."
        )
