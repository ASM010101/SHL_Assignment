"""
Catalog loader for SHL Assessment Recommender.

Loads the SHL product catalog from JSON and builds lookup structures
for fast name/URL/type resolution.

Design Decision: Pre-build all lookup dicts at startup for O(1) access.
Improves: Performance (no runtime search), Hard evals (URL validation).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.config import CATALOG_PATH, TEST_TYPE_MAP
from app.utils.helpers import get_test_type_code
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class CatalogItem:
    """A single SHL assessment product.

    Attributes:
        entity_id: Unique identifier from catalog.
        name: Assessment name.
        link: Canonical URL to product page.
        description: Full description text.
        keys: Category labels (e.g., "Knowledge & Skills").
        test_type_code: Short code (e.g., "K", "P", "A").
        job_levels: Applicable job levels.
        languages: Supported languages.
        duration: Approximate completion time.
        remote: Whether remote testing is supported.
        adaptive: Whether test is adaptive.
        enriched_text: Combined searchable text (name + description + enrichment).
    """

    entity_id: str
    name: str
    link: str
    description: str
    keys: list[str] = field(default_factory=list)
    test_type_code: str = ""
    job_levels: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    duration: str = ""
    remote: str = ""
    adaptive: str = ""
    enriched_text: str = ""

    def to_retrieval_text(self) -> str:
        """Create combined text for embedding/retrieval.

        Combines name, description, keys, job levels for rich semantic matching.
        Improves: Recall@10 (richer embeddings).
        """
        parts = [
            f"Assessment: {self.name}",
            f"Description: {self.description}",
            f"Category: {', '.join(self.keys)}",
            f"Job Levels: {', '.join(self.job_levels)}",
            f"Languages: {', '.join(self.languages[:5])}",
        ]
        if self.duration:
            parts.append(f"Duration: {self.duration}")
        if self.enriched_text:
            parts.append(f"Additional: {self.enriched_text}")
        return " | ".join(parts)

    def to_context_string(self) -> str:
        """Create a concise context string for LLM prompts.

        Used when presenting assessment details to the LLM for ranking/comparison.
        """
        return (
            f"Name: {self.name}\n"
            f"URL: {self.link}\n"
            f"Type: {self.test_type_code} ({', '.join(self.keys)})\n"
            f"Description: {self.description}\n"
            f"Duration: {self.duration or 'N/A'}\n"
            f"Job Levels: {', '.join(self.job_levels)}\n"
            f"Languages: {', '.join(self.languages[:5])}"
            f"{' (+' + str(len(self.languages) - 5) + ' more)' if len(self.languages) > 5 else ''}\n"
            f"Remote: {self.remote} | Adaptive: {self.adaptive}"
        )


class CatalogStore:
    """In-memory catalog store with precomputed lookup structures.

    Purpose: Central source of truth for all catalog data.
    Alternatives considered: Database (overkill for 377 items), file reads (too slow).
    Trade-offs: ~2MB memory for instant lookups. Worth it.
    Failure modes: Corrupted JSON → startup failure with clear error.
    """

    def __init__(self) -> None:
        self.items: list[CatalogItem] = []
        self.by_name: dict[str, CatalogItem] = {}
        self.by_url: dict[str, CatalogItem] = {}
        self.by_id: dict[str, CatalogItem] = {}
        self.name_to_url: dict[str, str] = {}
        self.url_to_name: dict[str, str] = {}
        self.url_to_type: dict[str, str] = {}
        self.name_to_type: dict[str, str] = {}
        self.all_urls: set[str] = set()
        self.all_names: set[str] = set()
        self.all_names_list: list[str] = []

    def load(self, catalog_path: Optional[str] = None) -> None:
        """Load catalog from JSON file and build lookup structures.

        Args:
            catalog_path: Path to catalog JSON. Defaults to config value.

        Raises:
            FileNotFoundError: If catalog file doesn't exist.
            json.JSONDecodeError: If JSON is malformed.
        """
        path = Path(catalog_path or CATALOG_PATH)
        if not path.exists():
            raise FileNotFoundError(f"Catalog not found at {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        logger.info("Loading catalog from %s (%d items)", path, len(raw_data))

        for item_data in raw_data:
            keys = item_data.get("keys", [])
            test_type_code = get_test_type_code(keys)

            item = CatalogItem(
                entity_id=str(item_data.get("entity_id", "")),
                name=item_data.get("name", ""),
                link=item_data.get("link", ""),
                description=item_data.get("description", ""),
                keys=keys,
                test_type_code=test_type_code,
                job_levels=item_data.get("job_levels", []),
                languages=item_data.get("languages", []),
                duration=item_data.get("duration", ""),
                remote=item_data.get("remote", ""),
                adaptive=item_data.get("adaptive", ""),
            )

            self.items.append(item)
            self.by_name[item.name] = item
            self.by_name[item.name.lower()] = item  # case-insensitive lookup
            self.by_url[item.link] = item
            self.by_id[item.entity_id] = item
            self.name_to_url[item.name] = item.link
            self.url_to_name[item.link] = item.name
            self.url_to_type[item.link] = item.test_type_code
            self.name_to_type[item.name] = item.test_type_code
            self.all_urls.add(item.link)
            self.all_names.add(item.name)

        self.all_names_list = sorted(self.all_names)
        logger.info(
            "Catalog loaded: %d items, %d unique URLs, %d unique names",
            len(self.items),
            len(self.all_urls),
            len(self.all_names),
        )

    def get_by_name(self, name: str) -> Optional[CatalogItem]:
        """Lookup assessment by name (case-insensitive).

        Args:
            name: Assessment name to look up.

        Returns:
            CatalogItem if found, None otherwise.
        """
        return self.by_name.get(name) or self.by_name.get(name.lower())

    def get_by_url(self, url: str) -> Optional[CatalogItem]:
        """Lookup assessment by URL.

        Args:
            url: Catalog URL to look up.

        Returns:
            CatalogItem if found, None otherwise.
        """
        return self.by_url.get(url)

    def search_by_name_substring(self, query: str, limit: int = 10) -> list[CatalogItem]:
        """Search assessments by name substring match.

        Args:
            query: Search query.
            limit: Max results to return.

        Returns:
            List of matching CatalogItems.
        """
        query_lower = query.lower()
        results = [
            item for item in self.items
            if query_lower in item.name.lower() or query_lower in item.description.lower()
        ]
        return results[:limit]

    def get_retrieval_texts(self) -> list[str]:
        """Get all retrieval texts for embedding.

        Returns:
            List of combined texts, one per catalog item.
        """
        return [item.to_retrieval_text() for item in self.items]
