"""
Catalog enrichment for SHL Assessment Recommender.

Enriches each catalog item with inferred metadata:
- Technologies, domains, behavioral traits, skills
- This enrichment is used to create richer embeddings and improve retrieval.

Design Decision: Rule-based enrichment (no LLM cost at runtime).
Improves: Recall@10 (richer embeddings catch more queries).
"""

import json
import re
from pathlib import Path
from typing import Optional

from app.catalog.loader import CatalogItem, CatalogStore
from app.config import ENRICHED_CATALOG_PATH
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

# ─── Enrichment Rules ────────────────────────────────────────────────────────
# Rule-based enrichment: map catalog categories and name patterns to
# additional searchable terms. This is deterministic and requires no LLM.

TECHNOLOGY_PATTERNS: dict[str, list[str]] = {
    r"java\b": ["Java", "JVM", "backend", "object-oriented programming"],
    r"\.net|asp\.net|c#": [".NET", "C#", "Microsoft", "backend"],
    r"python": ["Python", "scripting", "automation", "data science"],
    r"javascript|angular|react|node": ["JavaScript", "frontend", "web development"],
    r"angular": ["Angular", "TypeScript", "SPA", "frontend framework"],
    r"react": ["React", "JSX", "frontend", "component-based"],
    r"sql\b": ["SQL", "database", "relational", "data querying"],
    r"docker": ["Docker", "containerization", "DevOps", "deployment"],
    r"aws|amazon web": ["AWS", "cloud computing", "Amazon", "infrastructure"],
    r"azure": ["Azure", "Microsoft cloud", "cloud computing"],
    r"spring": ["Spring", "Java framework", "backend", "microservices"],
    r"hadoop|hbase|hive|pig|spark": ["big data", "distributed computing", "data engineering"],
    r"kafka": ["streaming", "messaging", "event-driven", "data pipeline"],
    r"html|css": ["web development", "frontend", "markup"],
    r"php": ["PHP", "web development", "backend"],
    r"ruby": ["Ruby", "web development", "scripting"],
    r"swift|ios": ["iOS", "mobile development", "Apple"],
    r"android": ["Android", "mobile development", "Google"],
    r"kotlin": ["Kotlin", "Android", "JVM", "mobile"],
    r"salesforce": ["Salesforce", "CRM", "cloud", "business"],
    r"sap": ["SAP", "ERP", "enterprise", "business"],
    r"wordpress": ["WordPress", "CMS", "web", "content management"],
    r"photoshop|illustrator": ["Adobe", "design", "creative", "graphics"],
    r"excel|word|powerpoint|office": ["Microsoft Office", "productivity", "business"],
    r"autocad": ["AutoCAD", "design", "engineering", "CAD"],
    r"git\b": ["Git", "version control", "DevOps", "collaboration"],
    r"linux|unix": ["Linux", "Unix", "system administration", "operating system"],
    r"selenium": ["Selenium", "testing", "automation", "QA"],
    r"jenkins": ["Jenkins", "CI/CD", "DevOps", "automation"],
    r"rest\b|restful": ["REST", "API", "web services", "microservices"],
    r"machine learning|data science": ["ML", "AI", "data analysis", "statistics"],
    r"blockchain": ["blockchain", "distributed ledger", "cryptocurrency"],
    r"cybersecurity|security": ["security", "cybersecurity", "information security"],
}

DOMAIN_PATTERNS: dict[str, list[str]] = {
    r"contact.?cent|call.?cent": ["customer service", "contact center", "call handling", "inbound support"],
    r"sales": ["sales", "commercial", "business development", "revenue"],
    r"customer.?serv": ["customer service", "support", "client relations"],
    r"accounting|payable|receivable|financial|bookkeeping": ["finance", "accounting", "bookkeeping"],
    r"engineering": ["engineering", "technical", "STEM"],
    r"mechanical|electrical|civil|aeronaut|aerospace": ["engineering", "STEM", "technical discipline"],
    r"nurs|medical|health|pharmac": ["healthcare", "medical", "clinical"],
    r"manag": ["management", "leadership", "supervisory"],
    r"clerical|admin|filing|typing": ["administrative", "clerical", "office support"],
    r"retail": ["retail", "consumer", "store operations"],
    r"insurance": ["insurance", "underwriting", "risk"],
    r"banking": ["banking", "financial services"],
    r"legal": ["legal", "compliance", "regulatory"],
}

BEHAVIORAL_PATTERNS: dict[str, list[str]] = {
    r"opq|personality|behavior": [
        "personality assessment", "workplace behavior", "behavioral style",
        "interpersonal skills", "communication", "leadership potential",
        "teamwork", "strategic thinking", "decision making",
    ],
    r"motivation|mq\b": ["motivation", "drive", "engagement", "career aspirations"],
    r"competenc": ["competency assessment", "skill evaluation", "performance prediction"],
    r"judgment|sjt|situational": [
        "situational judgment", "decision making", "workplace scenarios",
        "problem solving", "critical thinking",
    ],
    r"verify|cognitive|reasoning|aptitude|ability": [
        "cognitive ability", "reasoning", "problem solving",
        "learning agility", "mental aptitude", "analytical thinking",
    ],
    r"numerical": ["numerical reasoning", "quantitative", "math", "data interpretation"],
    r"verbal": ["verbal reasoning", "reading comprehension", "language", "communication"],
    r"inductive|deductive|logical": ["logical reasoning", "pattern recognition", "abstract thinking"],
    r"leadership": ["leadership", "management", "strategic", "executive"],
    r"simulation": ["job simulation", "realistic preview", "hands-on assessment"],
    r"development|360": ["development", "feedback", "coaching", "growth", "self-awareness"],
    r"global.?skill": [
        "global skills", "holistic assessment", "Great 8",
        "broad skills evaluation", "multi-dimensional",
    ],
    r"svar|spoken": ["spoken language", "verbal fluency", "pronunciation", "accent"],
}


def enrich_catalog_item(item: CatalogItem) -> str:
    """Generate enrichment text for a single catalog item.

    Applies rule-based pattern matching to add searchable terms.

    Args:
        item: CatalogItem to enrich.

    Returns:
        Enrichment text string to append to retrieval text.
    """
    enrichment_terms: list[str] = []
    combined_text = f"{item.name} {item.description}".lower()

    # Technology enrichment
    for pattern, terms in TECHNOLOGY_PATTERNS.items():
        if re.search(pattern, combined_text, re.IGNORECASE):
            enrichment_terms.extend(terms)

    # Domain enrichment
    for pattern, terms in DOMAIN_PATTERNS.items():
        if re.search(pattern, combined_text, re.IGNORECASE):
            enrichment_terms.extend(terms)

    # Behavioral enrichment
    for pattern, terms in BEHAVIORAL_PATTERNS.items():
        if re.search(pattern, combined_text, re.IGNORECASE):
            enrichment_terms.extend(terms)

    # Seniority enrichment from job_levels
    for level in item.job_levels:
        level_lower = level.lower()
        if "entry" in level_lower or "graduate" in level_lower:
            enrichment_terms.extend(["entry-level", "graduate", "junior", "early career"])
        elif "executive" in level_lower or "director" in level_lower:
            enrichment_terms.extend(["executive", "senior leadership", "C-suite", "strategic"])
        elif "manager" in level_lower or "supervisor" in level_lower:
            enrichment_terms.extend(["management", "team lead", "supervisory"])
        elif "mid" in level_lower:
            enrichment_terms.extend(["mid-career", "experienced", "professional"])

    # Deduplicate
    seen: set[str] = set()
    unique: list[str] = []
    for term in enrichment_terms:
        if term.lower() not in seen:
            seen.add(term.lower())
            unique.append(term)

    return " ".join(unique)


def enrich_catalog(catalog: CatalogStore) -> None:
    """Enrich all catalog items with additional metadata.

    Modifies items in-place by setting enriched_text field.

    Args:
        catalog: Loaded CatalogStore instance.
    """
    logger.info("Enriching %d catalog items...", len(catalog.items))
    enrichment_count = 0

    for item in catalog.items:
        enrichment = enrich_catalog_item(item)
        if enrichment:
            item.enriched_text = enrichment
            enrichment_count += 1

    logger.info(
        "Enrichment complete: %d/%d items enriched",
        enrichment_count,
        len(catalog.items),
    )


def save_enriched_catalog(catalog: CatalogStore, path: Optional[str] = None) -> None:
    """Save enriched catalog data to JSON for persistence.

    Args:
        catalog: Enriched CatalogStore.
        path: Output file path.
    """
    output_path = Path(path or ENRICHED_CATALOG_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for item in catalog.items:
        data.append({
            "entity_id": item.entity_id,
            "name": item.name,
            "link": item.link,
            "description": item.description,
            "keys": item.keys,
            "test_type_code": item.test_type_code,
            "job_levels": item.job_levels,
            "languages": item.languages,
            "duration": item.duration,
            "remote": item.remote,
            "adaptive": item.adaptive,
            "enriched_text": item.enriched_text,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info("Enriched catalog saved to %s", output_path)
