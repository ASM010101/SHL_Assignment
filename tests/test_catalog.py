"""
Catalog loading and integrity tests for SHL Assessment Recommender.

Ensures the catalog data structures load correctly, map short codes
properly, and contain valid URLs and categories.

Improves: Hard evals (grounding check, no hallucinations).
"""

import pytest
from app.catalog.loader import CatalogStore, CatalogItem
from app.utils.helpers import get_test_type_code


def test_test_type_mapping():
    # Verify standard keys convert to single code
    assert get_test_type_code(["Knowledge & Skills"]) == "K"
    assert get_test_type_code(["Personality & Behavior"]) == "P"
    assert get_test_type_code(["Ability & Aptitude"]) == "A"
    assert get_test_type_code(["Competencies"]) == "C"
    assert get_test_type_code(["Simulations"]) == "S"
    assert get_test_type_code(["Biodata & Situational Judgment"]) == "B"
    assert get_test_type_code(["Development & 360"]) == "D"
    assert get_test_type_code(["Assessment Exercises"]) == "E"

    # Multiple keys mapped as comma-separated unique ordered codes
    assert get_test_type_code(["Personality & Behavior", "Competencies"]) == "P, C"
    # Fallback to K if keys are empty or unmatched
    assert get_test_type_code([]) == "K"
    assert get_test_type_code(["Unknown Category"]) == "K"


def test_catalog_store_loading():
    store = CatalogStore()
    store.load("data/enriched_catalog.json")

    # Catalog must be populated
    assert len(store.items) == 377
    assert len(store.all_urls) == 377
    assert len(store.all_names) == 377

    # Test O(1) lookups
    item = store.get_by_name("Java 8 (New)")
    assert item is not None
    assert item.link == "https://www.shl.com/products/product-catalog/view/java-8-new/"
    assert item.test_type_code == "K"

    # URL lookup
    item_url = store.get_by_url("https://www.shl.com/products/product-catalog/view/java-8-new/")
    assert item_url is not None
    assert item_url.name == "Java 8 (New)"

    # Verify no invalid characters in URLs
    for url in store.all_urls:
        assert url.startswith("https://www.shl.com/products/product-catalog/")
        assert " " not in url
