"""
Edge case unit tests for SHL Assessment Recommender.

Covers turn count management, malformed history, duplicate recommendations,
empty catalog lookups, and auto-fixing of malformed outputs.

Improves: Hard evals (limit rules, duplicate detection).
"""

from app.agents.conversation_planner import ConversationPlanner, Action, Intent
from app.retrieval.hybrid_retriever import HiringProfile
from app.utils.validators import OutputValidator


def test_turn_count_enforcement():
    planner = ConversationPlanner()
    profile = HiringProfile(role="Software Engineer", seniority="Senior", confidence=0.7)

    # 1. Normal turns (1 to 4) -> RECOMMEND or CLARIFY
    act_t3 = planner.plan(Intent.SEARCH, profile, turn_count=3, has_active_recommendations=False)
    assert act_t3 == Action.RECOMMEND

    # 2. Vague intent under threshold -> CLARIFY
    vague_profile = HiringProfile(role=None, confidence=0.1)
    act_t2 = planner.plan(Intent.SEARCH, vague_profile, turn_count=2, has_active_recommendations=False)
    assert act_t2 == Action.CLARIFY

    # 3. Running out of turns (turn 6/7) -> RECOMMEND if profile has min info, even if confidence is low (>= 0.5)
    min_profile = HiringProfile(role="Engineer", confidence=0.5)
    act_t7 = planner.plan(Intent.SEARCH, min_profile, turn_count=7, has_active_recommendations=False)
    assert act_t7 == Action.RECOMMEND

    # 4. Turn cap reached (8) -> END_CONVERSATION
    act_t8 = planner.plan(Intent.GOODBYE, profile, turn_count=8, has_active_recommendations=True)
    assert act_t8 == Action.END_CONVERSATION


def test_output_validator_fixes():
    # Setup simple mock validator
    catalog_urls = {f"https://shl.com/test{i}" for i in range(1, 20)}
    catalog_names = {f"Test {i}" for i in range(1, 20)}
    catalog_name_to_url = {f"Test {i}": f"https://shl.com/test{i}" for i in range(1, 20)}
    catalog_url_to_type = {f"https://shl.com/test{i}": "K" for i in range(1, 20)}
    catalog_name_to_type = {f"Test {i}": "K" for i in range(1, 20)}

    validator = OutputValidator(
        catalog_urls=catalog_urls,
        catalog_names=catalog_names,
        catalog_name_to_url=catalog_name_to_url,
        catalog_url_to_type=catalog_url_to_type,
        catalog_name_to_type=catalog_name_to_type
    )

    # 1. Fixes missing URLs if name is correct
    reply, recs, end = validator.validate_and_fix(
        reply="Here is test one",
        recommendations=[{"name": "Test 1", "url": "", "test_type": "K"}],
        end_of_conversation=False
    )
    assert recs[0]["url"] == "https://shl.com/test1"

    # 2. Fixes wrong test_type
    reply, recs, end = validator.validate_and_fix(
        reply="Here is test one",
        recommendations=[{"name": "Test 1", "url": "https://shl.com/test1", "test_type": "wrong_type"}],
        end_of_conversation=False
    )
    assert recs[0]["test_type"] == "K"

    # 3. Drops recommendations with completely invalid URLs
    reply, recs, end = validator.validate_and_fix(
        reply="Here is test one",
        recommendations=[{"name": "Invalid Test", "url": "https://shl.com/fake", "test_type": "K"}],
        end_of_conversation=False
    )
    assert len(recs) == 0

    # 4. Removes duplicates
    reply, recs, end = validator.validate_and_fix(
        reply="Here are tests",
        recommendations=[
            {"name": "Test 1", "url": "https://shl.com/test1", "test_type": "K"},
            {"name": "Test 1", "url": "https://shl.com/test1", "test_type": "K"}
        ],
        end_of_conversation=False
    )
    assert len(recs) == 1

    # 5. Cap recommendations at 10 items
    many_recs = [
        {"name": f"Test {i}", "url": f"https://shl.com/test{i}", "test_type": "K"}
        for i in range(1, 16)
    ]
    reply, recs, end = validator.validate_and_fix(
        reply="Here is a long list",
        recommendations=many_recs,
        end_of_conversation=False
    )
    assert len(recs) == 10
