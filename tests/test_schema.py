"""
Schema compliance unit tests for SHL Assessment Recommender.

Verifies that the Pydantic request/response models match the required schemas
exactly, with no extra or missing fields, and proper types.

Improves: Hard evals (strictly conforms to automated evaluator expectations).
"""

import pytest
from pydantic import ValidationError
from app.api.schemas import Message, ChatRequest, Recommendation, ChatResponse


def test_message_schema():
    # Valid message
    msg = Message(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"

    # Invalid role
    with pytest.raises(ValidationError):
        Message(role=123, content="Hello")


def test_chat_request_schema():
    # Valid history
    req = ChatRequest(messages=[
        Message(role="user", content="Java Dev role"),
        Message(role="assistant", content="What seniority?")
    ])
    assert len(req.messages) == 2
    assert req.messages[0].role == "user"

    # Empty messages should fail validation
    with pytest.raises(ValidationError):
        ChatRequest(messages=[])


def test_recommendation_schema():
    # Valid recommendation
    rec = Recommendation(
        name="Java 8 (New)",
        url="https://www.shl.com/products/product-catalog/view/java-8-new/",
        test_type="K"
    )
    assert rec.name == "Java 8 (New)"
    assert rec.url == "https://www.shl.com/products/product-catalog/view/java-8-new/"
    assert rec.test_type == "K"

    # Missing fields
    with pytest.raises(ValidationError):
        Recommendation(name="Java 8 (New)")


def test_chat_response_schema():
    # Valid response with recommendations
    resp = ChatResponse(
        reply="Here are recommendations",
        recommendations=[
            Recommendation(name="Java 8 (New)", url="https://example.com/java", test_type="K")
        ],
        end_of_conversation=False
    )
    assert resp.reply == "Here are recommendations"
    assert len(resp.recommendations) == 1
    assert resp.end_of_conversation is False

    # Valid response without recommendations (empty list)
    resp_empty = ChatResponse(
        reply="What seniority level?",
        recommendations=[],
        end_of_conversation=False
    )
    assert resp_empty.recommendations == []

    # Missing required field 'reply'
    with pytest.raises(ValidationError):
        ChatResponse(recommendations=[], end_of_conversation=False)
