"""
Pydantic schemas for SHL Assessment Recommender API.

These schemas are NON-NEGOTIABLE — they match exactly what the
assignment specifies. Deviating breaks the automated evaluator.

Schema from assignment:
  Request:  {"messages": [{"role": "user"|"assistant", "content": "..."}]}
  Response: {"reply": "...", "recommendations": [...], "end_of_conversation": bool}

Improves: Hard evals (schema compliance is a must-pass criterion).
"""

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single message in the conversation history.

    Attributes:
        role: Either "user" or "assistant".
        content: The message text.
    """

    role: str = Field(
        ...,
        description="Message role: 'user' or 'assistant'",
        examples=["user", "assistant"],
    )
    content: str = Field(
        ...,
        description="Message content text",
        examples=["I am hiring a Java developer"],
    )


class ChatRequest(BaseModel):
    """Request body for POST /chat.

    The API is stateless. Every request carries the full conversation history.

    Attributes:
        messages: Complete conversation history as list of messages.
    """

    messages: list[Message] = Field(
        ...,
        description="Full conversation history",
        min_length=1,
    )


class Recommendation(BaseModel):
    """A single assessment recommendation.

    Every field must be grounded in the SHL catalog.

    Attributes:
        name: Assessment name (must exist in catalog).
        url: Catalog URL (must exist in catalog).
        test_type: Short type code (K, P, A, C, S, B, D, E).
    """

    name: str = Field(
        ...,
        description="Assessment name from SHL catalog",
        examples=["Java 8 (New)"],
    )
    url: str = Field(
        ...,
        description="Catalog URL from SHL catalog",
        examples=["https://www.shl.com/products/product-catalog/view/java-8-new/"],
    )
    test_type: str = Field(
        ...,
        description="Test type code (K=Knowledge, P=Personality, A=Ability, etc.)",
        examples=["K"],
    )


class ChatResponse(BaseModel):
    """Response body for POST /chat.

    - recommendations is EMPTY when the agent is still gathering context or refusing.
    - recommendations is an array of 1-10 items when committed to a shortlist.
    - end_of_conversation is true only when the agent considers the task complete.

    Attributes:
        reply: Agent's text reply.
        recommendations: List of recommended assessments (empty or 1-10).
        end_of_conversation: Whether the conversation is complete.
    """

    reply: str = Field(
        ...,
        description="Agent's text response",
    )
    recommendations: list[Recommendation] = Field(
        default_factory=list,
        description="Recommended assessments (empty when gathering context, 1-10 when committed)",
    )
    end_of_conversation: bool = Field(
        default=False,
        description="True only when the agent considers the task complete",
    )


class HealthResponse(BaseModel):
    """Response body for GET /health.

    Attributes:
        status: Always "ok" when service is ready, or "loading" during initialization.
    """

    status: str = Field(default="ok", description="Service status ('ok' or 'loading')")
    step: str | None = Field(default=None, description="Current initialization step description")
    progress: int | None = Field(default=None, description="Initialization progress percentage (0-100)")
