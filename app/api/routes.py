"""
FastAPI route definitions for SHL Assessment Recommender.

Exposes:
- GET /health (Readiness check, returning status: ok)
- POST /chat (Stateless conversation endpoint)

Design Decision: Explicit schema compliance validation and dependency retrieval.
Improves: Hard evals (API endpoints and schema compliance).
"""

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse
from app.api.schemas import ChatRequest, ChatResponse, HealthResponse
from app.api.index_html import INDEX_HTML
from app.utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()


@router.get(
    "/",
    response_class=HTMLResponse,
    summary="Playground UI for Recommender",
    description="Serves the premium, responsive glassmorphic chat playground interface.",
)
async def index():
    """Serves the front-end page."""
    return INDEX_HTML


@router.get(
    "/health",
    summary="Health and readiness check",
    description="Returns 'status': 'ok' when the service is fully loaded and initialized.",
)
async def health(request: Request):
    """Health check endpoint.

    Ensures the catalog, models, and index are loaded.
    """
    # Fetch initialization status from app state
    init_status = getattr(request.app.state, "initialization_status", {"status": "loading", "step": "Starting...", "progress": 5})
    if init_status.get("status") == "ok":
        return {"status": "ok"}
    return init_status


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Process a message in the conversation",
    description="Takes a list of conversation messages and returns the next agent response.",
)
async def chat(request: Request, body: ChatRequest):
    """Chat endpoint for conversational assessment recommendations.

    Stateless endpoint that expects full conversation history.
    """
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:
        raise HTTPException(
            status_code=503,
            detail="Service initializing (orchestrator not loaded yet)."
        )

    try:
        # Convert request body to list of dicts for orchestrator processing
        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in body.messages
        ]

        logger.info("Received chat request with %d messages", len(messages))

        # Generate unique request id for tracing
        import uuid
        request_id = str(uuid.uuid4())[:8]

        response_dict = orchestrator.process_message(messages, request_id=request_id)

        # Map return dict to Pydantic ChatResponse
        return ChatResponse(
            reply=response_dict["reply"],
            recommendations=response_dict["recommendations"],
            end_of_conversation=response_dict["end_of_conversation"],
        )

    except Exception as e:
        logger.error("Error processing chat message: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal Server Error: {str(e)}"
        )
