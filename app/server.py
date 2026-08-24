"""FastAPI application for Aster & Row customer support AI agent."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from app.agent import handle_turn

# Configure logger
logger = logging.getLogger("aster_row_server")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Aster & Row Customer Support API",
    description="Backend API for Aster & Row outdoor gear customer support agent.",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


class ChatRequest(BaseModel):
    """Request payload for /api/chat."""

    session_id: str = Field(..., min_length=1, max_length=128, description="Unique session identifier.")
    message: str = Field(..., min_length=1, max_length=4000, description="User message text.")

    @field_validator("session_id", "message")
    @classmethod
    def validate_not_blank(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("Field cannot be empty or contain only whitespace.")
        return trimmed


class ChatResponse(BaseModel):
    """Response payload for /api/chat."""

    answer: str
    sources: list[str]
    tool_calls: list[dict[str, Any]]
    handoff: bool


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> dict[str, str]:
    """Health check endpoint returning API operational status."""
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    """Process a user chat turn through the Aster & Row customer support AI agent."""
    try:
        result = handle_turn(session_id=payload.session_id, user_message=payload.message)
        return ChatResponse(
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            tool_calls=result.get("tool_calls", []),
            handoff=bool(result.get("handoff", False)),
        )
    except Exception as e:
        logger.error("Unhandled exception in chat_endpoint: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred while processing your request. Please try again or contact customer support.",
        )


@app.get("/")
async def root():
    """Serve static frontend index.html if available, or return health summary."""
    index_path = STATIC_DIR / "index.html"
    if index_path.is_file():
        return FileResponse(index_path)
    return {"message": "Aster & Row Customer Support API is running. See /docs for API schema."}


# Mount static directory for JS/CSS assets
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
