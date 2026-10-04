"""FastAPI backend for the pre-RAG Gout chatbot prototype."""
from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app_service import ChatService


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=1, max_length=12000)


class ResetRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)


service = ChatService()
app = FastAPI(
    title="Gout Chat Backend (No RAG)",
    version="0.1.0",
    description="Streamlit-facing API using the existing ChatEngine. RAG is disabled.",
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": service.model_loaded,
        "rag_enabled": False,
        "backend": "existing_chat_engine",
    }


@app.post("/api/chat")
def chat(payload: ChatRequest) -> dict[str, Any]:
    try:
        return service.chat(payload.session_id, payload.question)
    except (ValueError, TypeError, NotImplementedError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        # Do not return a fake answer when inference fails.
        raise HTTPException(
            status_code=500,
            detail=f"Model/backend error: {type(exc).__name__}: {exc}",
        ) from exc


@app.post("/api/reset")
def reset(payload: ResetRequest) -> dict[str, Any]:
    try:
        service.reset(payload.session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", "session_id": payload.session_id}


@app.get("/api/session/{session_id}")
def session_state(session_id: str) -> dict[str, Any]:
    try:
        history = service.get_history(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "session_id": session_id,
        "history": history,
        "history_turns": len(history) // 2,
        "rag_enabled": False,
    }
