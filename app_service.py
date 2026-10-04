"""Prototype application service: session history -> existing ChatEngine, no RAG.

This module deliberately has no FastAPI/Streamlit dependency so the session logic can
be unit-tested without loading the web stack or the real model.
"""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import threading
from typing import Any, Callable

from unit03_chat_engine import append_turn, create_chat_engine, new_history

ROOT = Path(__file__).resolve().parent
LOCKED_CONFIG = ROOT / "configs/unit03_hf_locked_from_b.json"
DEFAULT_PROMPT = ROOT / "prompts/system_vi_v1.txt"


def create_app_engine():
    """Create the UI backend engine using the locked C-run model revision by default."""
    config_path = Path(os.environ.get("GOUT_MODEL_CONFIG", str(LOCKED_CONFIG)))
    prompt_path = Path(os.environ.get("GOUT_SYSTEM_PROMPT", str(DEFAULT_PROMPT)))
    return create_chat_engine(config_path=config_path, prompt_path=prompt_path)


class ChatService:
    """Keep independent in-memory histories and reuse one ChatEngine/model instance."""

    def __init__(self, engine_factory: Callable[[], Any] = create_app_engine):
        self._engine_factory = engine_factory
        self._engine: Any | None = None
        self._engine_lock = threading.Lock()
        self._sessions_lock = threading.RLock()
        self._generation_lock = threading.Lock()
        self._histories: dict[str, list[dict]] = {}

    @property
    def model_loaded(self) -> bool:
        return self._engine is not None

    def ensure_engine(self) -> Any:
        """Load the backend/model once and reuse it for all requests."""
        if self._engine is None:
            with self._engine_lock:
                if self._engine is None:
                    self._engine = self._engine_factory()
        return self._engine

    @staticmethod
    def _validate_session_id(session_id: str) -> str:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id không được rỗng")
        value = session_id.strip()
        if len(value) > 128:
            raise ValueError("session_id quá dài")
        return value

    @staticmethod
    def _validate_question(question: str) -> str:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Câu hỏi không được rỗng")
        value = question.strip()
        if len(value) > 12000:
            raise ValueError("Câu hỏi quá dài")
        return value

    def get_history(self, session_id: str) -> list[dict]:
        session_id = self._validate_session_id(session_id)
        with self._sessions_lock:
            return deepcopy(self._histories.get(session_id, new_history()))

    def reset(self, session_id: str) -> None:
        session_id = self._validate_session_id(session_id)
        with self._sessions_lock:
            self._histories.pop(session_id, None)

    def chat(self, session_id: str, question: str) -> dict:
        """Generate one answer and append only successful user/assistant pair.

        RAG is intentionally disabled: contexts is always []. Reference/ground_truth is
        never accepted by this service API.
        """
        session_id = self._validate_session_id(session_id)
        question = self._validate_question(question)

        # Serialize generation because one shared HF model instance is reused. This also
        # prevents two simultaneous requests for the same session from racing histories.
        with self._generation_lock:
            with self._sessions_lock:
                history = deepcopy(self._histories.get(session_id, new_history()))

            engine = self.ensure_engine()
            result = engine.chat(question, history, [])
            answer = result.get("answer")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("Backend model trả về câu trả lời rỗng")

            updated = append_turn(history, question, answer, status="ok")
            with self._sessions_lock:
                self._histories[session_id] = updated

            return {
                "session_id": session_id,
                "answer": answer,
                "sources": result.get("sources", []),
                "history_turns": len(updated) // 2,
                "rag_enabled": False,
                "reference_used": False,
                "call": result.get("call", {}),
            }
