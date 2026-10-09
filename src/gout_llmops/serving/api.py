from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="Gout LLMOps API", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok", "note": "Model engine is not auto-loaded by this scaffold."}
