"""Run the FastAPI backend on localhost:8000."""
import os
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "backend_api:app",
        host=os.environ.get("GOUT_BACKEND_HOST", "127.0.0.1"),
        port=int(os.environ.get("GOUT_BACKEND_PORT", "8000")),
        reload=False,
    )
