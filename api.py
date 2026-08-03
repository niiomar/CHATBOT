import asyncio
import logging
import os
from functools import lru_cache
from typing import Literal

import requests
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from rag import OLLAMA_BASE_URL, build_chain

os.environ["HF_HUB_OFFLINE"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("chatbot_api")

API_KEY = os.environ.get("API_KEY")
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost:8501").split(",") if o.strip()]
ASK_RATE_LIMIT = os.environ.get("ASK_RATE_LIMIT", "20/minute")

if not API_KEY:
    logger.warning("API_KEY is not set - /ask is unauthenticated. Set API_KEY for any non-local deployment.")

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Chatbot Assistant", version="1.0.0")
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded, try again shortly."})


app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


@lru_cache
def get_chain():
    return build_chain()


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(provided: str | None = Security(api_key_header)):
    if API_KEY and provided != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class Query(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    chat_history: list[ChatTurn] = Field(default_factory=list, max_length=40)


@app.post("/ask", dependencies=[Depends(require_api_key)])
@limiter.limit(ASK_RATE_LIMIT)
async def ask(request: Request, query: Query, chain=Depends(get_chain)):
    history = [turn.model_dump() for turn in query.chat_history]
    logger.info("question received (%d chars, %d history turns)", len(query.question), len(history))

    async def stream_response():
        try:
            async for chunk in chain.astream(query.question, history):
                if hasattr(chunk, "content"):
                    yield chunk.content
                else:
                    yield str(chunk)
        except Exception:
            logger.exception("error while streaming answer")
            yield "\n\n[An error occurred while generating the answer.]"

    return StreamingResponse(stream_response(), media_type="text/plain")


@app.post("/admin/refresh", dependencies=[Depends(require_api_key)])
async def refresh_index(chain=Depends(get_chain)):
    await asyncio.to_thread(chain.refresh_retriever)
    return {"status": "refreshed"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/health/ready")
async def readiness(chain=Depends(get_chain)):
    def check_ollama():
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        r.raise_for_status()

    try:
        await asyncio.to_thread(check_ollama)
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Ollama unreachable: {e}") from e

    return {"status": "ready"}
