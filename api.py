from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from rag import build_chain
import os

os.environ["HF_HUB_OFFLINE"] = "1"

app = FastAPI(title="NSB-AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

chain = build_chain()


class Query(BaseModel):
    question: str


@app.post("/ask")
async def ask(query: Query):
    async def stream_response():
        async for chunk in chain.astream(query.question):
            if hasattr(chunk, "content"):
                yield chunk.content
            else:
                yield str(chunk)

    return StreamingResponse(stream_response(), media_type="text/plain")


@app.get("/health")
async def health():
    return {"status": "ok"}
