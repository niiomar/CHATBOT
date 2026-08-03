import asyncio
import logging
import os

from langchain_chroma import Chroma
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

logger = logging.getLogger("chatbot_api.rag")

VECTORSTORE_PATH = "./vectorstore"
RERANKER_PATH = "./local_models/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_N = 6
# cross-encoder/ms-marco-MiniLM-L-6-v2 outputs a raw logit; scores above ~0 are the
# model's own rule-of-thumb boundary for "relevant". Below it, treat the retrieval as
# empty rather than let the LLM improvise around weak/irrelevant context.
RERANK_CONFIDENCE_THRESHOLD = 0.0
NO_CONTEXT_ANSWER = "I don't have information on that."
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

ANSWER_PROMPT = PromptTemplate.from_template("""You are a chatbot assistant for a private organization.
Answer using only the context below. Be concise but complete.
If the answer has multiple items, use bullet points. Group under headings where it makes sense.
Prefer National Signals Bureau Act 1040 or National Signals Bureau Regulations 2486 content when relevant.
If context is partial, give the best answer possible.
If the answer isn't in the context, say: "I don't have information on that."
Cite the sources you use inline with the bracketed numbers shown in the context, e.g. [1].

Chat History:
{chat_history}

Context:
{context}

Question: {question}

Answer:""")

CONDENSE_PROMPT = PromptTemplate.from_template("""Given the chat history and a follow-up question, rewrite the follow-up \
question as a standalone question that includes any context needed to understand it on its own \
(e.g. resolve pronouns or references to earlier topics). If the question is already standalone, \
or there is no chat history, return it unchanged. Return only the rewritten question, with no \
preamble or explanation.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:""")


def format_history(history):
    if not history:
        return "(none)"
    lines = []
    for turn in history:
        role = "User" if turn.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {turn.get('content', '')}")
    return "\n".join(lines)


def source_label(doc):
    source_file = doc.metadata.get("source_file", "unknown")
    page = doc.metadata.get("page")
    return f"{source_file}, p.{page + 1}" if page is not None else source_file


def format_context(docs):
    numbered = [f"[{i}] (Source: {source_label(d)})\n{d.page_content}" for i, d in enumerate(docs, start=1)]
    sources = list(dict.fromkeys(source_label(d) for d in docs))
    return "\n\n".join(numbered), sources


def build_retriever(db):
    vector_retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 20, "fetch_k": 100, "lambda_mult": 0.7},
    )

    stored = db.get()
    corpus = [
        Document(page_content=content, metadata=metadata or {})
        for content, metadata in zip(stored.get("documents") or [], stored.get("metadatas") or [], strict=True)
    ]

    if not corpus:
        logger.warning("vectorstore is empty, falling back to vector-only retrieval")
        return vector_retriever

    bm25_retriever = BM25Retriever.from_documents(corpus, k=20)
    return EnsembleRetriever(retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5])


class RagChain:
    def __init__(self):
        # Imported lazily: these pull in torch, which is an optional ("ml") extra so
        # that CI/tests can install and run without the multi-GB CUDA wheels.
        import torch
        from sentence_transformers import CrossEncoder

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("loading embedding model and vectorstore (device=%s)", device)

        self.embedding_function = HuggingFaceEmbeddings(
            cache_folder="models",
            model_name="./local_models/nomic-embed-text-v1.5",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.db = Chroma(
            persist_directory=VECTORSTORE_PATH, embedding_function=self.embedding_function
        )

        self.llm = ChatOllama(
            model="llama3.1",
            base_url=OLLAMA_BASE_URL,
            temperature=0,
            num_predict=500,
            num_ctx=100000,
            streaming=True,
        )

        self.retriever = build_retriever(self.db)
        self.reranker = CrossEncoder(RERANKER_PATH, device=device)

        self.condense_chain = CONDENSE_PROMPT | self.llm | StrOutputParser()
        logger.info("rag chain ready")

    def refresh_retriever(self):
        """Rebuild the hybrid retriever from what's currently in the vectorstore.

        The BM25 half is an in-memory snapshot taken at build time, so it goes stale
        after ingest.py adds/removes documents. Call this after re-ingesting instead
        of restarting the process.
        """
        self.retriever = build_retriever(self.db)
        logger.info("retriever refreshed")

    def _rerank(self, question, docs):
        if not docs:
            return [], None
        pairs = [[question, d.page_content] for d in docs]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(scores, docs, strict=True), key=lambda pair: pair[0], reverse=True)
        top = ranked[:RERANK_TOP_N]
        best_score = float(top[0][0])
        return [d for _, d in top], best_score

    async def _standalone_question(self, question, chat_history):
        if not chat_history:
            return question
        standalone = await self.condense_chain.ainvoke(
            {"chat_history": format_history(chat_history), "question": question}
        )
        logger.info("condensed question: %r -> %r", question, standalone)
        return standalone

    async def astream(self, question, chat_history=None):
        standalone_question = await self._standalone_question(question, chat_history)
        docs = await self.retriever.ainvoke(standalone_question)
        docs, best_score = await asyncio.to_thread(self._rerank, standalone_question, docs)

        if best_score is None or best_score < RERANK_CONFIDENCE_THRESHOLD:
            logger.info("no sufficiently relevant chunks found (best_score=%s)", best_score)
            yield NO_CONTEXT_ANSWER
            return

        context, sources = format_context(docs)
        logger.info("retrieved -> reranked to %d chunks from %d source(s), best_score=%.3f", len(docs), len(sources), best_score)

        prompt = ANSWER_PROMPT.format(
            chat_history=format_history(chat_history), context=context, question=question
        )

        async for chunk in self.llm.astream(prompt):
            yield chunk.content

        if sources:
            yield "\n\n**Sources:** " + ", ".join(sources)


def build_chain():
    return RagChain()
