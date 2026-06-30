from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import torch

VECTORSTORE_PATH = "./vectorstore"

PROMPT = """You are a chatbot assistant for a private organization.
Answer using only the context below. Be concise but complete.
If the answer has multiple items, use bullet points. Group under headings where it makes sense.
Prefer National Signals Bureau Act 1040 or National Signals Bureau Regulations 2486 content when relevant.
If context is partial, give the best answer possible.
If the answer isn't in the context, say: "I don't have information on that."

Context: {context}

Question: {question}

Answer:"""


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def build_chain():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    embedding_function = HuggingFaceEmbeddings(
        cache_folder="models",
        model_name="./local_models/nomic-embed-text-v1.5",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    db = Chroma(
        persist_directory=VECTORSTORE_PATH, embedding_function=embedding_function
    )

    llm = ChatOllama(
        model="llama3.1", temperature=0, num_predict=500, num_ctx=100000, streaming=True
    )

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 20, "fetch_k": 100, "lambda_mult": 0.7},
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PromptTemplate.from_template(PROMPT)
        | llm
    )

    return chain
