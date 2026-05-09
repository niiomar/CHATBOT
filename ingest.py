from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
import shutil
import torch

DOCS_PATH = "./docs"
VECTORSTORE_PATH = "./vectorstore"


def load_documents():
    docs = []
    files = [f for f in os.listdir(DOCS_PATH) if f.endswith((".pdf", ".docx", ".txt"))]

    if not files:
        print("No documents found in ./docs")
        return docs

    for filename in files:
        path = os.path.join(DOCS_PATH, filename)
        print(f"Loading: {filename}")
        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(path)
            else:
                loader = TextLoader(path)

            loaded = loader.load()
            for d in loaded:
                d.metadata["source_file"] = filename
            docs.extend(loaded)

        except Exception as e:
            print(f"Skipped {filename}: {e}")

    return docs


def ingest():
    if os.path.exists(VECTORSTORE_PATH):
        shutil.rmtree(VECTORSTORE_PATH)
        print("Cleared old vectorstore.")

    docs = load_documents()
    if not docs:
        return
    
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=200
    ).split_documents(docs)
    print(f"{len(chunks)} chunks ready.")
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        cache_folder="models",
        model_name="./local_models/nomic-embed-text-v1.5",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

    Chroma.from_documents(chunks, embeddings, persist_directory=VECTORSTORE_PATH)
    print(f"Done. Vectorstore saved to {VECTORSTORE_PATH}")


if __name__ == "__main__":
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    ingest()