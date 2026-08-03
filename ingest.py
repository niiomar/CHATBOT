from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import argparse
import hashlib
import json
import logging
import os
import shutil
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("chatbot_api.ingest")

DOCS_PATH = "./docs"
VECTORSTORE_PATH = "./vectorstore"
MANIFEST_PATH = os.path.join(VECTORSTORE_PATH, "manifest.json")
API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
API_KEY = os.environ.get("API_KEY")


def notify_api_refresh():
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    try:
        r = requests.post(f"{API_BASE_URL}/admin/refresh", headers=headers, timeout=5)
        r.raise_for_status()
        logger.info("notified running API to refresh its retrieval index")
    except requests.exceptions.RequestException:
        logger.info(
            "could not reach API at %s to refresh its index - restart it to pick up these changes", API_BASE_URL
        )

SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)


def list_doc_files():
    return [f for f in os.listdir(DOCS_PATH) if f.endswith((".pdf", ".docx", ".txt"))]


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def scan_files():
    return {f: file_hash(os.path.join(DOCS_PATH, f)) for f in list_doc_files()}


def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(manifest):
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def load_file(filename):
    path = os.path.join(DOCS_PATH, filename)
    logger.info("loading: %s", filename)
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
        return loaded

    except Exception:
        logger.exception("skipped %s", filename)
        return []


def load_documents():
    files = list_doc_files()

    if not files:
        logger.warning("no documents found in %s", DOCS_PATH)
        return []

    docs = []
    for filename in files:
        docs.extend(load_file(filename))
    return docs


def embeddings_client():
    import torch  # optional "ml" extra; kept out of module scope so light-weight code paths stay importable without it

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        cache_folder="models",
        model_name="./local_models/nomic-embed-text-v1.5",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def delete_source(db, filename):
    existing = db.get(where={"source_file": filename})
    ids = existing.get("ids") or []
    if ids:
        db.delete(ids=ids)
        logger.info("removed %d stale chunks for %s", len(ids), filename)


def full_rebuild():
    if os.path.exists(VECTORSTORE_PATH):
        shutil.rmtree(VECTORSTORE_PATH)
        logger.info("cleared old vectorstore")

    docs = load_documents()
    if not docs:
        return False

    chunks = SPLITTER.split_documents(docs)
    logger.info("%d chunks ready", len(chunks))

    Chroma.from_documents(chunks, embeddings_client(), persist_directory=VECTORSTORE_PATH)

    save_manifest(scan_files())
    logger.info("done, vectorstore saved to %s", VECTORSTORE_PATH)
    return True


def incremental_ingest():
    current_hashes = scan_files()
    if not current_hashes:
        logger.warning("no documents found in %s", DOCS_PATH)
        return False

    manifest = load_manifest()
    changed = [f for f, h in current_hashes.items() if manifest.get(f) != h]
    removed = [f for f in manifest if f not in current_hashes]

    if not changed and not removed:
        logger.info("no changes detected, vectorstore is up to date")
        return False

    db = Chroma(persist_directory=VECTORSTORE_PATH, embedding_function=embeddings_client())

    for filename in removed + changed:
        if filename in manifest:
            delete_source(db, filename)

    new_docs = []
    for filename in changed:
        new_docs.extend(load_file(filename))

    if new_docs:
        chunks = SPLITTER.split_documents(new_docs)
        db.add_documents(chunks)
        logger.info("added %d chunks from %d file(s)", len(chunks), len(changed))

    save_manifest(current_hashes)
    logger.info("done, vectorstore updated at %s", VECTORSTORE_PATH)
    return True


def ingest(rebuild=False):
    if rebuild or not os.path.exists(VECTORSTORE_PATH) or not os.path.exists(MANIFEST_PATH):
        changed = full_rebuild()
    else:
        changed = incremental_ingest()

    if changed:
        notify_api_refresh()


if __name__ == "__main__":
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    parser = argparse.ArgumentParser(description="Ingest documents from ./docs into the vectorstore.")
    parser.add_argument(
        "--rebuild", action="store_true", help="Wipe the vectorstore and re-ingest every document from scratch."
    )
    args = parser.parse_args()

    ingest(rebuild=args.rebuild)
