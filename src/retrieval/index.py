import json
import shutil
from pathlib import Path

import chromadb
import numpy as np

from retrieval.embeddings import CHUNKS_PATH, MODEL_NAME, OUTPUT_DIR, embed_texts, load_chunks, load_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = PROJECT_ROOT / "data" / "index" / "chroma"
COLLECTION_NAME = "eu_ai_act"
# Chroma only accepts str / int / float / bool metadata values.
META_KEYS = (
    "chunk_id",
    "chunk_index",
    "source_doc_id",
    "document",
    "source",
    "section_type",
    "binding",
    "page_number",
    "page_end",
    "parent_id",
    "is_partial",
    "article_number",
    "article_title",
    "chapter_number",
    "chapter_title",
    "section_number",
    "section_title",
    "annex_number",
    "annex_title",
    "paragraph",
    "point",
    "definition_number",
    "defined_term",
    "start_index",
)


def chroma_metadata(meta):
    return {key: meta[key] for key in META_KEYS if key in meta}


def load_or_compute_vectors(chunks):
    vectors_path = OUTPUT_DIR / "vectors.npy"
    ids_path = OUTPUT_DIR / "chunk_ids.json"
    chunk_ids = [chunk["metadata"]["chunk_id"] for chunk in chunks]
    if vectors_path.exists() and ids_path.exists():
        saved_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        if saved_ids == chunk_ids:
            return np.load(vectors_path)
    model = load_model()
    return embed_texts(model, [chunk["page_content"] for chunk in chunks])


def build_index(chunks_path=CHUNKS_PATH, index_dir=INDEX_DIR, reset=True):
    chunks = load_chunks(chunks_path)
    vectors = load_or_compute_vectors(chunks)
    if reset and index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(index_dir))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "model": MODEL_NAME},
    )
    batch = 100
    for start in range(0, len(chunks), batch):
        end = start + batch
        part = chunks[start:end]
        collection.add(
            ids=[chunk["metadata"]["chunk_id"] for chunk in part],
            documents=[chunk["page_content"] for chunk in part],
            embeddings=vectors[start:end].tolist(),
            metadatas=[chroma_metadata(chunk["metadata"]) for chunk in part],
        )
    return collection


def open_collection(index_dir=INDEX_DIR):
    if not index_dir.exists():
        raise FileNotFoundError(f"No Chroma index at {index_dir}. Run: uv run python -m retrieval.index")
    client = chromadb.PersistentClient(path=str(index_dir))
    return client.get_collection(COLLECTION_NAME)


def index_disk_bytes(index_dir=INDEX_DIR):
    return sum(path.stat().st_size for path in index_dir.rglob("*") if path.is_file())


def main():
    collection = build_index()
    size = index_disk_bytes()
    summary = {
        "collection": COLLECTION_NAME,
        "path": str(INDEX_DIR.relative_to(PROJECT_ROOT)),
        "num_vectors": collection.count(),
        "embedding_dimension": 384,
        "space": "cosine",
        "model": MODEL_NAME,
        "disk_bytes": size,
        "disk_mb": round(size / (1024 * 1024), 2),
    }
    (INDEX_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved to {INDEX_DIR}")


if __name__ == "__main__":
    main()
