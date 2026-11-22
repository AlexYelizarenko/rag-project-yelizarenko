import json
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = PROJECT_ROOT / "reports" / "lab-2" / "chunks.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "embeddings"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32


def load_chunks(path=CHUNKS_PATH):
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_model(model_name=MODEL_NAME):
    return SentenceTransformer(model_name, device="cpu")


def embed_texts(model, texts):
    # Unit-length vectors: cosine similarity equals the dot product, for both chunks and queries.
    return model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=len(texts) > BATCH_SIZE,
    )


def main():
    chunks = load_chunks()
    texts = [chunk["page_content"] for chunk in chunks]
    chunk_ids = [chunk["metadata"]["chunk_id"] for chunk in chunks]

    started = time.perf_counter()
    model = load_model()
    load_seconds = time.perf_counter() - started

    started = time.perf_counter()
    vectors = embed_texts(model, texts)
    encode_seconds = time.perf_counter() - started

    # Count tokens without special tokens; the model adds 2 ([CLS], [SEP]) within max_seq_length.
    token_counts = [len(model.tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"]) for text in texts]
    token_limit = model.max_seq_length - 2
    truncated = [chunk for chunk, count in zip(chunks, token_counts) if count > token_limit]

    summary = {
        "model": MODEL_NAME,
        "embedding_dimension": model.get_embedding_dimension(),
        "max_seq_length_tokens": model.max_seq_length,
        "normalized": True,
        "similarity": "cosine",
        "num_vectors": len(vectors),
        "vector_dtype": str(vectors.dtype),
        "source": str(CHUNKS_PATH.relative_to(PROJECT_ROOT)),
        "model_load_seconds": round(load_seconds, 1),
        "encode_seconds": round(encode_seconds, 1),
        "chunk_tokens": {
            "min": min(token_counts),
            "median": int(statistics.median(token_counts)),
            "mean": round(statistics.mean(token_counts), 1),
            "max": max(token_counts),
        },
        "chunks_truncated": len(truncated),
        "chunks_truncated_by_type": dict(Counter(chunk["metadata"]["section_type"] for chunk in truncated)),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "vectors.npy", vectors)
    (OUTPUT_DIR / "chunk_ids.json").write_text(json.dumps(chunk_ids, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
