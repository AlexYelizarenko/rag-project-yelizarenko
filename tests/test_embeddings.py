import json

import numpy as np
import pytest

from retrieval.embeddings import CHUNKS_PATH, OUTPUT_DIR, embed_texts, load_chunks, load_model


@pytest.fixture(scope="module")
def model():
    return load_model()


def test_model_output_shape_and_normalization(model):
    vectors = embed_texts(model, ["What is a deployer?", "Article 5 – Prohibited AI practices"])
    assert vectors.shape == (2, 384)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_embeddings_capture_meaning(model):
    query, related, unrelated = embed_texts(model, [
        "Which uses of AI are banned?",
        "The following AI practices shall be prohibited.",
        "Member States shall designate a single point of contact.",
    ])
    assert query @ related > query @ unrelated


@pytest.mark.skipif(not (OUTPUT_DIR / "vectors.npy").exists(), reason="run `python -m retrieval.embeddings` first")
def test_saved_embeddings_match_corpus():
    chunks = load_chunks(CHUNKS_PATH)
    vectors = np.load(OUTPUT_DIR / "vectors.npy")
    chunk_ids = json.loads((OUTPUT_DIR / "chunk_ids.json").read_text(encoding="utf-8"))
    assert vectors.shape == (len(chunks), 384)
    assert chunk_ids == [chunk["metadata"]["chunk_id"] for chunk in chunks]
    assert np.isfinite(vectors).all()
