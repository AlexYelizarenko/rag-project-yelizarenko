import json

import pytest

from retrieval.index import INDEX_DIR, META_KEYS, chroma_metadata, open_collection
from retrieval.search import dense_search


pytestmark = pytest.mark.skipif(
    not INDEX_DIR.exists(),
    reason="run `uv run python -m retrieval.index` first",
)


def test_index_has_all_chunks_and_metadata():
    collection = open_collection()
    assert collection.count() == 540
    sample = collection.get(ids=["article_3_def_4"], include=["metadatas", "documents"])
    meta = sample["metadatas"][0]
    assert meta["chunk_id"] == "article_3_def_4"
    assert meta["section_type"] == "article"
    assert meta["binding"] is True
    assert "deployer" in sample["documents"][0]


def test_chroma_metadata_only_allows_scalar_fields():
    meta = chroma_metadata({
        "chunk_id": "x",
        "binding": True,
        "article_number": 3,
        "nested": {"no": "way"},
        "missing_ok": None,
    })
    assert meta == {"chunk_id": "x", "binding": True, "article_number": 3}
    assert set(meta) <= set(META_KEYS)


def test_dense_search_returns_top_k_with_scores():
    hits = dense_search("What is a deployer under the AI Act?", k=5)
    assert [hit["rank"] for hit in hits] == [1, 2, 3, 4, 5]
    assert hits[0]["chunk_id"] == "article_3_def_4"
    assert 0.0 <= hits[-1]["score"] <= hits[0]["score"] <= 1.0
    assert set(hits[0]) == {"rank", "chunk_id", "score", "document", "metadata"}
