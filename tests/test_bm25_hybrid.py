import pytest

from retrieval.bm25 import BM25Index, tokenize
from retrieval.index import INDEX_DIR
from retrieval.search import hybrid_search, rrf

TOY_CHUNKS = [
    {"page_content": "Article 5 – Prohibited AI practices. Social scoring is prohibited.",
     "metadata": {"chunk_id": "a5", "binding": True}},
    {"page_content": "A deployer means a person using an AI system under its authority.",
     "metadata": {"chunk_id": "def_deployer", "binding": True}},
    {"page_content": "Recital about the deployer and the purpose of the rules.",
     "metadata": {"chunk_id": "recital", "binding": False}},
    {"page_content": "Member States shall designate market surveillance authorities.",
     "metadata": {"chunk_id": "a70", "binding": True}},
    # Okapi IDF is zero for a term in exactly half the documents, so the toy corpus needs filler.
    {"page_content": "Fines for infringements shall be effective and proportionate.",
     "metadata": {"chunk_id": "a99", "binding": True}},
    {"page_content": "This Regulation shall enter into force on the twentieth day.",
     "metadata": {"chunk_id": "a113", "binding": True}},
]


def test_tokenize_keeps_article_numbers_and_optionally_drops_stopwords():
    assert tokenize("Article 6(3) – High-Risk") == ["article", "6", "3", "high", "risk"]
    assert tokenize("What is a deployer?") == ["what", "is", "a", "deployer"]
    assert tokenize("What is a deployer under the AI Act?", remove_stopwords=True) == ["deployer", "under", "ai", "act"]
    assert tokenize("shall not apply", remove_stopwords=True) == ["shall", "not", "apply"]


def test_bm25_ranks_exact_term_and_respects_binding_filter():
    index = BM25Index(TOY_CHUNKS)
    hits = index.search("deployer", k=5)
    assert {hit["chunk_id"] for hit in hits} == {"def_deployer", "recital"}
    assert [hit["rank"] for hit in hits] == [1, 2]

    binding = index.search("deployer", k=5, binding_only=True)
    assert [hit["chunk_id"] for hit in binding] == ["def_deployer"]


def test_bm25_returns_nothing_for_unknown_terms():
    assert BM25Index(TOY_CHUNKS).search("zebra", k=5) == []


def test_rrf_matches_hand_computed_scores():
    fused = dict(rrf([["a", "b", "c"], ["b", "d"]], k=60))
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["d"] == pytest.approx(1 / 62)
    assert fused["c"] == pytest.approx(1 / 63)
    assert [chunk_id for chunk_id, _ in rrf([["a", "b", "c"], ["b", "d"]])] == ["b", "a", "d", "c"]


@pytest.mark.skipif(not INDEX_DIR.exists(), reason="run `uv run python -m retrieval.index` first")
def test_hybrid_search_on_real_corpus():
    hits = hybrid_search("What is a deployer under the AI Act?", k=5)
    assert [hit["rank"] for hit in hits] == [1, 2, 3, 4, 5]
    assert len({hit["chunk_id"] for hit in hits}) == 5
    assert hits[0]["score"] >= hits[-1]["score"]
    assert hits[0]["chunk_id"] == "article_3_def_4"
    assert all(hit["dense_rank"] or hit["bm25_rank"] for hit in hits)
