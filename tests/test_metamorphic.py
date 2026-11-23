import os

import pytest
from dotenv import load_dotenv

from rag.evaluate import load_gold_set, metamorphic, run
from retrieval.index import INDEX_DIR, PROJECT_ROOT

# Metamorphic relation: a paraphrase keeps the meaning, so retrieval and citations should barely change.
# Thresholds are fixed here, not tuned: Recall@5 may drop by at most 0.10, mean Jaccard overlap must be >= 0.5.
MAX_RECALL_DROP = 0.10
MIN_JACCARD = 0.5

load_dotenv(PROJECT_ROOT / ".env")
pytestmark = pytest.mark.skipif(not INDEX_DIR.exists(), reason="Chroma index not built")


@pytest.fixture(scope="module")
def paraphrase_items():
    return [item for item in load_gold_set() if item["paraphrases"]]


@pytest.fixture(scope="module")
def rag():
    from rag.ask import RAG
    return RAG()


@pytest.fixture(scope="module")
def retrieval_report(rag, paraphrase_items):
    rows, _ = run(paraphrase_items, rag, llm=False)
    return metamorphic(rows)


@pytest.mark.xfail(strict=True, reason="known failure mode: paraphrases lose the gold chunk (see reports/lab-5)")
def test_mr1_recall_at_5_is_invariant_to_paraphrasing(retrieval_report):
    original = retrieval_report["recall_at_5_original"]["value"]
    paraphrased = retrieval_report["recall_at_5_paraphrase"]["value"]
    assert original - paraphrased <= MAX_RECALL_DROP, retrieval_report["recall_flips"]


@pytest.mark.xfail(strict=True, reason="known failure mode: top-5 sets of paraphrases overlap little (see reports/lab-5)")
def test_mr2_top5_sets_overlap_for_paraphrases(retrieval_report):
    assert retrieval_report["top5_jaccard_mean"] >= MIN_JACCARD


@pytest.mark.llm
@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="needs GROQ_API_KEY")
def test_mr3_cited_chunks_overlap_for_paraphrases(rag, paraphrase_items):
    rows, stopped = run(paraphrase_items, rag, max_new=len(paraphrase_items) * 2)
    report = metamorphic(rows)
    if not report["cited_pairs"]:
        pytest.skip(f"no answered paraphrase pairs ({stopped})")
    assert report["cited_jaccard_mean"] >= MIN_JACCARD, report["cited_jaccard_per_pair"]
