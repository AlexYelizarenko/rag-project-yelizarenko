import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
EVAL_SET = ROOT / "tests" / "eval-set.jsonl"
CHUNKS = ROOT / "reports" / "lab-2" / "chunks.jsonl"

pytestmark = pytest.mark.skipif(not CHUNKS.exists(), reason=f"{CHUNKS} not found")


@pytest.fixture(scope="module")
def eval_set():
    return [json.loads(line) for line in EVAL_SET.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture(scope="module")
def chunks():
    rows = [json.loads(line) for line in CHUNKS.read_text(encoding="utf-8").splitlines()]
    return {row["metadata"]["chunk_id"]: row["page_content"] for row in rows}


def test_size_and_uniqueness(eval_set):
    assert 15 <= len(eval_set) <= 25
    assert len({item["id"] for item in eval_set}) == len(eval_set)
    assert len({item["query"] for item in eval_set}) == len(eval_set)
    assert len({item["relevant_chunk_id"] for item in eval_set}) == len(eval_set)
    assert len({item["topic"] for item in eval_set}) == len(eval_set)


def test_gold_chunks_exist_and_contain_evidence(eval_set, chunks):
    for item in eval_set:
        assert item["relevant_chunk_id"] in chunks, item["id"]
        text = " ".join(chunks[item["relevant_chunk_id"]].split())
        assert item["evidence"] in text, item["id"]
