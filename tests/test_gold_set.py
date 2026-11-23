import json
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GOLD_SET = ROOT / "tests" / "gold-set.jsonl"
CHUNKS = ROOT / "reports" / "lab-2" / "chunks.jsonl"


@pytest.fixture(scope="module")
def gold_set():
    return [json.loads(line) for line in GOLD_SET.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_size_mix_and_uniqueness(gold_set):
    types = Counter(item["type"] for item in gold_set)
    assert 30 <= len(gold_set) <= 40
    assert 5 <= types["multi_chunk"] <= 7 and 3 <= types["negative"] <= 5
    assert set(types) == {"simple", "multi_chunk", "negative"}
    queries = [item["query"] for item in gold_set] + [p for item in gold_set for p in item["paraphrases"]]
    assert len({item["id"] for item in gold_set}) == len(gold_set)
    assert len(set(queries)) == len(queries)


def test_item_shape(gold_set):
    for item in gold_set:
        assert item["expected_answer"], item["id"]
        if item["type"] == "negative":
            assert item["should_refuse"] and not item["relevant_chunk_ids"] and not item["key_points"], item["id"]
        else:
            assert not item["should_refuse"] and item["key_points"], item["id"]
            assert len(item["relevant_chunk_ids"]) == (1 if item["type"] == "simple" else 2), item["id"]


@pytest.mark.skipif(not CHUNKS.exists(), reason=f"{CHUNKS} not found")
def test_gold_chunks_exist_and_contain_evidence(gold_set):
    rows = [json.loads(line) for line in CHUNKS.read_text(encoding="utf-8").splitlines()]
    chunks = {row["metadata"]["chunk_id"]: " ".join(row["page_content"].split()) for row in rows}
    for item in gold_set:
        assert set(item["evidence"]) == set(item["relevant_chunk_ids"]), item["id"]
        for chunk_id, quote in item["evidence"].items():
            assert chunk_id in chunks, (item["id"], chunk_id)
            assert quote in chunks[chunk_id], (item["id"], chunk_id)
