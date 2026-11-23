import httpx
from groq import RateLimitError

from rag.evaluate import jaccard, key_point_hits, metamorphic, run

GOLD = [
    {"id": "a", "type": "simple", "query": "qa", "relevant_chunk_ids": ["c1"], "key_points": ["deployer"],
     "paraphrases": ["qa2"]},
    {"id": "b", "type": "simple", "query": "qb", "relevant_chunk_ids": ["c2"], "key_points": ["x"], "paraphrases": []},
]


class FakeRAG:
    """Answers the first `quota` queries, then raises the error Groq gives when the quota is used up."""

    def __init__(self, quota):
        self.quota, self.llm_calls = quota, 0

    def retrieve(self, query, k=None):
        return [{"chunk_id": "c1"}, {"chunk_id": "c3"}]

    def answer(self, query, cache_only=False):
        if self.llm_calls >= self.quota:
            response = httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com"))
            raise RateLimitError("Rate limit reached: tokens per day", response=response, body=None)
        self.llm_calls += 1
        return {"status": "ok", "retried": False, "cached": False, "retrieved": [{"chunk_id": "c1"}],
                "citations": [{"chunk_id": "c1"}], "answer": "A deployer [1].", "seconds": 1.0}


def test_run_stops_calling_the_model_on_rate_limit_and_keeps_results():
    rows, stopped = run(GOLD, FakeRAG(quota=1))
    assert stopped.startswith("Groq rate limit")
    assert [r["status"] for r in rows] == ["ok", None, None]
    assert all(r["top5"] for r in rows)  # retrieval metrics are still computed for every query


def test_key_points_ignore_case_hyphen_and_special_spaces():
    answer = "Up to EUR 35\u202f000\u202f000 and a Fundamental\u2011Rights check."
    assert key_point_hits(answer, ["35 000 000|35 million", "fundamental rights", "missing"]) == 2


def test_metamorphic_compares_original_and_paraphrase():
    rows, _ = run(GOLD, FakeRAG(quota=10))
    report = metamorphic(rows)
    assert report["pairs"] == 1 and report["top5_jaccard_mean"] == 1.0 and report["cited_jaccard_mean"] == 1.0
    assert jaccard(["a", "b"], ["b", "c"]) == 1 / 3
