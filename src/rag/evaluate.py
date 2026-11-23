import argparse
import json
import sys
from statistics import mean

from groq import APIError, RateLimitError

from rag.ask import RAG
from retrieval.index import PROJECT_ROOT

GOLD_SET_PATH = PROJECT_ROOT / "tests" / "gold-set.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "lab-5"
RECALL_K = 5
DEFAULT_MAX_NEW = 60
# The model writes "fundamental‑rights" (U+2011) or "35 000 000" (U+202F); hyphens and special spaces all become
# plain spaces on both sides, so "fundamental rights" and "fundamental-rights" match.
FOLD = str.maketrans({"-": " ", "\u2010": " ", "\u2011": " ", "\u2012": " ", "\u2013": " ", "\u00a0": " ",
                      "\u202f": " "})


def load_gold_set(path=GOLD_SET_PATH):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def key_point_hits(answer, key_points):
    """A key point is a set of '|'-separated alternatives; it is hit if any alternative occurs in the answer."""
    text = answer.translate(FOLD).lower()
    return sum(any(alt.translate(FOLD).lower() in text for alt in point.split("|")) for point in key_points)


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def run(gold_set, rag, max_new=DEFAULT_MAX_NEW, with_paraphrases=True, llm=True):
    """One row per query (originals first, then paraphrases). Stops calling Groq on a rate limit or API error."""
    jobs = [(item, "original", item["query"]) for item in gold_set]
    if with_paraphrases:
        jobs += [(item, "paraphrase", p) for item in gold_set for p in item["paraphrases"]]

    rows, new_answers, stopped = [], 0, None
    for item, variant, query in jobs:
        top = [hit["chunk_id"] for hit in rag.retrieve(query, k=RECALL_K)]
        gold = item["relevant_chunk_ids"]
        row = {"id": item["id"], "type": item["type"], "variant": variant, "query": query, "gold": gold,
               "top5": top, "recall_any": any(g in top for g in gold) if gold else None,
               "recall_all": all(g in top for g in gold) if gold else None, "status": None}
        if llm and stopped is None:
            try:
                cached_before = rag.llm_calls
                if new_answers >= max_new:
                    result = rag.answer(query, cache_only=True)
                else:
                    result = rag.answer(query)
                new_answers += rag.llm_calls > cached_before
            except RateLimitError as error:
                stopped, result = f"Groq rate limit: {error.message}", None
            except APIError as error:
                stopped, result = f"Groq API error: {error.message}", None
            if result:
                cited = [c["chunk_id"] for c in result["citations"]]
                row.update(status=result["status"], retried=result["retried"], cached=result["cached"],
                           retrieved=[h["chunk_id"] for h in result["retrieved"]], cited=cited,
                           cites_gold=any(g in cited for g in gold) if gold else None,
                           key_points=[item["key_points"], key_point_hits(result["answer"], item["key_points"])],
                           answer=result["answer"], seconds=result["seconds"])
        rows.append(row)
    return rows, stopped


def share(values):
    values = list(values)
    return {"value": round(sum(values) / len(values), 3), "n": len(values), "hits": sum(values)} if values else None


def summarize(rows, gold_set):
    originals = [r for r in rows if r["variant"] == "original"]
    positive = [r for r in originals if r["type"] != "negative"]
    answered = [r for r in originals if r["status"]]
    answered_positive = [r for r in answered if r["type"] != "negative"]
    negatives = [r for r in answered if r["type"] == "negative"]
    ok = [r for r in answered_positive if r["status"] == "ok"]
    lab3_ids = {f"q{n:02d}" for n in range(1, 21)}

    metrics = {
        "recall_at_5": {
            "all_positive": share(r["recall_any"] for r in positive),
            "simple": share(r["recall_any"] for r in positive if r["type"] == "simple"),
            "multi_chunk_any": share(r["recall_any"] for r in positive if r["type"] == "multi_chunk"),
            "multi_chunk_all": share(r["recall_all"] for r in positive if r["type"] == "multi_chunk"),
            "lab3_queries": share(r["recall_any"] for r in positive if r["id"] in lab3_ids),
            "new_queries": share(r["recall_any"] for r in positive if r["id"] not in lab3_ids),
        },
        "answered": {"originals": len(answered), "of": len(originals)},
        # Contract from Lab 4 held: a valid JSON reply whose citations all point to retrieved sources.
        "valid_citations": share(r["status"] != "uncertain" for r in answered),
        "valid_on_first_attempt": share(r["status"] != "uncertain" and not r["retried"] for r in answered),
        "correct_refusals": share(r["status"] == "insufficient_context" for r in negatives),
        "correct_refusals_borderline": share(r["status"] == "insufficient_context" for r in negatives
                                             if next(i for i in gold_set if i["id"] == r["id"]).get("borderline")),
        "false_refusals": share(r["status"] == "insufficient_context" for r in answered_positive),
        "uncertain": share(r["status"] == "uncertain" for r in answered),
        "blocked_by_guard": share(r["status"] == "blocked" for r in answered),
        "ok_answers_citing_gold_chunk": share(r["cites_gold"] for r in ok),
        "key_points_all_present": share(r["key_points"][1] == len(r["key_points"][0]) for r in ok),
        "key_points_mean_coverage": round(mean(r["key_points"][1] / len(r["key_points"][0]) for r in ok), 3) if ok else None,
        # Time of the original (uncached) model call, stored with the cached answer.
        "mean_seconds_per_answer": round(mean(r["seconds"] for r in answered), 2) if answered else None,
    }
    metrics["metamorphic"] = metamorphic(rows)
    return metrics


def metamorphic(rows):
    """Paraphrase invariance: Recall@5, overlap of top-5 chunk sets and of cited chunk sets."""
    by_id = {r["id"]: r for r in rows if r["variant"] == "original"}
    pairs = [(by_id[r["id"]], r) for r in rows if r["variant"] == "paraphrase"]
    if not pairs:
        return None
    cited_pairs = [(a, b) for a, b in pairs if a["status"] == "ok" and b["status"] == "ok"]
    return {
        "pairs": len(pairs),
        "recall_at_5_original": share(a["recall_any"] for a, _ in pairs),
        "recall_at_5_paraphrase": share(b["recall_any"] for _, b in pairs),
        "recall_flips": [a["id"] for a, b in pairs if a["recall_any"] != b["recall_any"]],
        "top5_jaccard_mean": round(mean(jaccard(a["top5"], b["top5"]) for a, b in pairs), 3),
        "top1_same": share(a["top5"][:1] == b["top5"][:1] for a, b in pairs),
        "cited_pairs": len(cited_pairs),
        "cited_jaccard_mean": round(mean(jaccard(a["cited"], b["cited"]) for a, b in cited_pairs), 3)
        if cited_pairs else None,
        "cited_jaccard_per_pair": {a["id"]: round(jaccard(a["cited"], b["cited"]), 2) for a, b in cited_pairs},
    }


def print_summary(metrics, stopped):
    def fmt(entry):
        return "n/a" if entry is None else f"{entry['value']:.2f} ({entry['hits']}/{entry['n']})"

    print("Recall@5 (hybrid, binding only)")
    for name, entry in metrics["recall_at_5"].items():
        print(f"  {name:<22} {fmt(entry)}")
    print(f"Answered originals: {metrics['answered']['originals']}/{metrics['answered']['of']}")
    for name in ("valid_citations", "valid_on_first_attempt", "correct_refusals", "correct_refusals_borderline",
                 "false_refusals", "uncertain", "blocked_by_guard", "ok_answers_citing_gold_chunk",
                 "key_points_all_present"):
        print(f"  {name:<30} {fmt(metrics[name])}")
    print(f"  key_points_mean_coverage       {metrics['key_points_mean_coverage']}")
    meta = metrics["metamorphic"]
    if meta:
        print(f"Metamorphic ({meta['pairs']} paraphrase pairs): Recall@5 {fmt(meta['recall_at_5_original'])} -> "
              f"{fmt(meta['recall_at_5_paraphrase'])}, top-5 Jaccard {meta['top5_jaccard_mean']}, "
              f"same top-1 {fmt(meta['top1_same'])}, cited Jaccard {meta['cited_jaccard_mean']} "
              f"({meta['cited_pairs']} pairs)")
    if stopped:
        print(f"\nStopped calling the model early ({stopped}). Results so far are saved; rerun later to continue.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate the RAG system on tests/gold-set.jsonl")
    parser.add_argument("--max-new", type=int, default=DEFAULT_MAX_NEW,
                        help=f"at most this many new (uncached) answers per run (default {DEFAULT_MAX_NEW})")
    parser.add_argument("--no-paraphrases", action="store_true")
    parser.add_argument("--retrieval-only", action="store_true", help="do not call the model at all")
    args = parser.parse_args(argv)

    gold_set = load_gold_set()
    rag = RAG()
    rows, stopped = run(gold_set, rag, max_new=args.max_new, with_paraphrases=not args.no_paraphrases,
                        llm=not args.retrieval_only)
    metrics = summarize(rows, gold_set)
    metrics.update(model=rag.model, mode=rag.mode, k=rag.k, binding_only=rag.binding_only, stopped_early=stopped)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "eval-rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(metrics, stopped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
