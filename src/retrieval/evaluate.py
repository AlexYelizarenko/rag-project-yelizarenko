import json

from retrieval.bm25 import BM25Index
from retrieval.embeddings import PROJECT_ROOT, load_model
from retrieval.index import open_collection
from retrieval.search import dense_search, hybrid_search

EVAL_SET = PROJECT_ROOT / "tests" / "eval-set.jsonl"
REPORT_DIR = PROJECT_ROOT / "reports" / "lab-4"
TOP_K = 10


def load_eval_set(path=EVAL_SET):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_experiment(name, eval_set, retrieve):
    ranks = []
    for item in eval_set:
        top = [hit["chunk_id"] for hit in retrieve(item["query"])]
        gold = item["relevant_chunk_id"]
        rank = top.index(gold) + 1 if gold in top else None
        ranks.append({"id": item["id"], "gold": gold, "rank": rank, "top": top})

    def recall(k):
        return round(sum(1 for row in ranks if row["rank"] is not None and row["rank"] <= k) / len(ranks), 3)

    return {
        "name": name,
        "recall_at_5": recall(5),
        "recall_at_10": recall(10),
        "misses_at_5": [row["id"] for row in ranks if row["rank"] is None or row["rank"] > 5],
        "misses_at_10": [row["id"] for row in ranks if row["rank"] is None],
        "ranks": ranks,
    }


def main():
    eval_set = load_eval_set()
    model = load_model()
    collection = open_collection()
    bm25 = BM25Index()
    bm25_no_stop = BM25Index(remove_stopwords=True)

    def dense(binding_only=False, prefix=""):
        return lambda q: dense_search(q, k=TOP_K, model=model, collection=collection,
                                      binding_only=binding_only, query_prefix=prefix)

    def hybrid(binding_only=False):
        return lambda q: hybrid_search(q, k=TOP_K, model=model, collection=collection, bm25=bm25,
                                       binding_only=binding_only)

    experiments = [
        run_experiment("dense", eval_set, dense()),
        run_experiment("dense_eu_ai_act_prefix", eval_set, dense(prefix="EU AI Act: ")),
        run_experiment("dense_regulation_prefix", eval_set, dense(prefix="Regulation (EU) 2024/1689: ")),
        run_experiment("dense_binding_only", eval_set, dense(binding_only=True)),
        run_experiment("bm25", eval_set, lambda q: bm25.search(q, k=TOP_K)),
        run_experiment("bm25_remove_stopwords", eval_set, lambda q: bm25_no_stop.search(q, k=TOP_K)),
        run_experiment("bm25_binding_only", eval_set, lambda q: bm25.search(q, k=TOP_K, binding_only=True)),
        run_experiment("hybrid", eval_set, hybrid()),
        run_experiment("hybrid_binding_only", eval_set, hybrid(binding_only=True)),
    ]
    best = max(experiments, key=lambda row: (row["recall_at_10"], row["recall_at_5"]))

    summary = {
        "best": best["name"],
        "notes": (
            "Best setting is chosen by (Recall@10, Recall@5). Exact gold chunk_id required; "
            "a neighbouring piece of the same article counts as a miss. Hybrid = RRF (k=60) over "
            "top-50 dense and top-50 BM25."
        ),
        "experiments": [{key: row[key] for key in ("name", "recall_at_5", "recall_at_10", "misses_at_5", "misses_at_10")}
                        for row in experiments],
        "ranks": {row["name"]: row["ranks"] for row in experiments},
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "recall.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{'setting':26} R@5    R@10   misses@10")
    for row in experiments:
        print(f"{row['name']:26} {row['recall_at_5']:.2f}   {row['recall_at_10']:.2f}   {', '.join(row['misses_at_10'])}")
    print(f"\nBest: {best['name']}")
    print(f"Saved detail to {out}")


if __name__ == "__main__":
    main()
