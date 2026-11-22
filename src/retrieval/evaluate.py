import json
from pathlib import Path

from retrieval.embeddings import PROJECT_ROOT, embed_texts, load_model
from retrieval.index import open_collection

EVAL_SET = PROJECT_ROOT / "tests" / "eval-set.jsonl"
REPORT_DIR = PROJECT_ROOT / "reports" / "lab-3"


def load_eval_set(path=EVAL_SET):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def recall_at_k(hits_by_query, gold_ids, k):
    hits = 0
    for retrieved, gold in zip(hits_by_query, gold_ids):
        if gold in {item["chunk_id"] for item in retrieved[:k]}:
            hits += 1
    return hits / len(gold_ids)


def run_experiment(name, eval_set, model, collection, query_prefix="", where=None):
    gold_ids = [item["relevant_chunk_id"] for item in eval_set]
    # Embed all queries once; search() would reload work per query.
    texts = [f"{query_prefix}{item['query']}" if query_prefix else item["query"] for item in eval_set]
    embeddings = embed_texts(model, texts)
    hits_by_query = []
    ranks = []
    for item, embedding in zip(eval_set, embeddings):
        kwargs = {
            "query_embeddings": [embedding.tolist()],
            "n_results": 10,
            "include": ["metadatas", "distances"],
        }
        if where is not None:
            kwargs["where"] = where
        result = collection.query(**kwargs)
        retrieved = [
            {"chunk_id": chunk_id, "score": round(1.0 - distance, 4)}
            for chunk_id, distance in zip(result["ids"][0], result["distances"][0])
        ]
        hits_by_query.append(retrieved)
        gold = item["relevant_chunk_id"]
        try:
            rank = next(i for i, hit in enumerate(retrieved, start=1) if hit["chunk_id"] == gold)
        except StopIteration:
            rank = None
        ranks.append({"id": item["id"], "gold": gold, "rank": rank, "top": [h["chunk_id"] for h in retrieved]})

    return {
        "name": name,
        "query_prefix": query_prefix,
        "n_queries": len(eval_set),
        "recall_at_5": round(recall_at_k(hits_by_query, gold_ids, 5), 3),
        "recall_at_10": round(recall_at_k(hits_by_query, gold_ids, 10), 3),
        "misses_at_5": [row for row in ranks if row["rank"] is None or row["rank"] > 5],
        "misses_at_10": [row for row in ranks if row["rank"] is None or row["rank"] > 10],
        "ranks": ranks,
    }


def main():
    eval_set = load_eval_set()
    model = load_model()
    collection = open_collection()

    experiments = [
        run_experiment("raw_query", eval_set, model, collection, query_prefix=""),
        run_experiment("eu_ai_act_prefix", eval_set, model, collection, query_prefix="EU AI Act: "),
        run_experiment("regulation_prefix", eval_set, model, collection, query_prefix="Regulation (EU) 2024/1689: "),
        # Drop non-binding recitals; hurts q20 (gold is a recital) but may help obligation queries.
        run_experiment("binding_only", eval_set, model, collection, where={"binding": True}),
    ]
    best = max(experiments, key=lambda row: (row["recall_at_10"], row["recall_at_5"]))

    summary = {
        "best": best["name"],
        "experiments": [
            {
                "name": row["name"],
                "query_prefix": row["query_prefix"],
                "recall_at_5": row["recall_at_5"],
                "recall_at_10": row["recall_at_10"],
                "misses_at_5": [m["id"] for m in row["misses_at_5"]],
                "misses_at_10": [m["id"] for m in row["misses_at_10"]],
            }
            for row in experiments
        ],
        "notes": (
            "Best setting is chosen by (Recall@10, Recall@5). "
            "Exact gold chunk_id required; a neighbouring piece of the same article counts as a miss."
        ),
        "best_detail": {
            "name": best["name"],
            "recall_at_5": best["recall_at_5"],
            "recall_at_10": best["recall_at_10"],
            "ranks": best["ranks"],
        },
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "recall.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({"best": summary["best"], "experiments": summary["experiments"]}, indent=2))
    print(f"\nSaved detail to {out}")
    print("\nBest setting misses @5:")
    for miss in best["misses_at_5"]:
        print(f"  {miss['id']} gold={miss['gold']} rank={miss['rank']} top={miss['top'][:3]}")


if __name__ == "__main__":
    main()
