import argparse
import sys

from retrieval.bm25 import BM25Index
from retrieval.embeddings import embed_texts, load_model
from retrieval.index import open_collection

RRF_K = 60
HYBRID_CANDIDATES = 50


def dense_search(query, k=5, model=None, collection=None, binding_only=False, query_prefix=""):
    if model is None:
        model = load_model()
    if collection is None:
        collection = open_collection()
    text = f"{query_prefix}{query}" if query_prefix else query
    embedding = embed_texts(model, [text])[0].tolist()
    kwargs = {
        "query_embeddings": [embedding],
        "n_results": k,
        "include": ["documents", "metadatas", "distances"],
    }
    if binding_only:
        kwargs["where"] = {"binding": True}
    result = collection.query(**kwargs)
    return [
        {
            "rank": rank,
            "chunk_id": chunk_id,
            # Cosine space: Chroma returns distance = 1 - cosine similarity.
            "score": round(1.0 - distance, 4),
            "document": document,
            "metadata": meta,
        }
        for rank, (chunk_id, document, meta, distance) in enumerate(
            zip(result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]),
            start=1,
        )
    ]


def rrf(rankings, k=RRF_K):
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank(d)), ranks from 1."""
    scores = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    # Ties are broken by chunk_id so the order is deterministic.
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def hybrid_search(query, k=5, model=None, collection=None, bm25=None, binding_only=False,
                  candidates=HYBRID_CANDIDATES, rrf_k=RRF_K):
    if bm25 is None:
        bm25 = BM25Index()
    dense_hits = dense_search(query, k=candidates, model=model, collection=collection, binding_only=binding_only)
    bm25_hits = bm25.search(query, k=candidates, binding_only=binding_only)
    by_id = {hit["chunk_id"]: hit for hit in bm25_hits + dense_hits}
    dense_rank = {hit["chunk_id"]: hit["rank"] for hit in dense_hits}
    bm25_rank = {hit["chunk_id"]: hit["rank"] for hit in bm25_hits}

    fused = rrf([[hit["chunk_id"] for hit in dense_hits], [hit["chunk_id"] for hit in bm25_hits]], k=rrf_k)
    return [
        {
            "rank": rank,
            "chunk_id": chunk_id,
            "score": round(score, 6),
            "dense_rank": dense_rank.get(chunk_id),
            "bm25_rank": bm25_rank.get(chunk_id),
            "document": by_id[chunk_id]["document"],
            "metadata": by_id[chunk_id]["metadata"],
        }
        for rank, (chunk_id, score) in enumerate(fused[:k], start=1)
    ]


def _preview(document):
    lines = [line.strip() for line in document.splitlines() if line.strip()]
    for line in lines[:4]:
        # Prefer a title line ("Article N – …") or a longer content line over a bare "Article N".
        if "–" in line or "—" in line or len(line) > 40:
            return line[:120]
    return (lines[0] if lines else "")[:120]


def format_hits(hits):
    lines = []
    for hit in hits:
        meta = hit["metadata"]
        line = (f"{hit['rank']}. chunk_id={hit['chunk_id']}  score={hit['score']:.4f}  "
                f"pages={meta.get('page_number')}-{meta.get('page_end')}  binding={meta.get('binding')}")
        if "dense_rank" in hit:
            line += f"  dense_rank={hit['dense_rank']}  bm25_rank={hit['bm25_rank']}"
        lines.append(line)
        lines.append(f"   {_preview(hit['document'])}")
    return "\n".join(lines)


SCORE_NOTES = {
    "dense": "score = cosine similarity = 1 - Chroma distance",
    "bm25": "score = BM25 (Okapi) score, not comparable across queries",
    "hybrid": f"score = RRF, sum of 1/({RRF_K} + rank) over dense and BM25 (top-{HYBRID_CANDIDATES} each)",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Search the EU AI Act chunk index")
    parser.add_argument("query", nargs="+", help="Search query")
    parser.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="dense")
    parser.add_argument("--k", type=int, default=5, help="Number of results (default 5)")
    parser.add_argument("--prefix", default="", help='Dense only: optional query prefix, e.g. "EU AI Act: "')
    parser.add_argument("--binding-only", action="store_true", help="Search only binding chunks (exclude recitals)")
    args = parser.parse_args(argv)
    query = " ".join(args.query)

    if args.mode == "dense":
        hits = dense_search(query, k=args.k, binding_only=args.binding_only, query_prefix=args.prefix)
    elif args.mode == "bm25":
        hits = BM25Index().search(query, k=args.k, binding_only=args.binding_only)
    else:
        hits = hybrid_search(query, k=args.k, binding_only=args.binding_only)

    print(f"Query: {query}")
    print(f"Mode: {args.mode}{' (binding only)' if args.binding_only else ''}  Top-{args.k} ({SCORE_NOTES[args.mode]})\n")
    print(format_hits(hits))


if __name__ == "__main__":
    main(sys.argv[1:])
