import argparse
import sys

from retrieval.embeddings import embed_texts, load_model
from retrieval.index import open_collection


def search(query, k=5, model=None, collection=None, query_prefix="", binding_only=False):
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
    hits = []
    for chunk_id, document, meta, distance in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        hits.append({
            "chunk_id": chunk_id,
            "score": round(1.0 - distance, 4),
            "distance": round(distance, 4),
            "page_number": meta.get("page_number"),
            "page_end": meta.get("page_end"),
            "section_type": meta.get("section_type"),
            "binding": meta.get("binding"),
            "preview": _preview(document),
            "document": document,
            "metadata": meta,
        })
    return hits


def _preview(document):
    lines = [line.strip() for line in document.splitlines() if line.strip()]
    for line in lines[:4]:
        if "–" in line or "—" in line or len(line) > 40:
            return line[:120]
    return (lines[0] if lines else "")[:120]


def format_hits(hits):
    lines = []
    for rank, hit in enumerate(hits, start=1):
        pages = f"{hit['page_number']}-{hit['page_end']}"
        lines.append(
            f"{rank}. chunk_id={hit['chunk_id']}  score={hit['score']:.4f}  "
            f"pages={pages}  binding={hit['binding']}"
        )
        lines.append(f"   {hit['preview']}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Search the EU AI Act chunk index")
    parser.add_argument("query", nargs="+", help="Search query")
    parser.add_argument("--k", type=int, default=5, help="Number of results (default 5)")
    parser.add_argument(
        "--prefix",
        default="",
        help='Optional query prefix, e.g. "EU AI Act: "',
    )
    parser.add_argument(
        "--binding-only",
        action="store_true",
        help="Search only binding chunks (exclude recitals)",
    )
    args = parser.parse_args(argv)
    query = " ".join(args.query)
    hits = search(query, k=args.k, query_prefix=args.prefix, binding_only=args.binding_only)
    print(f"Query: {args.prefix}{query}")
    print(f"Top-{args.k} (score = cosine similarity = 1 - Chroma distance)\n")
    print(format_hits(hits))


if __name__ == "__main__":
    main(sys.argv[1:])
