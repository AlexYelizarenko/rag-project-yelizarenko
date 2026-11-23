import argparse
import hashlib
import json
import sys
import time

from groq import RateLimitError

from rag.generate import (BLOCKED_ANSWER, DEFAULT_K, DISCLAIMER, SYSTEM_PROMPT, check_answer, check_query,
                          generate, load_llm, model_name)
from retrieval.bm25 import BM25Index
from retrieval.embeddings import load_model
from retrieval.index import PROJECT_ROOT, open_collection
from retrieval.search import dense_search, hybrid_search

CACHE_PATH = PROJECT_ROOT / "reports" / "lab-5" / "llm-cache.json"


class RAG:
    """Loads the retrievers once; answers are cached so repeated runs do not spend the Groq quota."""

    def __init__(self, mode="hybrid", k=DEFAULT_K, binding_only=True, llm=None, cache_path=CACHE_PATH, guard=True):
        self.mode, self.k, self.binding_only, self.guard = mode, k, binding_only, guard
        self.embedder, self.collection, self.bm25 = load_model(), open_collection(), BM25Index()
        self.model = model_name()
        self._llm = llm
        self.cache_path = cache_path
        self.cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path and cache_path.exists() else {}
        self.llm_calls = 0

    def retrieve(self, query, k=None):
        k = k or self.k
        if self.mode == "dense":
            return dense_search(query, k=k, model=self.embedder, collection=self.collection,
                                binding_only=self.binding_only)
        if self.mode == "bm25":
            return self.bm25.search(query, k=k, binding_only=self.binding_only)
        return hybrid_search(query, k=k, model=self.embedder, collection=self.collection, bm25=self.bm25,
                             binding_only=self.binding_only)

    def answer(self, query, use_cache=True, cache_only=False):
        """Retrieve + generate. Raises groq.RateLimitError when the quota is used up (callers stop there).
        With cache_only=True, returns None instead of calling the model for an uncached query."""
        hits = self.retrieve(query)
        if self.guard and (reason := check_query(query)):
            return self._blocked(query, hits, reason, {"attempts": [], "retried": False, "seconds": 0.0,
                                                       "disclaimer": DISCLAIMER, "cached": False})
        result = self._answer(query, hits, use_cache, cache_only)
        if result and self.guard and (reason := check_answer(result["answer"], result["status"])):
            return self._blocked(query, hits, reason, result)
        return result

    def _blocked(self, query, hits, reason, result):
        return {**result, "query": query, "mode": self.mode, "k": self.k, "binding_only": self.binding_only,
                "model": self.model, "retrieved": [{"rank": h["rank"], "chunk_id": h["chunk_id"], "score": h["score"]}
                                                   for h in hits],
                "status": "blocked", "answer": BLOCKED_ANSWER, "citations": [], "blocked_reason": reason,
                "sources": result.get("sources", [])}

    def _answer(self, query, hits, use_cache, cache_only):
        # The key covers everything the model sees, so a changed prompt or retrieval result is not served stale.
        key = hashlib.sha1(json.dumps([self.model, SYSTEM_PROMPT, query, [h["chunk_id"] for h in hits]])
                           .encode("utf-8")).hexdigest()
        if use_cache and key in self.cache:
            return {**self.cache[key], "cached": True}
        if cache_only:
            return None

        if self._llm is None:
            self._llm = load_llm(self.model)
        start = time.perf_counter()
        result = generate(query, hits, llm=self._llm)
        self.llm_calls += len(result["attempts"])
        result = {
            "query": query, "mode": self.mode, "k": self.k, "binding_only": self.binding_only, "model": self.model,
            "retrieved": [{"rank": h["rank"], "chunk_id": h["chunk_id"], "score": h["score"]} for h in hits],
            **result, "seconds": round(time.perf_counter() - start, 2),
        }
        if self.cache_path:
            self.cache[key] = result
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=1), encoding="utf-8")
        return {**result, "cached": False}


def format_result(result):
    lines = [f"Query: {result['query']}", f"Status: {result['status']}"
             + (" (after one retry)" if result["retried"] else "") + f"   model: {result['model']}", "",
             result["answer"], ""]
    if result.get("blocked_reason"):
        lines += [f"Blocked by the keyword guard: {result['blocked_reason']}", ""]
    if result["citations"]:
        lines.append("Cited sources:")
        lines += [f"  [{c['n']}] {c['source']}  ({c['chunk_id']})" for c in result["citations"]]
    lines.append("Retrieved: " + ", ".join(h["chunk_id"] for h in result["retrieved"]))
    lines += ["", result["disclaimer"]]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ask a question about the EU AI Act (retrieval + Groq).")
    parser.add_argument("query", nargs="+")
    parser.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="hybrid")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="sources in the prompt (3-5)")
    parser.add_argument("--include-recitals", action="store_true", help="also search non-binding recitals")
    parser.add_argument("--json", metavar="PATH", help="also write the full result to a JSON file")
    parser.add_argument("--no-cache", action="store_true", help="always call the model")
    parser.add_argument("--no-guard", action="store_true", help="disable the prompt-injection keyword guard")
    args = parser.parse_args(argv)

    rag = RAG(mode=args.mode, k=args.k, binding_only=not args.include_recitals, guard=not args.no_guard)
    try:
        result = rag.answer(" ".join(args.query), use_cache=not args.no_cache)
    except RateLimitError as error:
        print(f"Groq rate limit reached, no answer this time: {error.message}", file=sys.stderr)
        return 2
    except Exception as error:  # missing or wrong API key, network errors
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(format_result(result))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
