import re

import numpy as np
from rank_bm25 import BM25Okapi

from retrieval.embeddings import load_chunks

TOKEN_RE = re.compile(r"\w+")
# Question words ("what", "how") are rare in legal text, so BM25 gives them a high IDF. Removing them
# is optional: on tests/eval-set.jsonl it lowered BM25 recall, so it is off by default.
STOPWORDS = frozenset("""
a an the and or of to in on at by for from with about as into than then there
is are was were be been being am do does did done have has had having
i me my we our us you your he she it its they them their this that these those
what which who whom whose when where why how can could would should will
""".split())


def tokenize(text, remove_stopwords=False):
    # "Article 6(3)" -> ["article", "6", "3"], so article numbers stay searchable.
    tokens = TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]
    return tokens


class BM25Index:
    def __init__(self, chunks=None, remove_stopwords=False):
        self.chunks = chunks if chunks is not None else load_chunks()
        self.remove_stopwords = remove_stopwords
        self.bm25 = BM25Okapi([tokenize(chunk["page_content"], remove_stopwords) for chunk in self.chunks])

    def search(self, query, k=5, binding_only=False):
        scores = self.bm25.get_scores(tokenize(query, self.remove_stopwords))
        hits = []
        for index in np.argsort(-scores, kind="stable"):
            chunk = self.chunks[index]
            if binding_only and not chunk["metadata"].get("binding"):
                continue
            if scores[index] <= 0:
                break
            hits.append({
                "rank": len(hits) + 1,
                "chunk_id": chunk["metadata"]["chunk_id"],
                "score": round(float(scores[index]), 4),
                "document": chunk["page_content"],
                "metadata": chunk["metadata"],
            })
            if len(hits) == k:
                break
        return hits
