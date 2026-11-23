import argparse
import json
import os
import re
import sys

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from retrieval.index import PROJECT_ROOT

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_K = 4
# Most chunks are <= 2,000 chars; this only cuts the few very long ones (e.g. Article 5) to save tokens.
MAX_SOURCE_CHARS = 2500
INLINE_CITATION_RE = re.compile(r"\[(\d+)\]")
CITATION_ITEM_RE = re.compile(r"\[?(\d+)\]?")

DISCLAIMER = "This is general information based on the EU AI Act, not legal advice."
UNCERTAIN_ANSWER = (
    "I am not sure: I could not produce an answer with valid citations to the retrieved sources. "
    "Please check the sources listed below directly."
)

SYSTEM_PROMPT = """You explain the EU AI Act (Regulation (EU) 2024/1689) in plain language to people who are not lawyers.

Rules:
- Use only the numbered sources in the user message. Do not use outside knowledge and never invent provisions.
- Support every statement with the number of the source it comes from in square brackets, e.g. [1] or [2][3]. Only the numbers of the given sources exist.
- Use simple language, explain legal terms briefly, and keep short paragraphs. Keep exceptions and conditions; do not change the legal meaning.
- Sources marked "non-binding" are recitals: they explain the law but do not create obligations. Say so when you rely on them.
- If the sources do not contain enough information to answer, set "sufficient_context" to false and briefly say what is missing instead of guessing.

Reply with a JSON object only, in exactly this format:
{"answer": "<answer with [n] citations>", "citations": [<numbers of the sources you used>], "sufficient_context": true}"""


def load_llm(model=None):
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set. Put it into .env (see .env.example) or export it.")
    llm = ChatGroq(model=model or os.getenv("GROQ_MODEL", DEFAULT_MODEL), temperature=0, max_tokens=1024)
    # Groq JSON mode: the reply is guaranteed to be a JSON object (the prompt must mention JSON).
    return llm.bind(response_format={"type": "json_object"})


def source_label(metadata):
    section = metadata.get("section_type")
    if section == "article":
        label = f"Article {metadata['article_number']} – {metadata.get('article_title', '')}".rstrip(" –")
    elif section == "annex":
        label = f"Annex {metadata['annex_number']} – {metadata.get('annex_title', '')}".rstrip(" –")
    else:
        label = "Recital / preamble"
    pages = metadata.get("page_number")
    if pages is not None and metadata.get("page_end") not in (None, pages):
        pages = f"{pages}–{metadata['page_end']}"
    binding = "binding" if metadata.get("binding") else "non-binding"
    return f"{label} (EU AI Act, p. {pages}, {binding})"


def build_prompt(query, chunks):
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        text = chunk["document"]
        if len(text) > MAX_SOURCE_CHARS:
            text = text[:MAX_SOURCE_CHARS].rstrip() + " […]"
        blocks.append(f"[{number}] {source_label(chunk['metadata'])}\n{text}")
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {query}"


def parse_output(raw, n_sources):
    """Return (data, errors); data has "answer", "citations" (sorted unique ints) and "sufficient_context"."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None, ["the reply is not valid JSON"]
    if not isinstance(data, dict):
        return None, ["the reply must be a JSON object"]

    errors = []
    answer = data.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        errors.append('"answer" must be a non-empty string')
        answer = ""
    sufficient = data.get("sufficient_context")
    if not isinstance(sufficient, bool):
        errors.append('"sufficient_context" must be true or false')

    listed = data.get("citations", [])
    if not isinstance(listed, list):
        errors.append('"citations" must be a list of source numbers')
        listed = []
    cited = {int(number) for number in INLINE_CITATION_RE.findall(answer)}
    for item in listed:
        match = CITATION_ITEM_RE.fullmatch(str(item).strip()) if not isinstance(item, bool) else None
        if match:
            cited.add(int(match.group(1)))
        else:
            errors.append(f'"citations" contains {item!r}, which is not a source number')

    unknown = sorted(number for number in cited if not 1 <= number <= n_sources)
    if unknown:
        errors.append(f"sources {unknown} do not exist; only [1]–[{n_sources}] may be cited")
    if sufficient is True and not cited:
        errors.append("the answer cites no sources; cite the sources that support it, or set sufficient_context to false")

    return {"answer": answer.strip(), "citations": sorted(cited), "sufficient_context": sufficient}, errors


def generate(query, chunks, llm=None):
    """Answer `query` from the retrieved `chunks` (hits from retrieval.search). Retries once on a broken reply."""
    if llm is None:
        llm = load_llm()
    sources = [
        {"n": number, "chunk_id": chunk["chunk_id"], "source": source_label(chunk["metadata"])}
        for number, chunk in enumerate(chunks, start=1)
    ]
    result = {"status": "uncertain", "answer": UNCERTAIN_ANSWER, "citations": [], "sources": sources,
              "attempts": [], "retried": False, "disclaimer": DISCLAIMER}
    if not chunks:
        result["status"] = "insufficient_context"
        result["answer"] = "No relevant passages of the EU AI Act were found for this question."
        return result

    messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(build_prompt(query, chunks))]
    for _ in range(2):
        raw = llm.invoke(messages).content
        data, errors = parse_output(raw, len(chunks))
        result["attempts"].append({"raw": raw, "errors": errors})
        if not errors:
            result["status"] = "ok" if data["sufficient_context"] else "insufficient_context"
            result["answer"] = data["answer"]
            result["citations"] = [sources[number - 1] for number in data["citations"]]
            break
        messages += [
            AIMessage(raw),
            HumanMessage("Your reply broke the rules: " + "; ".join(errors)
                         + f". Answer again as the JSON object, citing only sources [1]–[{len(chunks)}]."),
        ]
    result["retried"] = len(result["attempts"]) > 1
    return result


def main(argv=None):
    from retrieval.search import hybrid_search

    parser = argparse.ArgumentParser(description="Answer a question about the EU AI Act with Groq.")
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="sources in the prompt (3-5)")
    args = parser.parse_args(argv)

    try:
        llm = load_llm()
        chunks = hybrid_search(args.query, k=args.k, binding_only=True)
        result = generate(args.query, chunks, llm=llm)
    except Exception as error:  # missing/fake API key, network or rate-limit errors from Groq
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
