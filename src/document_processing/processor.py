"""Run the full PDF -> chunks pipeline and save the result.

Usage: uv run python -m document_processing.processor [path/to/pdf]
"""

import json
import sys
from pathlib import Path

from langchain_core.documents import Document

from document_processing.chunker import build_documents
from document_processing.pdf_extractor import extract_lines
from document_processing.structure import detect_sections

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF = PROJECT_ROOT / "data" / "raw" / "eu_ai_act.pdf"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
JSONL_PATH = PROJECT_ROOT / "reports" / "lab-2" / "chunks.jsonl"


def process_pdf(pdf_path=DEFAULT_PDF) -> tuple[list[Document], dict]:
    lines, page_count, extraction_warnings = extract_lines(pdf_path)
    sections, structure_warnings = detect_sections(lines)
    documents = build_documents(sections)

    types = [d.metadata["section_type"] for d in documents]
    longest = sorted(documents, key=lambda d: len(d.page_content), reverse=True)[:10]
    summary = {
        "pdf": str(pdf_path),
        "total_pages": page_count,
        "introductory_chunks": types.count("introductory"),
        "articles": sum(1 for s in sections if s["section_type"] == "article"),
        "article_chunks": types.count("article"),
        "annexes": sum(1 for s in sections if s["section_type"] == "annex"),
        "annex_chunks": types.count("annex"),
        "total_chunks": len(documents),
        "chunks_over_2000_chars": sum(len(d.page_content) > 2000 for d in documents),
        "longest_chunks": [
            {
                "chunk_id": d.metadata["chunk_id"],
                "title": d.metadata.get("article_title") or d.metadata.get("annex_title", ""),
                "characters": len(d.page_content),
                "pages": f"{d.metadata['page_number']}-{d.metadata['page_end']}",
            }
            for d in longest
        ],
        "warnings": extraction_warnings + structure_warnings,
    }
    return documents, summary


def main():
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PDF
    documents, summary = process_pdf(pdf_path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    chunks = [{"page_content": d.page_content, "metadata": d.metadata} for d in documents]
    (OUTPUT_DIR / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSONL_PATH.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks), encoding="utf-8")

    for key, value in summary.items():
        if key not in ("warnings", "longest_chunks"):
            print(f"{key}: {value}")
    print("longest_chunks:")
    for chunk in summary["longest_chunks"]:
        print(f"  - {chunk['chunk_id']}: {chunk['characters']} chars, pages {chunk['pages']}")
    print(f"warnings: {len(summary['warnings'])}")
    for warning in summary["warnings"]:
        print(f"  - {warning}")
    print(f"Saved to {OUTPUT_DIR} and {JSONL_PATH}")


if __name__ == "__main__":
    main()
