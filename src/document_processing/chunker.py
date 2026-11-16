"""Turn detected sections into LangChain Documents with metadata."""

import re
from bisect import bisect_right

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_processing.pdf_extractor import Line

DOCUMENT_NAME = "EU AI Act"
SOURCE = "Regulation (EU) 2024/1689, OJ L, 12.7.2024"

INTRO_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
    keep_separator="end",  # keep the full stop with its sentence, not at the start of the next chunk
    add_start_index=True,
)

MAX_SECTION_CHARS = 2000
PARAGRAPH_RE = re.compile(r"^(\d+)\. ", re.M)  # "1. ", "2. " at the start of a line
DEFINITION_RE = re.compile(r"^\((\d+)\) ‘([^’]+)’", re.M)  # "(4) ‘deployer’ means ..." in Article 3

# Copied from a section into the Document metadata when present.
STRUCTURE_KEYS = [
    "article_number", "article_title", "chapter_number", "chapter_title",
    "section_number", "section_title", "annex_number", "annex_title",
]


def section_text(lines: list[Line]) -> tuple[str, list[int]]:
    """Join lines into text; also return the character offset at which each line starts."""
    parts, offsets, length = [], [], 0
    for i, line in enumerate(lines):
        if i > 0:
            separator = "\n\n" if line.paragraph_start else "\n"
            parts.append(separator)
            length += len(separator)
        offsets.append(length)
        parts.append(line.text)
        length += len(line.text)
    return "".join(parts), offsets


def split_section(text: str, is_definitions: bool, max_chars: int) -> list[dict]:
    """Split a long article/annex into numbered paragraphs (or Article 3 definitions).

    Only numbers continuing the sequence (or restarting at 1, as annex sections do) count as
    boundaries, so a line that happens to start with e.g. "2024. " is not treated as one.
    Text before the first number (heading, title, lead-in) stays with the first piece.
    Paragraphs are merged with their neighbours up to max_chars, but never across a restart
    of the numbering (a new annex section); definitions are not merged.
    Returns pieces with "start", "end" and "numbers"; empty list if there is nothing to split on.
    """
    pattern = DEFINITION_RE if is_definitions else PARAGRAPH_RE
    items, expected = [], 1
    for match in pattern.finditer(text):
        number = int(match.group(1))
        if number == expected or (number == 1 and items):
            items.append({"start": match.start(), "numbers": [number], "term": match.group(2) if is_definitions else None})
            expected = number + 1
    if len(items) < 2:
        return []

    items[0]["start"] = 0
    for item, following in zip(items, items[1:] + [{"start": len(text)}]):
        item["end"] = following["start"]

    if is_definitions:
        return items
    pieces = [items[0]]
    for item in items[1:]:
        if item["numbers"][0] != 1 and item["end"] - pieces[-1]["start"] <= max_chars:
            pieces[-1]["end"] = item["end"]
            pieces[-1]["numbers"] += item["numbers"]
        else:
            pieces.append(item)
    return pieces


def build_documents(sections: list[dict]) -> list[Document]:
    documents = []
    for section in sections:
        lines = section["lines"]
        if not lines:
            continue
        text, offsets = section_text(lines)
        base = {"document": DOCUMENT_NAME, "source": SOURCE, "section_type": section["section_type"]}

        if section["section_type"] == "introductory":
            # Recitals explain the purpose of the law but are not legally binding themselves.
            for n, chunk in enumerate(INTRO_SPLITTER.create_documents([text])):
                start = chunk.metadata["start_index"]
                end = start + len(chunk.page_content) - 1
                chunk.metadata = {
                    **base,
                    "chunk_id": f"intro_{n + 1}",
                    "binding": False,
                    "page_number": lines[bisect_right(offsets, start) - 1].page,
                    "page_end": lines[bisect_right(offsets, end) - 1].page,
                    "start_index": start,
                }
                documents.append(chunk)
            continue

        is_article = section["section_type"] == "article"
        if is_article:
            parent_id = f"article_{section['article_number']}"
            header = f"Article {section['article_number']} – {section['article_title']}"
        else:
            parent_id = f"annex_{section['annex_number']}"
            header = f"Annex {section['annex_number']} – {section['annex_title']}"
        base.update({
            "binding": True,
            **{key: section[key] for key in STRUCTURE_KEYS if key in section},
            "parent_id": parent_id,
        })

        is_definitions = is_article and section["article_number"] == 3
        budget = MAX_SECTION_CHARS - len(header) - 2  # room for the header repeated on later pieces
        pieces = split_section(text, is_definitions, budget) if len(text) > MAX_SECTION_CHARS else []
        if not pieces:
            documents.append(Document(page_content=text, metadata={
                **base,
                "chunk_id": parent_id,
                "is_partial": False,
                "page_number": lines[0].page,
                "page_end": lines[-1].page,
            }))
            continue

        for n, piece in enumerate(pieces):
            piece_text = text[piece["start"]:piece["end"]].rstrip()
            end = piece["start"] + len(piece_text) - 1
            numbers = piece["numbers"]
            label = str(numbers[0]) if len(numbers) == 1 else f"{numbers[0]}-{numbers[-1]}"

            metadata = {**base, "is_partial": True}
            if is_definitions:
                metadata.update({"chunk_id": f"{parent_id}_def_{label}", "definition_number": numbers[0], "defined_term": piece["term"]})
            elif is_article:
                metadata.update({"chunk_id": f"{parent_id}_part_{n + 1}", "paragraph": label})
            else:
                metadata.update({"chunk_id": f"{parent_id}_part_{n + 1}", "point": label})
            metadata["page_number"] = lines[bisect_right(offsets, piece["start"]) - 1].page
            metadata["page_end"] = lines[bisect_right(offsets, end) - 1].page

            # The first piece already starts with the original heading and title.
            content = piece_text if n == 0 else f"{header}\n\n{piece_text}"
            documents.append(Document(page_content=content, metadata=metadata))
    return documents
