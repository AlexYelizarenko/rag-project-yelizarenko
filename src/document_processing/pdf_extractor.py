"""Extract text lines from the EU AI Act PDF, keeping page numbers and basic formatting."""

import re
from dataclasses import dataclass

import pymupdf

# Running headers/footers of the Official Journal layout. Lines are only dropped when they
# sit in the top/bottom margin AND match one of these patterns; anything else is kept.
HEADER_FOOTER_RE = re.compile(
    r"EN|L series|Official Journal|of the European Union|OJ L, \d{1,2}\.\d{1,2}\.\d{4}"
    r"|ELI: \S+|\d+/\d+|\d{4}/\d+|\d{1,2}\.\d{1,2}\.\d{4}"
)
TOP_MARGIN = 75
BOTTOM_MARGIN = 805

BOLD_FLAG = 16
ITALIC_FLAG = 2


@dataclass
class Line:
    page: int  # 1-based physical PDF page
    text: str
    bold: bool
    italic: bool
    paragraph_start: bool  # visible vertical gap before this line


def extract_lines(pdf_path) -> tuple[list[Line], int, list[str]]:
    """Return content lines in reading order, the number of pages and extraction warnings."""
    lines: list[Line] = []
    warnings: list[str] = []

    with pymupdf.open(pdf_path) as doc:
        page_count = doc.page_count
        for page_index, page in enumerate(doc):
            page_number = page_index + 1
            prev_top = prev_bottom = None

            for block in page.get_text("dict")["blocks"]:
                for raw_line in block.get("lines", []):
                    spans = [s for s in raw_line["spans"] if s["text"].strip()]
                    if not spans:
                        continue
                    text = " ".join("".join(s["text"] for s in raw_line["spans"]).split())
                    top, bottom = raw_line["bbox"][1], raw_line["bbox"][3]

                    in_margin = top < TOP_MARGIN or top > BOTTOM_MARGIN
                    if in_margin:
                        if HEADER_FOOTER_RE.fullmatch(text):
                            continue
                        warnings.append(f"Page {page_number}: unexpected text in page margin kept: {text!r}")

                    # Numbers like "1." or "(12)" are separate PDF lines on the same baseline
                    # as their text; join them so numbering stays attached to its paragraph.
                    if prev_top is not None and abs(top - prev_top) < 3 and lines[-1].page == page_number:
                        lines[-1].text += " " + text
                        prev_bottom = max(prev_bottom, bottom)
                        continue

                    flags = spans[0]["flags"]
                    lines.append(Line(
                        page=page_number,
                        text=text,
                        bold=bool(flags & BOLD_FLAG),
                        italic=bool(flags & ITALIC_FLAG),
                        paragraph_start=prev_bottom is not None and top - prev_bottom > 4,
                    ))
                    prev_top, prev_bottom = top, bottom

    return lines, page_count, warnings
