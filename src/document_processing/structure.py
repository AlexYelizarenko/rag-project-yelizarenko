"""Split extracted lines into the legal structure: introductory part, articles and annexes."""

import re

from document_processing.pdf_extractor import Line

ARTICLE_RE = re.compile(r"Article (\d+)")
CHAPTER_RE = re.compile(r"CHAPTER ([IVXLC]+)")
SECTION_RE = re.compile(r"SECTION (\d+)")
ANNEX_RE = re.compile(r"ANNEX ([IVXLC]+)")

ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def roman_to_int(roman: str) -> int:
    total = 0
    for i, char in enumerate(roman):
        value = ROMAN_VALUES[char]
        if i + 1 < len(roman) and ROMAN_VALUES[roman[i + 1]] > value:
            total -= value
        else:
            total += value
    return total


def detect_sections(lines: list[Line]) -> tuple[list[dict], list[str]]:
    """Group lines into sections.

    Every line ends up in exactly one section. A section is a dict with "section_type"
    ("introductory", "article" or "annex"), "lines" and structural fields
    (number, title, chapter, section).

    A line counts as a heading only if it is exactly "Article N" / "CHAPTER X" / "SECTION N" /
    "ANNEX X", has heading formatting (headings are non-bold, titles below them are bold) and
    continues the expected numbering. This rejects references such as "Article 49" that end a
    bold annex title, or article references inside recitals.
    """
    sections = [{"section_type": "introductory", "lines": []}]
    warnings: list[str] = []
    pending: list[Line] = []  # chapter/section headings waiting for their first article
    chapter: dict | None = None
    section: dict | None = None
    next_article, next_chapter, next_annex = 1, 1, 1
    in_annexes = False

    def take_title(start: int) -> tuple[str, list[Line]]:
        # Titles are bold, except a few (e.g. Annex X) printed in italic; body text is regular.
        title_lines = []
        while start + len(title_lines) < len(lines):
            candidate = lines[start + len(title_lines)]
            is_heading = not candidate.bold and any(
                r.fullmatch(candidate.text) for r in (ARTICLE_RE, CHAPTER_RE, SECTION_RE, ANNEX_RE)
            )
            if is_heading or not (candidate.bold or candidate.italic):
                break
            title_lines.append(candidate)
        if not title_lines:
            warnings.append(f"Page {lines[start - 1].page}: no title found after {lines[start - 1].text!r}")
        return " ".join(l.text for l in title_lines), title_lines

    def flush_pending():
        if pending:
            warnings.append(f"Page {pending[0].page}: heading {pending[0].text!r} not followed by an article; kept as text")
            sections[-1]["lines"].extend(pending)
            pending.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        article = ARTICLE_RE.fullmatch(line.text)
        chapter_match = CHAPTER_RE.fullmatch(line.text)
        section_match = SECTION_RE.fullmatch(line.text)
        annex = ANNEX_RE.fullmatch(line.text)

        if article and line.italic and not line.bold:
            number = int(article.group(1))
            if not in_annexes and number == next_article:
                title, title_lines = take_title(i + 1)
                new = {
                    "section_type": "article",
                    "lines": [*pending, line, *title_lines],
                    "article_number": number,
                    "article_title": title,
                }
                if chapter:
                    new.update(chapter)
                if section:
                    new.update(section)
                sections.append(new)
                pending.clear()
                next_article += 1
                i += 1 + len(title_lines)
                continue
            warnings.append(f"Page {line.page}: {line.text!r} looks like a heading but breaks the article sequence (expected Article {next_article}); kept as text")

        elif chapter_match and not line.bold and not line.italic:
            roman = chapter_match.group(1)
            if not in_annexes and roman_to_int(roman) == next_chapter:
                title, title_lines = take_title(i + 1)
                flush_pending()
                chapter = {"chapter_number": next_chapter, "chapter_title": title}
                section = None
                pending.extend([line, *title_lines])
                next_chapter += 1
                i += 1 + len(title_lines)
                continue
            warnings.append(f"Page {line.page}: {line.text!r} looks like a chapter heading but is out of sequence; kept as text")

        elif section_match and line.italic and not line.bold and chapter and not in_annexes:
            title, title_lines = take_title(i + 1)
            section = {"section_number": int(section_match.group(1)), "section_title": title}
            pending.extend([line, *title_lines])
            i += 1 + len(title_lines)
            continue

        elif annex and line.italic and not line.bold:
            roman = annex.group(1)
            if next_article > 1 and roman_to_int(roman) == next_annex:
                flush_pending()
                title, title_lines = take_title(i + 1)
                sections.append({
                    "section_type": "annex",
                    "lines": [line, *title_lines],
                    "annex_number": roman,
                    "annex_title": title,
                })
                in_annexes = True
                next_annex += 1
                i += 1 + len(title_lines)
                continue
            warnings.append(f"Page {line.page}: {line.text!r} looks like an annex heading but is out of sequence; kept as text")

        flush_pending()
        sections[-1]["lines"].append(line)
        i += 1

    flush_pending()
    if next_article == 1:
        warnings.append("Article 1 heading not found; the whole document was treated as introductory text")
    if next_annex == 1:
        warnings.append("No annex headings found")
    return sections, warnings
