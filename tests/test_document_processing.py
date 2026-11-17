import re
import unicodedata

import pytest
from langchain_core.documents import Document

from document_processing.chunker import build_documents, section_text
from document_processing.pdf_extractor import extract_lines
from document_processing.processor import DEFAULT_PDF, process_pdf
from document_processing.structure import detect_sections, roman_to_int

pytestmark = pytest.mark.skipif(not DEFAULT_PDF.exists(), reason=f"{DEFAULT_PDF} not found")


@pytest.fixture(scope="module")
def extracted():
    lines, page_count, warnings = extract_lines(DEFAULT_PDF)
    sections, structure_warnings = detect_sections(lines)
    return lines, page_count, sections, warnings + structure_warnings


@pytest.fixture(scope="module")
def result():
    return process_pdf(DEFAULT_PDF)


def by_type(documents, section_type):
    return [d for d in documents if d.metadata["section_type"] == section_type]


def test_no_parsing_warnings(result):
    _, summary = result
    assert summary["warnings"] == []
    assert summary["total_pages"] == 144


def test_every_line_assigned_to_exactly_one_section(extracted):
    lines, _, sections, _ = extracted
    assigned = [line for section in sections for line in section["lines"]]
    assert assigned == lines


def test_page_headers_and_footers_removed(result):
    documents, _ = result
    for d in documents:
        assert not re.search(r"\b\d{1,3}/144\b", d.page_content)
        assert "ELI: http://data.europa.eu/eli/reg/2024/1689/oj" not in d.page_content


def test_text_is_normalized_and_long_enough(result):
    documents, _ = result
    for d in documents:
        assert len(d.page_content) >= 100, d.metadata["chunk_id"]
        assert unicodedata.normalize("NFC", d.page_content) == d.page_content
        assert "\xad" not in d.page_content
    assert any("experimentation facilities" in d.page_content for d in documents)


def test_source_id_and_chunk_index(result):
    documents, _ = result
    assert [d.metadata["chunk_index"] for d in documents] == list(range(len(documents)))
    assert {d.metadata["source_doc_id"] for d in documents} == {"eu_ai_act_2024_1689"}
    assert documents[0].metadata["chunk_id"] == "intro_1"
    assert documents[-1].metadata["annex_number"] == "XIII"


def test_valid_documents_and_metadata(result):
    documents, summary = result
    for d in documents:
        assert isinstance(d, Document)
        assert d.page_content.strip()
        assert d.metadata["document"] == "EU AI Act"
        assert all(isinstance(v, (str, int, float, bool)) for v in d.metadata.values())
        assert 1 <= d.metadata["page_number"] <= d.metadata["page_end"] <= summary["total_pages"]
    ids = [d.metadata["chunk_id"] for d in documents]
    assert len(ids) == len(set(ids))


def group_by_parent(documents):
    groups = {}
    for d in documents:
        if "parent_id" in d.metadata:
            groups.setdefault(d.metadata["parent_id"], []).append(d)
    return groups


def parent_pages(groups, parent_id):
    chunks = groups[parent_id]
    return chunks[0].metadata["page_number"], chunks[-1].metadata["page_end"]


def test_every_article_and_annex_present_in_order(result):
    documents, _ = result
    groups = group_by_parent(documents)
    article_ids = [p for p in groups if p.startswith("article_")]
    annex_ids = [p for p in groups if p.startswith("annex_")]
    assert article_ids == [f"article_{n}" for n in range(1, 114)]
    assert [roman_to_int(p.removeprefix("annex_")) for p in annex_ids] == list(range(1, 14))
    for chunks in groups.values():
        meta = chunks[0].metadata
        assert meta.get("article_title") or meta.get("annex_title")
        assert all(c.metadata["is_partial"] == (len(chunks) > 1) for c in chunks)


def test_only_long_sections_are_split(extracted, result):
    _, _, sections, _ = extracted
    documents, _ = result
    groups = group_by_parent(documents)
    for chunks in groups.values():
        if len(chunks) == 1:
            continue
        whole_text = "".join(c.page_content for c in chunks)
        assert len(whole_text) > 2000


def test_split_chunks_keep_their_header(result):
    documents, _ = result
    for chunks in group_by_parent(documents).values():
        first = chunks[0].metadata
        if "article_number" in first:
            assert f"Article {first['article_number']}\n" in chunks[0].page_content
            header = f"Article {first['article_number']} – {first['article_title']}\n\n"
        else:
            assert f"ANNEX {first['annex_number']}\n" in chunks[0].page_content
            header = f"Annex {first['annex_number']} – {first['annex_title']}\n\n"
        for c in chunks[1:]:
            assert c.page_content.startswith(header)


def test_article_3_split_by_definition(result):
    documents, _ = result
    definitions = group_by_parent(documents)["article_3"]
    assert [d.metadata["definition_number"] for d in definitions] == list(range(1, 69))
    by_number = {d.metadata["definition_number"]: d for d in definitions}
    assert by_number[1].metadata["defined_term"] == "AI system"
    assert by_number[4].metadata["defined_term"] == "deployer"
    assert "(4) ‘deployer’ means" in by_number[4].page_content
    assert "For the purposes of this Regulation" in by_number[1].page_content


def test_paragraph_split_of_long_article(result):
    documents, _ = result
    article_5 = group_by_parent(documents)["article_5"]
    assert len(article_5) > 1
    assert article_5[0].metadata["paragraph"].startswith("1")
    assert article_5[0].page_content.lstrip().startswith("CHAPTER II")
    labels = [c.metadata["paragraph"] for c in article_5]
    assert labels[-1].endswith("8")
    assert [c.metadata["chunk_id"] for c in article_5] == [f"article_5_part_{n}" for n in range(1, len(article_5) + 1)]


def test_merged_pieces_stay_within_limit(result):
    documents, _ = result
    for d in documents:
        label = d.metadata.get("paragraph") or d.metadata.get("point") or ""
        if "-" in label:
            assert len(d.page_content) <= 2000, d.metadata["chunk_id"]


def test_known_articles_and_annexes(result):
    documents, _ = result
    groups = group_by_parent(documents)
    first = {parent: chunks[0].metadata for parent, chunks in groups.items()}

    assert (first["article_1"]["chapter_number"], first["article_1"]["page_number"]) == (1, 44)
    assert first["article_5"]["article_title"] == "Prohibited AI practices"
    assert first["article_5"]["chapter_number"] == 2
    assert parent_pages(groups, "article_5") == (51, 53)
    assert first["article_6"]["section_number"] == 1
    assert first["article_50"]["chapter_number"] == 4
    assert first["article_113"]["chapter_number"] == 13

    assert first["annex_III"]["annex_title"] == "High-risk AI systems referred to in Article 6(2)"
    assert parent_pages(groups, "annex_III") == (127, 129)
    # Annex VIII's title ends with a line "Article 49"; it must not become an article.
    assert first["annex_VIII"]["annex_title"].endswith("in accordance with Article 49")
    assert first["annex_X"]["annex_title"].startswith("Union legislative acts on large-scale IT systems")
    assert first["annex_XIII"]["page_number"] == 144


def test_intro_stops_before_article_1(result):
    documents, _ = result
    intro = by_type(documents, "introductory")
    assert intro[-1].page_content.rstrip().endswith("HAVE ADOPTED THIS REGULATION:")
    assert all(d.metadata["page_end"] <= 44 for d in intro)
    assert not any("GENERAL PROVISIONS" in d.page_content for d in intro)
    assert all(d.metadata["binding"] is False for d in intro)


def test_intro_chunk_size_and_overlap(extracted, result):
    _, _, sections, _ = extracted
    documents, _ = result
    intro_text, _ = section_text(sections[0]["lines"])
    intro = by_type(documents, "introductory")
    assert all(len(d.page_content) <= 1500 for d in intro)

    overlapping = 0
    for prev, cur in zip(intro, intro[1:]):
        prev_end = prev.metadata["start_index"] + len(prev.page_content)
        overlap = prev_end - cur.metadata["start_index"]
        assert overlap <= 200
        if overlap > 0:
            overlapping += 1
        else:
            # Chunks only start without overlap on a clean paragraph (recital) boundary.
            assert intro_text[prev_end:cur.metadata["start_index"]] == "\n\n"
    assert overlapping > 0


def test_intro_text_fully_covered(extracted, result):
    _, _, sections, _ = extracted
    documents, _ = result
    intro_text, _ = section_text(sections[0]["lines"])
    covered_until = 0
    for d in by_type(documents, "introductory"):
        start = d.metadata["start_index"]
        assert intro_text[start:start + len(d.page_content)] == d.page_content
        assert not intro_text[covered_until:start].strip(), f"text lost before offset {start}"
        covered_until = max(covered_until, start + len(d.page_content))
    assert not intro_text[covered_until:].strip()


def test_intro_page_numbers_match_text(result):
    documents, _ = result
    recital_1 = next(d for d in by_type(documents, "introductory") if "(1) The purpose of this Regulation" in d.page_content)
    assert recital_1.metadata["page_number"] == 1
    recital_180 = next(d for d in by_type(documents, "introductory") if "(180) The European Data Protection" in d.page_content)
    assert recital_180.metadata["page_number"] == 44


def test_articles_and_annexes_contain_all_their_lines(extracted):
    _, _, sections, _ = extracted
    groups = group_by_parent(build_documents(sections[1:]))
    for section, chunks in zip(sections[1:], groups.values()):
        for line in section["lines"]:
            assert sum(line.text in c.page_content for c in chunks) >= 1, line.text


def test_split_pieces_do_not_duplicate_text(extracted):
    _, _, sections, _ = extracted
    groups = group_by_parent(build_documents(sections[1:]))
    for section, chunks in zip(sections[1:], groups.values()):
        if len(chunks) == 1:
            continue
        text, _ = section_text(section["lines"])
        first, *rest = chunks
        header_len = len(rest[0].page_content) - len(rest[0].page_content.split("\n\n", 1)[1]) if rest else 0
        rebuilt = first.page_content + "".join(c.page_content[header_len:] for c in rest)
        assert re.sub(r"\s+", "", rebuilt) == re.sub(r"\s+", "", text)
