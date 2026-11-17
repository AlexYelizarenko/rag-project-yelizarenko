# Data Card — EU AI Act corpus (Lab 2)

## Source and licence

**Source document:** Regulation (EU) 2024/1689 of the European Parliament and of the Council of 13 June 2024 (Artificial Intelligence Act), Official Journal of the European Union, L series, 12.7.2024.

**Official URL:** https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689

**ELI:** http://data.europa.eu/eli/reg/2024/1689/oj

**Local file:** `data/raw/eu_ai_act.pdf` (144 pages)

**Copyright holder:** European Union

**Reuse terms:** Reuse of EUR-Lex legal documents is permitted for commercial and non-commercial purposes under Commission Decision 2011/833/EU, without charge and without an individual application, unless a specific document states otherwise. See the [EUR-Lex legal notice](https://eur-lex.europa.eu/content/legal-notice/legal-notice.html).

**Attribution used in this project:**

> © European Union, 2024, https://eur-lex.europa.eu  
> Source: Regulation (EU) 2024/1689 (OJ L, 12.7.2024), EUR-Lex.  
> Reused under Commission Decision 2011/833/EU.

**Conditions observed:**

- The source is acknowledged in chunk metadata (`document`, `source`) and in this data card.
- The text is not rewritten in a way that changes legal meaning (PDF extraction and structure-aware chunking only).
- The corpus is for educational and informational use in a RAG system; it is not legal advice.

## Corpus size

| Stage | Size |
| --- | --- |
| Source documents | 1 PDF (Official Journal act) |
| Raw extracted text (PyMuPDF, before header/footer cleanup) | ~600 000 characters across 144 pages |
| After cleaning and structure-aware chunking | **540 chunks** |
| Structural units preserved | 113 articles, 13 annexes, plus introductory/recital material |

Cleaning applied today: Official Journal running headers/footers removed from page margins; lines joined into reading order; Unicode NFC normalization; one soft-hyphenated word rejoined; output written as UTF-8; legal structure detected (chapters, sections, articles, annexes). There is no HTML in this source, so HTML stripping does not apply. See the README section "Text cleaning and normalization" for details.

Ten random chunks for manual inspection: `reports/lab-2/random_chunks.md`.

All produced chunks are longer than 100 characters (shortest: 122 characters).

## Chunking summary

| Metric | Value |
| --- | --- |
| Total chunks | 540 |
| Introductory / recital chunks | 234 |
| Article chunks | 273 (from 113 articles; Article 3 split by definition) |
| Annex chunks | 33 (from 13 annexes) |
| Average chunk size | ~1 115 characters / ~172 words |
| Median chunk size | ~1 142 characters / ~175 words |
| Chunks over 2 000 characters | 15 |

**Chunking rules (current pipeline):**

- Intro and recitals: recursive character split, max 1 500 characters, overlap 200.
- Articles and annexes up to 2 000 characters: one chunk each.
- Longer articles/annexes: split on numbered paragraphs/points, merge neighbours up to ~2 000 characters; Article 3 is one definition per chunk.
- Metadata includes `source_doc_id` (`eu_ai_act_2024_1689`), `chunk_id` (stable global ID), `chunk_index` (position in the document, 0–539), `section_type`, `page_number` / `page_end`, and for articles/annexes `parent_id` and `is_partial`. Recitals are marked `binding: false`; articles and annexes `binding: true`.

## Top-20 words

Computed over all 540 chunk texts: lowercase alphabetic tokens of length ≥ 3, with common English stop words removed. Counts are inflated slightly by intro-chunk overlap and by repeated headers on split pieces.

| Rank | Word | Count |
| --- | --- | --- |
| 1 | systems | 827 |
| 2 | article | 817 |
| 3 | regulation | 742 |
| 4 | system | 715 |
| 5 | high-risk | 504 |
| 6 | union | 475 |
| 7 | data | 431 |
| 8 | market | 389 |
| 9 | authorities | 336 |
| 10 | law | 331 |
| 11 | use | 310 |
| 12 | including | 297 |
| 13 | referred | 295 |
| 14 | european | 289 |
| 15 | relevant | 284 |
| 16 | provider | 282 |
| 17 | commission | 279 |
| 18 | information | 268 |
| 19 | persons | 261 |
| 20 | out | 258 |

The vocabulary matches the AI Act domain (systems, high-risk, provider, market, authorities).

## Known limitations

1. **Single-document corpus.** The working set is currently one regulation. Broader Q&A will need additional official EU sources (guidelines, AI Office publications, implementing acts).
2. **Not HTML-native.** Lab-style HTML cleaning is unnecessary here; PDF layout quirks remain (e.g. a stray backtick in the Article 1 title in the source PDF).
3. **Fifteen oversized chunks.** Some numbered paragraphs (and Articles 66 and 108, which have no numbered paragraphs) still exceed 2 000 characters.
4. **Intro overlap and repeated headers.** Recital chunks can overlap; later article/annex pieces repeat a short header. Word-frequency counts and total character sums are therefore slightly inflated.
5. **English OJ text only.** Other language versions are not included.
6. **Informational use only.** Chunking and retrieval do not replace legal interpretation or professional advice.
7. **Binding vs guidance.** Recitals are labelled non-binding; future guidance documents must keep the same distinction in metadata.
