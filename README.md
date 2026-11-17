# EU AI Act Assistant

An AI assistant that explains the European Union Artificial Intelligence Act (Regulation (EU) 2024/1689) in plain language. It uses Retrieval-Augmented Generation (RAG), so answers are grounded in official EU documents instead of only the language model's general knowledge.

> **Status:** early development. PDF processing and chunking of the AI Act is implemented; embeddings, retrieval and generation are not implemented yet.

> **Disclaimer:** this tool is for educational and informational purposes only. It is not legal advice.

## Idea

The EU AI Act is a long and complex legal text. Most people who are affected by it (developers, companies, students, curious citizens) have no legal background. This project lets them ask questions in everyday language, for example:

- What is considered a high-risk AI system?
- Can I use AI to evaluate job applicants?
- Are any AI applications completely prohibited in the EU?
- Do I need to tell people they are interacting with an AI system?
- What is the difference between an AI provider and an AI deployer?
- What happens if a company doesn't follow the AI Act?
- Does the AI Act apply to companies outside the EU?

The assistant should feel like a friendly, knowledgeable person, not like a legal search engine.

## Target users

- Ordinary people interested in how AI is regulated in the EU
- AI developers who need to understand their obligations
- Companies building or using AI systems in the EU
- University students and researchers
- People working with AI who have no legal background

The system must not assume any knowledge of legal terminology or procedures.

## Knowledge base

**Primary source (central, always):**
[Regulation (EU) 2024/1689 – Artificial Intelligence Act](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689)

It covers risk classification, prohibited practices, high-risk requirements, obligations of providers and deployers, transparency, general-purpose AI models, AI literacy, market surveillance, enforcement and penalties.

**Additional sources (planned, official and reliable only):**

- European Commission publications and guidelines on the AI Act
- European AI Office publications
- Official explanatory documents
- Related EU regulations and implementing acts
- Official reports and studies on AI regulation and compliance
- Amendments and consolidated versions of the AI Act

Target: roughly 500-1,000 documents, subject to availability and relevance checks. The AI Act itself remains the main source.

## Answer requirements

Answers should:

- Use simple, natural language and explain any necessary legal term
- Prefer practical examples over abstract wording
- Be split into small, understandable sections
- Explain what a rule means in practice, not just repeat it
- Cite relevant articles, annexes and source documents (e.g. "Article 6 and Annex III")
- Distinguish **binding legislation** from **guidelines and recommendations**
- Never invent legal provisions or requirements
- Say clearly when the documents do not contain enough information
- Communicate uncertainty when a question needs legal interpretation
- Not oversimplify important exceptions or change the meaning of the law

Example:

> **Q:** What does high-risk AI actually mean? Is every AI chatbot high-risk?
>
> **A:** Not necessarily! Under the EU AI Act, AI systems are classified by the risks they may pose to people's health, safety and fundamental rights. For example, systems used in recruitment or for evaluating creditworthiness can be high-risk because their decisions significantly affect people's lives. A typical chatbot is not automatically high-risk just because it uses AI; it depends on its intended purpose. See Article 6 and Annex III of the EU AI Act.

## Scope and limitations

- Focused on EU AI regulation, primarily the AI Act
- Does not cover all European legislation
- Does not provide legal advice; it should say when professional interpretation is needed

## Evaluation criteria

- Retrieval relevance
- Answer correctness
- Source citation accuracy
- Clarity and simplicity of explanations
- Ability to explain legal terminology
- Avoiding unsupported answers and hallucinations
- Recognizing when information is insufficient

## Installation

Requirements: [uv](https://docs.astral.sh/uv/) (it will also fetch the right Python version, 3.14).

```bash
git clone git@github.com:AlexYelizarenko/rag-project-yelizarenko.git
cd RAG_system_LW

# create .venv and install dependencies from pyproject.toml
uv sync

# run
uv run python main.py
```

Add a new library:

```bash
uv add <package>
```

Activate the environment manually if you prefer:

```bash
source .venv/bin/activate
```

## Processing the AI Act PDF

```bash
uv run python -m document_processing.processor          # uses data/raw/eu_ai_act.pdf
uv run pytest                                           # run the tests
```

This writes:

- `data/processed/chunks.json`: all chunks with metadata, as one JSON array. This is a local working copy and is git-ignored.
- `data/processed/summary.json`: page count, chunk counts, longest chunks and parsing warnings. Also git-ignored.
- `reports/lab-2/chunks.jsonl`: the same 540 chunks in JSONL format, one JSON object per line. This is the committed deliverable for Lab 2.

Each JSONL line looks like this:

```json
{"page_content": "Article 3 – Definitions\n\n(4) ‘deployer’ means ...", "metadata": {"chunk_id": "article_3_def_4", "parent_id": "article_3", "section_type": "article", "page_number": 46, "page_end": 46, "...": "..."}}
```

The file is UTF-8 (non-ASCII characters such as `’` are stored as-is). It is about 0.8 MB, well below GitHub's 100 MB file limit, so it is committed directly to Git without Git LFS or external storage. It is regenerated on every run of the processor, so it always matches the current pipeline.

Chunking follows the legal structure:

| Part | Chunking | Key metadata |
| --- | --- | --- |
| Introduction and recitals (before Article 1) | `RecursiveCharacterTextSplitter`, 1,500 chars, 200 overlap | `binding: false` |
| Articles 1-113 | one chunk per article; articles over 2,000 chars split by numbered paragraphs | `article_number`, `article_title`, `chapter_number`, `section_number`, `paragraph` |
| Article 3 (Definitions) | one chunk per definition | `definition_number`, `defined_term` |
| Annexes I-XIII | one chunk per annex; annexes over 2,000 chars split by numbered points | `annex_number`, `annex_title`, `point` |

Every chunk has `document`, `source`, `source_doc_id` (`eu_ai_act_2024_1689`), `section_type`, `chunk_id` (stable global ID), `chunk_index` (position of the chunk in the document, from 0), `page_number` and `page_end` (physical 1-based PDF pages). Recitals are marked non-binding because, legally, they explain the regulation but are not binding provisions.

Article and annex chunks also have `parent_id` (e.g. `article_5`) and `is_partial`. Split pieces (`article_5_part_2`, `article_3_def_4`) repeat a header line such as `Article 5 – Prohibited AI practices`, and all pieces with the same `parent_id` together make up the full article.

### Text cleaning and normalization

The only source is the official EUR-Lex **PDF**, so the cleaning steps are chosen for PDF text:

- **No HTML cleanup.** The text comes from the PDF text layer, not from web pages, so it contains no HTML tags or entities to strip.
- **Page noise is removed.** Official Journal headers and footers (`EN`, `OJ L, 12.7.2024`, `ELI: …`, `12/144`) are dropped, but only when they sit in the page margin and match a known pattern (`pdf_extractor.py`).
- **No minimum-length filter.** All 540 chunks are longer than 100 characters (the shortest has 122), so a "drop documents shorter than 100 characters" step would remove nothing. A test checks this, so it fails if a future change produces a shorter chunk.
- **UTF-8 normalization:**
  - PyMuPDF returns Python Unicode strings.
  - Every line is normalized to Unicode NFC (`unicodedata.normalize("NFC", ...)`), so the same character always has one byte representation.
  - Whitespace is collapsed to single spaces.
  - Output files are written explicitly as UTF-8 (`encoding="utf-8"`, `ensure_ascii=False`), so characters like `’` and `—` are stored as-is instead of `\u2019` escapes.
- **Soft hyphens.** The PDF has one soft hyphen (U+00AD), where "experimentation" is split across two lines on page 90. The extractor rejoins such words.

Further normalizations were left out on purpose. We checked the extracted text and found that they are either unnecessary or harmful:

- **NFKC / ligature folding:** NFKC would change 0 characters, and the PDF has no ligatures such as `ﬁ`.
- **Converting curly quotes and dashes to ASCII:** the regulation marks defined terms with `‘…’` (for example ‘deployer’), and the Article 3 definition splitter relies on them. They are also part of the original legal text, which should stay unchanged for accurate quoting.
- **Lowercasing, stop-word or punctuation removal, stemming:** these would damage citations such as `Article 6(2)` and point labels like `(a)`. Modern embedding models work better with natural text.
- **Line-end de-hyphenation (regular `-`):** no line in the extracted text ends with a hyphen, so there is nothing to rejoin.

`reports/lab-2/random_chunks.md` contains 10 randomly chosen chunks for manual inspection. Regenerate it with `uv run python scripts/sample_chunks.py` (fixed seed).

## Project structure

```
RAG_system_LW/
├── src/document_processing/
│   ├── pdf_extractor.py   # PDF -> cleaned, NFC-normalized text lines with page numbers and bold/italic flags
│   ├── structure.py       # detects chapters, sections, articles and annexes
│   ├── chunker.py         # turns sections into LangChain Documents with metadata
│   └── processor.py       # runs the pipeline, writes data/processed/ and reports/lab-2/chunks.jsonl
├── tests/                 # tests against the real AI Act PDF
├── scripts/
│   └── sample_chunks.py   # writes 10 random chunks to reports/lab-2/ for manual checking
├── reports/lab-2/         # data card, chunks.jsonl (committed corpus) and random chunk sample
├── data/
│   ├── raw/eu_ai_act.pdf  # official PDF from EUR-Lex
│   └── processed/         # generated output (git-ignored)
├── main.py                # entry point (placeholder for now)
├── pyproject.toml         # project metadata and dependencies (managed by uv)
├── AGENTS.md              # guidance for AI coding agents
└── README.md
```

## Planned pipeline

1. **Ingest:** download official documents, extract text, split into chunks that respect legal structure (articles, paragraphs, annexes, recitals)
2. **Index:** embed chunks and store them in a vector store with metadata (document, article, type: binding / guidance)
3. **Retrieve:** find the most relevant chunks for a question
4. **Generate:** produce a plain-language answer with citations, using only retrieved context
5. **Evaluate:** test on a set of reference questions against the criteria above

Specific libraries (LLM, embeddings, vector store) are not chosen yet.
