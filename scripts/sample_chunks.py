"""Write 10 randomly chosen chunks to reports/lab-2/random_chunks.md for manual inspection.

Usage: uv run python scripts/sample_chunks.py
Run the processor first. The seed is fixed so the same chunks are picked every time.
"""

import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.json"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "lab-2" / "random_chunks.md"
SEED = 42
SAMPLE_SIZE = 10

chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
sample = random.Random(SEED).sample(chunks, SAMPLE_SIZE)

parts = [
    "# 10 random chunks after cleaning\n",
    f"Randomly selected from `data/processed/chunks.json` ({len(chunks)} chunks) with `random.Random({SEED})`. "
    "Regenerate with `uv run python scripts/sample_chunks.py`.\n",
]
for n, chunk in enumerate(sample, 1):
    meta = chunk["metadata"]
    parts.append(f"## {n}. `{meta['chunk_id']}`\n")
    parts.append(
        f"- Section type: {meta['section_type']}\n"
        f"- Pages: {meta['page_number']}–{meta['page_end']}\n"
        f"- Length: {len(chunk['page_content'])} characters\n"
    )
    parts.append(f"```text\n{chunk['page_content']}\n```\n")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text("\n".join(parts), encoding="utf-8")
print(f"Wrote {SAMPLE_SIZE} chunks to {OUTPUT_PATH}")
