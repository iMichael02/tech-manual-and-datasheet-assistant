"""PDF -> page text -> overlapping chunks with provenance metadata.

Uses poppler's ``pdftotext``. Reading-order mode is the default because the datasheets are
two-column and ``-layout`` interleaves columns and wrecks sentences. Pages that look like
tables (many lines with >=3 columns) are re-extracted with ``-layout`` so each row's label,
value and unit stay together (reading-order mode scatters them, which let the LLM attach
the wrong value to a label, e.g. MAX31855 0.25 vs 0.0625 resolution).
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from .catalog import CATALOG

CHUNK_WORDS = 180
OVERLAP_WORDS = 40


@dataclass
class Chunk:
    id: int
    doc: str          # file name
    title: str
    parts: list[str]  # aliases of the part(s) this doc covers
    page: int
    text: str


def _extract(pdf: Path, layout: bool) -> list[str]:
    cmd = ["pdftotext"] + (["-layout"] if layout else []) + [str(pdf), "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    pages = out.decode("utf-8", "replace").split("\f")
    return pages[:-1] if pages and not pages[-1].strip() else pages


_COLS = re.compile(r"\s{3,}")


def is_table_page(layout_text: str, min_rows: int = 8, min_ratio: float = 0.3) -> bool:
    lines = [l for l in layout_text.splitlines() if l.strip()]
    rows = sum(len(_COLS.split(l.strip())) >= 3 for l in lines)
    return rows >= min_rows and rows / len(lines) >= min_ratio


def _flatten_layout(text: str) -> str:
    """Keep row structure without adding standalone 'words' (they would eat chunk budget):
    the cell separator '|' is glued to the preceding cell and rows end with ';'."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if len(re.findall(r"[A-Za-z0-9]", line)) < 2:
            continue
        cells = ["°C" if c == "NC" else c for c in _COLS.split(line)]   # units column: NC = °C
        rows.append("| ".join(cells) + ";")
    return " ".join(rows)


def _pages(pdf: Path) -> list[str]:
    plain = _extract(pdf, layout=False)
    lay = _extract(pdf, layout=True)
    if len(lay) != len(plain):          # page split mismatch: be safe
        return plain
    return [_flatten_layout(l) if is_table_page(l) else p for p, l in zip(plain, lay)]


def _clean(text: str) -> str:
    text = text.replace("­", "").replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = re.sub(r"(\d)NC\b", r"\1°C", text)   # pdftotext renders the degree sign as 'N' on some PDFs
    # Rejoin words hyphenated across line breaks, then collapse whitespace.
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return re.sub(r"[ \t]*\n[ \t]*", " ", text).strip()


def chunk_page(text: str, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    words = _clean(text).split()
    if len(words) < 8:
        return []
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(1, len(words) - overlap), step)]


def ingest(data_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for pdf in sorted(data_dir.glob("*.pdf")):
        title, parts = CATALOG.get(pdf.name, (pdf.stem, [pdf.stem.upper()]))
        for pageno, ptext in enumerate(_pages(pdf), start=1):
            for piece in chunk_page(ptext):
                chunks.append(Chunk(len(chunks), pdf.name, title, parts, pageno, piece))
    return chunks


def save(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(c) for c in chunks], ensure_ascii=False))


def load(path: Path) -> list[Chunk]:
    return [Chunk(**d) for d in json.loads(path.read_text())]
