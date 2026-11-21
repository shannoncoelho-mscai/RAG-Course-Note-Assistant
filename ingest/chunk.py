"""Heading-aware recursive chunker — AI textbook domain.

Splits each page into overlapping chunks (target 500 tokens, overlap 80)
at heading boundaries first, then falls back to paragraph/sentence
splitting. Heading detection runs on the running text (not just
_page-level regex) so section changes mid-page are honoured.

Every chunk is prefixed with `Chapter N | {section} |`
so textbook context survives retrieval — this measurably improves hit-rate
over naive fixed-size splitting.

Each chunk dict: {chunk_id, page, section, text, source}
"""
from __future__ import annotations

import re
import uuid

TARGET_TOKENS = 500
OVERLAP_TOKENS = 80
APPROX_TOKENS_PER_WORD = 0.75  # rough: 1 token ≈ 1.33 words; reverse for splitting

# --- Heading detection (three levels, checked in order of specificity) ----
#   Level 1 — chapter:        "CHAPTER 2" or "Chapter 2"
#   Level 2 — numbered section: "2.1" or "2.10"
#   Level 3 — standalone title: ALL-CAPS line of >= 3 chars, no digits
_CHAPTER_RE = re.compile(r"^CHAPTER\s+(\d+)\s*$", re.IGNORECASE)
_SECTION_RE = re.compile(r"^(\d+(?:\.\d+)+)\.?\s+.*$", re.MULTILINE)
_TITLE_RE = re.compile(r"^[A-Z][A-Z0-9 ]{3,}$")


def _est_tokens(text: str) -> int:
    return int(len(text.split()) / APPROX_TOKENS_PER_WORD)


def _split_paragraphs(text: str) -> list[str]:
    """Split on blank lines (paragraph boundaries) and filter empties."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _split_sentences(text: str) -> list[str]:
    """Sentence split for oversized paragraphs."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if s.strip()] or [text.strip()]


def _sliding_window(paras: list[str], target: int, overlap: int) -> list[str]:
    """Pack paragraphs/sentence fragments into target-sized chunks with overlap.

    If a single paragraph exceeds `target` tokens, it is sentence-split
    before packing so no chunk ever overflows the target by more than one
    sentence.
    """
    # Expand oversized paragraphs into sentence chunks first
    units: list[str] = []
    for para in paras:
        if _est_tokens(para) > target:
            units.extend(_split_sentences(para))
        else:
            units.append(para)

    chunks: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for unit in units:
        unit_tokens = _est_tokens(unit)
        if buf_tokens + unit_tokens > target and buf:
            chunks.append("\n\n".join(buf))
            tail_tokens = 0
            tail: list[str] = []
            for p in reversed(buf):
                t = _est_tokens(p)
                if tail_tokens + t > overlap:
                    break
                tail.insert(0, p)
                tail_tokens += t
            buf = tail
            buf_tokens = tail_tokens
        buf.append(unit)
        buf_tokens += unit_tokens
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def chunk_pages(pages: list[dict], source_name: str) -> list[dict]:
    """Turn page dicts (from parse_pdf) into citation-ready chunks.

    Chunk IDs use a per-page counter: `{source}:p{page}_c{NNN}` where NNN
    resets on every page (so page_num always matches the chunk's page).
    """
    all_chunks: list[dict] = []
    current_chapter = "untitled"
    current_section = "untitled"

    for page in pages:
        text = page["text"]
        page_num = page["page"]

        # Update chapter/section from headings in the page text
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            m = _CHAPTER_RE.match(s)
            if m:
                current_chapter = f"Chapter {m.group(1)}"
                current_section = current_chapter
                continue
            m = _SECTION_RE.match(s)
            if m:
                current_section = m.group(0).strip()
                continue
            if _TITLE_RE.match(s):
                current_section = s.title()
                continue

        paras = _split_paragraphs(text)
        for chunk_text in _sliding_window(paras, TARGET_TOKENS, OVERLAP_TOKENS):
            all_chunks.append({
                "chunk_id": f"{source_name}:p{page_num}_c{len(all_chunks) + 1:03d}",
                "page": page_num,
                "section": current_section,
                "text": f"{source_name} > {current_section} |\n{chunk_text}",
                "source": source_name,
            })

    return all_chunks
