"""Digital PDF extraction, heuristically ordered blocks and legal clause-aware chunks.

Scanned-image pages are flagged and excluded; OCR is NOT implemented.
Tables are best-effort (merged cells and spanning headers require manual QA).
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import fitz

from .schema import Chunk, DocumentStats

SECTION_RE = re.compile(
    r"^(?:(?:ARTICLE|Article|Art\.)\s+\d+[A-Za-z]?|"
    r"(?:SECTION|Section)\s+\d+(?:\.\d+)*|"
    r"(?:CHAPTER|Chapter)\s+[IVXLC\d]+|"
    r"(?:CLAUSE|Clause)\s+\d+(?:\.\d+)*|"
    r"§\s*\d+(?:\.\d+)*|"
    r"\d+(?:\.\d+){1,3}(?:\s|\.|$))", re.UNICODE
)


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _reading_order(blocks: list[dict], width: float) -> list[dict]:
    """Column heuristic. Full-width blocks go first only if above the columns.

    Not a guaranteed reading-order parser for unusual layouts.
    """
    narrow = [b for b in blocks if b["bbox"][2] - b["bbox"][0] < width * .65]
    left = [b for b in narrow if (b["bbox"][0]+b["bbox"][2])/2 < width * .48]
    right = [b for b in narrow if (b["bbox"][0]+b["bbox"][2])/2 > width * .52]
    two_columns = len(left) >= 3 and len(right) >= 3
    if not two_columns:
        return sorted(blocks, key=lambda b: (round(b["bbox"][1] / 8), b["bbox"][0]))
    first_column_top = min(b["bbox"][1] for b in narrow)
    headings = [b for b in blocks if b not in narrow and b["bbox"][3] <= first_column_top]
    remaining = [b for b in blocks if b not in headings]
    def col_key(b):
        mid = (b["bbox"][0] + b["bbox"][2]) / 2
        return (0 if mid < width * .50 else 1, b["bbox"][1], b["bbox"][0])
    return sorted(headings, key=lambda b: b["bbox"][1]) + sorted(remaining, key=col_key)


def _merge_bbox(rectangles):
    if not rectangles:
        return None
    return [round(min(r[0] for r in rectangles), 2),
            round(min(r[1] for r in rectangles), 2),
            round(max(r[2] for r in rectangles), 2),
            round(max(r[3] for r in rectangles), 2)]


def _pieces(text: str, limit: int):
    """Prefer paragraph and sentence breaks to fixed character segmentation."""
    paragraphs = [s.strip() for s in re.split(r"\n\s*\n", text) if s.strip()]
    for para in paragraphs:
        if len(para) <= limit:
            yield para
        else:
            sentences = re.split(r"(?<=[.!?;])\s+(?=[A-Z(\d])", para)
            part = ""
            for sent in sentences:
                if part and len(part) + len(sent) + 1 > limit:
                    yield part
                    part = ""
                if len(sent) > limit:
                    if part:
                        yield part
                        part = ""
                    # Last-resort boundary split for unpunctuated PDF blocks
                    for i in range(0, len(sent), limit):
                        yield sent[i:i + limit]
                else:
                    part = (part + " " + sent).strip()
            if part:
                yield part


def extract_pdf(data: bytes, filename: str, include_tables: bool = True,
                max_chars: int = 1100) -> tuple[list[Chunk], DocumentStats]:
    """Returns chunks/statistics. Physical page numbers are used throughout."""
    digest = hashlib.sha256(data).hexdigest()
    doc = fitz.open(stream=data, filetype="pdf")
    if doc.needs_pass:
        raise ValueError(f"Password-protected PDF is unsupported: {filename}")
    if doc.page_count > 750:
        raise ValueError("PDF exceeds 750 pages; split it into smaller public files.")
    page_data = []
    repeats = Counter()
    scanned = []
    tables_seen = 0
    for pno, page in enumerate(doc):
        blocks = []
        for b in page.get_text("blocks"):
            if len(b) < 7 or b[6] != 0:
                continue
            txt = b[4].strip()
            if not txt:
                continue
            bbox = list(map(float, b[:4]))
            blocks.append({"text": txt, "bbox": bbox})
            # Header/footer recurrence across pages only
            if bbox[1] < 55 or bbox[3] > page.rect.height - 45:
                repeats[_clean(txt).lower()] += 1
        if len(page.get_text("words")) < 25 and len(page.get_images(full=True)) > 0:
            scanned.append(pno+1)
        page_data.append((page.rect.width, page.rect.height, blocks))
    chunks: list[Chunk] = []
    current_section = "Unlabeled section"
    section_labels: set[str] = set()
    seq = 0
    for pno, (width, height, blocks) in enumerate(page_data, start=1):
        if pno in scanned:
            continue  # Never treat short captions from an image-only page as full legal evidence.
        # Repeated running headers/footers removed; footnotes in content retained.
        cleaned = []
        for b in blocks:
            in_margin = b["bbox"][1] < 55 or b["bbox"][3] > height-45
            if in_margin and repeats[_clean(b["text"]).lower()] >= 3:
                continue
            cleaned.append(b)
        ordered = _reading_order(cleaned, width)
        buffer: list[str] = []
        rects = []
        def flush():
            nonlocal seq, buffer, rects
            if not buffer:
                return
            for item in _pieces("\n\n".join(buffer), max_chars):
                if len(item.strip()) < 12:
                    continue
                seq += 1
                chunks.append(Chunk(
                    id=f"{digest[:12]}-p{pno}-c{seq}", document=filename,
                    doc_hash=digest, page=pno, section=current_section,
                    text=item, bbox=_merge_bbox(rects), kind="text"))
            buffer, rects = [], []
        for b in ordered:
            raw = b["text"].strip()
            # PDF lines can contain multiple paragraphs; match top line for heading.
            first_line = raw.splitlines()[0].strip()
            match = SECTION_RE.match(first_line)
            if match and len(first_line) <= 110:
                flush()
                current_section = match.group(0).strip().rstrip(".")
                section_labels.add(current_section)
            if buffer and sum(map(len, buffer)) + len(raw) > max_chars:
                flush()
            buffer.append(raw)
            rects.append(b["bbox"])
        flush()  # never cite across physical page boundaries
        if include_tables and pno not in scanned:
            try:
                tables = doc[pno-1].find_tables()
                for table_no, tbl in enumerate(tables.tables, start=1):
                    matrix = tbl.extract()
                    lines = [" | ".join(_clean(str(x or "")) for x in row)
                             for row in matrix]
                    table_text = "\n".join(s for s in lines if s.strip(" |"))
                    if len(table_text) >= 15:
                        seq += 1
                        tables_seen += 1
                        chunks.append(Chunk(
                            id=f"{digest[:12]}-p{pno}-t{table_no}", document=filename,
                            doc_hash=digest, page=pno,
                            section=f"{current_section} / Table {table_no}",
                            text=table_text[:6000], bbox=list(tbl.bbox), kind="table"))
            except (ValueError, AttributeError, RuntimeError):
                # Not all PDF table structures can be parsed; non-table text remains.
                pass
    doc.close()
    stats = DocumentStats(document=filename, doc_hash=digest,
                          pages=len(page_data), chunks=len(chunks),
                          scanned_pages=scanned, tables_detected=tables_seen,
                          sections_detected=len(section_labels))
    return chunks, stats


def load_or_extract(data: bytes, filename: str, cache_dir: Path,
                    include_tables: bool = True):
    digest = hashlib.sha256(data).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"parse_{digest}_{int(include_tables)}_v2.json"
    if cache_path.exists():
        obj = json.loads(cache_path.read_text(encoding="utf-8"))
        chunks = [Chunk(**c) for c in obj["chunks"]]
        for c in chunks:
            c.document = filename
        st = DocumentStats(**obj["stats"])
        st.document = filename
        return chunks, st
    chunks, st = extract_pdf(data, filename, include_tables=include_tables)
    cache_path.write_text(json.dumps({"chunks": [c.as_dict() for c in chunks],
                                      "stats": vars(st)}, ensure_ascii=False),
                          encoding="utf-8")
    return chunks, st
