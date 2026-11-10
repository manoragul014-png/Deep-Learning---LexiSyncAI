"""Small serializable data models for legal evidence."""
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Chunk:
    id: str
    document: str
    doc_hash: str
    page: int  # Physical PDF page, 1-indexed
    section: str
    text: str
    bbox: Optional[list[float]] = None  # Extracted block region; may be approximate
    kind: str = "text"

    def as_dict(self):
        return asdict(self)


@dataclass
class DocumentStats:
    document: str
    doc_hash: str
    pages: int
    chunks: int
    scanned_pages: list[int]
    tables_detected: int
    sections_detected: int
    reading_order: str = "heuristic (two-column aware)"
