"""BM25 + transformer embeddings + reciprocal rank fusion + cross-encoder.

No hidden lexical-only fallback: missing model files cause a clear error.
"""
from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .schema import Chunk

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def tokenize(s: str):
    # Preserve references such as 83(4), 7.2, and exact clause numbers.
    return re.findall(r"\d+(?:\.\d+)*(?:\([a-z\d]+\))*|[a-z]+(?:'[a-z]+)?", s.lower())


class BM25:
    """BM25 Okapi, implemented without an extra retrieval package."""
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = .75):
        self.docs = [Counter(tokenize(s)) for s in texts]
        self.N = len(texts)
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avgdl = sum(self.lengths)/self.N if self.N else 1
        df = Counter()
        for d in self.docs:
            df.update(d.keys())
        self.idf = {t: math.log(1 + (self.N - n + .5)/(n + .5))
                    for t, n in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query: str) -> np.ndarray:
        terms = set(tokenize(query))
        out = np.zeros(self.N, dtype=np.float32)
        for i, doc in enumerate(self.docs):
            norm = self.k1 * (1 - self.b + self.b * self.lengths[i]/self.avgdl)
            for tok in terms:
                tf = doc.get(tok, 0)
                if tf:
                    out[i] += self.idf.get(tok, 0) * (tf * (self.k1 + 1))/(tf + norm)
        return out


def rrf_merge(ranked_lists: list[list[int]], limit: int = 40, k: int = 60):
    scores = Counter()
    for ranked in ranked_lists:
        for rank, index in enumerate(ranked, start=1):
            scores[index] += 1 / (k + rank)
    return [i for i, _ in scores.most_common(limit)], scores


class Retriever:
    def __init__(self, chunks: list[Chunk], cache_dir: Path,
                 embed_model: str = EMBED_MODEL, rerank_model: str = RERANK_MODEL):
        if not chunks:
            raise ValueError("No searchable text found. Check for scanned PDFs.")
        self.chunks = chunks
        self.bm25 = BM25([f"{c.section} {c.text}" for c in chunks])
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embed_model_name = embed_model
        self.rerank_model_name = rerank_model
        self._embedder = None
        self._reranker = None
        self.vectors = None
        self.index_seconds = None

    def _model(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RuntimeError("Install requirements.txt to use dense retrieval.") from e
            self._embedder = SentenceTransformer(self.embed_model_name)
        return self._embedder

    def _cross_encoder(self):
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:
                raise RuntimeError("Install requirements.txt to use cross-encoder reranking.") from e
            self._reranker = CrossEncoder(self.rerank_model_name, max_length=512)
        return self._reranker

    def build(self):
        start = time.perf_counter()
        checksum = hashlib.sha256()
        checksum.update(self.embed_model_name.encode())
        for c in self.chunks:
            checksum.update(c.id.encode())
            checksum.update(c.text.encode())
        cache_path = self.cache_dir / ("embeddings_" + checksum.hexdigest()[:24] + ".npy")
        if cache_path.exists():
            v = np.load(cache_path, allow_pickle=False)
            if v.shape[0] == len(self.chunks):
                self.vectors = v
        if self.vectors is None:
            model = self._model()
            self.vectors = model.encode(
                [f"{c.section}\n{c.text}" for c in self.chunks],
                show_progress_bar=False, batch_size=32,
                normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
            np.save(cache_path, self.vectors, allow_pickle=False)
        # Pre-warm both neural models before the professor starts asking questions.
        self._model()
        self._cross_encoder()
        self.index_seconds = time.perf_counter() - start
        return self

    def rank(self, query: str, mode: str = "hybrid_rerank", top_k: int = 8):
        """Returns [{'chunk': Chunk, 'score': float, 'index': int}], timing dict.
        Modes: bm25, dense, hybrid, hybrid_rerank. Same corpus, same query.
        """
        if self.vectors is None:
            raise RuntimeError("Call build() before searching.")
        n = len(self.chunks)
        ts = time.perf_counter()
        bm25_scores = self.bm25.scores(query)
        sparse_ids = np.argsort(-bm25_scores, kind="stable").tolist()
        dense_scores, dense_ids = None, []
        if mode != "bm25":
            qvec = self._model().encode([query], normalize_embeddings=True,
                                        convert_to_numpy=True)[0]
            dense_scores = self.vectors @ qvec
            dense_ids = np.argsort(-dense_scores, kind="stable").tolist()
        retrieval_seconds = time.perf_counter() - ts
        cross_seconds = 0.0
        if mode == "bm25":
            ids = sparse_ids[:top_k]
            ranked = [(int(i), float(bm25_scores[i])) for i in ids]
        elif mode == "dense":
            ids = dense_ids[:top_k]
            ranked = [(int(i), float(dense_scores[i])) for i in ids]
        elif mode in ("hybrid", "hybrid_rerank"):
            candidates, fused = rrf_merge([sparse_ids[:min(60, n)],
                                           dense_ids[:min(60, n)]], limit=min(40, n))
            if mode == "hybrid":
                ranked = [(int(i), float(fused[i])) for i in candidates[:top_k]]
            else:
                ts = time.perf_counter()
                # Only rerank top 24 fused candidates for fast in-class demos.
                candidates = candidates[:24]
                pairs = [(query, f"{self.chunks[i].section}: {self.chunks[i].text[:2200]}")
                         for i in candidates]
                cross = np.asarray(self._cross_encoder().predict(pairs), dtype=float).reshape(-1)
                ranked = [(int(candidates[j]), float(cross[j]))
                          for j in np.argsort(-cross, kind="stable")]
                cross_seconds = time.perf_counter() - ts
                # Diversify physical pages so a multi-hop answer has a chance.
                diverse, by_page = [], Counter()
                for row in ranked:
                    c = self.chunks[row[0]]
                    key = (c.doc_hash, c.page)
                    if by_page[key] < 3:
                        diverse.append(row)
                        by_page[key] += 1
                    if len(diverse) >= top_k:
                        break
                ranked = diverse
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")
        return ([{"index": idx, "chunk": self.chunks[idx], "score": score}
                 for idx, score in ranked[:top_k]],
                {"retrieval_seconds": round(retrieval_seconds, 3),
                 "reranking_seconds": round(cross_seconds, 3)})
