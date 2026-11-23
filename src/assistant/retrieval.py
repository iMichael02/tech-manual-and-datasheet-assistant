"""BM25 retrieval with part-number routing and an out-of-scope gate.

Why not embeddings? Datasheet questions hinge on exact tokens (``0x76``, ``DS18B20``,
``I2C``, ``tCONV``) which lexical search handles well, and BM25 needs no model
download or GPU. The Retriever interface is small, so a dense/hybrid retriever
can be dropped in (see README, "Next steps").
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .catalog import NOT_PARTS
from .ingest import Chunk

TOKEN = re.compile(r"[a-z0-9]+(?:[.\-/][a-z0-9]+)*|[±°µΩ%]", re.I)
STOP = set("""a an and are as at be by can do does for from has have how i in is it its of on or
that the this to was what when where which who why will with you your should would could my me
use used using get set need between about than then into per""".split())
PART_LIKE = re.compile(r"\b[A-Za-z]{2,6}-?\d{3,5}[A-Za-z]{0,3}\d*\b|\b[A-Za-z]{2,4}\d{2,3}[A-Za-z]\d{0,3}\b")


def tokenize(text: str) -> list[str]:
    toks = []
    for t in TOKEN.findall(text.lower()):
        toks.append(t)
        if re.search(r"[.\-/]", t):  # also index the sub-pieces: "1-wire" -> 1, wire
            toks.extend(p for p in re.split(r"[.\-/]", t) if p)
    return [t for t in toks if t not in STOP]


@dataclass
class Hit:
    chunk: Chunk
    score: float


@dataclass
class Retrieval:
    hits: list[Hit]
    parts: list[str]            # known parts detected in the question
    unknown_parts: list[str]    # part-like tokens we have no document for
    coverage: float             # idf-weighted share of query terms found in hits
    in_scope: bool
    reason: str = ""


class Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75,
                 part_routing: bool = True, min_coverage: float = 0.55):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.part_routing, self.min_coverage = part_routing, min_coverage
        self.tf = [Counter(tokenize(c.text)) for c in chunks]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / max(1, len(self.len))
        df: Counter = Counter()
        for t in self.tf:
            df.update(t.keys())
        n = len(chunks)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}
        self.max_idf = max(self.idf.values())
        self.known = {a.upper(): None for c in chunks for a in c.parts}
        self.vocab_parts = set(self.known)

    # -- scoring -----------------------------------------------------------
    def _bm25(self, q: list[str], i: int) -> float:
        tf, dl, s = self.tf[i], self.len[i], 0.0
        for w in q:
            f = tf.get(w)
            if f:
                s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
        return s

    def detect_parts(self, question: str) -> tuple[list[str], list[str]]:
        up = question.upper()
        known = [p for p in self.vocab_parts if re.search(rf"(?<![A-Z0-9]){re.escape(p)}(?![A-Z0-9])", up)]
        unknown = []
        for m in PART_LIKE.findall(question):
            u = m.upper()
            if u in NOT_PARTS or any(u == k or u.startswith(k) or k.startswith(u) for k in self.vocab_parts):
                continue
            if any(ch.isdigit() for ch in u) and any(ch.isalpha() for ch in u):
                unknown.append(u)
        return sorted(known), sorted(set(unknown))

    def search(self, question: str, k: int = 5) -> Retrieval:
        parts, unknown = self.detect_parts(question)
        # Drop part names from the lexical query when routing: they'd only re-add noise
        # (e.g. "DS1000E" appears on every page header) — routing handles them instead.
        q = tokenize(question)
        pool = range(len(self.chunks))
        if self.part_routing and parts:
            pool = [i for i in pool if set(p.upper() for p in self.chunks[i].parts) & set(parts)]
            routed = {t.lower() for p in parts for t in tokenize(p)}
            q = [t for t in q if t not in routed] or q
        scored = sorted(((self._bm25(q, i), i) for i in pool), reverse=True)[:k]
        hits = [Hit(self.chunks[i], s) for s, i in scored if s > 0]

        # Coverage: how much of the (idf-weighted) question is found in the retrieved text?
        # Words absent from every document are *uncovered* evidence, so they count at max idf.
        qset = set(q)
        w = lambda t: self.idf.get(t, self.max_idf)  # noqa: E731
        total = sum(w(t) for t in qset) or 1.0
        # Per-chunk (best chunk), not pooled: "capital of France" must not pass just because
        # "capital" and "France" each occur somewhere in different, unrelated pages.
        coverage = max((sum(w(t) for t in qset if t in self.tf[h.chunk.id]) / total for h in hits),
                       default=0.0)

        in_scope, reason = True, ""
        if unknown and not parts:
            in_scope, reason = False, f"no document covers {', '.join(unknown)}"
        elif not hits:
            in_scope, reason = False, "no matching passages"
        elif coverage < self.min_coverage and not parts:
            in_scope, reason = False, f"question terms barely appear in the documents (coverage {coverage:.2f})"
        return Retrieval(hits, parts, unknown, coverage, in_scope, reason)
