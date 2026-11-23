from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from . import ingest as ing
from .generate import REFUSAL, is_refusal, make_backend, sources
from .retrieval import Retrieval, Retriever

# src/assistant/pipeline.py -> repo root is three levels up (editable install); set DATASHEET_ASSISTANT_ROOT to point at a repo checkout otherwise
ROOT = Path(os.environ.get("DATASHEET_ASSISTANT_ROOT") or Path(__file__).resolve().parents[2])
DATA_DIR = ROOT / "data" / "datasheets"
INDEX = ROOT / "data" / "index" / "chunks.json"


@dataclass
class Answer:
    text: str
    sources: list[str]
    retrieval: Retrieval
    refused: bool = False


@dataclass
class Assistant:
    retriever: Retriever
    backend: object
    top_k: int = 5
    history: list = field(default_factory=list)

    @classmethod
    def build(cls, backend: str = "auto", rebuild: bool = False, **kw) -> "Assistant":
        if rebuild or not INDEX.exists():
            ing.save(ing.ingest(DATA_DIR), INDEX)
        return cls(Retriever(ing.load(INDEX), **kw), make_backend(backend))

    def ask(self, question: str, remember: bool = False, rag: bool = True) -> Answer:
        if not rag:
            # Baseline: no retrieval, no scope gate, no excerpts. Refusal is whatever the model decides.
            text = self.backend.answer(question, None, self.history if remember else None)
            return Answer(text, [], Retrieval([], [], [], 0.0, True), refused=is_refusal(text))
        r = self.retriever.search(question, k=self.top_k)
        if not r.in_scope:
            return Answer(REFUSAL.format(reason=r.reason), [], r, refused=True)
        text = self.backend.answer(question, r, self.history if remember else None)
        if remember:
            self.history += [{"role": "user", "content": question}, {"role": "assistant", "content": text}]
            self.history = self.history[-6:]
        return Answer(text, sources(r), r)
