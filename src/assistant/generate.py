"""Answer generation backends.

* ``anthropic`` – Claude via the Messages API (set ANTHROPIC_API_KEY)
* ``ollama``    – local model via Ollama's HTTP API (set OLLAMA_MODEL)
* ``extractive``– no LLM: returns the most relevant sentences verbatim, with citations.
                  Zero-hallucination fallback and the offline baseline for evaluation.
"""
from __future__ import annotations

import os
import re

import requests

from .retrieval import Retrieval, tokenize

REFUSAL = ("I can't answer that from the available documentation. "
           "I only cover the datasheets and manuals in the knowledge base ({reason}).")

SYSTEM_PROMPT = """You are a technical assistant for engineers and technicians. You answer questions about \
electronic components, sensors and instruments using ONLY the numbered excerpts from manuals and datasheets \
provided in the user message.

Rules:
1. Ground every claim in the excerpts. Cite sources inline as [n] using the excerpt numbers.
2. Quote numeric values, units, addresses and register names exactly as written. Never round or convert silently.
3. If the excerpts do not contain the answer, say so plainly instead of guessing or using outside knowledge.
4. If several devices are involved or the excerpts disagree (e.g. typical vs. maximum values, different variants), \
say which device/condition each value belongs to.
5. For troubleshooting, give short numbered steps. Otherwise be concise: the answer first, then supporting detail."""


REFUSAL_PREFIX = "I can't answer that from the available documentation"

# Baseline for evaluation ("pre-prompting only"): same persona and scope, but no excerpts are provided.
NORAG_SYSTEM_PROMPT = """You are a technical assistant for engineers and technicians. You answer questions about \
electronic components, sensors and instruments. Your knowledge base covers ONLY these documents: {docs}.

Rules:
1. Answer concisely from your own knowledge. Quote numeric values, units and addresses exactly; do not guess.
2. If the question is about a device or topic not covered by those documents, or you are not sure of the answer, \
reply starting exactly with: "{prefix}".
3. For troubleshooting, give short numbered steps."""


def norag_system_prompt() -> str:
    from .catalog import CATALOG
    docs = "; ".join(sorted({title for title, _ in CATALOG.values()}))
    return NORAG_SYSTEM_PROMPT.format(docs=docs, prefix=REFUSAL_PREFIX)


def is_refusal(text: str) -> bool:
    return text.strip().startswith(REFUSAL_PREFIX)


def format_context(r: Retrieval) -> str:
    return "\n\n".join(f"[{i}] ({h.chunk.title} — {h.chunk.doc}, p.{h.chunk.page})\n{h.chunk.text}"
                       for i, h in enumerate(r.hits, 1))


def sources(r: Retrieval) -> list[str]:
    return [f"[{i}] {h.chunk.title} ({h.chunk.doc}), p.{h.chunk.page}" for i, h in enumerate(r.hits, 1)]


def _user_msg(question: str, r: Retrieval | None) -> str:
    if r is None:                      # no-RAG baseline: bare question
        return question
    return f"Excerpts:\n\n{format_context(r)}\n\nQuestion: {question}"


class Extractive:
    name = "extractive"
    WIN, STRIDE = 45, 15

    def answer(self, question: str, r: Retrieval | None, history=None) -> str:
        if r is None:
            raise ValueError("the extractive backend has no model, so it cannot run in no-RAG mode")
        q = set(tokenize(question))
        cands = []
        for n, h in enumerate(r.hits[:3], 1):
            words = h.chunk.text.split()
            for i in range(0, max(1, len(words) - self.WIN + self.STRIDE), self.STRIDE):
                w = words[i:i + self.WIN]
                toks = set(tokenize(" ".join(w)))
                cands.append((len(q & toks) / (len(q) or 1) + 0.02 * any(c.isdigit() for c in " ".join(w)), n, i, " ".join(w)))
        cands.sort(reverse=True)
        picked, used = [], set()
        for sc, n, i, text in cands:
            if (n, i // self.WIN) in used:
                continue
            used.add((n, i // self.WIN))
            picked.append(f"- …{text}… [{n}]")
            if len(picked) == 3:
                break
        return "Most relevant passages (extractive mode, no LLM configured):\n" + "\n".join(picked)


class Anthropic:
    name = "anthropic"

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")
        self.key = os.environ["ANTHROPIC_API_KEY"]

    def answer(self, question: str, r: Retrieval | None, history=None) -> str:
        system = SYSTEM_PROMPT if r is not None else norag_system_prompt()
        msgs = list(history or []) + [{"role": "user", "content": _user_msg(question, r)}]
        resp = requests.post("https://api.anthropic.com/v1/messages", timeout=120,
                             headers={"x-api-key": self.key, "anthropic-version": "2023-06-01"},
                             json={"model": self.model, "max_tokens": 800, "temperature": 0,
                                   "system": system, "messages": msgs})
        resp.raise_for_status()
        return "".join(b["text"] for b in resp.json()["content"] if b["type"] == "text")


class Ollama:
    name = "ollama"

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
        self.host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def answer(self, question: str, r: Retrieval | None, history=None) -> str:
        system = SYSTEM_PROMPT if r is not None else norag_system_prompt()
        msgs = [{"role": "system", "content": system}] + list(history or []) + \
               [{"role": "user", "content": _user_msg(question, r)}]
        resp = requests.post(f"{self.host}/api/chat", timeout=300,
                             json={"model": self.model, "messages": msgs, "stream": False,
                                   "options": {"temperature": 0}})
        resp.raise_for_status()
        return resp.json()["message"]["content"]


def make_backend(name: str = "auto"):
    if name == "auto":
        if os.environ.get("ANTHROPIC_API_KEY"):
            name = "anthropic"
        else:
            try:
                requests.get(os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/tags", timeout=1)
                name = "ollama"
            except requests.RequestException:
                name = "extractive"
    return {"anthropic": Anthropic, "ollama": Ollama, "extractive": Extractive}[name]()
