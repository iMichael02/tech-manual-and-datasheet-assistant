"""Evaluation harness.

Retrieval (offline, deterministic):
  hit@k   – an expected document appears in the top-k hits
  ev@k    – the gold evidence pattern is present in the top-k text (what the LLM actually sees)
  MRR     – reciprocal rank of the first hit that contains the evidence
Scope:
  refusal rate on out-of-scope questions, false-refusal rate on answerable ones
Generation (optional, --generation): fraction of answers containing the gold keywords
  (case-insensitive regexes) and refusing correctly on out-of-scope questions.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from assistant.pipeline import Assistant

QS = json.loads((Path(__file__).parent / "questions.json").read_text())


def _has(text: str, pats: list[str]) -> bool:
    return all(re.search(p, text, re.I) for p in pats)


def retrieval_eval(a: Assistant, label: str, ks=(1, 3, 5)) -> dict:
    hit = {k: 0 for k in ks}; ev = {k: 0 for k in ks}; mrr = 0.0; miss = []
    n = len(QS["answerable"])
    for item in QS["answerable"]:
        r = a.retriever.search(item["q"], k=max(ks))
        rank = None
        for i, h in enumerate(r.hits, 1):
            if h.chunk.doc in item["docs"] and _has(h.chunk.text, item["evidence"]):
                rank = i; break
        for k in ks:
            hit[k] += any(h.chunk.doc in item["docs"] for h in r.hits[:k])
            ev[k] += rank is not None and rank <= k
        mrr += 1 / rank if rank else 0
        if rank is None or rank > 3:
            miss.append((item["q"], rank))
    return {"label": label, "hit": {k: v / n for k, v in hit.items()}, "ev": {k: v / n for k, v in ev.items()},
            "mrr": mrr / n, "miss": miss}


def scope_eval(a: Assistant) -> tuple[float, float, list[str], list[str]]:
    oos = QS["out_of_scope"]
    wrong_in = [q for q in oos if a.retriever.search(q).in_scope]
    false_ref = [i["q"] for i in QS["answerable"] if not a.retriever.search(i["q"]).in_scope]
    return 1 - len(wrong_in) / len(oos), len(false_ref) / len(QS["answerable"]), wrong_in, false_ref


def corpus_sanity(a: Assistant) -> list[str]:
    """Every gold item must be answerable from its stated docs; otherwise the gold set is wrong."""
    bad = []
    for i in QS["answerable"]:
        if not any(c.doc in i["docs"] and _has(c.text, i["evidence"]) for c in a.retriever.chunks):
            bad.append(i["q"])
    return bad


def generation_eval(a: Assistant, rag: bool = True) -> None:
    ok = 0
    for i in QS["answerable"]:
        ans = a.ask(i["q"], rag=rag)
        good = (not ans.refused) and _has(ans.text, i["answer"])
        ok += good
        if not good:
            print(f"  ✗ {i['q']}\n      -> {ans.text[:160]!r}")
    refused = sum(a.ask(q, rag=rag).refused for q in QS["out_of_scope"])
    n = len(QS["answerable"])
    print(f"\nGeneration [{a.backend.name}, {'RAG' if rag else 'no RAG'}]: keyword-correct {ok}/{n} ({ok/n:.0%}); "
          f"correct refusals {refused}/{len(QS['out_of_scope'])}")
    if a.backend.name == "extractive":
        print("  (extractive mode only quotes passages; use an LLM backend for a meaningful number)")


def run(backend: str = "auto", generation: bool = False, norag: bool = False) -> None:
    configs = [("BM25, no routing", dict(part_routing=False)),
               ("BM25 + part routing", dict(part_routing=True))]
    print(f"{len(QS['answerable'])} answerable + {len(QS['out_of_scope'])} out-of-scope questions\n")
    base = Assistant.build("extractive")
    bad = corpus_sanity(base)
    print(f"Gold-set sanity: {len(QS['answerable']) - len(bad)}/{len(QS['answerable'])} items verifiable in corpus")
    for q in bad:
        print("  ! evidence not found in corpus:", q)
    print(f"\n{'config':<22}{'hit@1':>7}{'hit@3':>7}{'hit@5':>7}{'ev@1':>7}{'ev@3':>7}{'ev@5':>7}{'MRR':>7}")
    last = None
    for label, kw in configs:
        a = Assistant.build("extractive", **kw)
        m = retrieval_eval(a, label)
        print(f"{label:<22}" + "".join(f"{m['hit'][k]:>7.0%}" for k in (1, 3, 5)) +
              "".join(f"{m['ev'][k]:>7.0%}" for k in (1, 3, 5)) + f"{m['mrr']:>7.2f}")
        last = m
    for q, rank in last["miss"]:
        print(f"  miss (rank {rank}): {q}")
    a = Assistant.build("extractive")
    ref, falseref, wrong_in, false_ref = scope_eval(a)
    print(f"\nOut-of-scope refusal: {ref:.0%}   False refusals on answerable: {falseref:.0%}")
    for q in wrong_in: print("  answered instead of refusing:", q)
    for q in false_ref: print("  wrongly refused:", q)
    if generation:
        a = Assistant.build(backend)
        generation_eval(a)
        if norag and a.backend.name == "extractive":
            print("\n(--norag skipped: the extractive backend has no model)")
        elif norag:
            print("\n--- baseline: same model, no retrieval (pre-prompt only) ---")
            generation_eval(a, rag=False)
