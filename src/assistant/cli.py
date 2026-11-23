from __future__ import annotations

import argparse

from .pipeline import Assistant


def show(a) -> None:
    print("\n" + a.text)
    if a.sources:
        print("\nSources:\n  " + "\n  ".join(a.sources))
    print()


def main() -> None:
    p = argparse.ArgumentParser(prog="assistant", description="Datasheet & manual assistant")
    p.add_argument("--backend", default="auto", choices=["auto", "anthropic", "ollama", "extractive"])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="(re)build the chunk index from data/datasheets/")
    ask = sub.add_parser("ask", help="ask one question")
    ask.add_argument("question", nargs="+")
    ask.add_argument("--no-rag", action="store_true", help="baseline: no retrieval, answer from the model alone")
    sub.add_parser("chat", help="interactive session")
    ev = sub.add_parser("eval", help="run the evaluation suite")
    ev.add_argument("--generation", action="store_true", help="also grade generated answers (uses the backend)")
    ev.add_argument("--norag", action="store_true",
                    help="with --generation: also run the no-retrieval baseline for comparison")
    args = p.parse_args()

    if args.cmd == "ingest":
        a = Assistant.build(args.backend, rebuild=True)
        print(f"Indexed {len(a.retriever.chunks)} chunks.")
    elif args.cmd == "ask":
        show(Assistant.build(args.backend).ask(" ".join(args.question), rag=not args.no_rag))
    elif args.cmd == "chat":
        a = Assistant.build(args.backend)
        print(f"Backend: {a.backend.name}. Ctrl-D to quit.")
        try:
            while q := input("? ").strip():
                show(a.ask(q, remember=True))
        except (EOFError, KeyboardInterrupt):
            pass
    elif args.cmd == "eval":
        from .evaluation.run import run
        run(args.backend, args.generation, args.norag)


if __name__ == "__main__":
    main()
