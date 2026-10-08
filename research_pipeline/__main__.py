import argparse
import json
import os
import sys

from .llm import AnthropicLLM, MockLLM
from .pipeline import Config, ResearchPipeline
from .sources import LocalCorpus, PubMed
from .store import Store
from .taxonomy import load_taxonomy
from .understand import plan_reach
from .viewer import write_viewer


def build(a):
    llm = AnthropicLLM() if a.llm == "anthropic" else MockLLM()
    srcs = ([LocalCorpus(a.corpus)] if a.corpus else []) + ([PubMed()] if a.pubmed else [])
    cfg = Config(threshold=a.threshold, max_attempts=a.attempts, require_clearance=False, loop_depth=getattr(a, "depth", 0), max_questions=getattr(a, "max_questions", 8))
    return ResearchPipeline(llm, srcs, Store(a.db), load_taxonomy(a.taxonomy), cfg)


def show(res):
    print(f"\n[{res.status}] depth={res.depth} attempts={res.attempts} score={res.score.composite} umbrella={res.umbrella}\nQ: {res.question}\n{res.answer}")
    print("scores:", res.score.dims)
    if res.followups:
        print("follow-ups:", *res.followups, sep="\n  - ")


def main(argv=None):
    p = argparse.ArgumentParser(prog="research_pipeline")
    p.add_argument("--db", default="research.db")
    p.add_argument("--corpus", help="folder of .txt/.md evidence")
    p.add_argument("--pubmed", action="store_true", help="also search PubMed (needs network)")
    p.add_argument("--llm", choices=["mock", "anthropic"], default="mock")
    p.add_argument("--taxonomy", help="JSON file extending the specialty tree")
    p.add_argument("--threshold", type=float, default=0.72)
    p.add_argument("--attempts", type=int, default=3)
    p.add_argument("--viz", help="write an interactive HTML replay of the run(s) to this path")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ask"); s.add_argument("question")
    s = sub.add_parser("loop"); s.add_argument("question"); s.add_argument("--depth", type=int, default=2); s.add_argument("--max-questions", type=int, default=8)
    s = sub.add_parser("train"); s.add_argument("file", help="one question per line"); s.add_argument("--loop", action="store_true")
    s = sub.add_parser("serve"); s.add_argument("--port", type=int, default=8765); s.add_argument("--host", default="127.0.0.1")
    s = sub.add_parser("understand"); s.add_argument("question"); s.add_argument("--answer", action="append", default=[], help="slot=text, e.g. acuity=chronic")
    s = sub.add_parser("library"); s.add_argument("scope", nargs="?", default="overview", help="overview | umb:<umbrella> | <role id>"); s.add_argument("--ingest", help="folder of .txt/.md to add"); s.add_argument("--to", help="scopes for --ingest, comma separated")
    sub.add_parser("stats")
    s = sub.add_parser("export"); s.add_argument("--sft", default="sft.jsonl"); s.add_argument("--prefs", default="prefs.jsonl")
    a = p.parse_args(argv)
    if a.cmd == "serve":
        from .server import App, serve
        return serve(App(a.db, a.corpus, a.taxonomy), a.host, a.port)
    pl = build(a)
    results = []
    if a.cmd == "understand":
        u = pl.understand(a.question, dict(x.split("=", 1) for x in a.answer))
        print(u.message)
        print("\n--- extended reach would add:", plan_reach(u, pl.tax)["reasons"])
        return
    if a.cmd == "library":
        if a.ingest:
            from .sources import LocalCorpus
            scopes = (a.to or "").split(",")
            n = 0
            for t, x, f, q in LocalCorpus(a.ingest).docs:
                from .sources import Evidence
                pl.library.ingest(Evidence("", t, x, "local:" + f, q), [s for s in scopes if s])
                n += 1
            print(f"ingested {n} documents into {scopes}")
        print(json.dumps(pl.library.report(a.scope) if a.scope != "overview" else dict(overview=pl.library.overview(), total=pl.library.size()), indent=1)[:6000])
        return
    if a.cmd == "ask":
        results = [pl.ask(a.question)]
    elif a.cmd == "loop":
        results = pl.loop(a.question)
    elif a.cmd == "train":
        qs = [l.strip() for l in open(a.file) if l.strip()]
        pl.train(qs, with_loop=a.loop, on_result=results.append)
        print(json.dumps(pl.store.stats(), indent=1))
    elif a.cmd == "stats":
        print(json.dumps(pl.store.stats(), indent=1))
    elif a.cmd == "export":
        print("sft:", pl.store.export_sft(a.sft), "prefs:", pl.store.export_preferences(a.prefs))
    if a.cmd in ("ask", "loop"):
        for r in results:
            show(r)
    if a.viz and results:
        write_viewer(results, a.viz)
        print("viewer ->", a.viz)


if __name__ == "__main__":
    main()
