"""Local web app: `python -m research_pipeline serve` -> control panel (prompt, roles, expertise demands, parameters,
knowledge) + live node-graph replay of the real pipeline run. Stdlib only."""
from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import agents
from .llm import AnthropicLLM, MockLLM
from .pipeline import Config, ResearchPipeline
from .sources import Inline, LocalCorpus, PubMed
from .store import Store
from .taxonomy import FACETS, load_taxonomy
from .viewer import TEMPLATE, run_json

DEFAULTS = dict(threshold=0.72, max_attempts=3, top_internal=4, top_external=2, evidence_n=8, loop_depth=2, max_questions=6,
                min_quality=0.0, llm="mock", model="claude-sonnet-5-5", pubmed=False, umbrella="auto")


class App:
    def __init__(self, db=":memory:", corpus=None, taxonomy=None):
        self.store, self.tax, self.corpus = Store(db), load_taxonomy(taxonomy), corpus
        self.lock = threading.Lock()
        self.train_state = dict(running=False, done=0, total=0, log=[])

    def meta(self):
        return dict(
            umbrellas=[dict(id=u.id, name=u.name, category=u.category, body_areas=list(u.body_areas), adjacent=list(u.adjacent),
                            specialties=[dict(id=s.id, name=s.name, keywords=list(s.keywords[:8]), body_areas=list(s.body_areas), tier=s.tier,
                                              scope=s.scope, extended=s.extended, refer=s.refer) for s in u.specialties])
                       for u in self.tax.values()],
            defaults=dict(DEFAULTS, system_prompt=agents.DEFAULT_SYSTEM), llm_ready=bool(os.environ.get("ANTHROPIC_API_KEY")),
            corpus=self.corpus, stats=self.store.stats())

    def pipeline(self, req) -> ResearchPipeline:
        p = {**DEFAULTS, **req.get("params", {})}
        if p["llm"] == "anthropic":
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise ValueError("ANTHROPIC_API_KEY is not set on the server; use the mock LLM or export the key and restart.")
            llm = AnthropicLLM(model=p["model"])
        else:
            llm = MockLLM()
        roles, exp = req.get("roles", {}), req.get("expertise", {})
        srcs = [Inline(req.get("knowledge", []))] + ([LocalCorpus(self.corpus)] if self.corpus else []) + ([PubMed()] if p["pubmed"] else [])
        cfg = Config(threshold=float(p["threshold"]), max_attempts=int(p["max_attempts"]), top_internal=int(p["top_internal"]),
                     top_external=int(p["top_external"]), evidence_n=int(p["evidence_n"]), loop_depth=int(p["loop_depth"]),
                     max_questions=int(p["max_questions"]), min_quality=float(p["min_quality"]), umbrella=p["umbrella"],
                     system_prompt=roles.get("system_prompt") or agents.DEFAULT_SYSTEM, role_prompts=roles.get("role_prompts", {}),
                     demands=exp.get("demands", ""), required=exp.get("required", []), excluded=exp.get("excluded", []))
        return ResearchPipeline(llm, srcs, self.store, self.tax, cfg)

    def run(self, req):
        q = (req.get("question") or "").strip()
        if not q:
            raise ValueError("Enter a question / prompt first.")
        with self.lock:
            pl = self.pipeline(req)
            res = pl.loop(q) if req.get("mode") == "loop" else [pl.ask(q)]
        return dict(runs=[run_json(r) for r in res], stats=self.store.stats())

    def train(self, req):
        qs = [x.strip() for x in req.get("questions", []) if x.strip()]
        if self.train_state["running"] or not qs:
            raise ValueError("Training already running." if qs else "Add at least one training question.")
        pl = self.pipeline(req)
        st = self.train_state
        st.update(running=True, done=0, total=len(qs), log=[])

        def work():
            try:
                for q in qs:
                    with self.lock:
                        for r in (pl.loop(q) if req.get("mode") == "loop" else [pl.ask(q)]):
                            st["log"].append(dict(q=r.question, score=r.score.composite, status=r.status, attempts=r.attempts))
                    st["done"] += 1
            finally:
                st["running"] = False
        threading.Thread(target=work, daemon=True).start()
        return st

    def export(self):
        import tempfile
        d = tempfile.mkdtemp()
        n1, n2 = self.store.export_sft(d + "/s.jsonl"), self.store.export_preferences(d + "/p.jsonl")
        return dict(sft=open(d + "/s.jsonl").read(), prefs=open(d + "/p.jsonl").read(), n_sft=n1, n_prefs=n2)


def serve(app: App, host="127.0.0.1", port=8765):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            b = body if isinstance(body, bytes) else (body if isinstance(body, str) else json.dumps(body)).encode()
            self.send_response(code); self.send_header("content-type", ctype + "; charset=utf-8"); self.send_header("content-length", str(len(b))); self.end_headers(); self.wfile.write(b)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, TEMPLATE.replace("__DATA__", "[]"), "text/html")
            if self.path == "/api/meta":
                return self._send(200, app.meta())
            if self.path == "/api/train":
                return self._send(200, app.train_state)
            self._send(404, {"error": "not found"})

        def do_POST(self):
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("content-length", 0)) or 2) or b"{}")
                fn = {"/api/run": lambda: app.run(req), "/api/train": lambda: app.train(req), "/api/export": app.export}.get(self.path)
                if not fn:
                    return self._send(404, {"error": "not found"})
                self._send(200, fn())
            except ValueError as e:
                self._send(400, {"error": str(e)})
            except Exception as e:  # surface pipeline/LLM errors to the UI instead of dropping the connection
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer((host, port), H)
    print(f"Research pipeline UI -> http://{host}:{port}")
    srv.serve_forever()
