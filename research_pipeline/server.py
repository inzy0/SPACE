"""Local web app: `python -m research_pipeline serve` -> control panel (prompt, roles, expertise demands, parameters,
knowledge) + live node-graph replay of the real pipeline run. Stdlib only."""
from __future__ import annotations

import json
import os
import threading
from urllib.parse import parse_qs, urlparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import agents
from .llm import AnthropicLLM, MockLLM
from .pipeline import Config, ResearchPipeline
from .sources import Inline, LocalCorpus, PubMed
from .library import Library
from .store import Store
from .taxonomy import load_taxonomy
from .terms import terms_for_role
from .viewer import TEMPLATE, run_json

DEFAULTS = dict(threshold=0.72, max_attempts=3, top_internal=4, top_external=2, evidence_n=8, loop_depth=2, max_questions=6,
                min_quality=0.0, llm="mock", model="claude-sonnet-5-5", pubmed=False, umbrella="auto",
                require_clearance=True, auto_reach=True, max_reach_extra=3, use_library=True, consult_all=False, llm_understanding=False)


class App:
    def __init__(self, db=":memory:", corpus=None, taxonomy=None):
        self.store, self.tax, self.corpus = Store(db), load_taxonomy(taxonomy), corpus
        self.library = Library(self.store.db, self.tax)
        self.lock = threading.Lock()
        self.train_state = dict(running=False, done=0, total=0, log=[])

    def meta(self):
        return dict(
            umbrellas=[dict(id=u.id, name=u.name, category=u.category, body_areas=list(u.body_areas), adjacent=list(u.adjacent),
                            specialties=[dict(id=s.id, name=s.name, keywords=list(s.keywords[:8]), body_areas=list(s.body_areas), tier=s.tier,
                                              scope=s.scope, extended=s.extended, refer=s.refer, workup=s.workup, terms=terms_for_role(s.id)) for s in u.specialties])
                       for u in self.tax.values()],
            defaults=dict(DEFAULTS, system_prompt=agents.DEFAULT_SYSTEM), llm_ready=bool(os.environ.get("ANTHROPIC_API_KEY")),
            corpus=self.corpus, stats=self.store.stats(), library=self.library.overview(), library_items=self.library.size())

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
                     demands=exp.get("demands", ""), required=exp.get("required", []), excluded=exp.get("excluded", []),
                     require_clearance=bool(p["require_clearance"]), auto_reach=bool(p["auto_reach"]), max_reach_extra=int(p["max_reach_extra"]),
                     use_library=bool(p["use_library"]), consult_all=bool(p["consult_all"]), llm_understanding=bool(p["llm_understanding"]) and p["llm"] != "mock")
        return ResearchPipeline(llm, srcs, self.store, self.tax, cfg, library=self.library)

    def run(self, req):
        q = (req.get("question") or "").strip()
        if not q:
            raise ValueError("Enter a question / prompt first.")
        ctx = req.get("context") or {}
        with self.lock:
            pl = self.pipeline(req)
            res = pl.loop(q, context=ctx) if req.get("mode") == "loop" else [pl.ask(q, context=ctx)]
        if res and res[0].awaiting:  # clearance 1: send the confirmation back instead of researching
            return dict(awaiting=True, understanding=res[0].understanding, trace=res[0].trace, runs=[], stats=self.store.stats())
        return dict(runs=[run_json(r) for r in res], stats=self.store.stats(), library=self.library.overview())

    def understand(self, req):
        q = (req.get("question") or "").strip()
        if not q:
            raise ValueError("Enter a question / prompt first.")
        pl = self.pipeline(req)
        return dict(understanding=pl.understand(q, (req.get("context") or {}).get("answers")).to_dict())

    def library_view(self, scope):
        if scope in ("", "overview"):
            return dict(overview=self.library.overview(), total=self.library.size())
        if scope.startswith("umb:") and scope[4:] not in self.tax and scope != "umb:":
            raise ValueError("Unknown umbrella.")
        return self.library.report(scope)

    def train(self, req):
        qs = [x.strip() for x in req.get("questions", []) if x.strip()]
        if self.train_state["running"] or not qs:
            raise ValueError("Training already running." if qs else "Add at least one training question.")
        req = dict(req, params={**req.get("params", {}), "require_clearance": False})  # training runs unattended
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
            if self.path.startswith("/api/library"):
                try:
                    return self._send(200, app.library_view(parse_qs(urlparse(self.path).query).get("scope", [""])[0]))
                except ValueError as e:
                    return self._send(400, {"error": str(e)})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("content-length", 0)) or 2) or b"{}")
                fn = {"/api/run": lambda: app.run(req), "/api/train": lambda: app.train(req), "/api/understand": lambda: app.understand(req), "/api/export": app.export}.get(self.path)
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
