from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field

from . import agents
from .router import route as do_route
from .scoring import STYLE_DIMS, Score, score as do_score
from .sources import gather
from .store import Store
from .taxonomy import load_taxonomy
from .text import jaccard
from .trace import Tracer


@dataclass
class Config:
    threshold: float = 0.72
    max_attempts: int = 3
    top_internal: int = 4
    top_external: int = 2
    evidence_n: int = 8
    loop_depth: int = 2
    max_questions: int = 8
    dedupe: float = 0.6
    system_prompt: str = agents.DEFAULT_SYSTEM  # global instructions for every role
    demands: str = ""  # expertise demands injected into every prompt, e.g. "RCT-level evidence only; address children"
    role_prompts: dict = field(default_factory=dict)  # specialty id -> extra instruction for that role
    required: list = field(default_factory=list)  # specialty ids that must sit on the panel
    excluded: list = field(default_factory=list)
    umbrella: str = "auto"
    min_quality: float = 0.0  # evidence quality floor


@dataclass
class RunResult:
    question: str
    answer: str
    score: Score
    status: str  # accepted | needs_review
    attempts: int
    followups: list[str]
    trace: list[dict]
    umbrella: str
    depth: int = 0
    run_id: int = 0
    parent: int | None = None


class ResearchPipeline:
    def __init__(self, llm, sources, store: Store | None = None, taxonomy=None, cfg: Config | None = None, judge=None):
        self.llm, self.sources = llm, sources
        self.store = store or Store()
        self.tax = taxonomy or load_taxonomy()
        self.cfg = cfg or Config()
        self.judge = judge  # optional callable(question, answer, Score) -> Score, e.g. LLM-as-judge blend

    # ---- one question -------------------------------------------------------------------------------------------
    def ask(self, question, parent=None, depth=0) -> RunResult:
        c, tr = self.cfg, Tracer()
        tr.emit("in:question", "Question", 0, outputs={"question": question})
        tr.emit("in:sources", "Sources", 0, outputs={"sources": [type(s).__name__ for s in self.sources]})
        pre = do_route(question, self.tax, umbrella=c.umbrella)
        weights = self.store.expert_weights(pre.umbrella.id)
        examples = self.store.best_examples(pre.umbrella.id)
        tr.emit("in:memory", "Learned memory", 0, inputs={"umbrella": pre.umbrella.id},
                outputs={"expert_weights": {k: round(v, 2) for k, v in weights.items()}, "few_shot_examples": len(examples)})
        r = do_route(question, self.tax, weights, c.top_internal, c.top_external, c.required, c.excluded, c.umbrella)
        tr.emit("route", "Route", 2, ["in:question", "in:memory"], {"question": question},
                {"umbrella": r.umbrella.name, "body_areas": r.body_areas, "internal": [s.name for s in r.internal],
                 "external": [s.name for s in r.external], "umbrella_scores": {k: v for k, v in r.scores.items() if v}})
        experts = [s.id for s in r.internal + r.external]

        attempts, best, rows, feedback, query, n_ev, chair_only = 0, None, [], None, question, c.evidence_n, False
        evidence, ops, last_score = [], [], ""
        while attempts < c.max_attempts:
            attempts += 1
            sfx = "" if attempts == 1 else f"#{attempts}"
            if not chair_only:
                evidence = gather(self.sources, query, n_ev, c.min_quality)
                tr.emit("gather" + sfx, "Gather evidence" + (f" (attempt {attempts})" if sfx else ""), 1, ["in:question", "in:sources"],
                        {"query": query, "n": n_ev}, {"evidence": [dict(id=e.id, title=e.title, q=e.quality, source=e.source) for e in evidence]},
                        status="ok" if evidence else "warn", note="" if evidence else "no evidence found")
                ops = self._panel(tr, question, r, evidence, sfx)
            ans, cprompt = agents.chair(self.llm, question, evidence, ops, examples, feedback, c.system_prompt, c.demands)
            up = [last_score] if chair_only else ["lead" + sfx, *[f"ext:{s.id}{sfx}" for s in r.external]]
            tr.emit("chair" + sfx, "Chair synthesis", 7, up,
                    {"positions": len(ops), "revision_feedback": feedback or "", "prompt": cprompt[:1800]}, {"answer": ans})
            sc = do_score(ans, question, evidence, ops, c.threshold)
            if self.judge:
                sc = self.judge(question, ans, sc)
            last_score = "score" + sfx
            tr.emit("score" + sfx, "Score & filter", 8, ["chair" + sfx], {"threshold": c.threshold}, {"dims": sc.dims, "composite": sc.composite, "failing": sc.failing},
                    status="ok" if sc.passed else "fail", note="PASS" if sc.passed else "REGENERATE", score=sc.composite)
            rows.append((attempts, ans, sc.composite, str(sc.dims), ",".join(sc.failing), sc.feedback, str([e.to_dict() for e in evidence])))
            if best is None or sc.composite > best[1].composite:
                best = (ans, sc)
            if sc.passed:
                break
            feedback = sc.feedback
            chair_only = set(sc.failing) <= STYLE_DIMS and bool(evidence)
            if not chair_only:  # needs new material: broaden retrieval with the panel's own keywords
                query = question + " " + " ".join(k for s in r.internal[:3] for k in s.keywords[:3])
                n_ev += 4
        ans, sc = best
        status = "accepted" if sc.passed else "needs_review"
        fups = agents.followups(self.llm, question, ans, system=c.system_prompt)
        fups += [f"What additional {s.name} evidence addresses: {question}" for s in r.internal if s.id.split('.')[-1] in sc.failing][:1]
        rid = self.store.log_run(question, r.umbrella.id, parent, depth, ans, sc.composite, status, attempts, experts, rows)
        tr.emit("store", "Learn: store & weights", 9, [last_score],
                {"status": status, "composite": sc.composite}, {"run_id": rid, "attempts_logged": attempts, "export": "sft + preference pairs"}, status="ok")
        tr.emit("followups", "Follow-up questions", 9, [last_score], {"open_issues": sc.failing}, {"queue": fups})
        return RunResult(question, ans, sc, status, attempts, fups, tr.events, r.umbrella.id, depth, rid, parent)

    def _panel(self, tr, question, r, evidence, sfx):
        def logger(stage, lane):
            def f(spec, mine, op, prompt):
                tr.emit(f"{stage}:{spec.id}{sfx}", spec.name, lane, [("gather" + sfx) if stage == "spec" else f"spec:{spec.id}{sfx}"] if stage != "ext" else ["lead" + sfx],
                        {"evidence": [e.id for e in mine], "body_areas": list(spec.body_areas), "prompt": prompt[:1500]},
                        {"position": op.text, "confidence": op.confidence, "cites": op.evidence_ids})
            return f
        kw = dict(system=self.cfg.system_prompt, role_prompts=self.cfg.role_prompts, demands=self.cfg.demands)
        first = agents.ask_experts(self.llm, question, r.internal, evidence, "specialist", trace=logger("spec", 3), **kw)
        deb = agents.ask_experts(self.llm, question, r.internal, evidence, "debate", prior=first, trace=logger("deb", 4), **kw)
        internal = first + deb
        conf = [o.confidence for o in first]
        tr.emit("lead" + sfx, f"{r.umbrella.name} lead: internal consensus", 5, [f"deb:{s.id}{sfx}" for s in r.internal],
                {"positions": [o.role for o in first]},
                {"mean_confidence": round(sum(conf) / len(conf), 2) if conf else 0, "cited": sorted({e for o in first for e in o.evidence_ids})})
        ext = agents.ask_experts(self.llm, question, r.external, evidence, "external", prior=internal, trace=logger("ext", 6), **kw)
        return first + ext + deb

    # ---- loop: discussion -> follow-up questions -> discussion ---------------------------------------------------
    def loop(self, seed, on_result=None) -> list[RunResult]:
        c, queue, seen, out = self.cfg, deque([(seed, 0, None)]), [], []
        while queue and len(out) < c.max_questions:
            q, d, parent = queue.popleft()
            if any(jaccard(q, s) >= c.dedupe for s in seen):
                continue
            seen.append(q)
            res = self.ask(q, parent, d)
            out.append(res)
            if on_result:
                on_result(res)
            if d < c.loop_depth:
                queue.extend((f, d + 1, res.run_id) for f in res.followups)
        return out

    # ---- background training --------------------------------------------------------------------------------------
    def train(self, questions, background=False, with_loop=False, on_result=None):
        """Replay a question bank: every run is scored and written to the store (expert weights, few-shot memory,
        SFT/preference exports). Re-queues previously failed ('hard') questions first."""
        def work():
            for q in list(dict.fromkeys(self.store.hard_questions() + list(questions))):
                for res in (self.loop(q, on_result) if with_loop else [self.ask(q)]):
                    if on_result and not with_loop:
                        on_result(res)
        if not background:
            return work()
        t = threading.Thread(target=work, daemon=True)
        t.start()
        return t
