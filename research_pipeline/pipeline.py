from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field

from . import agents
from .router import route as do_route
from .scoring import STYLE_DIMS, Score, score as do_score
from .library import Library, umb
from .sources import gather
from .store import Store
from .taxonomy import load_taxonomy
from .text import jaccard
from .trace import Tracer
from .understand import plan_reach, refine_with_llm, understand


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
    require_clearance: bool = False  # True: stop after the pyramid + confirmation message until the user confirms (clearance 2)
    auto_reach: bool = True  # extend the panel (paediatrics, geriatrics, pharmacy, emergency, public health ...) to the question's demands
    max_reach_extra: int = 3
    use_library: bool = True  # per-specialty + per-umbrella evidence score library
    llm_understanding: bool = False  # let the LLM refine the pyramid slots (needs a real LLM)


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
    understanding: dict | None = None
    reach: dict | None = None
    awaiting: bool = False


class _LibSource:
    """Adapter: the evidence library as a Source, ranked for the panel's scopes (specialties + umbrella common library)."""

    def __init__(self, lib, scopes):
        self.lib, self.scopes = lib, scopes

    def search(self, query, n=8):
        return self.lib.search(query, self.scopes, n)


class ResearchPipeline:
    def __init__(self, llm, sources, store: Store | None = None, taxonomy=None, cfg: Config | None = None, judge=None, library=None):
        self.llm, self.sources = llm, sources
        self.store = store or Store()
        self.tax = taxonomy or load_taxonomy()
        self.cfg = cfg or Config()
        self.judge = judge  # optional callable(question, answer, Score) -> Score, e.g. LLM-as-judge blend
        self.library = library or Library(self.store.db, self.tax)

    # ---- one question -------------------------------------------------------------------------------------------
    def understand(self, question, answers=None):
        """Language-pyramid pass (clearance 0 -> 1). Pure and cheap, so the UI can call it before any research."""
        u = understand(question, self.tax, answers)
        if self.cfg.llm_understanding:
            extra = refine_with_llm(self.llm, question, u)
            if extra:
                u = understand(question, self.tax, {**extra, **(answers or {})})
        return u

    def ask(self, question, parent=None, depth=0, context=None) -> RunResult:
        """context = {"answers": {slot: text}, "confirmed": bool, "waive": bool} from the clearance-1 exchange."""
        c, tr, ctx = self.cfg, Tracer(), context or {}
        answers = ctx.get("answers", {})
        tr.emit("in:question", "Question", 0, outputs={"question": question, **({"your_answers": answers} if answers else {})})
        u = self.understand(question, answers)
        prev = "in:question"
        for n, key in ((1, "u:lexical"), (2, "u:syntactic"), (3, "u:semantic"), (4, "u:pragmatic"), (5, "u:context")):
            lv = u.levels[n]
            tr.emit(key, f"{n} · {lv['name']}", 1, [prev], {"reads": "the question" if n == 1 else u.levels[n - 1]["name"] + " layer"}, {"summary": lv["summary"], **{f"· {i + 1}": it for i, it in enumerate(lv["items"][:6])}})
            prev = key
        confirmed = bool(ctx.get("confirmed") or ctx.get("waive"))
        gate = c.require_clearance and not confirmed
        tr.emit("clearance", "Clearance 1: confirm understanding" if gate else "Clearance 2: confirmed", 2, [prev],
                {"critical_gaps": [q["group"] for q in u.questions if q["critical"]], "confirmed": confirmed, "urgency": u.urgency},
                {"restated": u.restatement, "questions": [f"[{q['group']}] {q['text']}" for q in u.questions], "clearance_level": 1 if gate else 2}, status="warn" if gate else "ok",
                note="WAITING FOR YOU" if gate else "PROCEED")
        if gate:
            u.clearance = 1
            return RunResult(question, "", Score({}, 0.0, False, [], ""), "awaiting_confirmation", 0, [], tr.events, "", depth, 0, parent, u.to_dict(), None, True)
        u.clearance = 2
        # ---- route (multi-term concepts) + reach extension
        pre = do_route(question, self.tax, umbrella=c.umbrella, concepts=u.concepts)
        weights = self.store.expert_weights(pre.umbrella.id)
        examples = self.store.best_examples(pre.umbrella.id)
        reach = plan_reach(u, self.tax, c.max_reach_extra, c.excluded) if c.auto_reach else dict(extra_required=[], reasons=[], top_external_delta=0, evidence_delta=0, scale="", multimorbidity=False, dropped=[])
        tr.emit("reach", "Reach: extend to the question's demands", 2, ["clearance"], {"scale": reach["scale"], "auto_reach": c.auto_reach},
                {"extra_experts": reach["reasons"] or ["none needed"], "top_external +": reach["top_external_delta"], "evidence +": reach["evidence_delta"], "multimorbidity": reach["multimorbidity"]})
        required = list(dict.fromkeys(list(c.required) + reach["extra_required"]))
        r = do_route(question, self.tax, weights, c.top_internal, c.top_external + reach["top_external_delta"], required, c.excluded, c.umbrella, concepts=u.concepts)
        tr.emit("in:memory", "Learned memory", 0, inputs={"umbrella": pre.umbrella.id},
                outputs={"expert_weights": {k: round(v, 2) for k, v in weights.items()}, "few_shot_examples": len(examples), "library_items": self.library.size() if c.use_library else 0})
        tr.emit("route", "Route (multi-term)", 4, ["reach", "in:memory"], {"terms": [f"{x['surface']}→{x['canonical']}" for x in u.concepts if not x["negated"]][:8], "required": required},
                {"umbrella": r.umbrella.name, "body_areas": r.body_areas, "internal": [s.name for s in r.internal], "external": [s.name for s in r.external],
                 "umbrella_scores": {k: round(v, 1) for k, v in r.scores.items() if v}})
        experts = [s.id for s in r.internal + r.external]
        scopes = experts + [umb(r.umbrella.id)]
        frame, cterms = u.frame(), u.frame_terms()
        concept_q = " ".join(dict.fromkeys(x["canonical"] for x in u.concepts if not x["negated"]))

        attempts, best, rows, feedback, query, n_ev, chair_only = 0, None, [], None, (question + " " + concept_q + " " + " ".join(answers.values())).strip(), c.evidence_n + reach["evidence_delta"], False
        evidence, ops, last_score = [], [], ""
        while attempts < c.max_attempts:
            attempts += 1
            sfx = "" if attempts == 1 else f"#{attempts}"
            if not chair_only:
                srcs = list(self.sources) + ([_LibSource(self.library, scopes)] if c.use_library else [])
                evidence = gather(srcs, query, n_ev, c.min_quality)
                if c.use_library:  # ingest into every panelist's library, then attach per-specialty scores for this panel
                    for e in evidence:
                        e.lib_id = self.library.ingest(e, scopes)
                        e.lib = {s: self.library.score(e.lib_id, s)["overall"] for s in scopes}
                tr.emit("gather" + sfx, "Gather evidence" + (f" (attempt {attempts})" if sfx else ""), 3, ["route"] if not sfx else ["in:question"],
                        {"query": query[:140], "n": n_ev, "library scopes": len(scopes) if c.use_library else 0},
                        {"evidence": [dict(id=e.id, title=e.title, q=e.quality, design=e.design, lib=e.lib.get(umb(r.umbrella.id)), source=e.source) for e in evidence]},
                        status="ok" if evidence else "warn", note="" if evidence else "no evidence found")
                ops = self._panel(tr, question, r, evidence, sfx, frame)
            ans, cprompt = agents.chair(self.llm, question, evidence, ops, examples, feedback, c.system_prompt, c.demands, frame, umb(r.umbrella.id))
            up = [last_score] if chair_only else ["lead" + sfx, *[f"ext:{s.id}{sfx}" for s in r.external]]
            tr.emit("chair" + sfx, "Chair synthesis", 9, up,
                    {"positions": len(ops), "revision_feedback": feedback or "", "prompt": cprompt[:1800]}, {"answer": ans})
            sc = do_score(ans, question, evidence, ops, c.threshold, cterms)
            if self.judge:
                sc = self.judge(question, ans, sc)
            last_score = "score" + sfx
            tr.emit("score" + sfx, "Score & filter", 10, ["chair" + sfx], {"threshold": c.threshold}, {"dims": sc.dims, "composite": sc.composite, "failing": sc.failing},
                    status="ok" if sc.passed else "fail", note="PASS" if sc.passed else "REGENERATE", score=sc.composite)
            rows.append((attempts, ans, sc.composite, str(sc.dims), ",".join(sc.failing), sc.feedback, str([e.to_dict() for e in evidence])))
            if best is None or sc.composite > best[1].composite:
                best = (ans, sc, ops, list(evidence))
            if sc.passed:
                break
            feedback = sc.feedback
            chair_only = set(sc.failing) <= STYLE_DIMS and bool(evidence)
            if not chair_only:  # needs new material: broaden retrieval with the panel's own keywords
                query = query + " " + " ".join(k for s in r.internal[:3] for k in s.keywords[:3])
                n_ev += 4
        ans, sc, bops, bev = best
        status = "accepted" if sc.passed else "needs_review"
        if c.use_library:  # teach the library which evidence each specialty actually relied on
            by_e = {e.id: e.lib_id for e in bev}
            cited = {}
            for o in bops:
                if o.stage != "debate":
                    cited.setdefault(o.expert_id, set()).update(by_e[i] for i in o.evidence_ids if i in by_e)
            allc = set().union(*cited.values()) if cited else set()
            cited[umb(r.umbrella.id)] = allc
            self.library.feedback(cited, status == "accepted", sc.composite)
        fups = agents.followups(self.llm, question, ans, system=c.system_prompt)
        fups += [f"What additional {s.name} evidence addresses: {question}" for s in r.internal if s.id.split('.')[-1] in sc.failing][:1]
        rid = self.store.log_run(question, r.umbrella.id, parent, depth, ans, sc.composite, status, attempts, experts, rows)
        tr.emit("store", "Learn: store, weights, library", 11, [last_score],
                {"status": status, "composite": sc.composite}, {"run_id": rid, "attempts_logged": attempts, "library_updated": c.use_library, "export": "sft + preference pairs"}, status="ok")
        tr.emit("followups", "Follow-up questions", 11, [last_score], {"open_issues": sc.failing}, {"queue": fups})
        return RunResult(question, ans, sc, status, attempts, fups, tr.events, r.umbrella.id, depth, rid, parent, u.to_dict(), reach)

    def _panel(self, tr, question, r, evidence, sfx, frame=""):
        def logger(stage, lane):
            def f(spec, mine, op, prompt):
                tr.emit(f"{stage}:{spec.id}{sfx}", spec.name, lane, [("gather" + sfx) if stage == "spec" else f"spec:{spec.id}{sfx}"] if stage != "ext" else ["lead" + sfx],
                        {"evidence": [e.id for e in mine], "body_areas": list(spec.body_areas), "prompt": prompt[:1500]},
                        {"position": op.text, "confidence": op.confidence, "cites": op.evidence_ids})
            return f
        kw = dict(system=self.cfg.system_prompt, role_prompts=self.cfg.role_prompts, demands=self.cfg.demands, frame=frame)
        first = agents.ask_experts(self.llm, question, r.internal, evidence, "specialist", trace=logger("spec", 5), **kw)
        deb = agents.ask_experts(self.llm, question, r.internal, evidence, "debate", prior=first, trace=logger("deb", 6), **kw)
        internal = first + deb
        conf = [o.confidence for o in first]
        tr.emit("lead" + sfx, f"{r.umbrella.name} lead: internal consensus", 7, [f"deb:{s.id}{sfx}" for s in r.internal],
                {"positions": [o.role for o in first]},
                {"mean_confidence": round(sum(conf) / len(conf), 2) if conf else 0, "cited": sorted({e for o in first for e in o.evidence_ids})})
        ext = agents.ask_experts(self.llm, question, r.external, evidence, "external", prior=internal, trace=logger("ext", 8), **kw)
        return first + ext + deb

    # ---- loop: discussion -> follow-up questions -> discussion ---------------------------------------------------
    def loop(self, seed, on_result=None, context=None) -> list[RunResult]:
        c, queue, seen, out = self.cfg, deque([(seed, 0, None)]), [], []
        while queue and len(out) < c.max_questions:
            q, d, parent = queue.popleft()
            if any(jaccard(q, s) >= c.dedupe for s in seen):
                continue
            seen.append(q)
            res = self.ask(q, parent, d, context if d == 0 else dict(context or {}, confirmed=True))  # follow-ups inherit the confirmed context
            out.append(res)
            if res.awaiting:
                break
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
