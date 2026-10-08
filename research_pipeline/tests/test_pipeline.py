import json
import tempfile
import unittest
from pathlib import Path

from research_pipeline import Config, ResearchPipeline, Store
from research_pipeline.llm import MockLLM
from research_pipeline.router import route
from research_pipeline.server import App
from research_pipeline.sources import Inline
from research_pipeline.taxonomy import load_taxonomy
from research_pipeline.terms import CONCEPTS, find_concepts
from research_pipeline.understand import plan_reach, understand

DOCS = [("Surgery for brain injury", "Decompressive craniectomy lowers intracranial pressure after severe traumatic brain injury.", 0.9),
        ("Stroke rehabilitation", "Early physiotherapy and exercise improve gait and mobility after stroke and brain injury.", 0.8),
        ("Metal neurotoxicity", "Lead and mercury exposure cause cognitive impairment; toxicity depends on dose.", 0.6)]
Q = "surgical rehabilitation and toxicity after severe traumatic brain injury"


def pipe(**cfg):
    return ResearchPipeline(MockLLM(), [Inline([dict(title=t, text=x, quality=q) for t, x, q in DOCS])], Store(), cfg=Config(**cfg))


class Tests(unittest.TestCase):
    def test_routing_neuro_roles(self):
        r = route(Q, load_taxonomy())
        self.assertEqual(r.umbrella.id, "neuro")
        names = " ".join(s.name for s in r.internal)
        for w in ("Neurosurgeon", "physiotherapist", "Neurotoxicologist"):
            self.assertIn(w, names)

    def test_required_excluded_and_override(self):
        r = route(Q, load_taxonomy(), required=["cardio.ep"], excluded=["neuro.toxicologist"])
        self.assertIn("cardio.ep", [s.id for s in r.external])
        self.assertNotIn("neuro.toxicologist", [s.id for s in r.internal])
        self.assertEqual(route(Q, load_taxonomy(), umbrella="cardio").umbrella.id, "cardio")

    def test_regeneration_then_accept(self):
        res = pipe().ask(Q)
        self.assertEqual(res.status, "accepted")
        self.assertEqual(res.attempts, 2)  # mock's first draft lacks limitations/disclaimer -> gate fails -> regenerate
        self.assertTrue(any(e["node"].startswith("score") and e["status"] == "fail" for e in res.trace))
        self.assertTrue(all("inputs" in e and "outputs" in e for e in res.trace))

    def test_no_evidence_needs_review(self):
        res = ResearchPipeline(MockLLM(), [], Store(), cfg=Config(max_attempts=2)).ask(Q)
        self.assertEqual(res.status, "needs_review")

    def test_demands_and_role_prompts_reach_prompt(self):
        res = pipe(demands="RCT only", role_prompts={"neuro.surgeon": "Focus on timing"}).ask(Q)
        p = next(e for e in res.trace if e["node"].startswith("spec:neuro.surgeon"))["inputs"]["prompt"]
        self.assertIn("RCT only", p)
        self.assertIn("Focus on timing", p)

    def test_min_quality_filters_evidence(self):
        res = pipe(min_quality=0.85).ask(Q)
        ev = next(e for e in res.trace if e["node"] == "gather")["outputs"]["evidence"]
        self.assertTrue(ev and all(x["q"] >= 0.85 for x in ev))

    def test_learning_loop_and_exports(self):
        p = pipe(loop_depth=1, max_questions=3)
        out = p.loop(Q)
        self.assertGreaterEqual(len(out), 2)
        st = p.store.stats()
        self.assertEqual(st["runs"], len(out))
        self.assertTrue(st["experts"])
        self.assertTrue(p.store.expert_weights("neuro"))
        d = tempfile.mkdtemp()
        self.assertGreaterEqual(p.store.export_sft(d + "/s.jsonl"), 1)
        self.assertGreaterEqual(p.store.export_preferences(d + "/p.jsonl"), 1)
        self.assertIn("feedback", json.loads(Path(d, "p.jsonl").read_text().splitlines()[0]))

    def test_background_training(self):
        p = pipe()
        p.train([Q, "stroke rehabilitation exercise"], background=True).join(10)
        self.assertEqual(p.store.stats()["runs"], 2)

    def test_server_api(self):
        app = App()
        req = dict(question=Q, knowledge=[dict(title=t, text=x, quality=q) for t, x, q in DOCS], params=dict(top_internal=3),
                   expertise=dict(required=["neuro.dermato"]))
        first = app.run(req)  # clearance 1: the confirmation comes back, no research yet
        self.assertTrue(first["awaiting"])
        self.assertEqual(first["runs"], [])
        out = app.run(dict(req, context=dict(confirmed=True)))
        r = out["runs"][0]
        self.assertEqual(r["status"], "accepted")
        self.assertIn("Neuro-dermatology liaison", next(e for e in r["trace"] if e["node"] == "route")["outputs"]["internal"])
        with self.assertRaises(ValueError):
            app.run(dict(question=" "))
        self.assertTrue(app.meta()["umbrellas"])


    def test_catalog_covers_all_healthcare_with_scope(self):
        tax = load_taxonomy()
        self.assertGreaterEqual(len(tax), 30)
        cats = {u.category for u in tax.values()}
        for c in ("Medical specialties", "Surgical & perioperative", "Nursing & midwifery", "Pharmacy & medicines", "Dental & oral health",
                  "Allied health & rehabilitation", "Public health & evidence", "Palliative care & ethics", "Emergency & critical care"):
            self.assertIn(c, cats)
        ids = [s.id for u in tax.values() for s in u.specialties]
        self.assertEqual(len(ids), len(set(ids)))
        for u in tax.values():
            self.assertTrue(all(a in tax for a in u.adjacent), u.id)
            for s in u.specialties:
                self.assertTrue(s.scope and s.extended and s.refer, s.id)

    def test_routing_across_professions(self):
        tax = load_taxonomy()
        cases = {"Management of periodontitis and when is a root canal indicated": ("dental", "Periodontist"),
                 "Can nurse practitioners safely prescribe for chronic hypertension": ("nursing", "Nurse practitioner"),
                 "Should a pharmacist adjust warfarin dose when starting an antibiotic": ("pharm", "Clinical pharmacist"),
                 "Gestational diabetes screening and labour management": ("obgyn", "Obstetrician"),
                 "Palliative sedation and consent for a patient lacking capacity": ("pall", "Palliative-medicine physician"),
                 "Return to play after ACL reconstruction in an athlete": ("sportsmed", "Sports physiotherapist"),
                 "How is sepsis recognised in the emergency department": ("emerg", "Emergency physician")}
        for q, (umb, role) in cases.items():
            r = route(q, tax)
            self.assertEqual(r.umbrella.id, umb, q)
            self.assertTrue(any(role in s.name for s in r.internal), (q, [s.name for s in r.internal]))

    def test_scope_of_practice_in_prompt(self):
        res = pipe().ask(Q)
        p = next(e for e in res.trace if e["node"].startswith("spec:neuro.surgeon"))["inputs"]["prompt"]
        for w in ("SCOPE OF PRACTICE", "EXTENDED SCOPE", "REFER ON"):
            self.assertIn(w, p)


    # ---------------------------------------------------------------- language pyramid, clearance, reach, library
    def test_pyramid_levels_and_lay_terms(self):
        u = understand("My mum is 72 and had a brain attack last week, she is on warfarin. What are her chances of recovery and what rehab in a care home?", load_taxonomy())
        self.assertEqual(sorted(u.levels), [1, 2, 3, 4, 5])
        self.assertIn("stroke", [c["canonical"] for c in u.concepts])  # lay term -> clinical concept
        self.assertEqual(u.perspective, "caregiver")
        self.assertEqual(u.intent, "prognosis")
        self.assertEqual(u.slots["acuity"].value, "subacute")  # "last week"
        self.assertEqual(u.slots["place"].value, "care home")  # longest setting phrase wins over "home"
        self.assertIn("older adult", u.slots["who"].value)
        self.assertIn("recovery", u.slots["chance"].value)
        self.assertFalse(u.emergent)  # a week ago is not happening now

    def test_emergency_banner_only_for_current_events(self):
        tax = load_taxonomy()
        self.assertTrue(understand("I am having crushing chest pain right now", tax).emergent)
        self.assertIn("emergency", understand("I am having crushing chest pain right now", tax).message.lower())
        self.assertFalse(understand("What does the evidence say about chest pain pathways?", tax).emergent)

    def test_confirmation_asks_when_how_why_where_place_situation(self):
        u = understand("back pain", load_taxonomy())
        groups = {q["group"] for q in u.questions}
        self.assertTrue({"WHEN", "WHY", "PLACE"} <= groups, groups)
        self.assertIn("Clearance 1", u.message)
        u2 = understand("Is MS safe in pregnancy?", load_taxonomy())
        self.assertTrue(any(q["group"] == "CLARIFY" for q in u2.questions))  # ambiguous abbreviation

    def test_answers_fill_slots_and_remove_questions(self):
        tax = load_taxonomy()
        u = understand("knee pain", tax, {"acuity": "chronic, 2 years", "why": "treatment options"})
        self.assertEqual(u.slots["acuity"].conf, 1.0)
        self.assertNotIn("acuity", [q["slot"] for q in u.questions])

    def test_future_horizon_is_not_an_onset(self):
        u = understand("What is the risk of stroke after a TIA within 5 years?", load_taxonomy())
        self.assertNotEqual(u.slots["acuity"].value, "chronic")
        self.assertIn("5 years", u.slots["chance"].value)

    def test_clearance_gate_blocks_until_confirmed(self):
        p = pipe(require_clearance=True)
        r = p.ask(Q)
        self.assertTrue(r.awaiting)
        self.assertEqual(r.status, "awaiting_confirmation")
        self.assertEqual(p.store.stats()["runs"], 0)  # nothing researched or logged
        self.assertEqual(next(e for e in r.trace if e["node"] == "clearance")["outputs"]["clearance_level"], 1)
        r2 = p.ask(Q, context=dict(confirmed=True))
        self.assertEqual(r2.status, "accepted")
        self.assertEqual(next(e for e in r2.trace if e["node"] == "clearance")["outputs"]["clearance_level"], 2)
        self.assertEqual(p.store.stats()["runs"], 1)
        self.assertEqual(len([e for e in r2.trace if e["node"].startswith("u:")]), 5)

    def test_reach_extends_panel_to_question_demands(self):
        txt = "My mum is 72, on warfarin, had a stroke last week. What are her chances of recovery and which rehab should she get in a care home?"
        res = pipe().ask(txt)
        ext = " ".join(next(e for e in res.trace if e["node"] == "route")["outputs"]["external"])
        for w in ("Geriatrician", "pharmacist"):
            self.assertIn(w, ext)
        u = understand(txt, load_taxonomy())
        self.assertLessEqual(len(plan_reach(u, load_taxonomy(), max_extra=2)["extra_required"]), 2)
        off = pipe(auto_reach=False).ask(txt)
        self.assertNotIn("Geriatrician", " ".join(next(e for e in off.trace if e["node"] == "route")["outputs"]["external"]))

    def test_confirmed_context_reaches_prompts_and_scoring(self):
        res = pipe().ask(Q, context=dict(answers={"acuity": "chronic", "place": "home"}, confirmed=True))
        p = next(e for e in res.trace if e["node"].startswith("spec:neuro.surgeon"))["inputs"]["prompt"]
        self.assertIn("CONFIRMED CONTEXT", p)
        self.assertIn("TIME COURSE: chronic", p)
        self.assertIn("EXAMINATION & WORKUP", p)
        self.assertIn("context_fit", res.score.dims)

    def test_multi_term_routing(self):
        tax = load_taxonomy()
        ids = set(tax) | {s.id for u in tax.values() for s in u.specialties}
        self.assertGreaterEqual(len(CONCEPTS), 100)
        self.assertEqual([(c.canonical, r) for c in CONCEPTS for r in c.routes if r not in ids], [])  # every route is a real umbrella / role
        self.assertEqual(route("my kid has a bad rash and itchy skin", tax, concepts=find_concepts("my kid has a bad rash and itchy skin")).umbrella.id, "derm")
        self.assertEqual(route("brain attack at home", tax, concepts=find_concepts("brain attack at home")).umbrella.id, "neuro")
        self.assertTrue([c for c in find_concepts("no chest pain but dizzy") if c["canonical"] == "chest pain" and c["negated"]])

    def test_evidence_library_per_specialty_and_umbrella(self):
        p = pipe()
        for _ in range(3):
            p.ask(Q)
        lib = p.library
        self.assertGreaterEqual(lib.size(), 3)
        spec_scope = next(e for e in p.ask(Q).trace if e["node"] == "route")["outputs"]["internal"]  # sanity: panel exists
        self.assertTrue(spec_scope)
        rep = lib.report("neuro.surgeon")
        self.assertTrue(rep["items"])
        self.assertTrue(all(0 <= i["overall"] <= 1 for i in rep["items"]))
        ucommon = lib.report("umb:neuro")
        self.assertTrue(ucommon["items"] and "by_specialty" in ucommon["items"][0])
        top = ucommon["items"][0]
        self.assertGreater(top["cited"], 0)  # usage was learned
        self.assertIn({"scope": "umb:neuro", "name": "Neuro", "items": ucommon["count"]}, [o for o in lib.overview() if o["scope"] == "umb:neuro"])

    def test_library_feedback_moves_scores_and_ranks_search(self):
        import sqlite3
        from research_pipeline.library import Library
        from research_pipeline.sources import Evidence
        lib = Library(sqlite3.connect(":memory:"), load_taxonomy())
        good = lib.ingest(Evidence("E1", "Meta-analysis of craniectomy", "Craniectomy lowers pressure in brain injury.", "pubmed:1", 0.9, "", 2024), ["neuro.surgeon", "umb:neuro"])
        weak = lib.ingest(Evidence("E2", "Case report of craniectomy", "A craniectomy in one brain injury patient.", "user", 0.4, "", 2001), ["neuro.surgeon", "umb:neuro"])
        self.assertGreater(lib.score(good, "neuro.surgeon")["overall"], lib.score(weak, "neuro.surgeon")["overall"])
        self.assertEqual(lib._item(good)[6], "meta-analysis")  # study design inferred from text
        before = lib.score(good, "neuro.surgeon")["overall"]
        lib.feedback({"neuro.surgeon": {good}}, True, 0.97)
        self.assertGreaterEqual(lib.score(good, "neuro.surgeon")["overall"], before)  # accepted + cited does not lower it
        lib.feedback({"neuro.surgeon": {weak}}, False, 0.3)
        lib.feedback({"neuro.surgeon": {weak}}, False, 0.3)
        self.assertLess(lib.score(weak, "neuro.surgeon")["overall"], lib.score(weak, "neuro.surgeon")["static"])
        self.assertEqual(lib.search("craniectomy brain injury", ["neuro.surgeon", "umb:neuro"])[0].title, "Meta-analysis of craniectomy")

    def test_loop_inherits_confirmed_context(self):
        p = pipe(loop_depth=1, max_questions=2, require_clearance=True)
        self.assertTrue(p.loop(Q)[0].awaiting)  # nothing runs until the first question is confirmed
        out = p.loop(Q, context=dict(confirmed=True))
        self.assertGreaterEqual(len(out), 2)
        self.assertTrue(all(not r.awaiting for r in out))

    def test_server_understand_and_library_endpoints(self):
        app = App()
        d = app.understand(dict(question="chronic knee pain in a 60 year old after running"))
        self.assertEqual(d["understanding"]["clearance"], 1)
        self.assertTrue(d["understanding"]["questions"])
        app.run(dict(question=Q, knowledge=[dict(title=t, text=x, quality=q) for t, x, q in DOCS], context=dict(confirmed=True)))
        self.assertTrue(app.library_view("overview")["overview"])
        self.assertTrue(app.library_view("umb:neuro")["items"])
        with self.assertRaises(ValueError):
            app.library_view("umb:nonexistent")
        self.assertTrue(all("workup" in s and "terms" in s for u in app.meta()["umbrellas"] for s in u["specialties"]))


if __name__ == "__main__":
    unittest.main()
