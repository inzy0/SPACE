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
        out = app.run(dict(question=Q, knowledge=[dict(title=t, text=x, quality=q) for t, x, q in DOCS], params=dict(top_internal=3),
                           expertise=dict(required=["neuro.dermato"])))
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


if __name__ == "__main__":
    unittest.main()
