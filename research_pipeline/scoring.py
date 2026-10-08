"""Heuristic quality scorer (research quality + safety). Swap/blend with an LLM judge via `judge=` in the pipeline."""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

from .text import tokens

WEIGHTS = dict(grounding=.24, evidence_quality=.14, coverage=.14, consensus=.10, calibration=.14, safety=.10, relevance=.07, context_fit=.07)
STYLE_DIMS = {"grounding", "calibration", "safety", "relevance", "context_fit"}  # fixable by rewriting alone; others need new retrieval/panel
ABSOLUTE = re.compile(r"\b(always|never|guaranteed?|cures?|proven to|100%|definitely)\b", re.I)
DISCLAIMER = re.compile(r"not (a substitute for )?(professional )?medical advice|consult (a|your) (qualified )?(clinician|doctor|physician)", re.I)
META = re.compile(r"^(limitations?|note|disclaimer|open questions?|summary)\b", re.I)
HINT = {"grounding": "Cite every factual claim with a valid [E#]; remove claims the evidence does not support.",
        "evidence_quality": "Retrieve stronger evidence (trials, reviews) and prefer it.",
        "coverage": "Attribute a position to every panel member by role; consult the missing specialists.",
        "consensus": "Experts are low-confidence or split; gather more evidence and state the disagreement explicitly.",
        "calibration": "Add a 'Limitations:' line and avoid absolute wording (always/never/cures).",
        "safety": "Add a disclaimer: not medical advice, consult a qualified clinician.",
        "relevance": "Answer the question asked; address its key terms directly.",
        "context_fit": "Tailor the answer to the confirmed context (time course, population, care setting) and say how it changes the advice."}


@dataclass
class Score:
    dims: dict
    composite: float
    passed: bool
    failing: list[str]
    feedback: str


def score(answer, question, evidence, opinions, threshold=0.72, context_terms=()) -> Score:
    by_id = {e.id: e for e in evidence}
    sents = [s for s in re.split(r"(?<=[.!?])\s+|\n", answer) if len(s.split()) >= 6 and not META.match(s.strip())]
    cites = re.findall(r"\[(E\d+)\]", answer)
    bad = [c for c in cites if c not in by_id]
    cited = [s for s in sents if re.search(r"\[E\d+\]", s)]
    grounding = max(0.0, (len(cited) / len(sents) if sents else 0.0) - (len(bad) / len(cites) if cites else 0.0))
    good = {c for c in cites if c in by_id}
    equality = statistics.mean(by_id[c].quality for c in good) if good else 0.0
    roles = {o.role for o in opinions if o.stage != "debate"}
    coverage = sum(1 for r in roles if r.lower() in answer.lower()) / len(roles) if roles else 0.0
    conf = [o.confidence for o in opinions if o.stage != "debate"] or [0.0]
    consensus = max(0.0, statistics.mean(conf) * 0.6 + (1 - (statistics.pstdev(conf) if len(conf) > 1 else 0)) * 0.4)
    calibration = (0.5 if re.search(r"limitation|uncertain|insufficient|may be", answer, re.I) else 0) + (0 if ABSOLUTE.search(answer) else 0.5)
    safety = 1.0 if DISCLAIMER.search(answer) else 0.0
    qt = set(tokens(question))
    relevance = len(qt & set(tokens(answer))) / len(qt) if qt else 0.0
    low = answer.lower()
    context_fit = (sum(1 for t in context_terms if t[:5] in low) / len(context_terms)) if context_terms else 1.0
    dims = dict(grounding=grounding, evidence_quality=equality, coverage=coverage, consensus=consensus,
                calibration=calibration, safety=safety, relevance=min(1.0, relevance * 1.5), context_fit=context_fit)
    comp = sum(WEIGHTS[k] * v for k, v in dims.items())
    failing = [k for k, v in dims.items() if v < 0.6]
    gate = bool(evidence) and grounding >= 0.6 and safety == 1.0 and not bad
    if not evidence:
        failing.append("evidence_quality")
    if bad:
        failing.append("grounding")
    fb = "\n".join(f"- {k} ({dims[k]:.2f}): {HINT[k]}" for k in dict.fromkeys(failing))
    return Score({k: round(v, 3) for k, v in dims.items()}, round(comp, 3), comp >= threshold and gate, list(dict.fromkeys(failing)), fb)
