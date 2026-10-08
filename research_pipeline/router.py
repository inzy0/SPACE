"""Question -> primary umbrella, internal mini-experts, external experts from adjacent umbrellas."""
from __future__ import annotations

from dataclasses import dataclass

from .taxonomy import FACETS, Specialty, Umbrella
from .text import tokens


@dataclass
class Route:
    umbrella: Umbrella
    internal: list[Specialty]
    external: list[Specialty]
    scores: dict[str, float]
    body_areas: list[str]


def _hits(q: str, qt: set[str], kws) -> int:
    return sum(1 for k in kws if (k in q if " " in k else k in qt or any(t.startswith(k) for t in qt)))


def route(question, taxonomy, weights=None, top_internal=4, top_external=2, required=(), excluded=(), umbrella=None) -> Route:
    """weights: expert_id -> learned multiplier from past scored runs (Store.expert_weight)."""
    q, qt = question.lower(), set(tokens(question))
    weights = weights or {}
    scores = {u.id: _hits(q, qt, u.keywords) * 2 + _hits(q, qt, [a for a in u.body_areas]) for u in taxonomy.values()}
    primary = taxonomy[umbrella] if umbrella in taxonomy else taxonomy[max(scores, key=lambda k: (scores[k], k == "neuro"))]
    allspec = {s.id: s for u in taxonomy.values() for s in u.specialties}
    excluded, required = set(excluded), [i for i in required if i in allspec and i not in excluded]

    def spec_score(s: Specialty) -> float:
        base = 0.3 if s.umbrella == primary.id else 0.0  # members of the primary umbrella always qualify, ranked by facet match
        areas = 0.5 * _hits(q, qt, s.body_areas) if s.id.split('.')[-1] not in FACETS else 0  # facets share the umbrella's areas
        return (_hits(q, qt, s.keywords) + areas + base) * weights.get(s.id, 1.0)

    ranked = sorted((s for s in primary.specialties if s.id not in excluded), key=lambda s: (-spec_score(s), s.id))
    req_int = [allspec[i] for i in required if allspec[i].umbrella == primary.id]
    internal = req_int + [s for s in ranked if s not in req_int][:max(0, top_internal - len(req_int))]
    req_ext = [allspec[i] for i in required if allspec[i].umbrella != primary.id]
    ext_pool = []
    for uid, sc in scores.items():
        if uid != primary.id and (sc > 0 or uid in primary.adjacent):
            for s in taxonomy[uid].specialties:
                if s.id not in excluded and s not in req_ext:
                  ext_pool.append((spec_score(s) + (sc + (1 if uid in primary.adjacent else 0)) * 0.1, s))
    ext_pool.sort(key=lambda x: (-x[0], x[1].id))
    external = req_ext + [s for sc, s in ext_pool if sc > 0.1][:max(0, top_external - len(req_ext))]
    areas = sorted({a for s in internal for a in s.body_areas if a.lower() in q or _hits(q, qt, [a])}) or list(primary.body_areas[:3])
    return Route(primary, internal, external, scores, areas)
