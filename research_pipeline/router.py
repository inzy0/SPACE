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
    return sum(1 for k in kws if (k in q if " " in k else k in qt or (len(k) >= 5 and any(t.startswith(k) for t in qt))))


def route(question, taxonomy, weights=None, top_internal=4, top_external=2, required=(), excluded=(), umbrella=None, concepts=None) -> Route:
    """weights: expert_id -> learned multiplier from past scored runs (Store.expert_weight)."""
    concepts = [c for c in (concepts or []) if not c.get("negated")]
    # lay words / synonyms are expanded to their canonical terms so "brain attack" routes like "stroke"
    q = (question + " " + " ".join(c["canonical"] for c in concepts)).lower()
    qt = set(tokens(q))
    weights = weights or {}
    boost_u, boost_s = {}, {}
    for c in concepts:
        for rt in c["routes"]:
            if "." in rt:
                boost_s[rt] = boost_s.get(rt, 0) + 2
                boost_u[rt.split(".")[0]] = boost_u.get(rt.split(".")[0], 0) + 1.5
            else:
                boost_u[rt] = boost_u.get(rt, 0) + 3
                if rt in taxonomy and taxonomy[rt].specialties:  # an umbrella-level term always brings that umbrella's lead (generalist) role
                    g = taxonomy[rt].specialties[0].id
                    boost_s[g] = boost_s.get(g, 0) + 1.5
    scores = {u.id: _hits(q, qt, u.keywords) * 2 + _hits(q, qt, [a for a in u.body_areas]) + boost_u.get(u.id, 0) for u in taxonomy.values()}
    lead = {k: v * (0.7 if taxonomy[k].support else 1.0) for k, v in scores.items()}  # support umbrellas (pharmacy, nursing, labs...) rarely lead
    primary = taxonomy[umbrella] if umbrella in taxonomy else taxonomy[max(lead, key=lambda k: (lead[k], k == "neuro"))]
    allspec = {s.id: s for u in taxonomy.values() for s in u.specialties}
    order = {s.id: i for u in taxonomy.values() for i, s in enumerate(u.specialties)}  # umbrella order = generalist first
    BASE = {"dermato": 0.1, "toxicologist": 0.2}  # facets that join a panel by default only when the question points at them
    excluded, required = set(excluded), [i for i in required if i in allspec and i not in excluded]

    def spec_score(s: Specialty) -> float:
        base = BASE.get(s.id.split('.')[-1], 0.3) if s.umbrella == primary.id else 0.0  # members of the primary umbrella always qualify, ranked by facet match
        areas = 0.25 * _hits(q, qt, s.body_areas) if s.id.split('.')[-1] not in FACETS else 0  # facets share the umbrella's areas
        return (_hits(q, qt, s.keywords) + areas + base + boost_s.get(s.id, 0)) * weights.get(s.id, 1.0)

    ranked = sorted((s for s in primary.specialties if s.id not in excluded), key=lambda s: (-spec_score(s), order[s.id]))
    req_int = [allspec[i] for i in required if allspec[i].umbrella == primary.id]
    internal = req_int + [s for s in ranked if s not in req_int][:max(0, top_internal - len(req_int))]
    req_ext = [allspec[i] for i in required if allspec[i].umbrella != primary.id]
    ext_pool = []
    for uid, u in taxonomy.items():
        if uid == primary.id:
            continue
        adj = 1.0 if uid in primary.adjacent else 0.0
        if scores[uid] == 0 and not adj:
            continue
        cands = [(spec_score(s), s) for s in u.specialties if s.id not in excluded and s not in req_ext]
        if cands:  # one voice per umbrella keeps the external panel diverse
            hits, best = max(cands, key=lambda x: (x[0], -order[x[1].id]))
            ext_pool.append((scores[uid] + 0.5 * adj + hits * 0.5 if (scores[uid] or hits >= 2) else 0, best))
    ext_pool.sort(key=lambda x: (-x[0], order[x[1].id]))
    external = req_ext + [s for sc, s in ext_pool if sc > 0.5][:max(0, top_external - len(req_ext))]
    areas = sorted({a for s in internal for a in s.body_areas if a.lower() in q or _hits(q, qt, [a])}) or list(primary.body_areas[:3])
    return Route(primary, internal, external, scores, areas)
