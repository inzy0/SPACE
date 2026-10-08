"""Language-pyramid understanding of a health question, the clearance-1 confirmation message and reach planning.

Pyramid (base -> apex):  1 Lexical (words, terms, abbreviations)  2 Syntactic (structure, question form, negation)
3 Semantic (conditions, interventions, body areas, population)  4 Pragmatic (intent, perspective, urgency, chance)
5 Contextual (when/acuity, where/place, situation, who, scale).

Clearance levels:  0 = raw question   1 = understood; confirmation + clarifying questions sent, awaiting reply
                   2 = confirmed (or waived) -> research may run.
Everything here is deterministic (offline). An LLM may refine slots through `refine_with_llm`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .terms import ABBREV_AMBIGUOUS, find_concepts, body_areas

WH = ["what", "when", "how", "why", "where", "who", "which"]
INTENTS = {  # intent -> trigger phrases / words
    "diagnosis": ["diagnos", "what is wrong", "what causes", "cause of", "causes", "could it be", "symptoms of", "signs of", "why do i", "why does", "differential", "test for", "tested"],
    "treatment": ["treat", "treatment", "manage", "management", "therapy", "cure", "best option", "what should", "how to", "how can", "options for", "help with", "operation", "surgery for", "remedy"],
    "prevention": ["prevent", "avoid", "risk reduction", "reduce the risk", "screening", "protect", "lower my risk", "stop it happening"],
    "prognosis": ["chance", "chances", "risk of", "likelihood", "probability", "odds", "survive", "survival", "outcome", "recover", "recovery", "prognosis", "recurrence", "relapse", "come back", "life expectancy", "success rate"],
    "safety": ["safe", "safety", "side effect", "interaction", "dose", "dosing", "can i take", "contraindicat", "harm", "toxic", "adverse", "mix with", "together with"],
    "rehabilitation": ["rehab", "recover function", "regain", "exercise", "physio", "get back to", "return to", "strength", "mobility"],
    "explain": ["what is", "what are", "explain", "mechanism", "how does", "how do", "meaning of", "understand"],
    "compare": [" vs ", "versus", "compare", "better than", "difference between", "or should", "which is better"],
    "evidence": ["evidence", "studies", "trial", "research", "guideline", "meta-analysis", "systematic review", "literature", "policy", "population", "cost-effective"],
}
INTENT_LABEL = {"diagnosis": "work out what is going on (diagnosis / cause)", "treatment": "choose or understand treatment / management",
                "prevention": "prevent it or reduce risk", "prognosis": "understand the chance / likely outcome (risk, prognosis)",
                "safety": "check safety (drugs, doses, interactions, harms)", "rehabilitation": "recover function (rehabilitation)",
                "explain": "understand what it is and how it works", "compare": "compare options", "evidence": "see what the evidence / guidelines say"}
CRITICAL = {  # slots that matter most per intent
    "diagnosis": ["acuity", "how", "who", "where"], "treatment": ["acuity", "who", "place", "how"], "prevention": ["who", "situation", "chance"],
    "prognosis": ["chance", "acuity", "who"], "safety": ["who", "how", "situation"], "rehabilitation": ["acuity", "who", "place", "how"],
    "explain": [], "compare": ["who"], "evidence": [],
}
ACUTE = ["sudden", "suddenly", "just now", "right now", "today", "this morning", "tonight", "hours", "minutes", "immediate", "worst ever", "severe", "emergency", "urgent", "acute", "currently having", "is having", "collapsed", "started"]
SUBACUTE = ["few days", "this week", "last week", "couple of weeks", "few weeks", "recent", "recently", "subacute"]
CHRONIC = ["chronic", "long-term", "long term", "ongoing", "persistent", "recurring", "recurrent", "lifelong", "for years", "for months", "since childhood", "for ages", "always had", "long standing", "longstanding"]
FLARE = ["flare", "flare-up", "exacerbation", "worsening", "relapse", "getting worse", "acute on chronic", "deteriorat"]
SETTINGS = {"home": ["at home", "home care", "home"], "primary care": ["gp", "family doctor", "primary care", "general practice", "pharmacy", "chemist"],
            "clinic / outpatient": ["clinic", "outpatient", "out-patient", "appointment", "specialist clinic"], "emergency department": ["emergency department", "a&e", "er ", "the er", "emergency room", "ambulance", "paramedic"],
            "ward / inpatient": ["ward", "inpatient", "in hospital", "admitted", "hospital"], "intensive care": ["icu", "intensive care", "ventilator", "critical care"],
            "care home": ["care home", "nursing home", "residential care"], "school / workplace": ["school", "workplace", "at work", "office", "factory"],
            "community / telehealth": ["community", "telehealth", "video call", "online consultation", "remote"], "dental practice": ["dentist", "dental practice"], "sports setting": ["sports field", "training ground", "gym", "pitch"]}
SITUATIONS = {"after surgery": ["after surgery", "post-op", "postoperative", "after my operation", "after the operation", "following surgery"], "after injury / trauma": ["after a fall", "after an accident", "after injury", "after trauma", "road traffic", "car crash", "hit my head", "fell"],
              "during pregnancy / postpartum": ["pregnan", "postpartum", "after birth", "breastfeeding", "postnatal"], "on medication": ["on warfarin", "taking", "on medication", "prescribed", "started on", "currently on"],
              "cancer treatment": ["chemo", "radiotherapy", "during cancer treatment", "undergoing treatment"], "travel": ["travel", "abroad", "holiday", "flight"], "work-related": ["at work", "work-related", "occupational", "workplace"],
              "second opinion / decision": ["second opinion", "deciding", "decision", "should i have", "whether to"], "caring for someone": ["my mother", "my father", "my mum", "my dad", "my son", "my daughter", "my child", "my wife", "my husband", "my patient", "caring for"],
              "after infection / illness": ["after covid", "after infection", "after the flu", "following illness"], "sport / exercise": ["during training", "playing", "marathon", "match"],
              "end of life": ["end of life", "dying", "hospice", "terminal"]}
CHANCE_W = ["chance", "chances", "risk", "likelihood", "probability", "odds", "prognosis", "survival", "recurrence", "relapse", "success rate", "complication", "come back"]
RED_FLAG_NOW = ["right now", "currently", "is having", "am having", "just happened", "can't breathe", "cannot breathe", "not breathing", "unconscious", "unresponsive", "crushing", "worst headache", "face drooping", "slurred", "want to die", "kill myself", "end my life", "took too many", "overdose", "bleeding heavily"]
NUM = r"(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|few|couple of)"
UNIT = {"minute": 1 / 1440, "hour": 1 / 24, "day": 1, "week": 7, "month": 30, "year": 365}
WORDNUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "few": 3, "couple of": 2}
GROUPS = {"acuity": "WHEN", "how": "HOW", "why": "WHY", "where": "WHERE", "place": "PLACE", "situation": "SITUATION", "who": "WHO", "chance": "CHANCE", "scale": "SCALE", "ambiguity": "CLARIFY"}
OPTIONS = {"acuity": ["acute (hours–days)", "subacute (weeks)", "chronic (months–years)", "flare of a long-term problem"],
           "why": [INTENT_LABEL[k] for k in ("diagnosis", "treatment", "prevention", "prognosis", "safety", "rehabilitation", "explain", "compare", "evidence")],
           "place": ["home", "primary care / GP", "clinic / outpatient", "emergency department", "ward / inpatient", "intensive care", "care home", "community / telehealth"],
           "scale": ["one person", "a group / service", "whole population / policy"], "how": ["mild", "moderate", "severe", "unsure"]}
QUESTION = {
    "acuity": "WHEN did it start, and how has it progressed — sudden/acute (hours–days), subacute (weeks), chronic (months–years), or a flare of a long-term problem?",
    "how": "HOW does it present or how bad is it — symptoms, severity, how it affects daily life — or how is it being managed now?",
    "why": "WHY are you asking — what decision or goal is this for (diagnosis, treatment choice, prevention, risk/prognosis, safety, rehabilitation, understanding, evidence review)?",
    "where": "WHERE in the body is it (and, if guidelines matter, which country/region)?",
    "place": "WHERE is care happening — PLACE of care (home, GP/primary care, clinic, emergency department, ward, ICU, care home, community)?",
    "situation": "What is the SITUATION or context (e.g. after surgery or injury, during pregnancy, on certain medicines, at work, travelling, second opinion, caring for someone)?",
    "who": "WHO is this about — age group, sex, pregnancy, main conditions and current medicines?",
    "chance": "What CHANCE / risk do you mean — likelihood of which outcome (recovery, recurrence, complication, survival), over what time frame?",
    "scale": "SCALE: is this about one person, a group/service, or the whole population?",
}


@dataclass
class Slot:
    key: str
    label: str
    value: str = ""
    conf: float = 0.0
    evidence: list = field(default_factory=list)
    critical: bool = False

    @property
    def known(self):
        return bool(self.value) and self.conf >= 0.5


@dataclass
class Understanding:
    text: str
    levels: dict
    slots: dict
    concepts: list
    intent: str
    perspective: str
    urgency: str
    emergent: bool
    clearance: int
    questions: list
    message: str
    restatement: str
    ambiguities: list
    umbrella_hits: dict

    def frame(self) -> str:
        """Compact confirmed-understanding block injected into every expert prompt."""
        s = self.slots
        rows = [("INTENT", INTENT_LABEL.get(self.intent, "")), ("PERSPECTIVE", self.perspective), ("POPULATION", s["who"].value), ("TIME COURSE", s["acuity"].value),
                ("BODY AREA / REGION", s["where"].value), ("CARE SETTING", s["place"].value), ("SITUATION", s["situation"].value), ("CHANCE / RISK FRAMING", s["chance"].value),
                ("HOW (severity / approach)", s["how"].value), ("SCALE", s["scale"].value), ("URGENCY", self.urgency if self.urgency != "routine" else "")]
        return "\n".join(f"{k}: {v}" for k, v in rows if v)

    def frame_terms(self) -> list[str]:
        out = []
        for k in ("acuity", "place", "who"):
            v = self.slots[k].value
            if v and self.slots[k].conf >= 0.5:
                out += [w for w in re.findall(r"[a-z]{5,}", v.lower())][:3]
        return list(dict.fromkeys(out))

    def to_dict(self) -> dict:
        return dict(text=self.text, levels=self.levels, slots={k: dict(label=v.label, value=v.value, conf=round(v.conf, 2), evidence=v.evidence, critical=v.critical) for k, v in self.slots.items()},
                    concepts=[{k: c[k] for k in ("canonical", "type", "surface", "negated", "emergent")} for c in self.concepts], intent=self.intent, perspective=self.perspective,
                    urgency=self.urgency, emergent=self.emergent, clearance=self.clearance, questions=self.questions, message=self.message, restatement=self.restatement,
                    ambiguities=self.ambiguities, ready=not any(q["critical"] for q in self.questions), frame=self.frame())


def _has(text, words):
    return [w for w in words if w in text]


REL = [("this morning", 0.2), ("tonight", 0.2), ("last night", 1), ("yesterday", 1), ("this week", 3), ("last week", 7), ("last month", 30), ("last year", 365), ("this year", 120)]


def _duration_days(text):
    best = None
    for ph, d in REL:
        if ph in text:
            best = max(best or 0, d)
    for m in re.finditer(NUM + r"\s*(minute|hour|day|week|month|year)s?(?:\s+ago)?", text):
        if re.search(r"(within|next|over the|in the|for the next|by|after)\s*(the\s*)?(next\s*)?$", text[max(0, m.start() - 16):m.start()]):
            continue  # a future horizon ("within 5 years"), not an onset
        n = m.group(1)
        v = float(n) if n.isdigit() else WORDNUM.get(n, 1)
        best = max(best or 0, v * UNIT[m.group(2)])
    m = re.search(r"since (?:the )?(?:age of )?(\d{4})", text)
    if m and 1900 < int(m.group(1)) <= 2100:
        best = max(best or 0, (2026 - int(m.group(1))) * 365)
    return best


def _acuity(low):
    flare, cues = _has(low, FLARE), []
    dur = _duration_days(low)
    chronic, sub, acute = _has(low, CHRONIC), _has(low, SUBACUTE), _has(low, ACUTE)
    if dur is not None:
        lab = "acute" if dur <= 2 else "subacute" if dur <= 84 else "chronic"
        cues.append(f"duration ≈ {dur:g} days")
    else:
        lab = "chronic" if chronic else "subacute" if sub and not acute else "acute" if acute else ""
    if flare and (lab == "chronic" or chronic):
        lab = "flare of a long-term problem"
    elif flare and not lab:
        lab = "flare of a long-term problem"
    cues += flare + (chronic if lab.startswith(("chronic", "flare")) else sub if lab == "subacute" else acute if lab == "acute" else [])
    conf = 0.9 if dur is not None else 0.75 if cues else 0.0
    return lab, conf, cues[:4]


def _population(low, concepts):
    parts, ev = [], []
    for pat, lab in ((r"\b(infant|baby|newborn|neonat\w*)\b", "infant / newborn"), (r"\b(toddler|child|children|kid|kids|paediatric|pediatric)\b", "child"), (r"\b(teen\w*|adolescent\w*)\b", "adolescent"),
                     (r"\b(elderly|older adult\w*|frail|geriatric|care home|nursing home|over 65)\b", "older adult")):
        m = re.search(pat, low)
        if m:
            parts.append(lab); ev.append(m.group(0))
    m = (re.search(r"\b(\d{1,3})\s*[- ]?(?:year[- ]old|yo|y/o|yrs?|years? old)\b", low) or re.search(r"\baged? (\d{1,3})\b", low)
         or re.search(r"\b(?:is|was|am|turned|turning) (\d{2,3})\b(?!\s*(?:mg|mcg|%|kg|mmhg|ml|units|bpm))", low))
    if m:
        a = int(m.group(1)); parts.append(f"age {a}"); ev.append(m.group(0))
        if a >= 65 and "older adult" not in parts: parts.append("older adult")
        elif a < 18 and not any(p in parts for p in ("child", "infant / newborn", "adolescent")): parts.append("child / adolescent")
    for pat, lab in ((r"\b(pregnan\w*|antenatal|postpartum|breastfeeding|expecting)\b", "pregnant / postpartum"), (r"\b(female|woman|women|girl|she|her|mum|mother|wife|daughter)\b", "female"),
                     (r"\b(male|man|men|boy|he|his|dad|father|husband|son)\b", "male")):
        m = re.search(pat, low)
        if m and not (lab in ("female", "male") and any("pregnant" in p for p in parts)):
            parts.append(lab); ev.append(m.group(0))
    meds = [c["surface"] for c in concepts if c["type"] == "drug" and not c["negated"]]
    if meds:
        parts.append("on " + ", ".join(meds[:3])); ev += meds[:3]
    comorb = [c["canonical"] for c in concepts if c["type"] == "condition" and not c["negated"]][1:3]
    if comorb:
        parts.append("also: " + ", ".join(comorb))
    return "; ".join(dict.fromkeys(parts)), (0.8 if parts else 0.0), ev[:4]


def _perspective(low):
    if re.search(r"\b(my patient|patients with|our patients|in my practice|we see|clinic|caseload|on the ward)\b", low):
        return "clinician"
    if re.search(r"\b(my (mum|mother|dad|father|son|daughter|wife|husband|child|baby|partner|friend|grandmother|grandfather))\b", low):
        return "caregiver"
    if re.search(r"\b(i|my|me|i'm|ive|i've)\b", low):
        return "patient"
    if re.search(r"\b(evidence|studies|trial|review|meta-analysis|guideline|literature|research)\b", low):
        return "researcher / student"
    return "unspecified"


def _intent(low):
    sc = {k: sum(low.count(p) for p in v) for k, v in INTENTS.items()}
    sc = {k: v for k, v in sc.items() if v}
    if not sc:
        return "", [], sc
    order = sorted(sc, key=lambda k: -sc[k])
    return order[0], order[1:3], sc


def understand(question: str, taxonomy=None, answers: dict | None = None) -> Understanding:
    answers = {k: v for k, v in (answers or {}).items() if str(v).strip()}
    full = question.strip() + (". " + ". ".join(str(v) for v in answers.values()) if answers else "")
    low = " " + full.lower() + " "
    concepts = find_concepts(full)
    SYMPTOM = re.search(r"\b(pain|ache|aching|sore|swelling|swollen|stiff|stiffness|numb|numbness|tingling|weakness|rash|itch\w*|lump|bleeding|cough|fever|nausea|burning)\b", full.lower())
    areas0 = body_areas(full)
    if SYMPTOM and areas0 and not any(c["type"] in ("condition", "symptom") and not c["negated"] for c in concepts):
        # generic fallback: "knee pain", "sore throat" ... a body area plus a symptom word is still a symptom, routed by the area's umbrella keywords
        concepts.append(dict(canonical=f"{areas0[0]} {SYMPTOM.group(1)}", type="symptom", surface=f"{areas0[0]} {SYMPTOM.group(1)}", start=0,
                             negated=False, routes=[], emergent=False))
    live = [c for c in concepts if not c["negated"]]

    # ---- 1 lexical
    toks = re.findall(r"[A-Za-z][A-Za-z'\-]*", full)
    amb = [dict(abbr=a.upper(), options=o) for a, o in ABBREV_AMBIGUOUS.items() if re.search(rf"(?<![a-z]){a}(?![a-z])", low) and a not in {c["surface"] for c in concepts}]
    lex = dict(name="Lexical", summary=f"{len(toks)} words · {len(live)} medical terms recognised" + (f" · {len(amb)} ambiguous abbreviation(s)" if amb else ""),
               items=[f"“{c['surface']}” → {c['canonical']} ({c['type']})" + (" [negated]" if c["negated"] else "") for c in concepts] + [f"{a['abbr']}? could be {' / '.join(a['options'])}" for a in amb])
    # ---- 2 syntactic
    whs = [w for w in WH if re.search(rf"\b{w}\b", low)]
    clauses = [c for c in re.split(r"[,;]| and | but | or | while | because | although ", full) if len(c.split()) > 2]
    qtype = "open (wh-)" if whs else "polar (yes/no)" if re.search(r"\b(is|are|can|could|should|does|do|will|would|may)\b", low[:40]) else "statement / request"
    negs = re.findall(r"\b(no|not|without|never|denies|none)\b", low)
    modal = sorted(set(re.findall(r"\b(can|could|should|must|may|might|would|will)\b", low)))
    syn = dict(name="Syntactic", summary=f"{qtype} question · {len(clauses) or 1} clause(s)" + (f" · asks {', '.join(whs).upper()}" if whs else ""),
               items=[f"question form: {qtype}", f"wh-words: {', '.join(whs) or 'none'}", f"clauses: {len(clauses) or 1}", f"negations: {len(negs)}" + (f" ({', '.join(sorted(set(negs)))})" if negs else ""),
                      f"modality: {', '.join(modal) or 'none'}", "conditional (“if …”) present" if re.search(r"\bif\b", low) else "no conditional"])
    # ---- 3 semantic
    by = {t: [c["canonical"] for c in live if c["type"] == t] for t in ("condition", "symptom", "intervention", "drug", "test", "population", "outcome")}
    areas = body_areas(full)
    if taxonomy:
        areas = list(dict.fromkeys(areas + [a for u in taxonomy.values() for a in u.body_areas if a.lower() in low and a != "whole body"]))
    umb_hits: dict = {}
    for c in live:
        for r in c["routes"]:
            u = r.split(".")[0]
            umb_hits[u] = umb_hits.get(u, 0) + 1
    sem = dict(name="Semantic", summary="; ".join(f"{k}: {', '.join(v)}" for k, v in by.items() if v) or "no recognised medical concept",
               items=[f"{k}: {', '.join(v)}" for k, v in by.items() if v] + ([f"body areas: {', '.join(areas)}"] if areas else []) + ([f"likely specialties: {', '.join(sorted(umb_hits, key=lambda k: -umb_hits[k])[:5])}"] if umb_hits else []))
    # ---- 4 pragmatic
    intent, second, isc = _intent(low)
    persp = _perspective(low)
    now = _has(low, RED_FLAG_NOW)
    dur0 = _duration_days(low)
    flagged = any(c["emergent"] for c in live)
    past = dur0 is not None and dur0 > 2  # "a week ago" is not happening now
    emergent = bool(flagged and not past and (now or (persp in ("patient", "caregiver") and (_has(low, ACUTE) or dur0 is None and re.search(r"\b(is|am|has|have|having|am having)\b", low)))))
    urgency = "possible emergency" if emergent else "urgent-sounding" if _has(low, ["urgent", "asap", "worried", "scared"]) else "routine"
    chance_words = _has(low, CHANCE_W)
    prag = dict(name="Pragmatic", summary=f"intent: {INTENT_LABEL.get(intent, 'unclear')} · speaker: {persp} · urgency: {urgency}",
                items=[f"primary intent: {INTENT_LABEL.get(intent, 'unclear')}"] + ([f"also: {', '.join(INTENT_LABEL[s] for s in second)}"] if second else []) + [f"perspective: {persp}", f"urgency: {urgency}"] + ([f"risk/chance wording: {', '.join(chance_words)}"] if chance_words else []))

    # ---- 5 contextual -> slots
    ac, acc, acue = _acuity(low)
    pop, popc, popev = _population(low, live)
    place, placec, placev, best = "", 0.0, [], 0
    for lab, ws in SETTINGS.items():  # the most specific (longest) phrase wins, so "care home" beats "home"
        for w in ws:
            if len(w.strip()) > best and re.search(r"(?<![a-z])" + re.escape(w.strip()) + r"(?![a-z])", low):
                place, placec, placev, best = lab, 0.8, [w.strip()], len(w.strip())
    sit = [lab for lab, ws in SITUATIONS.items() if any(w in low for w in ws)]
    geo = re.findall(r"\bin (?:the )?([A-Z][a-z]{2,}(?: [A-Z][a-z]+)?)", full)
    geo = [g for g in geo if g.lower() not in {c["surface"] for c in concepts} and g.lower() not in {"the", "a"}]
    where_v = "; ".join(filter(None, [", ".join(areas), ("region: " + geo[0]) if geo else ""]))
    scale_v = "population / policy" if re.search(r"\b(population|policy|service|nhs|health system|public health|screening programme|national|cost-effective|guideline)\b", low) else "a group / service" if re.search(r"\b(patients|our unit|our ward|our service|cohort|people with)\b", low) else "one person" if persp in ("patient", "caregiver") else ""
    how_v = ("severe" if re.search(r"\b(severe|worst|unbearable|excruciating|disabling)\b", low) else "mild" if re.search(r"\b(mild|slight|minor)\b", low) else "moderate" if re.search(r"\b(moderate)\b", low) else "")
    how_ev = [w for w in ("severe", "worst", "mild", "slight", "minor", "moderate") if w in low][:2]
    chance_v = ""
    if chance_words or intent == "prognosis":
        tf = re.findall(r"(?:within|over|in the next|after|for)\s+(?:the next\s+)?\d+\s*(?:day|week|month|year)s?", low)
        outw = list(dict.fromkeys(re.findall(r"\b(recovery|recurrence|relapse|survival|complications?|success|side effects?|death|disability|remission|cure|healing|stroke|fracture)\b", low)))[:2]
        cond = [c["canonical"] for c in live if c["type"] in ("condition", "symptom")][:1]
        out = outw or cond
        chance_v = ("chance of " + ", ".join(out) + (f" in {', '.join(cond)}" if outw and cond and cond[0] not in outw else "") + (f" ({tf[0].strip()})" if tf else "")) if (tf or out) else ""
    slots = {
        "acuity": Slot("acuity", "When · acute → chronic", ac, acc, acue),
        "how": Slot("how", "How · presentation / severity", how_v, 0.7 if how_v else 0.0, how_ev),
        "why": Slot("why", "Why · goal", INTENT_LABEL.get(intent, ""), 0.8 if intent and isc.get(intent, 0) >= 1 and len(isc) == 1 else 0.55 if intent else 0.0, [intent] if intent else []),
        "where": Slot("where", "Where · body area / region", where_v, 0.8 if areas else 0.0, areas[:3]),
        "place": Slot("place", "Place · care setting", place, placec, placev),
        "situation": Slot("situation", "Situation · scenario", "; ".join(sit), 0.7 if sit else 0.0, sit[:3]),
        "who": Slot("who", "Who · population", pop, popc, popev),
        "chance": Slot("chance", "Chance · risk / likelihood", chance_v, 0.8 if chance_v else 0.0, chance_words[:3]),
        "scale": Slot("scale", "Scale · individual → population", scale_v, 0.6 if scale_v else 0.0, []),
    }
    for k, a in answers.items():  # the user's replies override inference with full confidence
        if k in slots:
            slots[k].value, slots[k].conf = str(a), 1.0
    crit = CRITICAL.get(intent, []) if intent else (CRITICAL["treatment"] if any(c["type"] in ("condition", "symptom") for c in live) else [])  # no stated intent: ask like for a management question
    for k in crit:
        slots[k].critical = True
    ctx = dict(name="Contextual", summary=" · ".join(f"{k}: {s.value}" for k, s in slots.items() if s.known and k in ("acuity", "who", "place", "situation", "where")) or "no context stated yet",
               items=[f"{s.label}: {s.value or '— not stated —'}" for s in slots.values()])

    # ---- clarifying questions (WHEN / HOW / WHY / WHERE / PLACE / SITUATION / WHO / CHANCE / SCALE)
    qs = []
    for a in amb:
        qs.append(dict(key="ambiguity:" + a["abbr"].lower(), slot="ambiguity", group="CLARIFY", critical=True, text=f"By “{a['abbr']}” do you mean: {' / '.join(a['options'])}?", options=a["options"]))
    if len({c["canonical"] for c in live}) == 0 and not amb:
        qs.append(dict(key="what", slot="what", group="WHAT", critical=True, text="WHAT condition, symptom, treatment or topic is this about? I did not recognise a specific medical term.", options=[]))
    for k in ("acuity", "who", "why", "how", "where", "place", "situation", "chance", "scale"):
        s = slots[k]
        wanted = s.critical or k in ("why",) or (k == "chance" and intent == "prognosis")
        optional = k in ("how", "where", "place", "situation", "scale") and not s.critical
        if not s.known and (wanted or (optional and live)):
            qs.append(dict(key=k, slot=k, group=GROUPS[k], critical=s.critical, text=QUESTION[k], options=OPTIONS.get(k, [])))
    qs.sort(key=lambda q: (not q["critical"], list(GROUPS).index(q["slot"]) if q["slot"] in GROUPS else 99))
    qs = qs[:3] if emergent else qs[:7]  # in a possible emergency keep the reply short

    # ---- restatement + confirmation message (clearance 1)
    topic = ", ".join(dict.fromkeys(c["canonical"] for c in live)) or "(topic not recognised)"
    bits = [f"you want to {INTENT_LABEL[intent]}" if intent else "you have a health question"]
    bits.append(f"about **{topic}**")
    if slots["who"].known: bits.append(f"for {slots['who'].value}")
    if slots["acuity"].known: bits.append(f"— time course: {slots['acuity'].value}")
    if slots["place"].known: bits.append(f"— in {slots['place'].value}")
    if slots["situation"].known: bits.append(f"— situation: {slots['situation'].value}")
    restatement = " ".join(bits) + "."
    lines = []
    if emergent:
        lines.append("⚠ **This may be an emergency.** If this is happening now (e.g. chest pain, stroke signs, trouble breathing, overdose, thoughts of suicide), call your local emergency number or go to the nearest emergency department now. The research below is not a substitute for urgent care.\n")
    lines.append("**Clearance 1 — here is how I read your message**")
    lines.append(f"- *Words*: {lex['summary']}\n- *Structure*: {syn['summary']}\n- *Meaning*: {sem['summary']}\n- *Intent*: {prag['summary']}\n- *Context*: {ctx['summary']}")
    lines.append(f"\nSo I understand that {restatement}")
    if qs:
        lines.append("\nTo answer the right question I'd like to confirm / learn (skip what you don't know):\n" + "\n".join(f"{i}. [{q['group']}] {q['text']}" for i, q in enumerate(qs, 1)))
    else:
        lines.append("\nI have enough context to proceed.")
    lines.append("\nReply **confirm** to start the research, or correct me / answer the questions above. Research results are decision support, not medical advice.")
    ready = not any(q["critical"] for q in qs)
    return Understanding(full, {1: lex, 2: syn, 3: sem, 4: prag, 5: ctx}, slots, concepts, intent, persp, urgency, emergent, 1, qs, "\n".join(lines), restatement, amb, umb_hits)


# ------------------------------------------------------------------ reach: extend the panel to the question's demands
def plan_reach(u: Understanding, taxonomy, max_extra=3, excluded=()):
    """Return extra required role ids + panel/evidence growth, with the reason for each (scale: individual -> wider)."""
    s, low, extra, why = u.slots, " " + u.text.lower() + " ", [], []
    types = {c["type"] for c in u.concepts if not c["negated"]}

    def add(rid, reason):
        if rid in {r for r, _ in extra} or rid in excluded or not any(sp.id == rid for um in taxonomy.values() for sp in um.specialties):
            return
        extra.append((rid, reason))
    pop = s["who"].value.lower()
    if "infant" in pop or "newborn" in pop: add("paeds.neonatal", "newborn / infant population")
    elif "child" in pop or "adolescent" in pop: add("paeds.physician", "child / adolescent population")
    if "older adult" in pop or re.search(r"age (6[5-9]|[7-9]\d|1\d\d)\b", pop): add("geri.physician", "older-adult population")
    if "pregnan" in pop: add("obgyn.physician", "pregnancy / postpartum")
    if "drug" in types or u.intent == "safety" or "polypharmacy" in {c["canonical"] for c in u.concepts}: add("pharm.clinical", "medicines / safety question")
    if u.emergent or s["acuity"].value == "acute" and s["place"].value in ("emergency department", "intensive care"): add("emerg.em", "acute / possible emergency")
    if s["acuity"].value.startswith(("chronic", "flare")) and u.perspective != "researcher / student": add("nursing.cns", "long-term condition management")
    if u.intent == "diagnosis" or "test" in types: add("dx.rad", "diagnostic work-up")
    if u.intent == "rehabilitation": add("rehab.physiatrist", "rehabilitation goal")
    if u.intent in ("prognosis", "compare", "evidence"): add("pubh.stats", "risk / comparison needs evidence appraisal")
    if "population" in s["scale"].value: add("pubh.epi", "population-level scale")
    if re.search(r"palliative|end of life|dying|hospice", low): add("pall.phys", "end-of-life context")
    if re.search(r"depress|anxi|low mood|panic", low) and u.umbrella_hits.get("psych", 0) == 0: add("psych.psychologist", "psychological burden alongside the main problem")
    families = {k for k, v in u.umbrella_hits.items() if v}
    multi = len(families) >= 3
    kept = extra[:max_extra]
    return dict(extra_required=[r for r, _ in kept], reasons=[f"{r}: {w}" for r, w in kept], top_external_delta=1 if multi else 0, evidence_delta=2 if multi or len(kept) >= 2 else 0,
                scale=s["scale"].value or "individual / general", multimorbidity=multi, dropped=[r for r, _ in extra[max_extra:]])


def refine_with_llm(llm, question: str, u: Understanding) -> dict:
    """Optional: ask a real LLM to fill/correct slots as JSON. Returns {slot: value}; the offline mock returns {}."""
    prompt = (f"TASK: understand\nQUESTION: {question}\nCurrent slots: " + json.dumps({k: s.value for k, s in u.slots.items()}) +
              "\nReturn ONLY JSON mapping any of [acuity, how, why, where, place, situation, who, chance, scale] to your best reading of the question; omit unknowns.")
    try:
        raw = llm.complete("You are a careful medical-language analyst. Do not invent facts not implied by the text.", prompt)
        m = re.search(r"\{.*\}", raw, re.S)
        d = json.loads(m.group(0)) if m else {}
        return {k: str(v) for k, v in d.items() if k in u.slots and v}
    except Exception:
        return {}
