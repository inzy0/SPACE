"""Healthcare roster: umbrella -> mini-expertise roles, each with tier, scope of practice, extended scope and referral limits.

Organ-system umbrellas are generated from shared *facets* (physician, surgeon, physiotherapist, toxicologist, diagnostics,
dermatology liaison); profession umbrellas (nursing, pharmacy, dental, allied health ...) are hand-listed. The definitions
live in `catalog.py`; extend them at runtime with a JSON file (see `load_taxonomy`).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

# tier -> (core scope, extended / advanced-practice scope, refer-on boundary)
TIERS = {
    "physician": ("Assess, diagnose and medically manage conditions in the specialty; order and interpret investigations; prescribe; lead the treatment plan and prognosis discussion.",
                  "Sub-specialty and advanced procedural practice, supervision and delegation to advanced practitioners, guideline and clinical-trial leadership.",
                  "Refer for surgery/procedures, other organ systems, rehabilitation, palliative care, or ethical/legal decisions."),
    "surgeon": ("Pre-operative assessment, operative decision-making, performing procedures, post-operative care and complication recognition.",
                "Advanced minimally-invasive, robotic or reconstructive techniques; surgical-trial design; supervising trainees and surgical assistants.",
                "Refer non-operative management to the treating physician, rehabilitation to allied health, peri-operative optimisation to anaesthesia."),
    "advanced": ("Holistic assessment, diagnosis of common presentations, ordering and interpreting tests, managing stable chronic conditions under protocols.",
                 "Independent or supplementary prescribing, minor procedures, discharge and nurse/PA-led clinics where local law and credentialing allow (jurisdiction-dependent).",
                 "Refer undifferentiated, high-risk or complex cases to the responsible physician or specialist."),
    "nurse": ("Nursing assessment, monitoring, medicines administration, patient education, care planning and escalation of deterioration.",
              "Specialist nurse roles: protocol-based treatment, nurse-led triage and clinics, advanced assessment, care coordination.",
              "Escalate diagnosis and treatment changes to the prescriber; refer to allied health and social care as needed."),
    "allied": ("Assess function and impairment, deliver evidence-based therapy and rehabilitation, set measurable goals, educate and prevent recurrence within professional standards.",
               "Advanced practice: direct referral for imaging/tests, injection or prescribing rights in some jurisdictions, first-contact clinics, clinical specialism.",
               "Refer red flags and diagnostic uncertainty to the medical team; do not change medical treatment."),
    "diagnostic": ("Select and interpret investigations, report findings with uncertainty, advise on test choice, accuracy and limits (sensitivity/specificity).",
                   "Image-guided/interventional procedures, reporting radiographers, molecular and AI-assisted diagnostics, MDT case-conference leadership.",
                   "Defer treatment decisions and prognosis to the treating clinician."),
    "scientist": ("Interpret mechanisms, pharmacokinetics, toxic exposure and dose-response evidence; assess causality and risk; no individual treatment orders.",
                  "Clinical toxicology consultation, poison-centre advice, therapeutic drug monitoring and pharmacogenomic interpretation.",
                  "Refer individual patient management to the treating clinician or poison centre."),
    "pharmacist": ("Medication review, interactions, dose adjustment, adverse effects, adherence, and formulary/evidence appraisal.",
                   "Independent/supplementary prescribing, minor-ailment and chronic-disease clinics, deprescribing, pharmacogenomic-guided therapy, vaccination (jurisdiction-dependent).",
                   "Refer diagnosis and non-drug management to the prescriber."),
    "public": ("Population-level evidence, risk factors, prevention, screening, surveillance, equity and health-system implications; study-design and bias appraisal.",
               "Policy design, outbreak response, health-technology assessment and implementation science.",
               "Defer individual clinical decisions to treating clinicians."),
    "dental": ("Oral and dental diagnosis, prevention, restorative and periodontal care, local anaesthesia and oral surgery within dental regulation.",
               "Expanded duties for therapists/hygienists (local anaesthesia, fluoride, radiographs), sedation, implants, oral-medicine prescribing (jurisdiction-dependent).",
               "Refer systemic disease, maxillofacial pathology and complex surgery to medical or maxillofacial colleagues."),
    "psychosocial": ("Psychological and psychosocial assessment, evidence-based psychotherapy, safeguarding and support navigation.",
                     "Specialist therapy (e.g. trauma-focused), case management, capacity and safeguarding assessment; prescribing psychologists in some jurisdictions.",
                     "Refer medication, medical differential and acute risk to psychiatry or the medical team."),
    "acute": ("Rapid assessment and resuscitation, triage, stabilisation, time-critical decisions, escalation and disposition.",
              "Advanced airway, point-of-care ultrasound, procedural sedation, prehospital advanced life support, critical-care outreach.",
              "Hand over definitive and long-term management to specialty teams."),
    "integrative": ("Describe what traditional/complementary approaches are used for and critically appraise their evidence, safety and interactions; never substitute for evidence-based care.",
                    "Integrative-care coordination alongside conventional treatment, where regulated.",
                    "Refer diagnosis and disease-modifying treatment to conventional clinicians; flag herb-drug interactions to pharmacy."),
}


WORKUP = {
    "physician": "History (onset, course, severity, red flags), examination, baseline investigations, comorbidities and current medicines, then a stepwise differential and management options.",
    "surgeon": "Operative candidacy: indication, imaging and fitness assessment, anaesthetic/medical risk, alternatives to surgery, expected benefit vs complication risk, post-operative course.",
    "advanced": "Structured history and focused examination, point-of-care tests, red-flag screening, protocol-based plan with explicit escalation criteria.",
    "nurse": "Observations and early-warning scores, symptom and risk screening, medicines and allergy check, care-plan goals, escalation triggers.",
    "allied": "Functional assessment with validated outcome measures, baseline strength/mobility/participation, goals set with the patient, red-flag screen, home and work context.",
    "diagnostic": "Choose the test by the clinical question, pre-test probability and test accuracy; consider radiation/contrast risk, timing, and what result would change management.",
    "scientist": "Exposure, dose and timing history; mechanism and plausibility; dose-response and causality criteria; relevant levels or biomarkers; susceptible groups.",
    "pharmacist": "Full medicines reconciliation; indication and dose check against kidney/liver function; interactions, adverse effects, adherence and a monitoring plan.",
    "public": "Define population and outcome; check study design, bias and confounding; absolute vs relative risk; applicability and equity.",
    "dental": "Dental/oral history and pain assessment, intra- and extra-oral examination, radiographs as indicated, periodontal status, caries risk, treatment sequencing.",
    "psychosocial": "Biopsychosocial assessment, risk assessment, validated symptom measures, supports and stressors, goals and safety planning.",
    "acute": "Primary survey (ABCDE), vital signs, time-critical diagnoses first, immediate stabilisation, escalation and disposition.",
    "integrative": "Document what is used (product, dose, practitioner); check evidence grade and interactions with prescribed medicines; safety red flags; never delay effective care.",
}


@dataclass(frozen=True)
class Specialty:
    id: str
    name: str
    umbrella: str
    keywords: tuple[str, ...]
    body_areas: tuple[str, ...] = ()
    tier: str = "physician"
    scope: str = ""
    extended: str = ""
    refer: str = ""
    workup: str = ""


@dataclass(frozen=True)
class Umbrella:
    id: str
    name: str
    keywords: tuple[str, ...]
    body_areas: tuple[str, ...]
    specialties: tuple[Specialty, ...]
    adjacent: tuple[str, ...] = ()  # umbrellas consulted as *external* experts
    category: str = "Medical specialties"
    support: bool = False  # cross-cutting umbrellas (pharmacy, nursing, labs ...) rarely lead a question


# facet id -> (role template, keywords, tier)
FACETS = {
    "physician": ("{n} physician", ("diagnosis", "medication", "drug", "therapy", "treatment", "disease", "syndrome", "management", "symptoms"), "physician"),
    "surgeon": ("{n} surgeon", ("surgery", "surgical", "resection", "operative", "implant", "decompression", "bypass", "graft", "postoperative"), "surgeon"),
    "physio": ("{n} physiotherapist", ("rehabilitation", "physiotherapy", "exercise", "mobility", "gait", "recovery", "function", "strength"), "allied"),
    "toxicologist": ("{n} toxicologist", ("toxic", "toxicity", "poisoning", "exposure", "overdose", "adverse", "heavy", "metal", "solvent", "pesticide"), "scientist"),
    "diagnostics": ("{n} diagnostics specialist", ("mri", "ct", "imaging", "biomarker", "eeg", "emg", "scan", "test", "screening", "biopsy"), "diagnostic"),
    "dermato": ("{n} dermatology liaison", ("skin", "rash", "cutaneous", "lesion", "pruritus", "dermatitis", "neurocutaneous"), "physician"),
}
ALL_FACETS = tuple(FACETS)


def _mk(uid, s_id, name, kw, areas, tier, detail, umb_name):
    core, ext, ref = TIERS[tier]
    return Specialty(f"{uid}.{s_id}", name, uid, tuple(kw), tuple(areas), tier, f"{core} Focus: {detail or umb_name}.", ext, ref, WORKUP[tier])


def make_umbrella(uid, name, keywords, body_areas, adjacent=(), names=None, extras=(), cat="Medical specialties",
                  facets=ALL_FACETS, support=False):
    """names: facet id -> role name (None skips that facet). extras: (id, role, keywords, body_areas, tier[, focus])."""
    names, specs = names or {}, []
    for fid in facets:
        tmpl, fkw, tier = FACETS[fid]
        role = names.get(fid, tmpl.format(n=name))
        if role:
            specs.append(_mk(uid, fid, role, fkw, body_areas, tier, None, name))
    for e in extras:
        eid, role, kw, areas, tier, *rest = e
        specs.append(_mk(uid, eid, role, kw, areas, tier, rest[0] if rest else None, name))
    return Umbrella(uid, name, tuple(keywords), tuple(body_areas), tuple(specs), tuple(adjacent), cat, support)


def load_taxonomy(path: str | None = None) -> dict[str, Umbrella]:
    """Built-in catalog, optionally extended/overridden by JSON:
    {"umbrellas":[{"id","name","keywords","body_areas","adjacent","category","support","facets":[..],"names":{facet:role},
                   "extras":[[id, role, [kw], [areas], tier, optional focus]]}]}"""
    from .catalog import CATALOG
    tax = {u.id: u for u in CATALOG}
    if path:
        for u in json.load(open(path))["umbrellas"]:
            tax[u["id"]] = make_umbrella(u["id"], u["name"], u["keywords"], u["body_areas"], u.get("adjacent", ()), u.get("names"),
                                         [tuple(e) for e in u.get("extras", [])], u.get("category", "Medical specialties"),
                                         tuple(u.get("facets", ALL_FACETS)), u.get("support", False))
    return tax
