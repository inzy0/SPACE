"""Umbrella specialty -> mini-expertise tree, linked to body areas.

Every organ-system umbrella is generated from the same set of *facets* (physician, surgeon, physiotherapist,
toxicologist, diagnostics, dermatology-link) so the pattern "neuro -> neuro surgeon / physician / physio /
toxicologist / diagnostics / dermato" works for any system. Add umbrellas in code or via a JSON file
(see `load_taxonomy`).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Specialty:
    id: str
    name: str
    umbrella: str
    keywords: tuple[str, ...]
    body_areas: tuple[str, ...] = ()


@dataclass(frozen=True)
class Umbrella:
    id: str
    name: str
    keywords: tuple[str, ...]
    body_areas: tuple[str, ...]
    specialties: tuple[Specialty, ...]
    adjacent: tuple[str, ...] = ()  # umbrellas consulted as *external* experts


# facet id -> (default role template, facet keywords)
FACETS = {
    "physician": ("{n} physician", ("diagnosis", "medication", "drug", "therapy", "treatment", "disease", "syndrome", "management", "symptoms")),
    "surgeon": ("{n} surgeon", ("surgery", "surgical", "resection", "operative", "implant", "decompression", "bypass", "graft", "postoperative")),
    "physio": ("{n} physiotherapist", ("rehabilitation", "physiotherapy", "exercise", "mobility", "gait", "recovery", "function", "strength")),
    "toxicologist": ("{n} toxicologist", ("toxic", "toxicity", "poisoning", "exposure", "overdose", "adverse", "heavy", "metal", "solvent", "pesticide")),
    "diagnostics": ("{n} diagnostics specialist", ("mri", "ct", "imaging", "biomarker", "eeg", "emg", "scan", "test", "screening", "biopsy")),
    "dermato": ("{n} dermatology liaison", ("skin", "rash", "cutaneous", "lesion", "pruritus", "dermatitis", "neurocutaneous")),
}


def make_umbrella(uid, name, keywords, body_areas, adjacent=(), names=None, extras=()):
    """names: facet id -> role name override; extras: (id, role, keywords, body_areas) tuples."""
    names = names or {}
    specs = []
    for fid, (tmpl, fkw) in FACETS.items():
        role = names.get(fid, tmpl.format(n=name))
        specs.append(Specialty(f"{uid}.{fid}", role, uid, tuple(fkw), tuple(body_areas)))
    for eid, role, kw, areas in extras:
        specs.append(Specialty(f"{uid}.{eid}", role, uid, tuple(kw), tuple(areas)))
    return Umbrella(uid, name, tuple(keywords), tuple(body_areas), tuple(specs), tuple(adjacent))


NEURO = make_umbrella(
    "neuro", "Neuro",
    ("brain", "neuro", "neurological", "nerve", "spinal", "cerebral", "stroke", "epilepsy", "seizure", "dementia", "alzheimer",
     "parkinson", "migraine", "neuropathy", "multiple sclerosis", "cognitive", "concussion", "tbi", "cord", "cranial"),
    ("brain", "spinal cord", "peripheral nerves", "cranial nerves", "neuromuscular junction", "autonomic nervous system",
     "optic pathway", "vestibular system", "skin (neurocutaneous)"),
    adjacent=("cardio", "psych", "immuno", "onco", "endo", "msk"),
    names={"physician": "Neurologist (neuro physician)", "surgeon": "Neurosurgeon", "physio": "Neuro-physiotherapist",
           "toxicologist": "Neurotoxicologist", "diagnostics": "Neuro-diagnostics specialist (neuroradiology / neurophysiology)",
           "dermato": "Neuro-dermatology liaison"},
    extras=(
        ("opthal", "Neuro-ophthalmologist", ("vision", "optic", "visual", "papilledema", "diplopia", "nystagmus"), ("eye", "optic pathway")),
        ("psychol", "Neuropsychologist", ("memory", "cognition", "cognitive", "attention", "executive", "behavior"), ("brain",)),
        ("onco", "Neuro-oncologist", ("glioma", "glioblastoma", "tumor", "tumour", "metastasis", "meningioma"), ("brain", "spinal cord")),
        ("immuno", "Neuro-immunologist", ("autoimmune", "demyelinating", "encephalitis", "multiple sclerosis", "myasthenia"), ("brain", "spinal cord", "neuromuscular junction")),
        ("pain", "Neuropathic pain specialist", ("pain", "neuralgia", "neuropathic", "analgesic", "gabapentin"), ("peripheral nerves", "spinal cord")),
        ("neurotology", "Neuro-otologist", ("vertigo", "dizziness", "hearing", "tinnitus", "vestibular"), ("vestibular system", "cranial nerves")),
    ),
)

CARDIO = make_umbrella(
    "cardio", "Cardio", ("heart", "cardiac", "cardiovascular", "hypertension", "arrhythmia", "atrial", "coronary", "myocardial", "blood pressure", "heart failure"),
    ("heart", "coronary arteries", "aorta", "peripheral vessels"), adjacent=("neuro", "endo", "nephro", "pulmo"),
    names={"physician": "Cardiologist", "surgeon": "Cardiac surgeon", "physio": "Cardiac rehabilitation physiotherapist",
           "toxicologist": "Cardiotoxicologist", "diagnostics": "Cardiac imaging / ECG specialist", "dermato": "Cardio-dermatology liaison"},
    extras=(("ep", "Electrophysiologist", ("arrhythmia", "ablation", "pacemaker", "fibrillation", "qt"), ("heart",)),),
)
ONCO = make_umbrella(
    "onco", "Onco", ("cancer", "tumor", "tumour", "carcinoma", "metastasis", "chemotherapy", "radiotherapy", "oncology", "malignant", "lymphoma", "leukemia"),
    ("any organ", "lymphatic system", "bone marrow"), adjacent=("immuno", "neuro", "endo"),
    names={"physician": "Medical oncologist", "surgeon": "Surgical oncologist", "physio": "Oncology rehabilitation physiotherapist",
           "toxicologist": "Oncology toxicologist", "diagnostics": "Oncologic imaging / pathology specialist", "dermato": "Onco-dermatology liaison"},
    extras=(("radonc", "Radiation oncologist", ("radiotherapy", "radiation", "brachytherapy", "dose"), ("any organ",)),),
)
IMMUNO = make_umbrella(
    "immuno", "Immuno", ("immune", "autoimmune", "lupus", "arthritis", "allergy", "inflammation", "cytokine", "vaccine", "immunodeficiency", "antibody"),
    ("immune system", "joints", "skin"), adjacent=("neuro", "onco", "gastro", "msk"),
    names={"physician": "Clinical immunologist", "surgeon": "Transplant surgeon", "physio": "Immunology rehabilitation physiotherapist",
           "toxicologist": "Immunotoxicologist", "diagnostics": "Immunology laboratory diagnostics specialist", "dermato": "Immuno-dermatologist"},
)
ENDO = make_umbrella(
    "endo", "Endo", ("diabetes", "thyroid", "insulin", "hormone", "metabolic", "obesity", "adrenal", "pituitary", "glucose", "endocrine"),
    ("pancreas", "thyroid", "adrenal glands", "pituitary"), adjacent=("cardio", "neuro", "nephro"),
    names={"physician": "Endocrinologist", "surgeon": "Endocrine surgeon", "physio": "Metabolic rehabilitation physiotherapist",
           "toxicologist": "Endocrine-disruptor toxicologist", "diagnostics": "Endocrine diagnostics specialist", "dermato": "Endocrine-dermatology liaison"},
)
PSYCH = make_umbrella(
    "psych", "Psych", ("depression", "anxiety", "psychiatric", "schizophrenia", "bipolar", "adhd", "autism", "ptsd", "mood", "psychosis"),
    ("brain", "behavior"), adjacent=("neuro", "endo"),
    names={"physician": "Psychiatrist", "surgeon": "Neuromodulation / functional neurosurgeon", "physio": "Psychomotor therapist",
           "toxicologist": "Psychopharmacology toxicologist", "diagnostics": "Psychiatric assessment specialist", "dermato": "Psychodermatology liaison"},
)
MSK = make_umbrella(
    "msk", "Musculoskeletal", ("bone", "joint", "muscle", "fracture", "osteoporosis", "tendon", "spine", "orthopedic", "back pain", "arthritis"),
    ("bones", "joints", "muscles", "spine"), adjacent=("neuro", "immuno", "endo"),
    names={"physician": "Rheumatologist / sports physician", "surgeon": "Orthopedic surgeon", "physio": "Musculoskeletal physiotherapist",
           "toxicologist": "Musculoskeletal toxicologist", "diagnostics": "Musculoskeletal imaging specialist", "dermato": "Rheumato-dermatology liaison"},
)
GASTRO = make_umbrella(
    "gastro", "Gastro", ("gut", "bowel", "liver", "colitis", "crohn", "reflux", "ibs", "celiac", "microbiome", "hepatic", "digestive"),
    ("esophagus", "stomach", "intestines", "liver", "pancreas"), adjacent=("immuno", "onco", "neuro"),
    names={"physician": "Gastroenterologist", "surgeon": "GI surgeon", "physio": "Pelvic-floor / GI physiotherapist",
           "toxicologist": "Hepatotoxicologist", "diagnostics": "Endoscopy / GI diagnostics specialist", "dermato": "Gastro-dermatology liaison"},
)
PULMO = make_umbrella(
    "pulmo", "Pulmo", ("lung", "asthma", "copd", "pneumonia", "respiratory", "breathing", "oxygen", "bronchi", "pulmonary", "apnea"),
    ("lungs", "airways", "pleura"), adjacent=("cardio", "immuno", "onco"),
    names={"physician": "Pulmonologist", "surgeon": "Thoracic surgeon", "physio": "Respiratory physiotherapist",
           "toxicologist": "Inhalation toxicologist", "diagnostics": "Pulmonary function / imaging specialist", "dermato": "Pulmo-dermatology liaison"},
)
NEPHRO = make_umbrella(
    "nephro", "Nephro", ("kidney", "renal", "dialysis", "creatinine", "proteinuria", "nephritis", "urinary", "bladder"),
    ("kidneys", "ureters", "bladder"), adjacent=("cardio", "endo"),
    names={"physician": "Nephrologist", "surgeon": "Urologic / transplant surgeon", "physio": "Renal rehabilitation physiotherapist",
           "toxicologist": "Nephrotoxicologist", "diagnostics": "Renal diagnostics specialist", "dermato": "Renal-dermatology liaison"},
)

DEFAULT = {u.id: u for u in (NEURO, CARDIO, ONCO, IMMUNO, ENDO, PSYCH, MSK, GASTRO, PULMO, NEPHRO)}


def load_taxonomy(path: str | None = None) -> dict[str, Umbrella]:
    """Built-in taxonomy, optionally extended/overridden by a JSON file:
    {"umbrellas":[{"id","name","keywords","body_areas","adjacent","names":{facet:role},"extras":[[id,role,[kw],[areas]]]}]}"""
    tax = dict(DEFAULT)
    if path:
        for u in json.load(open(path))["umbrellas"]:
            tax[u["id"]] = make_umbrella(u["id"], u["name"], u["keywords"], u["body_areas"], u.get("adjacent", ()),
                                         u.get("names"), [tuple(e) for e in u.get("extras", [])])
    return tax
