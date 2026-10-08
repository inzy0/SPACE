"""Term lexicon: lay words, synonyms and abbreviations -> canonical concept -> umbrellas / specialties to route to.

Line format:  canonical | type | synonym; synonym | route, route      (route = umbrella id or umbrella.role id)
Ambiguous abbreviations (ms, pe, mi, ra, ca, cp, ed, hf ...) are deliberately NOT mapped: they trigger a clarifying question.
`!` after the type marks concepts that can signal an emergency when they describe a current situation.
"""
from __future__ import annotations

import re

RAW = """
stroke | condition! | cva; brain attack; cerebrovascular accident; ischemic stroke; ischaemic stroke; haemorrhagic stroke; hemorrhagic stroke; tia; mini stroke; transient ischemic attack; face drooping; sudden weakness one side | neuro, cardio.physician, rehab
traumatic brain injury | condition | tbi; head injury; brain injury; head trauma; concussion | neuro, emerg, rehab, sportsmed
epilepsy | condition | seizure; seizures; fits; convulsion; convulsions | neuro, paeds
migraine | condition | headache; migraines; cluster headache; severe headache | neuro, neuro.pain
dementia | condition | alzheimer; alzheimers; alzheimer's disease; memory loss; cognitive decline; vascular dementia; forgetfulness | neuro, geri, psych
parkinson's disease | condition | parkinson; parkinsons; tremor; shaking | neuro, rehab
multiple sclerosis | condition | demyelinating disease; optic neuritis | neuro.immuno, neuro
neuropathy | condition | nerve damage; peripheral neuropathy; pins and needles; numbness; tingling; sciatica; nerve pain | neuro, neuro.pain, msk.spine
back pain | symptom | lower back pain; backache; slipped disc; herniated disc; lumbago; neck pain | msk, msk.spine, msk.fcp, anaesth.pain
vertigo | symptom | dizziness; spinning; giddiness; balance problems; dizzy | ent, neuro.neurotology, ent.vest
heart attack | condition! | myocardial infarction; stemi; nstemi; acute coronary syndrome; coronary thrombosis | cardio, emerg
chest pain | symptom! | angina; chest tightness; crushing chest | cardio, emerg
high blood pressure | condition | hypertension; htn; raised blood pressure | cardio, nursing.np
heart failure | condition | cardiac failure; fluid overload; weak heart; swollen ankles | cardio, cardio.hf
atrial fibrillation | condition | afib; irregular heartbeat; palpitations; arrhythmia; racing heart | cardio.ep, cardio, haem
high cholesterol | condition | hyperlipidaemia; hyperlipidemia; dyslipidaemia; statin; statins; ldl | cardio.prevent, pharm.clinical
blood clot | condition! | dvt; deep vein thrombosis; pulmonary embolism; thrombosis; vte; clot in leg | vasc, haem, emerg
varicose veins | condition | varicose; venous insufficiency; leg ulcer | vasc
cardiac arrest | condition! | cpr; resuscitation; collapsed and not breathing | emerg, cardio
cancer | condition | tumor; tumour; malignancy; carcinoma; oncology; chemo; chemotherapy; metastatic; metastasis; leukaemia; leukemia; lymphoma; radiotherapy; immunotherapy | onco, haem, pall
breast cancer | condition | breast lump; mastectomy; breast tumour | onco, surgery.breast, obgyn
lung cancer | condition | lung tumour; lung nodule | onco, pulmo
colorectal cancer | condition | colon cancer; bowel cancer; rectal cancer | onco, surgery.colorectal, gastro
prostate cancer | condition | psa; raised psa | onco, uro
skin cancer | condition | melanoma; basal cell carcinoma; squamous cell carcinoma; suspicious mole | derm, onco
diabetes | condition | type 2 diabetes; type 1 diabetes; t2dm; t1dm; high blood sugar; hyperglycaemia; hyperglycemia; hba1c; insulin; blood glucose | endo, endo.dnurse, nutri, pharm.clinical
gestational diabetes | condition | gdm; diabetes in pregnancy | obgyn, endo
thyroid disease | condition | hypothyroidism; hyperthyroidism; goitre; goiter; thyroid | endo
obesity | condition | overweight; weight loss; bmi; glp-1; semaglutide; tirzepatide; bariatric | nutri.obesity, endo, surgery.bariatric
osteoporosis | condition | brittle bones; bone density; fragility fracture; low bone mass | endo, msk, geri
irritable bowel syndrome | condition | ibs; bloating; constipation; diarrhoea; diarrhea; stomach cramps | gastro, nutri
inflammatory bowel disease | condition | crohn's disease; crohns; ulcerative colitis; ibd; colitis | gastro, immuno
acid reflux | condition | gerd; heartburn; reflux; indigestion; dyspepsia | gastro
liver disease | condition | cirrhosis; fatty liver; hepatitis; nafld; jaundice | gastro.hepat, gastro
coeliac disease | condition | celiac; gluten intolerance; gluten | gastro, nutri
asthma | condition | wheeze; wheezing; inhaler; inhalers | pulmo, paeds, pharm.community
copd | condition | emphysema; chronic bronchitis; smoker's cough | pulmo, pulmo.rt
pneumonia | condition | chest infection; lung infection | pulmo, infect
sleep apnoea | condition | sleep apnea; snoring; osa; cpap | pulmo.sleep
covid-19 | condition | covid; coronavirus; long covid; sars-cov-2 | infect, pulmo, pubh
shortness of breath | symptom! | breathlessness; dyspnoea; dyspnea; can't breathe; difficulty breathing | pulmo, cardio, emerg
chronic kidney disease | condition | ckd; kidney failure; renal failure; dialysis; kidney disease | nephro
urinary tract infection | condition | uti; cystitis; burning urine; burning when i pee | uro, infect, nursing.np
kidney stones | condition | renal colic; kidney stone | uro, nephro
prostate enlargement | condition | bph; enlarged prostate; urinary frequency; weak urine stream | uro
incontinence | symptom | leaking urine; bladder control; pelvic floor; urine leakage | uro.physio, obgyn.physio
eczema | condition | atopic dermatitis; dermatitis; itchy skin; rash; skin rash | derm, paeds
psoriasis | condition | plaque psoriasis; scaly patches | derm, immuno
acne | condition | spots; pimples; breakouts | derm
pressure ulcer | condition | pressure sore; bedsore; chronic wound; non-healing wound; burn; scar | derm.wound, nursing.tv
glaucoma | condition | cataract; macular degeneration; blurred vision; vision loss; dry eye; red eye; eye pain | ophth
hearing loss | symptom | tinnitus; ringing in ears; deafness; hearing aid | ent.audio, ent
sinusitis | condition | sore throat; tonsillitis; ear infection; otitis; blocked nose; hoarse voice | ent
arthritis | condition | osteoarthritis; rheumatoid arthritis; joint pain; gout; stiff joints; swollen joints | msk, immuno
fracture | condition | broken bone; broken arm; broken leg; hip fracture; broken wrist | msk, emerg, geri.ortho
acl injury | condition | torn acl; cruciate ligament; meniscus tear; sprain; strain; tendinopathy; tennis elbow; return to play | sportsmed, msk
fibromyalgia | condition | chronic widespread pain; chronic pain; persistent pain | anaesth.pain, msk
depression | condition | low mood; feeling down; depressed; hopeless | psych, psych.psychologist
anxiety | condition | panic attacks; panic attack; worry; generalised anxiety; gad | psych, psych.psychologist
adhd | condition | attention deficit; hyperactivity; inattention | psych.cap, paeds.devpaed
autism | condition | asd; autistic; autism spectrum | paeds.devpaed, psych.cap
schizophrenia | condition | psychosis; hallucinations; delusions; hearing voices | psych
bipolar disorder | condition | bipolar; manic; mania | psych
ptsd | condition | post-traumatic stress; flashbacks; trauma symptoms | psych
suicidal thoughts | symptom! | suicide; self-harm; self harm; want to die; kill myself; end my life | psych, emerg
addiction | condition | alcohol dependence; drug use; substance abuse; substance use; opioid use disorder; smoking cessation; vaping; quit smoking | psych.addiction
insomnia | symptom | cannot sleep; trouble sleeping; poor sleep; can't sleep | neuro.sleep, psych, pulmo.sleep
eating disorder | condition | anorexia; bulimia; binge eating | psych, paeds.adolescent
sepsis | condition! | blood poisoning; septic shock | emerg.icu, infect
hiv | condition | aids; prep | infect.sti, infect
tuberculosis | condition | tb | infect, pulmo
sexually transmitted infection | condition | sti; std; chlamydia; gonorrhoea; syphilis | infect.sti
malaria | condition | tropical disease; travel vaccine; travelling abroad | infect.travel
vaccination | intervention | vaccine; vaccines; immunisation; immunization; booster; jab; flu shot | pubh.vax, infect, immuno
antibiotic resistance | condition | amr; antimicrobial resistance; superbug | infect.micro, pharm
pregnancy | population | pregnant; expecting; antenatal; prenatal; trimester; morning sickness; breastfeeding | obgyn, obgyn.midwife, obgyn.mfm
preeclampsia | condition | pre-eclampsia; pregnancy hypertension; high blood pressure in pregnancy | obgyn.mfm
menopause | condition | hot flushes; hrt; perimenopause; hormone replacement | obgyn.meno, endo
endometriosis | condition | pelvic pain; painful periods; period pain; heavy periods | obgyn
pcos | condition | polycystic ovary; polycystic ovaries | obgyn, endo
infertility | condition | ivf; trying to conceive; fertility; cannot conceive | obgyn.rei, uro.andro
contraception | intervention | birth control; the pill; iud; contraceptive | obgyn
miscarriage | condition | pregnancy loss; stillbirth | obgyn
infant | population | baby; newborn; neonate; premature baby; preterm baby | paeds, paeds.neonatal
child | population | children; kid; kids; toddler; paediatric; pediatric; schoolchild | paeds
adolescent | population | teenager; teen; teens; puberty | paeds.adolescent
older adult | population | elderly; frailty; frail; falls; ageing; aging; care home; nursing home; geriatric | geri
polypharmacy | condition | many medications; multiple medications; deprescribing; too many tablets | pharm.clinical, geri
palliative care | intervention | end of life; dying; hospice; terminal illness; advance care plan; comfort care | pall
surgery | intervention | operation; operative; post-op; postoperative; preoperative; anaesthetic; anesthetic | surgery, anaesth
hernia | condition | inguinal hernia; umbilical hernia | surgery.general
appendicitis | condition! | appendix; appendectomy | surgery.general, emerg
gallstones | condition | gallbladder; cholecystectomy; biliary colic | surgery.general, gastro
organ transplant | intervention | transplant; kidney transplant; liver transplant; donor organ | surgery.transplant, immuno
rehabilitation | intervention | rehab; physio; physiotherapy; occupational therapy; recovery of function; regain function | rehab, rehab.physio, rehab.ot
speech problems | symptom | aphasia; stammer; swallowing difficulty; dysphagia; slurred speech | rehab.slt, ent
amputation | condition | prosthesis; prosthetic; limb loss; artificial limb | rehab.po
diet | intervention | nutrition; food; vitamins; supplements; malnutrition; fasting; intermittent fasting; mediterranean diet; calories | nutri, nutri.rd
food allergy | condition! | peanut allergy; anaphylaxis; allergic reaction; hives; nut allergy | immuno.allergy, immuno
toothache | condition | tooth decay; cavity; dental pain; gum disease; gingivitis; periodontitis; wisdom tooth; root canal; braces; dentist; gums; teeth | dental, dental.gdp
blood thinner | drug | warfarin; anticoagulant; anticoagulants; apixaban; rivaroxaban; heparin | pharm.clinical, haem
pain relief | drug | painkiller; painkillers; analgesic; opioid; opioids; paracetamol; acetaminophen; ibuprofen; nsaid | pharm.clinical, anaesth.pain
antidepressant | drug | ssri; sertraline; fluoxetine; antidepressants | psych, pharm.clinical
antibiotic | drug | antibiotics; amoxicillin; antibacterial | pharm.clinical, infect
steroid | drug | prednisolone; corticosteroid; steroids; prednisone | pharm.clinical, immuno
overdose | condition! | poisoning; swallowed; toxic exposure; poison; took too many | emerg.tox, emerg
side effect | outcome | adverse reaction; adverse effect; adverse effects; drug interaction; interaction; side effects | pharm.mgmt, pharm.clinical
herbal medicine | intervention | herbal; turmeric; st john's wort; cannabis; cbd; natural remedy | integ.herbal, pharm
acupuncture | intervention | traditional chinese medicine; tcm; ayurveda; homeopathy; naturopathy; yoga; tai chi | integ
mri scan | test | mri; ct scan; x-ray; xray; ultrasound; pet scan; imaging; scan | dx.rad
blood test | test | bloods; lab test; biomarker; blood work; cholesterol test; blood tests | dx.biochem
biopsy | test | histology; pathology; tissue sample | dx.path
screening | test | screening test; mammogram; smear test; cervical screening; colonoscopy | pubh.phys, dx
genetic testing | test | dna test; genetic; brca; hereditary; gene; genes; inherited | genetics
evidence | outcome | studies; trial; trials; randomised; randomized; meta-analysis; systematic review; guideline; guidelines; research | pubh.stats, pubh
outbreak | condition | epidemic; pandemic; incidence; prevalence; surveillance | pubh.epi
health policy | outcome | health system; cost-effectiveness; funding; access to care; commissioning | pubh.hta, pubh
emergency | condition! | accident; collapse; collapsed; unconscious; unresponsive; severe bleeding; bleeding heavily | emerg
athlete | population | sports injury; marathon; training load; sport; sports | sportsmed
return to work | outcome | workplace; occupational; sick leave; fit note | pubh.occenv, rehab.vr
"""

ABBREV_AMBIGUOUS = {
    "ms": ["multiple sclerosis", "mitral stenosis", "musculoskeletal"],
    "pe": ["pulmonary embolism", "pre-eclampsia", "physical examination"],
    "mi": ["myocardial infarction (heart attack)", "mitral insufficiency"],
    "ra": ["rheumatoid arthritis", "right atrium"],
    "ca": ["cancer", "calcium", "coronary artery"],
    "cp": ["cerebral palsy", "chest pain"],
    "ed": ["emergency department", "erectile dysfunction", "eating disorder"],
    "hf": ["heart failure", "high frequency"],
    "af": ["atrial fibrillation", "atrial flutter"],
    "pd": ["Parkinson's disease", "peritoneal dialysis", "personality disorder"],
}

BODY = {
    "head": "head / brain", "brain": "brain", "skull": "head / brain", "neck": "neck / cervical spine", "throat": "throat", "face": "face",
    "eye": "eye", "eyes": "eye", "ear": "ear", "ears": "ear", "nose": "nose / sinuses", "mouth": "mouth / teeth", "tooth": "teeth", "teeth": "teeth", "jaw": "jaw",
    "chest": "chest", "heart": "heart", "lung": "lungs", "lungs": "lungs", "breast": "breast", "abdomen": "abdomen", "stomach": "stomach", "belly": "abdomen",
    "gut": "gut", "bowel": "bowel", "liver": "liver", "kidney": "kidneys", "kidneys": "kidneys", "bladder": "bladder", "back": "back / spine", "spine": "back / spine",
    "shoulder": "shoulder", "arm": "arm", "elbow": "elbow", "wrist": "wrist / hand", "hand": "wrist / hand", "hip": "hip", "knee": "knee", "leg": "leg", "ankle": "ankle / foot",
    "foot": "ankle / foot", "feet": "ankle / foot", "skin": "skin", "joint": "joints", "joints": "joints", "muscle": "muscles", "muscles": "muscles", "bone": "bones",
    "bones": "bones", "nerve": "nerves", "nerves": "nerves", "blood": "blood", "prostate": "prostate", "uterus": "uterus", "ovary": "ovaries", "thyroid": "thyroid",
}


class Concept:
    __slots__ = ("canonical", "type", "emergent", "surfaces", "routes")

    def __init__(self, canonical, type_, emergent, surfaces, routes):
        self.canonical, self.type, self.emergent, self.surfaces, self.routes = canonical, type_, emergent, surfaces, routes


def _parse():
    out = []
    for line in RAW.strip().splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 4:
            continue
        name, typ, syn, routes = parts
        em = typ.endswith("!")
        out.append(Concept(name, typ.rstrip("!"), em, [name] + [s.strip() for s in syn.split(";") if s.strip()], [r.strip() for r in routes.split(",") if r.strip()]))
    return out


CONCEPTS = _parse()
_PAT = [(c, re.compile(r"(?<![a-z0-9])(?:" + "|".join(re.escape(s) for s in sorted(c.surfaces, key=len, reverse=True)) + r")(?:s|es)?(?![a-z0-9])", re.I)) for c in CONCEPTS]
NEG = re.compile(r"\b(no|not|without|denies|denied|never|negative for|ruled out|free of|none)\W+(?:\w+\W+){0,2}$", re.I)


def find_concepts(text: str) -> list[dict]:
    """All concepts mentioned in text with surface form, span and negation flag (longest match wins per span)."""
    hits, taken = [], []
    cand = []
    for c, pat in _PAT:
        for m in pat.finditer(text):
            cand.append((m.end() - m.start(), m.start(), m.end(), c, m.group(0)))
    for ln, s, e, c, surf in sorted(cand, key=lambda x: (-x[0], x[1])):
        if any(not (e <= ts or s >= te) for ts, te in taken):
            continue
        taken.append((s, e))
        hits.append(dict(canonical=c.canonical, type=c.type, surface=surf.lower(), start=s, negated=bool(NEG.search(text[max(0, s - 28):s])),
                         routes=c.routes, emergent=c.emergent))
    hits.sort(key=lambda h: h["start"])
    seen, uniq = set(), []
    for h in hits:  # one entry per canonical concept
        if h["canonical"] not in seen:
            seen.add(h["canonical"]); uniq.append(h)
    return uniq


def body_areas(text: str) -> list[str]:
    toks = re.findall(r"[a-z]+", text.lower())
    return sorted({BODY[t] for t in toks if t in BODY})


def terms_for_role(role_id: str) -> list[str]:
    """Every surface term that routes to this role or its umbrella (for UI transparency)."""
    umb = role_id.split(".")[0]
    own = [s for c in CONCEPTS if role_id in c.routes for s in c.surfaces[:4]]
    um = [c.canonical for c in CONCEPTS if umb in c.routes]
    return list(dict.fromkeys(own + um))[:14]
