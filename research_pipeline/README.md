# Healthcare research pipeline — all in one

```bash
python healthcare_research.pyz          # ONE file, stdlib only, Python 3.10+  ->  full app at http://127.0.0.1:8765
python healthcare_research.pyz --open --corpus ./my_papers --db research.db     # + your own evidence folder, persistent learning
python research_pipeline/build_single_file.py                                     # rebuild the .pyz after editing the source
```

The one file contains everything below: language-pyramid understanding and clearance, the 35-umbrella / 263-role roster with scope of
practice, multi-term routing, reach extension, the multi-specialist discussion, the scorer and regenerate loop, the evidence score
libraries, background training, the CLI and the live node-graph UI.

Multi-specialist research pipeline with scoring, regeneration, learning and a live node-graph replay.

```
question ─▶ LANGUAGE PYRAMID (words · structure · meaning · intent · context)
        ─▶ CLEARANCE 1: "here is what I understood" + when/how/why/where/place/situation/who/chance questions
        ─▶ you confirm / answer ─▶ CLEARANCE 2 ─▶ REACH (extend the panel to the question's demands)
        ─▶ multi-term route (umbrella + body areas) ─▶ gather evidence (+ evidence library)
        ─▶ internal specialists ─▶ internal debate ─▶ umbrella lead
        ─▶ external experts (adjacent umbrellas + reach) ─▶ chair
        ─▶ score & filter ──fail──▶ regenerate (rewrite, or re-gather with a broader query + new panel)
                         └─pass──▶ store / learn (weights + library) ─▶ follow-up questions ─▶ (loop, context inherited)
```

## Language pyramid and clearance (`understand.py`, `terms.py`)

Five layers, base to apex: **1 Lexical** (words, lay terms, abbreviations) → **2 Syntactic** (question form, clauses, negation,
modality) → **3 Semantic** (conditions, symptoms, interventions, drugs, tests, body areas, population) → **4 Pragmatic** (intent,
speaker, urgency, risk/"chance" wording) → **5 Contextual** (slots below). Deterministic and offline; `llm_understanding` lets a real
LLM refine the slots.

| Slot | Asked as | Examples read from the text |
|---|---|---|
| WHEN · acute → chronic | onset and progression | "last week", "for 3 years", "flare", "sudden" |
| HOW | presentation, severity, management | "severe", "mild" |
| WHY | goal / decision | diagnosis, treatment, prevention, prognosis, safety, rehab, evidence |
| WHERE | body area, region | brain, knee, "in the UK" |
| PLACE | care setting | home, GP, ED, ICU, care home (longest phrase wins) |
| SITUATION | scenario | after surgery, pregnancy, on warfarin, caring for someone |
| WHO | population | age, sex, pregnancy, comorbidities, medicines |
| CHANCE | risk / likelihood | "chance of recovery", "within 5 years" |
| SCALE | one person → population | policy, service, cohort |

**Clearance:** level 0 raw → **1** (pyramid built, confirmation message + questions sent, nothing researched) → **2** (you confirmed,
answered, or skipped). Questions are ranked by what the intent needs; ambiguous abbreviations (MS, PE, MI, RA, ED ...) always get a
"did you mean" question; a current emergency (chest pain *right now*, stroke signs, overdose, suicidal thoughts) gets a banner and a
short reply. Past events ("a week ago") do not trigger it. Your answers become a **confirmed context frame** that is injected into
every expert and chair prompt, expands retrieval, and is scored (`context_fit`).

**Multi-term routing:** 120 concepts (conditions, symptoms, interventions, drugs, tests, populations) with lay terms, synonyms and
abbreviations map to umbrellas and specific roles, so "brain attack" routes like "stroke". Negation is handled ("no chest pain").

**Reach:** the context extends the panel: child → paediatrics, older adult → geriatrics, pregnancy → obstetrics, drugs/safety →
clinical pharmacist, acute/ED → emergency, chronic → clinical nurse specialist, diagnosis → imaging, rehab → rehabilitation medicine,
risk/comparison → evidence methodologist, population scale → epidemiology, end of life → palliative, multimorbidity → bigger panel and more
evidence. Capped by `max_reach_extra`; every addition is listed with its reason in the trace.

## Evidence score library (`library.py`)

One library **per specialty** and a **common library per umbrella** (all its specialties merged). Every item that reaches a panel is
ingested into the library of each panelist and the umbrella. Scores are explainable and per scope:
`static` = 0.45 quality + 0.20 study design (inferred from text) + 0.10 recency + 0.10 source reliability + 0.15 fit to the scope;
`learned` = how often that specialty cited the item in an answer that passed the gate (smoothed toward the static score).
Umbrella score = 0.5 best specialty + 0.3 mean of the top three + 0.2 breadth. Retrieval ranks by relevance × score and experts see their
own specialty's score, so useful evidence rises and weak evidence sinks as runs accumulate. Browse it in the UI (section 6),
`python -m research_pipeline library umb:neuro`, or `/api/library?scope=...`; add documents with `library --ingest folder --to neuro.surgeon,umb:neuro`.

## Run it

```bash
python -m research_pipeline --corpus ./my_papers --db research.db serve      # UI at http://127.0.0.1:8765
python -m research_pipeline --corpus ./my_papers ask "..." --viz run.html    # CLI + static replay
python -m research_pipeline --corpus ./my_papers loop "..." --depth 2
python -m research_pipeline understand "My mum is 72, had a brain attack last week..." --answer acuity=subacute   # clearance 1 message only
python -m research_pipeline library overview
python -m research_pipeline train questions.txt --loop                        # replay a question bank
python -m research_pipeline export                                            # sft.jsonl + prefs.jsonl
python -m unittest research_pipeline.tests.test_pipeline
```

Run from the repo root. Offline by default (`--llm mock`: extractive, deterministic, for testing the machinery).
For real answers: `export ANTHROPIC_API_KEY=...` and pick *Anthropic API* in the UI (or `--llm anthropic`).
`--pubmed` adds PubMed search. Python 3.10+, stdlib only.

## What you control (UI "Setup" panel = the `/api/run` request)

| Control | Effect |
|---|---|
| **Prompt** | the question; *single* or *loop* mode (follow-ups are discussed in turn, deduped, depth-capped) |
| **Roles** | global system prompt; per-role instruction; per-role `auto / required / excluded` |
| **Expertise demands** | free text injected into every expert and chair prompt; primary-umbrella override; min evidence quality; panel sizes |
| **Clearance & reach** | require confirmation, auto-extend reach, max extra experts, evidence library on/off; "Understand & confirm first" button |
| **Parameters** | pass threshold, max regenerations, evidence per attempt, loop depth/size, LLM + model, PubMed |
| **Knowledge** | pasted/uploaded documents with a quality weight (plus the server `--corpus` folder) |
| **Learning** | background training on a question list; export SFT + preference pairs |

## Healthcare roster (`taxonomy.py` + `catalog.py`)

35 umbrellas / 260+ roles in 9 categories, every role linked to body areas and carrying a **tier** with its
**scope of practice** (core), **extended / advanced-practice scope** (e.g. nurse and pharmacist prescribing, extended-scope
physiotherapy, jurisdiction-dependent) and **refer-on limits**. These go into the role's prompt so experts stay in scope and
say "refer to ..." instead of opining outside it.

| Category | Umbrellas |
|---|---|
| Medical specialties | neuro, cardio, vascular, onco, haematology, immuno/allergy, endocrine, gastro/hepatology, pulmo/sleep, nephro, urology, dermatology, eye, ENT/audiology, musculoskeletal, infectious disease, geriatrics, genetics, sports & exercise |
| Surgical & perioperative | general surgery (colorectal, HPB, bariatric, plastic, breast, transplant ...), anaesthesia & pain |
| Women's & children's | obstetrics/gynaecology/midwifery/fertility, paediatrics/neonatology/adolescent |
| Emergency & critical care | emergency, intensive care, trauma, paramedic, poison centre, disaster |
| Mental health | psychiatry, clinical psychology, addiction, child & adolescent, counselling, social work |
| Nursing & midwifery | nurse practitioner, physician associate, clinical nurse specialist, community, tissue viability ... |
| Pharmacy & medicines | clinical / prescribing / community pharmacist, clinical pharmacologist, pharmacogenomics, pharmacovigilance |
| Diagnostics & laboratory | radiology, interventional, nuclear medicine, pathology, biochemistry, microbiology, medical physics, clinical AI |
| Allied health & rehabilitation | rehab medicine, physio, OT, speech & language, prosthetics, podiatry, dietetics/nutrition, exercise physiology |
| Dental & oral | general, perio, endo, ortho, prostho, paediatric, OMFS, oral medicine, hygiene/therapy |
| Public health & evidence | epidemiology, biostatistics/EBM, occupational & environmental, health economics, vaccinology, informatics |
| Palliative care & ethics | palliative medicine, hospice nursing, clinical ethics, chaplaincy, bereavement |
| Integrative (evidence-graded) | acupuncture, herbal, TCM/Ayurveda, mind-body, supplements: appraised for evidence and interactions, never a substitute for care |

Organ-system umbrellas are generated from shared facets (physician, surgeon, physiotherapist, toxicologist, diagnostics,
dermatology liaison) plus custom extras. The router picks the primary umbrella, the best internal roles, and **one best expert per
relevant other umbrella** as the external panel. Support umbrellas (pharmacy, nursing, labs ...) rarely lead but join when relevant.
Add or override umbrellas with `--taxonomy file.json` (see `load_taxonomy`).

## Scoring & learning

Composite of grounding (valid `[E#]` citations), evidence quality, specialist coverage, consensus, calibration,
safety (disclaimer) and relevance. Hard gates: grounding >= 0.6, disclaimer present, no invented citations.
Failures produce targeted feedback for the rewrite. Every attempt is stored: expert weights adapt to past scores,
top answers become few-shot examples, (rejected -> accepted) pairs export as preference data, and questions that never
pass are re-queued first in training. Blend an LLM judge in with `ResearchPipeline(judge=...)`.

Research aid, not medical advice: output is only as good as the evidence supplied.
