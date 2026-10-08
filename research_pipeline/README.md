# Research pipeline

Multi-specialist research pipeline with scoring, regeneration, learning and a live node-graph replay.

```
question ─▶ gather evidence ─▶ route (umbrella + body areas)
        ─▶ internal specialists ─▶ internal debate ─▶ umbrella lead
        ─▶ external experts (adjacent umbrellas) ─▶ chair
        ─▶ score & filter ──fail──▶ regenerate (rewrite, or re-gather with a broader query + new panel)
                         └─pass──▶ store / learn ─▶ follow-up questions ─▶ (loop back to the top)
```

## Run it

```bash
python -m research_pipeline --corpus ./my_papers --db research.db serve      # UI at http://127.0.0.1:8765
python -m research_pipeline --corpus ./my_papers ask "..." --viz run.html    # CLI + static replay
python -m research_pipeline --corpus ./my_papers loop "..." --depth 2
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
