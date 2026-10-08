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

## Specialty tree (`taxonomy.py`)

Each umbrella (neuro, cardio, onco, immuno, endo, psych, msk, gastro, pulmo, nephro) is generated from the same facets
(physician, surgeon, physiotherapist, toxicologist, diagnostics, dermatology liaison) plus custom extras
(neuro-ophthalmologist, neuro-oncologist, neuropathic pain, neuro-otologist, ...), each linked to body areas.
Add umbrellas with `--taxonomy file.json` (see `load_taxonomy`).

## Scoring & learning

Composite of grounding (valid `[E#]` citations), evidence quality, specialist coverage, consensus, calibration,
safety (disclaimer) and relevance. Hard gates: grounding >= 0.6, disclaimer present, no invented citations.
Failures produce targeted feedback for the rewrite. Every attempt is stored: expert weights adapt to past scores,
top answers become few-shot examples, (rejected -> accepted) pairs export as preference data, and questions that never
pass are re-queued first in training. Blend an LLM judge in with `ResearchPipeline(judge=...)`.

Research aid, not medical advice: output is only as good as the evidence supplied.
