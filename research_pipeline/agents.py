"""Discussion: internal specialists -> internal debate -> umbrella lead -> external experts -> chair."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .text import overlap, tokens

DEFAULT_SYSTEM = SYSTEM = ("You are a member of a multi-specialist clinical research panel. Use ONLY the supplied evidence; cite it as [E#]. "
          "State uncertainty. Never give individual medical directives.")


@dataclass
class Opinion:
    expert_id: str
    role: str
    stage: str  # specialist | debate | external
    text: str
    confidence: float
    evidence_ids: list[str]


def _ev_block(ev):
    return "\n".join(f"[{e.id}] (q={e.quality:.2f}) {e.title} — {e.text}" for e in ev) or "(none)"


def _relevant(spec, question, evidence, k=4):
    qt = set(tokens(question)) | set(spec.keywords)
    return sorted(evidence, key=lambda e: -(overlap(qt, e.title + " " + e.text) * (0.5 + e.quality)))[:k]


def _parse(expert, stage, raw):
    m = re.search(r"CONFIDENCE:\s*([0-9.]+)", raw)
    return Opinion(expert.id, expert.name, stage, re.sub(r"\s*CONFIDENCE:.*$", "", raw).strip(),
                   min(1.0, float(m.group(1))) if m else 0.5, sorted(set(re.findall(r"\[(E\d+)\]", raw))))


def ask_experts(llm, question, experts, evidence, stage, prior=None, trace=None, system=DEFAULT_SYSTEM, role_prompts=None, demands=''):
    out = []
    for s in experts:
        mine = _relevant(s, question, evidence)
        peers = "\n".join(f"- {o.role}: {o.text}" for o in (prior or []) if o.expert_id != s.id)
        extra = (role_prompts or {}).get(s.id, "").strip()
        scope = (f"SCOPE OF PRACTICE ({s.tier}): {s.scope}\nEXTENDED SCOPE (advanced practice, jurisdiction-dependent): {s.extended}\n"
                 f"REFER ON: {s.refer} Stay inside this scope; for points outside it, say 'refer to <role>' instead of opining.\n")
        prompt = (f"TASK: {stage}\nROLE: {s.name}\n" + scope + (f"ROLE INSTRUCTIONS: {extra}\n" if extra else "") +
                  (f"EXPERTISE DEMANDS: {demands}\n" if demands else "") + f"BODY AREAS: {', '.join(s.body_areas)}\nQUESTION: {question}\n"
                  f"EVIDENCE:\n{_ev_block(mine)}\n" + (f"PEER POSITIONS:\n{peers}\n" if peers else "") +
                  "Give your position in 2-4 sentences, cite [E#], end with 'CONFIDENCE: 0-1'.")
        op = _parse(s, stage, llm.complete(system, prompt))
        out.append(op)
        if trace:
            trace(s, mine, op, prompt)
    return out


def chair(llm, question, evidence, opinions, examples=(), feedback=None, system=DEFAULT_SYSTEM, demands=''):
    panel = "\n".join(f"PANEL: {o.role}" for o in opinions if o.stage != "debate")
    positions = "\n".join(f"- {o.role} ({o.stage}, conf {o.confidence:.2f}): {o.text}" for o in opinions)
    shots = "\n\n".join(f"HIGH-SCORING EXAMPLE:\n{x[:600]}" for x in examples)
    prompt = (f"TASK: chair\nROLE: Chair\n" + (f"EXPERTISE DEMANDS: {demands}\n" if demands else "") + f"QUESTION: {question}\nEVIDENCE:\n{_ev_block(evidence)}\n{panel}\nPOSITIONS:\n{positions}\n"
              f"{shots}\n" + (f"REVISION FEEDBACK (fix all of these):\n{feedback}\n" if feedback else "") +
              "Write the final answer: attribute each position by role, cite every claim [E#], note disagreements, "
              "add a 'Limitations:' line and a medical-advice disclaimer.")
    return llm.complete(system, prompt).strip(), prompt


def followups(llm, question, answer, max_n=3, system=DEFAULT_SYSTEM):
    raw = llm.complete(system, f"TASK: followups\nQUESTION: {question}\nANSWER:\n{answer}\nList open questions as 'FOLLOWUP: ...' lines.")
    return [m.strip() for m in re.findall(r"FOLLOWUP:\s*(.+)", raw)][:max_n]
