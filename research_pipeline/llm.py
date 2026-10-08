"""LLM backends. Prompts start with `TASK: <kind>` so any backend (or the offline mock) can dispatch on it."""
from __future__ import annotations

import json
import os
import re
import urllib.request


class LLM:
    def complete(self, system: str, prompt: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class AnthropicLLM(LLM):
    def __init__(self, model=None, key=None, max_tokens=1500):
        self.model = model or os.environ.get("PIPELINE_MODEL", "claude-sonnet-5-5")
        self.key = key or os.environ["ANTHROPIC_API_KEY"]
        self.max_tokens = max_tokens

    def complete(self, system, prompt):
        body = json.dumps({"model": self.model, "max_tokens": self.max_tokens, "system": system,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", body,
                                     {"x-api-key": self.key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return "".join(b.get("text", "") for b in json.load(r)["content"])


EV = re.compile(r"^\[E(\d+)\] \(q=([\d.]+)\) (.*?) — (.*)$", re.M)


class MockLLM(LLM):
    """Deterministic offline stand-in. Extractive: it only restates the evidence it is shown, with citations.
    With first_draft_flaws=True the first chair draft omits limitations/disclaimer, so the
    score -> feedback -> regenerate loop is exercised end to end."""

    def __init__(self, first_draft_flaws=True):
        self.flaws = first_draft_flaws

    def complete(self, system, prompt):
        task = re.search(r"^TASK: (\w+)", prompt, re.M).group(1)
        role = (re.search(r"^ROLE: (.*)$", prompt, re.M) or [0, "Expert"])[1]
        ev = [(int(i), float(q), t, x) for i, q, t, x in EV.findall(prompt)]
        cite = lambda e: f"{e[3].split('. ')[0].rstrip('.')} [E{e[0]}]."
        if task in ("specialist", "external"):
            if not ev:
                return f"{role}: no relevant evidence was supplied, so I cannot take a position. CONFIDENCE: 0.2"
            body = " ".join(cite(e) for e in ev[:2])
            return f"{role}: {body} CONFIDENCE: {min(0.9, 0.45 + 0.15 * len(ev)):.2f}"
        if task == "debate":
            return f"{role}: I reviewed the peer positions and found no direct conflict with my reading of the evidence. CONFIDENCE: 0.70"
        if task == "followups":
            return "FOLLOWUP: What evidence is missing to resolve the main uncertainty?\nFOLLOWUP: Which patient subgroups respond differently?"
        # chair
        lines = [f"{m}: {' '.join(cite(e) for e in ev[:1])}" for m in re.findall(r"^PANEL: (.*)$", prompt, re.M)]
        out = ["Summary of expert discussion.", *lines]
        if not (self.flaws and "REVISION FEEDBACK" not in prompt):
            out += ["Limitations: the evidence above may be incomplete and findings are uncertain for individual patients.",
                    "Note: this is not medical advice; consult a qualified clinician."]
        return "\n".join(out)
