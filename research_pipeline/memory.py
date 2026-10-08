"""Optional Cognee graph memory (https://github.com/topoteretes/cognee). Off by default, never required, never fatal.

    pip install cognee            # then tick "Cognee graph memory" in the UI, or pass --cognee

What it does when enabled:
  * after an ACCEPTED answer the question, answer and the evidence the experts relied on are `remember`-ed into one Cognee dataset per
    umbrella (`<prefix>_<umbrella>`): a graph-backed common library for that umbrella that survives across sessions;
  * before gathering, `recall` over that dataset is added as one more evidence source (source = "cognee:<dataset>").
Any import / runtime / dataset error is swallowed and reported in the trace, so the pipeline behaves exactly as without it.
Cognee API used (see the repo README): `await cognee.remember(text, dataset_name=...)`, `await cognee.recall(query, datasets=[...], top_k=n)`.
"""
from __future__ import annotations

import asyncio
import importlib
import threading

from .sources import Evidence


def _text(r) -> str:
    if isinstance(r, str):
        return r
    if isinstance(r, dict):
        return str(r.get("text") or r.get("content") or r.get("search_result") or r)
    return str(getattr(r, "text", None) or r)


class CogneeMemory:
    def __init__(self, prefix="healthcare", module=None):
        self.prefix, self._mod, self._tried = prefix, module, module is not None
        self.error = ""
        self.lock = threading.Lock()  # cognee holds global state; serialise our calls

    @property
    def cognee(self):
        if not self._tried:
            self._tried = True
            try:
                self._mod = importlib.import_module("cognee")
            except Exception as e:  # not installed / broken install
                self._mod, self.error = None, f"{type(e).__name__}: {e}"
        return self._mod

    def available(self) -> bool:
        return self.cognee is not None

    def dataset(self, umbrella: str) -> str:
        return f"{self.prefix}_{umbrella}"

    def _run(self, coro):
        with self.lock:
            return asyncio.run(coro)

    def remember(self, text: str, umbrella: str) -> bool:
        if not self.available():
            return False
        try:
            self._run(self.cognee.remember(text, dataset_name=self.dataset(umbrella)))
            return True
        except Exception as e:
            self.error = f"remember: {type(e).__name__}: {e}"
            return False

    def recall(self, query: str, umbrella: str, n=5) -> list[Evidence]:
        if not self.available():
            return []
        try:
            res = self._run(self.cognee.recall(query, datasets=[self.dataset(umbrella)], top_k=n))
        except Exception as e:  # e.g. the dataset does not exist yet
            self.error = f"recall: {type(e).__name__}: {e}"
            return []
        out = []
        for r in list(res or [])[:n]:
            t = _text(r).strip()
            if t:
                out.append(Evidence("", t.split("\n")[0][:90], t[:900], f"cognee:{self.dataset(umbrella)}", 0.5))
        return out


class CogneeSource:
    """Adapter so recall() can sit in the normal list of evidence sources."""

    def __init__(self, memory: CogneeMemory, umbrella: str):
        self.memory, self.umbrella = memory, umbrella

    def search(self, query, n=8):
        return self.memory.recall(query, self.umbrella, n)
