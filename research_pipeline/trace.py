"""Run trace: every pipeline step records its inputs and outputs so a run can be replayed visually (viewer.py)."""
from __future__ import annotations

import time


class Tracer:
    def __init__(self):
        self.events: list[dict] = []
        self.t0 = time.time()

    def emit(self, node, label, lane, after=(), inputs=None, outputs=None, status="ok", note="", score=None):
        """lane = column in the graph (0 inputs ... 9 outputs); after = upstream node ids (edges)."""
        self.events.append(dict(seq=len(self.events), t=round(time.time() - self.t0, 3), node=node, label=label, lane=lane,
                                after=list(after), inputs=inputs or {}, outputs=outputs or {}, status=status, note=note, score=score))
        return node


class NullTracer(Tracer):
    def emit(self, *a, **k):
        return k.get("node") or a[0]
