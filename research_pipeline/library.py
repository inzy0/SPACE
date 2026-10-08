"""Evidence score library: one library per specialty (scope = role id) and a common library per umbrella (scope = "umb:<id>").

Every evidence item that reaches a panel is ingested. Its score is explainable and per scope:
  static  = 0.45 quality + 0.20 study design + 0.10 recency + 0.10 source reliability + 0.15 fit to the scope's terms
  learned = Laplace-smoothed usefulness: how often the item was cited by that specialty in an answer that passed the gate
  overall = (1-w) static + w learned,  w = min(0.5, 0.1 * citations)
Umbrella score = 0.5 best specialty + 0.3 mean of top-3 specialties + 0.2 breadth (how many specialties use it).
Retrieval ranks by query relevance x score, so what the experts found useful gets found first next time.
"""
from __future__ import annotations

import hashlib
import re
import time

from .sources import Evidence
from .text import overlap, tokens

DESIGN = [("meta-analysis", 1.0), ("systematic review", 0.95), ("guideline", 0.9), ("randomized", 0.9), ("randomised", 0.9), ("clinical trial", 0.8),
          ("trial", 0.75), ("cohort", 0.65), ("case-control", 0.55), ("case control", 0.55), ("cross-sectional", 0.45), ("review", 0.6),
          ("case series", 0.4), ("case report", 0.35), ("expert opinion", 0.3), ("editorial", 0.2), ("animal", 0.25), ("in vitro", 0.2)]
SOURCE_REL = {"pubmed": 0.9, "guideline": 0.9, "local": 0.6, "user": 0.5}


def infer_design(text: str):
    low = text.lower()
    for name, w in DESIGN:
        if name in low:
            return name, w
    return "unspecified", 0.5


def recency(year):
    if not year:
        return 0.5
    age = max(0, time.gmtime().tm_year - int(year))
    return 1.0 if age <= 3 else max(0.4, 1.0 - 0.03 * (age - 3))


def umb(uid: str) -> str:
    return "umb:" + uid


class Library:
    def __init__(self, db, taxonomy):
        self.db, self.tax = db, taxonomy
        self.db.executescript("""
        create table if not exists lib_items(id integer primary key, key text unique, title text, text text, source text, url text, year integer,
            design text, quality real, added real);
        create table if not exists lib_scope(item integer, scope text, cited integer default 0, accepted integer default 0, comp_sum real default 0,
            primary key(item, scope));
        """)
        self.kw = {}
        for u in taxonomy.values():
            self.kw[umb(u.id)] = set(tokens(" ".join(u.keywords + u.body_areas)))
            for s in u.specialties:
                self.kw[s.id] = set(tokens(" ".join(s.keywords + s.body_areas + (s.name,))))

    # ---- write ---------------------------------------------------------------------------------------------------
    def ingest(self, ev: Evidence, scopes) -> int:
        key = hashlib.sha1(re.sub(r"\W+", "", ev.title.lower()).encode()).hexdigest()
        design, _ = infer_design(ev.title + " " + ev.text)
        row = self.db.execute("select id, quality from lib_items where key=?", (key,)).fetchone()
        if row:
            iid = row[0]
            if ev.quality > row[1]:
                self.db.execute("update lib_items set quality=? where id=?", (ev.quality, iid))
        else:
            iid = self.db.execute("insert into lib_items values(null,?,?,?,?,?,?,?,?,?)", (key, ev.title, ev.text, ev.source, ev.url, ev.year, design, ev.quality, time.time())).lastrowid
        for s in scopes:
            self.db.execute("insert or ignore into lib_scope(item, scope) values(?,?)", (iid, s))
        self.db.commit()
        ev.design = design
        return iid

    def feedback(self, cited: dict, accepted: bool, composite: float):
        """cited: {scope: set(item ids)} -- items an expert (scope) relied on in this run."""
        for scope, ids in cited.items():
            for i in ids:
                self.db.execute("insert or ignore into lib_scope(item, scope) values(?,?)", (i, scope))
                self.db.execute("update lib_scope set cited=cited+1, accepted=accepted+?, comp_sum=comp_sum+? where item=? and scope=?",
                                (1 if accepted else 0, composite if accepted else 0.0, i, scope))
        self.db.commit()

    # ---- score ---------------------------------------------------------------------------------------------------
    def _item(self, iid):
        return self.db.execute("select id, title, text, source, url, year, design, quality from lib_items where id=?", (iid,)).fetchone()

    def _static(self, it, scope):
        _, title, text, source, _, year, design, quality = it
        dw = dict(DESIGN).get(design, 0.5)
        sr = SOURCE_REL.get(source.split(":")[0], 0.5)
        fit = min(1.0, overlap(self.kw.get(scope, set()), title + " " + text) * 3)
        parts = dict(quality=quality, design=dw, recency=recency(year), source=sr, fit=fit)
        return 0.45 * quality + 0.20 * dw + 0.10 * parts["recency"] + 0.10 * sr + 0.15 * fit, parts

    def score(self, iid, scope) -> dict:
        it = self._item(iid)
        if scope.startswith("umb:"):
            return self._umbrella(it, scope[4:])
        static, parts = self._static(it, scope)
        row = self.db.execute("select cited, comp_sum from lib_scope where item=? and scope=?", (iid, scope)).fetchone() or (0, 0.0)
        cited, comp = row
        learned = (comp + 2 * static) / (cited + 2)  # prior = the static score, worth two citations
        w = min(0.5, 0.1 * cited)
        return dict(overall=round((1 - w) * static + w * learned, 3), static=round(static, 3), learned=round(learned, 3), cited=cited, parts={k: round(v, 2) for k, v in parts.items()})

    def _umbrella(self, it, uid):
        ids = [s.id for s in self.tax[uid].specialties]
        scores, cited, users = [], 0, 0
        for sid in ids:
            row = self.db.execute("select cited, comp_sum from lib_scope where item=? and scope=?", (it[0], sid)).fetchone()
            if row is None:
                continue
            users += 1
            static, _ = self._static(it, sid)
            c, comp = row
            w = min(0.5, 0.1 * c)
            scores.append((1 - w) * static + w * (comp + 2 * static) / (c + 2))
            cited += c
        ustatic, parts = self._static(it, umb(uid))
        if not scores:
            return dict(overall=round(ustatic, 3), static=round(ustatic, 3), learned=round(ustatic, 3), cited=0, specialties=0, parts={k: round(v, 2) for k, v in parts.items()})
        top = sorted(scores, reverse=True)
        overall = 0.5 * top[0] + 0.3 * (sum(top[:3]) / len(top[:3])) + 0.2 * min(1.0, users / 3)
        return dict(overall=round(overall, 3), static=round(ustatic, 3), learned=round(top[0], 3), cited=cited, specialties=users, parts={k: round(v, 2) for k, v in parts.items()})

    # ---- read ----------------------------------------------------------------------------------------------------
    def size(self):
        return self.db.execute("select count(*) from lib_items").fetchone()[0]

    def search(self, query: str, scopes, n=8) -> list[Evidence]:
        """Items relevant to the query, ranked by relevance x best score across the given scopes (specialties + umbrella)."""
        qt = set(tokens(query))
        out = []
        for iid, title, text, source, url, year, design, quality in self.db.execute("select id, title, text, source, url, year, design, quality from lib_items"):
            rel = overlap(qt, title + " " + text)
            if rel <= 0:
                continue
            sc = {s: self.score(iid, s)["overall"] for s in scopes}
            out.append((rel * (0.4 + max(sc.values() or [0.5])), Evidence("", title, text, "library:" + source, quality, url, year), sc, design))
        out.sort(key=lambda x: -x[0])
        res = []
        for _, e, sc, design in out[:n]:
            e.lib, e.design = sc, design
            res.append(e)
        return res

    def report(self, scope: str, limit=25) -> dict:
        """The library for a scope: a specialty library or an umbrella's common library with a per-specialty breakdown."""
        if scope.startswith("umb:"):
            uid = scope[4:]
            ids = [s.id for s in self.tax[uid].specialties]
            q = ",".join("?" * len(ids))
            items = [r[0] for r in self.db.execute(f"select distinct item from lib_scope where scope in ({q}) or scope=?", (*ids, scope))]
        else:
            items = [r[0] for r in self.db.execute("select item from lib_scope where scope=?", (scope,))]
        rows = []
        for iid in items:
            it = self._item(iid)
            sc = self.score(iid, scope)
            row = dict(id=iid, title=it[1], snippet=it[2][:200], source=it[3], year=it[5], design=it[6], quality=it[7], **sc)
            if scope.startswith("umb:"):
                row["by_specialty"] = sorted(({"id": sid, "score": self.score(iid, sid)["overall"]} for (sid,) in self.db.execute(
                    "select scope from lib_scope where item=? and scope not like 'umb:%' and scope like ?", (iid, scope[4:] + ".%"))), key=lambda x: -x["score"])[:5]
            rows.append(row)
        rows.sort(key=lambda r: -r["overall"])
        return dict(scope=scope, count=len(rows), items=rows[:limit], total_library=self.size())

    def overview(self) -> list[dict]:
        out = []
        for u in self.tax.values():
            n = self.db.execute("select count(distinct item) from lib_scope where scope like ?", (u.id + ".%",)).fetchone()[0]
            if n:
                out.append(dict(scope=umb(u.id), name=u.name, items=n))
        return out
