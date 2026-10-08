"""SQLite memory + the 'training' signal: every attempt is scored and kept, expert weights adapt, and
accepted/rejected pairs are exportable for few-shot memory or fine-tuning."""
from __future__ import annotations

import json
import sqlite3
import time


class Store:
    def __init__(self, path=":memory:"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript("""
        create table if not exists runs(id integer primary key, ts real, question text, umbrella text, parent integer, depth integer,
            final_answer text, composite real, status text, attempts integer, experts text);
        create table if not exists attempts(id integer primary key, run integer, n integer, answer text, composite real, dims text,
            failing text, feedback text, evidence text);
        create table if not exists expert_stats(expert text, umbrella text, n integer, mean real, primary key(expert, umbrella));
        """)

    def log_run(self, question, umbrella, parent, depth, answer, composite, status, attempts, experts, attempt_rows):
        c = self.db.execute("insert into runs values(null,?,?,?,?,?,?,?,?,?,?)",
                            (time.time(), question, umbrella, parent, depth, answer, composite, status, attempts, json.dumps(experts)))
        rid = c.lastrowid
        for r in attempt_rows:
            self.db.execute("insert into attempts values(null,?,?,?,?,?,?,?,?)", (rid, *r))
        for e in experts:  # running mean of composite for each participating expert
            row = self.db.execute("select n, mean from expert_stats where expert=? and umbrella=?", (e, umbrella)).fetchone()
            n, m = row or (0, 0.0)
            self.db.execute("insert or replace into expert_stats values(?,?,?,?)", (e, umbrella, n + 1, (m * n + composite) / (n + 1)))
        self.db.commit()
        return rid

    def expert_weights(self, umbrella):
        """0.7-1.3 multiplier from smoothed mean score; unseen experts = 1.0 (shrunk toward 0.7 prior)."""
        out = {}
        for e, n, m in self.db.execute("select expert, n, mean from expert_stats where umbrella=?", (umbrella,)):
            out[e] = 0.7 + 0.6 * ((m * n + 0.7 * 3) / (n + 3))
        return out

    def best_examples(self, umbrella, k=2, min_score=0.8):
        return [r[0] for r in self.db.execute(
            "select final_answer from runs where umbrella=? and composite>=? and status='accepted' order by composite desc limit ?",
            (umbrella, min_score, k))]

    def export_sft(self, path, min_score=0.8):
        n = 0
        with open(path, "w") as f:
            for q, a, c, u in self.db.execute("select question, final_answer, composite, umbrella from runs where status='accepted' and composite>=?", (min_score,)):
                f.write(json.dumps(dict(question=q, answer=a, score=c, umbrella=u)) + "\n"); n += 1
        return n

    def export_preferences(self, path):
        """(rejected earlier attempt, accepted later attempt) for the same question = preference pair."""
        n = 0
        with open(path, "w") as f:
            for rid, q in self.db.execute("select id, question from runs where status='accepted' and attempts>1"):
                rows = self.db.execute("select answer, composite, feedback from attempts where run=? order by n", (rid,)).fetchall()
                f.write(json.dumps(dict(question=q, rejected=rows[0][0], chosen=rows[-1][0], feedback=rows[0][2],
                                        margin=round(rows[-1][1] - rows[0][1], 3))) + "\n"); n += 1
        return n

    def hard_questions(self, limit=20):
        return [r[0] for r in self.db.execute("select question from runs where status!='accepted' order by composite limit ?", (limit,))]

    def stats(self):
        r = self.db.execute("select count(*), avg(composite), sum(status='accepted'), avg(attempts) from runs").fetchone()
        return dict(runs=r[0], mean_score=round(r[1] or 0, 3), accepted=r[2] or 0, mean_attempts=round(r[3] or 0, 2),
                    experts=[dict(expert=e, umbrella=u, n=n, mean=round(m, 3)) for e, u, n, m in
                             self.db.execute("select * from expert_stats order by mean desc limit 15")])
