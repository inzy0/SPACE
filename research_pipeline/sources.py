"""Evidence sources. Anything with `search(query, n) -> list[Evidence]` works."""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .text import overlap, tokens


@dataclass
class Evidence:
    id: str  # assigned per run: E1, E2, ...
    title: str
    text: str
    source: str
    quality: float = 0.5  # 0..1 study-design weight
    url: str = ""
    year: int | None = None
    design: str = ""
    lib_id: int = 0
    lib: dict = field(default_factory=dict)  # scope -> library score (specialty ids and umb:<id>)

    def to_dict(self):
        return asdict(self)


class LocalCorpus:
    """Folder of .txt/.md files (first line = title). Optional `quality: 0.9` line anywhere sets the weight."""

    def __init__(self, folder):
        self.docs = []
        for p in sorted(Path(folder).rglob("*")):
            if p.suffix.lower() in (".txt", ".md") and p.is_file():
                raw = p.read_text(errors="ignore")
                q = re.search(r"^quality:\s*([\d.]+)\s*$", raw, re.M)
                body = re.sub(r"^quality:.*$", "", raw, flags=re.M).strip()
                title, _, rest = body.partition("\n")
                self.docs.append((title.strip("# ").strip(), rest.strip() or title, p.name, float(q.group(1)) if q else 0.5))

    def search(self, query, n=8):
        qt = set(tokens(query))
        ranked = sorted(self.docs, key=lambda d: -overlap(qt, d[0] + " " + d[1]))
        return [Evidence("", t, x[:900], f"local:{f}", q) for t, x, f, q in ranked[:n] if overlap(qt, t + " " + x) > 0]


class PubMed:
    WEIGHT = {"meta-analysis": 1.0, "systematic review": 0.95, "randomized controlled trial": 0.9, "clinical trial": 0.8,
              "review": 0.65, "case reports": 0.35, "editorial": 0.2}
    BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    def search(self, query, n=8):
        try:
            ids = json.load(urllib.request.urlopen(self.BASE + "esearch.fcgi?" + urllib.parse.urlencode(
                {"db": "pubmed", "term": query, "retmax": n, "retmode": "json", "sort": "relevance"}), timeout=20))["esearchresult"]["idlist"]
            if not ids:
                return []
            xml = urllib.request.urlopen(self.BASE + "efetch.fcgi?" + urllib.parse.urlencode(
                {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}), timeout=30).read()
        except Exception:
            return []  # offline / blocked: pipeline degrades to other sources
        out = []
        for a in ET.fromstring(xml).iter("PubmedArticle"):
            title = "".join(a.find(".//ArticleTitle").itertext()) if a.find(".//ArticleTitle") is not None else ""
            abstract = " ".join("".join(t.itertext()) for t in a.iter("AbstractText"))
            types = [("".join(t.itertext())).lower() for t in a.iter("PublicationType")]
            q = max([w for k, w in self.WEIGHT.items() if any(k in t for t in types)] or [0.5])
            pmid = a.findtext(".//PMID")
            yr = a.findtext(".//PubDate/Year")
            if abstract:
                out.append(Evidence("", title, abstract[:900], f"pubmed:{pmid}", q, f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", int(yr) if yr and yr.isdigit() else None))
        return out


class Inline:
    """Knowledge pasted/uploaded by the user: [{title, text, quality}]."""

    def __init__(self, docs):
        self.docs = [(d.get("title") or d["text"][:60], d["text"], "user", float(d.get("quality", 0.5))) for d in docs if d.get("text", "").strip()]

    search = LocalCorpus.search


def gather(sources, query, n=8, min_quality=0.0):
    """Query every source, dedupe by title, rank by quality, assign stable ids E1..En."""
    seen, pool = set(), []
    for s in sources:
        for e in s.search(query, n):
            if e.quality < min_quality:
                continue
            k = re.sub(r"\W+", "", e.title.lower())
            if k not in seen:
                seen.add(k)
                pool.append(e)
    pool.sort(key=lambda e: -e.quality)
    for i, e in enumerate(pool[:n], 1):
        e.id = f"E{i}"
    return pool[:n]
