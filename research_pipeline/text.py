import re

STOP = set("""a an and are as at be but by can do does for from has have how i if in into is it its may more most of on
or our should than that the their there these this to was were what when which who why will with would about after
before between during over under use used using vs versus""".split())


def tokens(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2 and w not in STOP]


def jaccard(a: str, b: str) -> float:
    x, y = set(tokens(a)), set(tokens(b))
    return len(x & y) / len(x | y) if x | y else 0.0


def overlap(query_tokens: set[str], text: str) -> float:
    t = set(tokens(text))
    return len(query_tokens & t) / (len(query_tokens) or 1)
