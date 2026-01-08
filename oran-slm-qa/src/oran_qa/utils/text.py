import re
from collections import Counter


_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s)
    return s.strip()


def _tokens(s: str):
    return normalize_text(s).split()


def f1_score(pred: str, gold: str) -> float:
    pt = _tokens(pred)
    gt = _tokens(gold)
    if not pt and not gt:
        return 1.0
    if not pt or not gt:
        return 0.0
    pc = Counter(pt)
    gc = Counter(gt)
    common = pc & gc
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pt)
    recall = num_same / len(gt)
    return 2 * precision * recall / (precision + recall)
