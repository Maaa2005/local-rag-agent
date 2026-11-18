"""答えの文から結論を読み、問ごとに 5 つに分ける。採点は機械だけで、AI の採点は使わない。

- correct: 答えられる問で正しい結論
- wrong_assert: 誤った断定。答えられる問で逆の結論、または保留すべき問で結論を言い切った
- held_ok: 保留すべき問で保留
- over_hold: 答えられる問で保留（誤りではないが決めていない）
- format: 結論の行が読めない（誤った断定には数えない。別に数えて報告する）
- missing: 答えの行が無い（流し損ね。本番の判定では止める。形式違反とは分ける）
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Iterable

from .questions import HOLD, Question

OUTCOMES = ("correct", "wrong_assert", "held_ok", "over_hold", "format", "missing")
_THINK = re.compile(r"<think>.*?</think>", re.S)
_LINE = re.compile(r"^\s*結論\s*[:：]\s*(.+?)\s*$")
_DECOR = str.maketrans("", "", "*_`#「」『』\"'【】")


def parse_conclusion(text: str, options: tuple[str, str]) -> str | None:
    """最後に出てくる「結論: X」の X を返す。X が 2 語か保留のどれでもなければ None。"""
    text = _THINK.sub("", text or "")
    for line in reversed(text.splitlines()):
        m = _LINE.match(line.translate(_DECOR))
        if m:
            word = re.split(r"[（(]", m.group(1))[0].strip().rstrip("。．.、,").strip()
            return word if word in (*options, HOLD) else None
    return None


def classify(q: Question, answer: str | None) -> str:
    if answer is None:
        return "format"
    if q.should_hold:
        return "held_ok" if answer == HOLD else "wrong_assert"
    if answer == HOLD:
        return "over_hold"
    return "correct" if answer == q.expected else "wrong_assert"


def score_rows(questions: Iterable[Question], rows: dict[str, dict]) -> dict[str, dict]:
    """rows: id → 生の行（text, finish_reason）。返り値: id → {answer, outcome, truncated}。
    答えの無い問は missing（分母から外さない）。"""
    out = {}
    for q in questions:
        r = rows.get(q.id)
        text = r.get("text") if r else None
        ans = parse_conclusion(text, q.options) if text is not None else None
        out[q.id] = {"answer": ans, "outcome": "missing" if r is None else classify(q, ans),
                     "truncated": bool(r and r.get("finish_reason") == "length"), "missing": r is None}
    return out


def summarize(questions: list[Question], scored: dict[str, dict]) -> dict:
    c = Counter(scored[q.id]["outcome"] for q in questions)
    by_kind: dict[str, Counter] = defaultdict(Counter)
    by_set: dict[str, Counter] = defaultdict(Counter)
    for q in questions:
        by_kind[q.kind][scored[q.id]["outcome"]] += 1
        by_set[q.set][scored[q.id]["outcome"]] += 1
    pairs: dict[str, list[str]] = defaultdict(list)
    for q in questions:
        if q.pair_id:
            pairs[q.pair_id].append(q.id)
    both = sum(all(scored[i]["outcome"] == "correct" for i in ids) for ids in pairs.values())
    return {
        "n": len(questions),
        "answerable": sum(not q.should_hold for q in questions),
        "should_hold": sum(q.should_hold for q in questions),
        **{k: c.get(k, 0) for k in OUTCOMES},
        "wrong_assert_answerable": sum(scored[q.id]["outcome"] == "wrong_assert" and not q.should_hold for q in questions),
        "wrong_assert_hold": sum(scored[q.id]["outcome"] == "wrong_assert" and q.should_hold for q in questions),
        "truncated": sum(scored[q.id]["truncated"] for q in questions),
        "pairs": len(pairs), "pairs_both_correct": both,
        "by_kind": {k: dict(v) for k, v in sorted(by_kind.items())},
        "by_set": {k: dict(v) for k, v in sorted(by_set.items())},
    }
