"""教師の答えを残すかどうかの照合（docs/verified-distill-plan.md §3）。すべてコードで決める。

1. 結論: 最後の「結論: X」が正解と完全一致（断りの問は metrics.is_refusal が真、資料の文を答える問は
   正解の語句をすべて含み、断りでない）
2. 引用: 規程の問と資料の文を答える問は、答えの条を含む塊の番号 [n] を引用している。断りの問は引用を求めない
   代わりに、断りの問の答えに数字（参考情報の値）を書いていない
3. 形: 上限で切れていない・300 トークン以下・日本語を含む
4. 境目の問の理由: 本文に問の値と境目の値の書き方が両方出る。境目ちょうどの問で、境目を越えた側と書く語が無い
5. 重複: 同じ問の 2 つの答えの似かた（difflib の比）が 0.9 以上なら 1 つだけ残す
"""
from __future__ import annotations

import difflib
import re

from evaluation.metrics import final_conclusion, is_refusal

MAX_TOKENS_KEPT = 300
DUP_RATIO = 0.9
_JA = re.compile(r"[぀-ヿ一-鿿]")
_DIGIT = re.compile(r"[0-9０-９]")
# 境目ちょうどの時に書いてはいけない語（境目を越えた側と言っている）。語ごとに、ちょうどの値は越えていない
WRONG_AT = {
    "超える": ("超えている", "超えています", "超えており", "超えるため", "超えます"),
    "未満": ("未満です", "未満であり", "未満のため", "未満となり"),
    "を下回る": ("下回っている", "下回っています", "下回るため", "下回ります"),
    "に満たない": ("満たないため", "満たない」に該当", "満たないので", "満たない条件に該当", "満たないことになり"),
}


def cited(answer: str) -> set[int]:
    return {int(x) for x in re.findall(r"\[(\d+)\]", answer)}


def accept(row: dict, answer: str, finish_reason: str | None, completion_tokens: int | None) -> tuple[bool, str]:
    """(残すか, 理由)。理由は捨てた時の一番最初の理由。"""
    if not answer or finish_reason == "length":
        return False, "切れた・空"
    if completion_tokens is not None and completion_tokens > MAX_TOKENS_KEPT:
        return False, "長すぎる"
    if not _JA.search(answer):
        return False, "日本語でない"
    cat = row["category"]
    if cat == "refusal":
        if not is_refusal(answer):
            return False, "断っていない"
        if _DIGIT.search(re.sub(r"\[\d+\]", "", answer)):
            return False, "断りに数字"
        return True, "ok"
    if cat == "describe":
        if is_refusal(answer) or not all(k in answer for k in row["keywords"]):
            return False, "説明の語が無い"
        if not cited(answer):
            return False, "引用なし"
        return True, "ok"
    if final_conclusion(answer) != row["expected"]:
        return False, "結論が違う"
    if not (cited(answer) & set(row["rule_chunk_ids"])):
        return False, "条の引用なし"
    chk = row.get("check") or {}
    if cat in ("boundary", "conversion") and chk.get("value_text") and chk.get("threshold_text"):
        if chk["value_text"] not in answer or chk["threshold_text"] not in answer:
            return False, "値か境目を書いていない"
        if chk.get("at") and any(w in answer for w in WRONG_AT.get(chk.get("word") or "", ())):
            return False, "ちょうどを越えた側と書いた"
    return True, "ok"


def dedupe(answers: list[str]) -> list[str]:
    kept: list[str] = []
    for a in answers:
        if all(difflib.SequenceMatcher(None, a, k).ratio() < DUP_RATIO for k in kept):
            kept.append(a)
    return kept
