"""教師の答えの採点（測る前に固定・life #24）。

合格の条件は 1 問ごとに次の 2 つを両方満たすこと（コードで照合し、人やモデルの採点を入れない）:
- 最後の「結論: X」行の X が正解と完全一致（evaluation.metrics.final_conclusion と同じ読み方）
- 最後の「根拠:」行に、正解の条（expected_articles）のどれか 1 つ以上が「第N条」の形で入っている
  （漢数字の 第一条〜第十条 も読む）

回答に失敗した問（error あり・本文なし）は不合格に数える。
"""
from __future__ import annotations

import re

from evaluation.metrics import final_conclusion

_GROUND_LINE = re.compile(r"^[\s*#>\-]*根拠\s*[:：]\s*(.*)$")
_ARTICLE = re.compile(r"第\s*([0-9０-９]+|[一二三四五六七八九十]+)\s*条")
_KANJI = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _to_int(s: str) -> int | None:
    s = s.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    if s.isdigit():
        return int(s)
    if s == "十":
        return 10
    if s.startswith("十") and len(s) == 2 and s[1] in _KANJI:
        return 10 + _KANJI[s[1]]
    if len(s) == 1 and s in _KANJI:
        return _KANJI[s]
    return None


def cited_articles(answer: str) -> list[int] | None:
    """最後の「根拠:」行に書かれた条の番号。根拠の行が無ければ None。"""
    for line in reversed(answer.splitlines()):
        m = _GROUND_LINE.match(line)
        if m:
            return [n for n in (_to_int(x) for x in _ARTICLE.findall(m.group(1))) if n is not None]
    return None


def grade(answer: str | None, expected_conclusion: str, expected_articles: list[int]) -> dict:
    if not answer:
        return {"conclusion": None, "articles": None, "conclusion_ok": False, "articles_ok": False, "ok": False}
    got = final_conclusion(answer)
    arts = cited_articles(answer)
    c_ok = got == expected_conclusion
    a_ok = bool(arts) and any(a in expected_articles for a in arts)
    return {"conclusion": got, "articles": arts, "conclusion_ok": c_ok, "articles_ok": a_ok, "ok": c_ok and a_ok}
