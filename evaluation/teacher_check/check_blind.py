"""確認役が正解を見ずに答えた結果と、コードの正解を照合する（封の前の点検）。

    python -m evaluation.teacher_check.check_blind <確認役の出力.md>

確認役の出力は `T001|第3条|結論` の行（1 問 1 行）。結論の食い違いと、根拠の条が正解の条と 1 つも重ならない問を出す。
食い違いが 0 件になるまで問か規程を直し、直した後は確認役に全問を解き直させる（直した問だけ差し替えない）。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from evaluation.teacher_check.questions import build
from evaluation.teacher_check.score import _ARTICLE, _to_int

_LINE = re.compile(r"^\s*(T\d{3})\s*\|\s*([^|]*)\|\s*(.+?)\s*$")


def parse(text: str) -> dict[str, dict]:
    out = {}
    for line in text.splitlines():
        m = _LINE.match(line)
        if m:
            arts = [n for n in (_to_int(x) for x in _ARTICLE.findall(m.group(2))) if n is not None]
            out[m.group(1)] = {"articles": arts, "conclusion": m.group(3).strip(" 「」*")}
    return out


def compare(text: str) -> dict:
    got = parse(text)
    rows = build()
    missing = [r["id"] for r in rows if r["id"] not in got]
    conclusion_diff = [(r["id"], r["expected_conclusion"], got[r["id"]]["conclusion"]) for r in rows
                       if r["id"] in got and got[r["id"]]["conclusion"] != r["expected_conclusion"]]
    article_diff = [(r["id"], r["expected_articles"], got[r["id"]]["articles"]) for r in rows
                    if r["id"] in got and not set(got[r["id"]]["articles"]) & set(r["expected_articles"])]
    return {"n": len(rows), "answered": len(got), "missing": missing,
            "conclusion_diff": conclusion_diff, "article_diff": article_diff}


def main() -> None:
    res = compare(Path(sys.argv[1]).read_text(encoding="utf-8"))
    print(f"答えた問 {res['answered']}/{res['n']}・答えが無い問 {res['missing']}")
    print(f"結論の食い違い {len(res['conclusion_diff'])} 件")
    for i, exp, got in res["conclusion_diff"]:
        print(f"  {i}: 正解 {exp} / 確認役 {got}")
    print(f"根拠の条が重ならない {len(res['article_diff'])} 件")
    for i, exp, got in res["article_diff"]:
        print(f"  {i}: 正解の条 {exp} / 確認役 {got}")


if __name__ == "__main__":
    main()
