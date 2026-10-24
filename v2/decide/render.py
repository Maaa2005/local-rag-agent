"""答えの文の組み立て（型は固定）。モデルは結論も説明も書かない。

中身は 規程の引用 / 質問から取り出した値と役割 / 計算と真偽 / 結論 だけ。
結論が「規程にそう書いてある側」でなく「その裏として推論した側」の時は、その旨を文に書く。
最後の行は必ず「結論: <選択肢>」の 1 行（engine が最後の行を検査する）。
"""
from __future__ import annotations

from typing import Any

from . import parse
from .records import Outcome, Record

_SYM = {">": ">", ">=": "≥", "<": "<", "<=": "≤"}


def doc_label(rec: Record) -> str:
    name = rec.doc_path.rsplit("/", 1)[-1]
    return name[:-3] if name.endswith(".md") else name


def render(rec: Record, dr: dict[str, Any], out: Outcome, key: str, assume: list[Any]) -> str:
    doc = doc_label(rec)
    lines: list[str] = ["【規程にもとづく判定】" + doc]
    lines.append("規程の記載:")
    seen = set()
    for c in dr["computation"]:
        if (c["quote"], c["line"]) in seen:
            continue
        seen.add((c["quote"], c["line"]))
        lines.append(f"- 「{c['quote']}」（{doc} {c['line']}行目）")
    lines.append("質問から取り出した値:")
    for e in dr["extracted"]:
        srcs = "・".join(dict.fromkeys(s["text"] for s in e["sources"]))
        lines.append(f"- {e['label']}: {e['value']}（質問の「{srcs}」）")
    lines.append("計算:")
    for c in dr["computation"]:
        lines.append(f"- {c['label']} {c['value']} {_SYM[c['op']]} {c['threshold']} → "
                     + ("成り立つ" if c["result"] else "成り立たない"))
    lines.append("全部の条件が成り立つ" if key == "all_true" else "条件のどれかが成り立たない")
    if out.basis == "explicit":
        lines[-1] += f"ので、規程に書かれた結論は「{out.value}」です。"
    else:
        lines[-1] += (f"ので、結論は「{out.value}」です。ただし規程に直接書かれているのは"
                      f"反対側の場合だけで、この側の結論はその裏として推論したものです。")
    if assume:
        descs = "・".join(o.description for o in assume)
        lines.append(f"（{descs}など）他の要件は満たしている前提です。")
    lines.append(f"結論: {out.value}")
    return "\n".join(lines)
