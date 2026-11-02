"""学習用・開発用と封（A・B・C）の重なりを機械で確かめる（docs/verified-distill-plan.md §1）。

- 題材: 家族の規程名・量の名前に、封と今までの評価の題材の語が入っていない
- 境目: 家族の境目の値（正規化）が封の境目と一致しない。割合 4/5・1/2・1/3 と期限の言い回しを使わない
- 13 文字の並び: NFKC で正規化した本文が、封の本文と 13 文字以上続けて一致しない（許す言い回しを除く）
意味の近さ（ruri-v3）は重いので、別の段（Colab）で測る。
"""
from __future__ import annotations

import json
import re
import unicodedata
from fractions import Fraction
from pathlib import Path

from evaluation.distill.gen.families import FAMILIES
from evaluation.distill.gen.spec import BANNED_DEADLINE_PHRASES, threshold_allowed, topic_allowed

REPO = Path(__file__).resolve().parents[3]
N = 13

# 決まった言い回し（本番の指示文・答えの形の指示・規程の決まった書き出し）。これらの中の並びは一致しても許す
ALLOWED_PHRASES = (
    "あなたは社内情報専門のアシスタントです。", "以下の【参考情報】のみを根拠として質問に回答してください。",
    "参考情報に含まれていない情報については「その情報は現在の資料には含まれていません」と回答してください。",
    "回答は必ず日本語で行ってください。", "回答中で根拠にした参考情報には、該当箇所の文末に [1] のようにその番号を付けて引用してください。",
    "最後の行には", "のどちらかだけを書いてください。", "のどれかだけを書いてください。", "「結論: 判断できない」",
    "第1条（目的）\nこの規程は、", "の場合は、この限りでない。", "次のすべてに当てはまる", "次のいずれかに当てはまる",
    "について定める。", "の決まりを定める。", "【参考情報】", "【質問】",
    "の場合は、この限りでない。", "この規程は、社員", "は、次のすべてに当てはまる", "は、次のいずれかに当てはまる",
)


def norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


_SKIP = re.compile(r"^(#|\[\d+\] |第\d+条（|第\d+条\(|【参考情報】|【質問】|最後の行には|答えの最後の2行|根拠: |結論: )")


def content_lines(text: str) -> list[str]:
    """見出し・参考情報の番号の行・答えの形の指示の行を除いた、内容の行（NFKC）。"""
    out = []
    for line in norm(text).splitlines():
        line = line.strip()
        if line and not _SKIP.match(line) and line not in ALLOWED_PHRASES:
            out.append(line)
    return out


def grams(s: str, n: int = N) -> set[str]:
    """行ごとの n 文字の並び（行をまたがない）。"""
    out: set[str] = set()
    for line in content_lines(s):
        out |= {line[i:i + n] for i in range(len(line) - n + 1)}
    return out


def allowed_grams() -> set[str]:
    out: set[str] = set()
    for p in ALLOWED_PHRASES:
        p = norm(p)
        out |= {p[i:i + N] for i in range(len(p) - N + 1)}
    return out


def sealed_texts() -> list[str]:
    texts = []
    for d in ("evaluation/corpus", "evaluation/teacher_check/corpus", "evaluation/distill/sealed_c/corpus"):
        texts += [p.read_text(encoding="utf-8") for p in sorted((REPO / d).rglob("*.md"))]
    for q in ("evaluation/dataset.jsonl", "evaluation/v2/numeric_boundary.jsonl",
              "evaluation/teacher_check/sealed/questions.jsonl", "evaluation/distill/sealed_c/sealed/questions.jsonl"):
        for line in (REPO / q).read_text(encoding="utf-8").splitlines():
            if line.strip():
                texts.append(json.loads(line)["question"])
    return texts


def sealed_grams() -> set[str]:
    out: set[str] = set()
    for t in sealed_texts():
        out |= grams(t)
    return out - allowed_grams()


def family_thresholds(f: dict) -> list[tuple[object, str]]:
    t = f["type"]
    if t in ("single", "exception"):
        return [(f["t"], f["unit"])]
    if t == "range":
        return [(f["lo"][0], f["unit"]), (f["hi"][0], f["unit"])]
    if t in ("and", "or"):
        return [(c["t"], c["unit"]) for c in (f["c1"], f["c2"])]
    if t == "ratio":
        return [(f["t"][0], f["t"][1])]
    if t == "deadline":
        return [(f["n"], "日")]
    if t == "age":
        return [(f["t"], "年" if f["basis"] == "entry" else "歳")]
    if t == "tier":
        return [(f["a"], f["unit"]), (f["b"], f["unit"])]
    raise ValueError(t)


def family_problems() -> list[str]:
    bad = []
    for f in FAMILIES:
        texts = [f["title"], f.get("subj", ""), f.get("part", ""), f.get("whole", ""), f.get("event", ""), f["purpose"]]
        texts += [c["subj"] for c in (f.get("c1"), f.get("c2")) if c]
        if not all(topic_allowed(x) for x in texts):
            bad.append(f"{f['id']}: 題材の語")
        for v, u in family_thresholds(f):
            if not threshold_allowed(Fraction(v), u):
                bad.append(f"{f['id']}: 境目 {v}{u}")
        rendered = json.dumps(f, ensure_ascii=False)
        if any(p in rendered for p in BANNED_DEADLINE_PHRASES):
            bad.append(f"{f['id']}: 期限の言い回し")
    return bad


def overlap(text: str, sealed: set[str]) -> list[str]:
    return sorted(grams(text) & sealed)
