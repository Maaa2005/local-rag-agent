"""学習用・開発用の問と封（A・B・C）の問の意味の近さ（docs/verified-distill-plan.md §1 の 5）。

cl-nagoya/ruri-v3-310m（接頭辞なし＝意味の近さの使い方）で、問の本文（答えの形の指示の行を除く）を埋め込み、
封の各問との余弦の近さの最大が 0.90 以上の問を捨てる。分布も返す。重いので Colab で動かす:

    python -m evaluation.distill.near --rows rows.jsonl --out near.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from evaluation.distill.gen.leak import REPO, content_lines

MODEL = "cl-nagoya/ruri-v3-310m"
LIMIT = 0.90
SEALED_QUESTIONS = ("evaluation/dataset.jsonl", "evaluation/v2/numeric_boundary.jsonl",
                    "evaluation/teacher_check/sealed/questions.jsonl", "evaluation/distill/sealed_c/sealed/questions.jsonl")


_FORMAT = re.compile(r"(最後の行には|答えの最後の2行).*$")


def body(question: str) -> str:
    """答えの形の指示（行の途中から始まる物も）を除いた問の本文。"""
    lines = (_FORMAT.sub("", x).strip() for x in content_lines(question))
    return "\n".join(x for x in lines if x)


def sealed_bodies() -> list[str]:
    out = []
    for q in SEALED_QUESTIONS:
        for line in (REPO / q).read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(body(json.loads(line)["question"]))
    return out


def summary(maxima: list[float]) -> dict:
    s = sorted(maxima)
    pick = lambda p: round(s[min(len(s) - 1, int(p * len(s)))], 4)
    return {"n": len(s), "p50": pick(0.5), "p90": pick(0.9), "p99": pick(0.99), "max": round(s[-1], 4),
            "over": sum(v >= LIMIT for v in s)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from sentence_transformers import SentenceTransformer

    rows = [json.loads(line) for line in a.rows.read_text(encoding="utf-8").splitlines() if line.strip()]
    model = SentenceTransformer(MODEL)
    sealed = model.encode(sealed_bodies(), normalize_embeddings=True, convert_to_tensor=True)
    mine = model.encode([body(r["question"]) for r in rows], normalize_embeddings=True, convert_to_tensor=True)
    sims = (mine @ sealed.T).max(dim=1).values.tolist()
    res = {"model": MODEL, "limit": LIMIT, "summary": summary(sims),
           "drop": [r["id"] for r, v in zip(rows, sims) if v >= LIMIT],
           "max_by_id": {r["id"]: round(v, 4) for r, v in zip(rows, sims)}}
    a.out.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    print("NEAR_JSON " + json.dumps({"summary": res["summary"], "drop": res["drop"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
