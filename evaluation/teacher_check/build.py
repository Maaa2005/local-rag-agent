"""教師の確認の問題集を書き出す。

    python -m evaluation.teacher_check.build --out evaluation/teacher_check/sealed
    python -m evaluation.teacher_check.build --blind <dir>   # 確認役に渡す分（正解・facts を入れない）

書くもの（--out）:
- questions.jsonl: 96 問（正解・facts・stage1 つき）
- manifest.json: 問題集と 6 規程の sha256。封をした後に 1 文字でも変われば照合で分かる
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from evaluation.teacher_check.questions import build

CORPUS = Path(__file__).resolve().parent / "corpus"
SYSTEM_PROMPT = ("あなたは会社の規程の担当者です。次の規程だけを根拠にして、社員からの質問に答えてください。"
                 "規程に書かれていない事を前提にしてはいけません。判断に必要な事実が質問に無く、結論が決まらない時は「判断できない」と答えてください。")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def doc_text(doc: str) -> str:
    return (CORPUS / f"{doc}.md").read_text(encoding="utf-8")


def prompt_messages(row: dict) -> list[dict]:
    """教師に渡す会話。検索は使わず、その問の規程の全文を渡す。"""
    return [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + doc_text(row["doc"])},
            {"role": "user", "content": row["question"]}]


def _dump(rows: list[dict], path: Path) -> bytes:
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    path.write_bytes(data)
    return data


def write_sealed(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rows = build()
    data = _dump(rows, out / "questions.jsonl")
    manifest = {
        "questions_sha256": sha256(data),
        "n": len(rows),
        "by_category": {c: sum(r["category"] == c for r in rows) for c in ("boundary", "paraphrase", "exception")},
        "stage1_ids": [r["id"] for r in rows if r["stage1"]],
        "corpus_sha256": {p.name: sha256(p.read_bytes()) for p in sorted(CORPUS.glob("*.md"))},
        "system_prompt_sha256": sha256(SYSTEM_PROMPT.encode("utf-8")),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def write_blind(out: Path) -> None:
    """確認役に渡す分。問と規程だけで、正解・facts・区分は入れない。"""
    out.mkdir(parents=True, exist_ok=True)
    rows = [{"id": r["id"], "doc": r["doc"], "question": r["question"]} for r in build()]
    _dump(rows, out / "questions_blind.jsonl")
    (out / "corpus").mkdir(exist_ok=True)
    for p in CORPUS.glob("*.md"):
        (out / "corpus" / p.name).write_bytes(p.read_bytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--blind", type=Path)
    args = ap.parse_args()
    if args.out:
        print(json.dumps(write_sealed(args.out), ensure_ascii=False, indent=2))
    if args.blind:
        write_blind(args.blind)


if __name__ == "__main__":
    main()
