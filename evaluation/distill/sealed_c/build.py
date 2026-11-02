"""封 C の書き出しと、正解を伏せた点検の照合。

    python -m evaluation.distill.sealed_c.build --blind <dir>         # 確認役に渡す分
    python -m evaluation.distill.sealed_c.build --check <確認役の出力>  # 照合
    python -m evaluation.distill.sealed_c.build --out evaluation/distill/sealed_c/sealed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.distill.sealed_c.questions import build
from evaluation.teacher_check.build import SYSTEM_PROMPT, sha256
from evaluation.teacher_check.check_blind import parse

CORPUS = Path(__file__).resolve().parent / "corpus"


def doc_text(doc: str) -> str:
    return (CORPUS / f"{doc}.md").read_text(encoding="utf-8")


def prompt_messages(row: dict) -> list[dict]:
    """封 B と同じ形（検索なし・規程の全文と問）。"""
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
    manifest = {"questions_sha256": sha256(data), "n": len(rows),
                "by_category": {c: sum(r["category"] == c for r in rows) for c in ("boundary", "paraphrase", "exception", "undecidable")},
                "corpus_sha256": {p.name: sha256(p.read_bytes()) for p in sorted(CORPUS.glob("*.md"))}}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def write_blind(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    _dump([{"id": r["id"], "doc": r["doc"], "question": r["question"]} for r in build()], out / "questions_blind.jsonl")
    (out / "corpus").mkdir(exist_ok=True)
    for p in CORPUS.glob("*.md"):
        (out / "corpus" / p.name).write_bytes(p.read_bytes())


def compare(text: str) -> dict:
    got = parse(text)
    rows = build()
    return {"n": len(rows), "answered": len(got), "missing": [r["id"] for r in rows if r["id"] not in got],
            "conclusion_diff": [(r["id"], r["expected_conclusion"], got[r["id"]]["conclusion"]) for r in rows
                                if r["id"] in got and got[r["id"]]["conclusion"] != r["expected_conclusion"]],
            "article_diff": [(r["id"], r["expected_articles"], got[r["id"]]["articles"]) for r in rows
                             if r["id"] in got and not set(got[r["id"]]["articles"]) & set(r["expected_articles"])]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--blind", type=Path)
    ap.add_argument("--check", type=Path)
    a = ap.parse_args()
    if a.blind:
        write_blind(a.blind)
    if a.check:
        print(json.dumps(compare(a.check.read_text(encoding="utf-8")), ensure_ascii=False, indent=1))
    if a.out:
        print(json.dumps(write_sealed(a.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
