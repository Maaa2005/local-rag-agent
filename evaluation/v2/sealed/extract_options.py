"""問の jsonl から、結論の選択肢の語の組 (conclusion_options) だけを取り出す (Issue #16 の本人の決定 2)。

評価を流す人が機械で取り出し、カードを作る人へ渡す。問の ID・本文・頻度・正解は出さない。
重複を除き、組は語の順に並べ、組の並びも並べ替える (問の順が分からないように)。

使い方: python evaluation/v2/sealed/extract_options.py <questions.jsonl> --out <options.json>
標準出力は組の数だけ。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FORMAT = "conclusion-options@1"


def extract_options(rows: list[dict]) -> tuple[list[list[str]], int]:
    """(語の組の一覧, 欄が無い・形が違う問の数)。"""
    found: set[tuple[str, ...]] = set()
    missing = 0
    for row in rows:
        opts = row.get("conclusion_options") if isinstance(row, dict) else None
        if not (isinstance(opts, (list, tuple)) and opts and all(isinstance(o, str) and o.strip() for o in opts)):
            missing += 1
            continue
        found.add(tuple(sorted({o.strip() for o in opts})))
    return [list(t) for t in sorted(found)], missing


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("questions")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    rows = [json.loads(line) for line in Path(a.questions).read_text(encoding="utf-8").splitlines() if line.strip()]
    pairs, missing = extract_options(rows)
    Path(a.out).write_text(json.dumps({"format": FORMAT, "conclusion_options": pairs}, ensure_ascii=False, indent=2)
                           + "\n", encoding="utf-8")
    if missing:
        print(f"conclusion_options が無い問: {missing}", file=sys.stderr)
    print(len(pairs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
