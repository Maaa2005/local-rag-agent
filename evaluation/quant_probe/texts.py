"""校正の文章（imatrix 用）と評価の文章（KLD・perplexity 用）を作る（計画 7 章）。

    python -m evaluation.quant_probe.texts --wikitext-test test.parquet --ja-wiki train-00014-of-00015.parquet \\
        --calib-en text_en_micro.parquet --calib-ja text_jp_micro.parquet --out <dir>

parquet を読むのに pyarrow が要る（単体テストでは使わない部分）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

from .questions import domain_text

JA_MIN_CHARS = 2000
JA_TOTAL_CHARS = 150_000
JA_SKIP_TITLE = ("一覧", "曖昧さ回避")


def pick_ja_articles(rows: Iterable[dict], min_chars: int = JA_MIN_CHARS,
                     total_chars: int = JA_TOTAL_CHARS) -> tuple[str, list[str]]:
    """並び順のまま、本文 min_chars 字以上で題に一覧・曖昧さ回避を含まない記事を、合計 total_chars 字まで取る。"""
    parts, titles, n = [], [], 0
    for r in rows:
        title, text = r.get("title") or "", (r.get("text") or "").strip()
        if len(text) < min_chars or any(w in title for w in JA_SKIP_TITLE):
            continue
        parts.append(text)
        titles.append(title)
        n += len(text)
        if n >= total_chars:
            break
    if n < total_chars:
        raise ValueError(f"日本語の記事が足りない: {n} 字")
    return "\n\n".join(parts) + "\n", titles


def _read_parquet(path: Path, columns: list[str]) -> list[dict]:
    import pyarrow.parquet as pq  # 実行時だけ

    t = pq.read_table(path, columns=columns)
    return t.to_pylist()


def _text_column(path: Path) -> str:
    import pyarrow.parquet as pq

    names = pq.read_schema(path).names
    for c in ("content", "text"):
        if c in names:
            return c
    raise ValueError(f"{path.name}: 文の列が無い {names}")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wikitext-test", type=Path, required=True)
    ap.add_argument("--ja-wiki", type=Path, required=True)
    ap.add_argument("--calib-en", type=Path, required=True)
    ap.add_argument("--calib-ja", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    en = "".join(r["text"] for r in _read_parquet(a.wikitext_test, ["text"]))
    ja, titles = pick_ja_articles(_read_parquet(a.ja_wiki, ["title", "text"]))
    cal = {}
    for lang, p in (("en", a.calib_en), ("ja", a.calib_ja)):
        col = _text_column(p)
        cal[lang] = "\n".join((r[col] or "").strip() for r in _read_parquet(p, [col])) + "\n"
    files = {"eval_en.txt": en, "eval_ja.txt": ja, "eval_dom.txt": domain_text(),
             "calib_en.txt": cal["en"], "calib_ja.txt": cal["ja"]}
    for name, text in files.items():
        (a.out / name).write_text(text, encoding="utf-8", newline="\n")
    man = {"sources": {k: {"file": str(v.name), "sha256": sha256(v)} for k, v in
                       (("wikitext_test", a.wikitext_test), ("ja_wiki", a.ja_wiki),
                        ("calib_en", a.calib_en), ("calib_ja", a.calib_ja))},
           "outputs": {n: {"sha256": sha256(a.out / n), "chars": len(t)} for n, t in files.items()},
           "ja_titles": titles}
    (a.out / "texts.manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    for n, t in files.items():
        print(n, len(t))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
