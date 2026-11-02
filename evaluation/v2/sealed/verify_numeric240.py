"""封の問（数値・Issue #16）の本体が numeric240.manifest.json と同じかを sha256 で確かめる。中身は表示しない。

使い方: python evaluation/v2/sealed/verify_numeric240.py <本体のフォルダ>
本体のフォルダには corpus/*.md・sealed/questions.jsonl・生成と検算のスクリプトがある（README.md）。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

MANIFEST = Path(__file__).with_name("numeric240.manifest.json")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_manifest(path: Path = MANIFEST) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify(root: Path, manifest: dict) -> list[str]:
    """一致しない点の一覧を返す（空なら一致）。問の文・正解は返さない。"""
    problems: list[str] = []
    q = root / "sealed" / "questions.jsonl"
    if not q.is_file():
        return ["sealed/questions.jsonl が無い"]
    if sha256(q) != manifest["questions_sha256"]:
        problems.append("sealed/questions.jsonl の sha256 が違う")
    rows = [json.loads(line) for line in q.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != manifest["n"]:
        problems.append(f"問の数が違う（{len(rows)}）")
    if sum(bool(r.get("should_hold")) for r in rows) != manifest["should_hold"]:
        problems.append("保留すべき問の数が違う")
    got = sorted(sha256(p) for p in (root / "corpus").glob("*.md"))
    if got != sorted(manifest["corpus_sha256"].values()):
        problems.append("規程（corpus/*.md）の sha256 の組が違う")
    for name, h in manifest["generator_sha256"].items():
        p = root / name
        if not p.is_file() or sha256(p) != h:
            problems.append(f"{name} が封をした時と違う")
    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    problems = verify(Path(argv[1]), load_manifest())
    if problems:
        print("NG")
        for p in problems:
            print(" -", p)
        return 1
    m = load_manifest()
    print(f"OK: {m['n']} 問・questions_sha256 {m['questions_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
