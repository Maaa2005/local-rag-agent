"""封の問（数値・Issue #16）の指紋 numeric240.manifest.json の形と、照合スクリプトの動きを確かめる。

本体（問・規程・正解）はリポに無いので、ここでは数と sha256 の形だけを見る。
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from evaluation.v2.sealed import verify_numeric240 as v

M = v.load_manifest()
PER_RULE = {"boundary": 3, "paraphrase": 2, "exception": 1, "exception_near": 1, "premise": 1, "multi": 1,
            "distractor": 1, "derived": 1, "unknown_determined": 1, "hold_missing": 1, "hold_unknown": 1,
            "hold_range": 1}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def test_counts_match_issue16():
    assert M["n"] == 240
    assert M["answerable"] == 192
    assert M["should_hold"] == 48
    assert M["by_rule"] == {f"R{i:02d}": 15 for i in range(1, 17)}
    assert M["by_group"] == {g: n * 16 for g, n in PER_RULE.items()}
    assert M["by_position"] == {"above": 16, "at": 16, "below": 16}
    assert M["paraphrase_pairs"] == 32
    assert sum(M["by_polarity"].values()) == 240 and M["by_polarity"]["hold"] == 48


def test_fingerprints_are_sha256():
    assert HEX64.match(M["questions_sha256"])
    assert len(M["corpus_sha256"]) == 16
    assert all(HEX64.match(h) for h in M["corpus_sha256"].values())
    assert set(M["generator_sha256"]) == {"rules.py", "items.py", "build.py", "check.py"}
    assert all(HEX64.match(h) for h in M["generator_sha256"].values())


def _keys(x):
    if isinstance(x, dict):
        for k, val in x.items():
            yield k
            yield from _keys(val)
    elif isinstance(x, list):
        for val in x:
            yield from _keys(val)


def test_manifest_has_no_content():
    # 問の文・正解・規程の名前を持たない（数と sha256 だけ）
    assert not {"question", "body", "expected_conclusion", "facts", "conclusion_options"} & set(_keys(M))
    assert "規程" not in json.dumps(M, ensure_ascii=False)


def _fake_seal(root: Path) -> dict:
    (root / "sealed").mkdir()
    (root / "corpus").mkdir()
    rows = [{"id": "S001", "should_hold": False}, {"id": "S002", "should_hold": True}]
    q = root / "sealed" / "questions.jsonl"
    q.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (root / "corpus" / "a.md").write_text("# a\n", encoding="utf-8")
    (root / "rules.py").write_text("x = 1\n", encoding="utf-8")
    h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    return {"n": 2, "should_hold": 1, "questions_sha256": h(q),
            "corpus_sha256": {"R01": h(root / "corpus" / "a.md")},
            "generator_sha256": {"rules.py": h(root / "rules.py")}}


def test_verify_ok_and_detects_change(tmp_path):
    m = _fake_seal(tmp_path)
    assert v.verify(tmp_path, m) == []
    (tmp_path / "corpus" / "a.md").write_text("# a 改\n", encoding="utf-8")
    (tmp_path / "rules.py").write_text("x = 2\n", encoding="utf-8")
    problems = v.verify(tmp_path, m)
    assert any("corpus" in p for p in problems)
    assert any("rules.py" in p for p in problems)


def test_verify_missing_body(tmp_path):
    assert v.verify(tmp_path, M) == ["sealed/questions.jsonl が無い"]
