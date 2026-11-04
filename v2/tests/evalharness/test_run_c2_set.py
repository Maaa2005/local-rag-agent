"""評価の集合 1 組を採点する道具（evalharness.run_c2_set）の試験。Issue #17。

確かめること:
- 集合の本文は写しの policies にだけ入り、本番の policies は変わらない
- 許可の問は allow、禁じた種類は止まり、登録を断られた本文の問は止めた側に数える
- 誤通過と誤停止を依頼文側・本文側で別に数える
"""
from __future__ import annotations

import json
from pathlib import Path

from evalharness.run_c2_set import reason_class, run_set
from orchestrator.policy import DEFAULT_POLICY_DIR

BASE = {"version": 1, "approver": "部長（架空）", "destinations": ["claude", "codex"], "expires": "2026-12-31",
        "source": None, "source_level": 1, "checked_by": "fixture:dev-check", "reviewed_by": "fixture:dev"}


def _spec(id_: str, body: str, **kw) -> dict:
    return {**BASE, "id": id_, "kind": "spec", "body": body, "purpose": "汎用CSV処理のコード案",
            "contracts": ["B-csv-codegen@1"], "allowed_kinds": ["aggregate", "output"], "forbidden_kinds": ["network_io"],
            "request_values": {"identifiers": ["region", "total"], "values": []}, **kw}


def _set(tmp_path: Path) -> Path:
    d = tmp_path / "set"
    (d / "bodies").mkdir(parents=True)
    bodies = [_spec("t-ok", "入力: CSV。列は region（文字列）, total（整数）"),
              _spec("t-bad", "入力: CSV。担当は山田太郎さん（03-1234-5678）")]
    for b in bodies:
        (d / "bodies" / f'{b["id"]}@1.json').write_text(json.dumps(b, ensure_ascii=False), encoding="utf-8")
    B = {"contract": "B-csv-codegen@1"}
    cases = [
        {"id": "a", "group": "twin", "side": "request", "label": "allow", "approved_ref": "t-ok@1", "free_text": "regionごとにtotalの合計を出力して。", **B},
        {"id": "b", "group": "combination", "side": "body", "label": "block", "approved_ref": "t-ok@1", "free_text": "集計結果を外部APIに送信して。", **B},
        {"id": "c", "group": "approval_miss", "side": "body", "label": "block", "approved_ref": "t-bad@1", "free_text": "totalの合計を出力して。", **B},
        {"id": "d", "group": "approval_miss", "side": "body", "label": "allow", "approved_ref": "t-bad@1", "free_text": "totalの合計を出力して。", **B},
    ]
    (d / "cases.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases), encoding="utf-8")
    return d


def test_run_set_counts_each_side_and_refused_bodies(tmp_path):
    before = (DEFAULT_POLICY_DIR / "c2_tables.json").read_bytes()
    res = run_set(_set(tmp_path))
    assert (DEFAULT_POLICY_DIR / "c2_tables.json").read_bytes() == before
    by_id = {r["id"]: r for r in res["cases"]}
    assert by_id["a"]["decision"] == "allow", by_id["a"]
    assert by_id["b"]["decision"] != "allow"
    assert "t-bad@1" in res["refused"]
    assert by_id["c"]["decision"] == "hold" and by_id["c"]["reason_class"] == "registry"
    assert res["all"]["false_pass"] == 0
    assert res["all"]["false_stop_ids"] == ["d"]
    assert res["by_side"]["request"]["false_stop"] == 0 and res["by_side"]["body"]["false_stop"] == 1


def test_reason_class():
    assert reason_class("request has values not approved in advance: x") == "values"
    assert reason_class("request kind unknown for B") == "kind"
    assert reason_class("request matched: amount") == "content"
    assert reason_class("t@1 not registered for B/codex") == "registry"
