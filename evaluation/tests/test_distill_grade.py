"""開発用の正誤と試作の選び方（evaluation/distill/grade.py）のテスト。"""
from __future__ import annotations

import json
from pathlib import Path

from evaluation.distill import grade, guard

REPO = Path(__file__).resolve().parents[2]
DEV = grade.dev_rows()
GUARD = [json.loads(line) for line in (REPO / "evaluation/distill/guard/guard.jsonl").read_text(encoding="utf-8").splitlines()]


def _row(cat, expected="支給される", keywords=()):
    return {"id": "x", "category": cat, "expected": expected, "keywords": list(keywords)}


def test_correct_by_category():
    assert grade.correct(_row("boundary"), "…と判断します。\n結論: 支給される")
    assert not grade.correct(_row("boundary"), "結論: 支給されない")
    assert not grade.correct(_row("boundary"), None) and not grade.correct(_row("boundary"), "")
    assert grade.correct(_row("refusal"), "その情報は現在の資料には含まれていません。")
    assert not grade.correct(_row("refusal"), "結論: 支給される")
    assert grade.correct(_row("describe", keywords=["5日"]), "申請は5日前までです[2]。")
    assert not grade.correct(_row("describe", keywords=["5日"]), "その情報は現在の資料には含まれていません。5日")


def test_split_is_disjoint_half_and_fixed():
    a, b = grade.split_dev(DEV)
    assert len(a) == len(b) == 150
    assert not {r["id"] for r in a} & {r["id"] for r in b}
    assert [r["id"] for r in a] == [r["id"] for r in grade.split_dev(DEV)[0]]
    assert len({r["id"] for r in DEV}) == 300
    assert grade.DEV_SEED == guard.DEV_SEED
    assert not {r["messages"][1]["content"] for r in DEV} & {r["messages"][1]["content"] for r in GUARD}


def test_accuracy_counts_missing_as_wrong():
    rows = [dict(_row("boundary"), id="a"), dict(_row("boundary"), id="b")]
    assert grade.accuracy(rows, {"a": "結論: 支給される"}) == 0.5
    assert grade.accuracy(rows, {"a": "結論: 支給される"}, category="boundary") == 0.5


def test_guard_ok_needs_all_kinds_and_time():
    passed = {r["id"]: True for r in GUARD}
    sec = {r["id"]: 2.0 for r in GUARD}
    ctrl = {r["id"]: 1.5 for r in GUARD}
    assert grade.guard_ok(GUARD, passed, sec, ctrl)["ok"]
    slow = {r["id"]: 2.3 for r in GUARD}
    assert not grade.guard_ok(GUARD, passed, slow, ctrl)["ok"]
    one_fail = dict(passed, **{next(r["id"] for r in GUARD if r["guard"] == "refusal"): False})
    res = grade.guard_ok(GUARD, one_fail, sec, ctrl)
    assert not res["ok"] and not res["kinds"]["refusal"] and res["kinds"]["describe"]


def test_choose_best_then_lower_lr():
    cands = [{"lr": 2e-4, "guard_ok": True, "select_acc": 0.8}, {"lr": 1e-4, "guard_ok": True, "select_acc": 0.8},
             {"lr": 5e-5, "guard_ok": False, "select_acc": 0.9}]
    assert grade.choose(cands)["lr"] == 1e-4
    assert grade.choose([dict(c, guard_ok=False) for c in cands]) is None


def test_near_bodies_drop_format_lines():
    from evaluation.distill import near
    b = near.body(DEV[0]["question"])
    assert b and "最後の行には" not in b and "結論:" not in b
    sealed = near.sealed_bodies()
    assert len(sealed) == 50 + 12 + 96 + 24 and all(sealed)
    assert not any("答えの最後の2行" in s or "結論:" in s for s in sealed)
    assert near.summary([0.1, 0.95, 0.5])["over"] == 1
