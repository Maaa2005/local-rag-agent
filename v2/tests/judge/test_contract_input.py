"""C1 が契約入力の中身を見る（多層防御の C1 層）。

- RuleC1Router: 元資料の直接指定（source）は reject、許容外キー・入力種別の欠落/None/空/非文字列は human
- 許容キーは契約ごと（judge.destination.allowed_input_keys）。free_text は依頼文を取る契約（B）だけ
- guarded_c1: 判定モデルが契約どおりの経路・宛先を返しても、契約入力が崩れていれば human（model=guard）
"""
from __future__ import annotations

import pytest

from judge.base import GUARD_MODEL, C1Decision, guarded_c1, is_guard_stop
from judge.destination import allowed_input_keys, contract_input_problem, lookup_contract
from judge.rules import RuleC1Router

A, B = "A-faq-format@1", "B-csv-codegen@1"
A_INPUT = {"contract": A, "explanation": "expl-keihi-001@1", "options": {"文体": "です・ます調", "長さ": "400字以内"}}
B_INPUT = {"contract": B, "spec": "spec-csv-001@1", "options": {"言語": "Python 3.11"}, "free_text": "月ごとに合計して"}
V072_INPUT = {"contract": A, "explanation": None, "source": "executive/M&A検討資料.md"}


def _c(key):
    c, why = lookup_contract(key)
    assert c is not None, why
    return c


def test_allowed_input_keys_per_contract():
    assert allowed_input_keys(_c(A)) == {"contract", "explanation", "options", "destination_hint"}
    assert allowed_input_keys(_c(B)) == {"contract", "spec", "options", "destination_hint", "free_text"}


def test_contract_input_problem_ok_for_normal_inputs():
    assert contract_input_problem(_c(A), A_INPUT) is None
    assert contract_input_problem(_c(B), B_INPUT) is None
    assert contract_input_problem(_c(B), {**B_INPUT, "free_text": ""}) is None  # run_contract_b の既定値
    assert contract_input_problem(_c(A), {"contract": A, "explanation": "expl-keihi-001@1"}) is None


# ---- RuleC1Router ----

def test_rules_c1_rejects_raw_source_v072():
    d = RuleC1Router().route("u_general", "M&A検討資料をFAQにして", V072_INPUT)
    assert (d.route, d.destination) == ("reject", None)
    assert "raw source reference" in d.reason and d.question_version.startswith("rules-c1@3#")


def test_rules_c1_rejects_source_even_with_valid_explanation():
    d = RuleC1Router().route("u_general", "x", {**A_INPUT, "source": "general/経費.md"})
    assert d.route == "reject"


@pytest.mark.parametrize("inp", [
    {**A_INPUT, "unknown": "x"},                     # 未知キー
    {**A_INPUT, "free_text": "やさしく"},             # 契約 A は free_text を取らない
    {**B_INPUT, "explanation": "expl-keihi-001@1"},  # 他契約の入力種別キー
    {"contract": A, "options": {}},                  # 入力種別の欠落
    {**A_INPUT, "explanation": None},
    {**A_INPUT, "explanation": ""},
    {**A_INPUT, "explanation": "   "},
    {**A_INPUT, "explanation": 123},
    {**A_INPUT, "explanation": ["expl-keihi-001@1"]},
    {**B_INPUT, "spec": None},
    {"contract": B, "free_text": "月ごとに合計して"},
])
def test_rules_c1_human_on_unusable_contract_input(inp):
    d = RuleC1Router().route("u_exec", "x", inp)
    assert (d.route, d.destination) == ("human", None) and "contract input not usable" in d.reason


def test_rules_c1_normal_inputs_unchanged():
    r = RuleC1Router()
    d = r.route("u_general", "FAQにして", A_INPUT)
    assert (d.route, d.destination) == ("A", "claude")
    d = r.route("u_manager", "x", B_INPUT)  # 契約 B は free_text を許す
    assert (d.route, d.destination) == ("B", "codex")
    d = r.route("u_exec", "x", {**A_INPUT, "destination_hint": "codex"})
    assert d.destination == "claude" and "hint codex is reference only" in d.reason


# ---- guarded_c1: 常に契約どおりの経路・宛先を返す偽ルーター ----

class AlwaysContract:
    """契約キーから正しい経路と唯一の宛先を返す（中身は見ない）。判定モデルの allow 側の誤りを模す。"""

    def route(self, user, request, input):  # noqa: A002
        c = _c(input["contract"])
        return C1Decision(route=c.route, destination=c.destinations[0], reason="fake", model="fake",
                          revision="r1", question_version="q@1")


def _g(inp):
    return guarded_c1(AlwaysContract(), "u", "x", inp, 5.0, 4000)


@pytest.mark.parametrize("inp", [
    V072_INPUT,
    {**A_INPUT, "source": "executive/M&A検討資料.md"},
    {**A_INPUT, "unknown": "x"},
    {**A_INPUT, "free_text": "やさしく"},
    {"contract": A, "options": {}},
    {**A_INPUT, "explanation": ""},
    {**A_INPUT, "explanation": 1},
    {**B_INPUT, "spec": None},
])
def test_guard_holds_unusable_contract_input_even_if_router_allows(inp):
    d = _g(inp)
    assert (d.route, d.destination, d.model) == ("human", None, GUARD_MODEL) and is_guard_stop(d)
    assert "unusable contract input" in d.reason
    assert "judge fake rev=r1 q=q@1" in d.reason  # 判定したモデル情報は理由に残す


def test_guard_passes_normal_contract_inputs():
    d = _g(A_INPUT)
    assert (d.route, d.destination, d.model) == ("A", "claude", "fake") and not is_guard_stop(d)
    d = _g(B_INPUT)
    assert (d.route, d.destination, d.model) == ("B", "codex", "fake") and not is_guard_stop(d)
    d = _g({**A_INPUT, "destination_hint": "codex"})
    assert d.destination == "claude" and not is_guard_stop(d)
