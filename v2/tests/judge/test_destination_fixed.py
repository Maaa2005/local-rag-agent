"""宛先固定（1 契約 1 宛先）の C1 側・送信資格側の検査。

- 契約の解決: 完全一致で引け、未知版・無効契約・宛先が 1 件でない契約・経路不一致・設定異常は保留
- guarded_c1: truncated・宛先欠落・宛先不一致・契約なしは human（model=guard）
- HTTP 版 C1 の truncated を結果に反映し guard で止める
- 送信資格の検査は逆の宛先・旧版・宛先 2 件の契約設定を拒否する
"""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime

import httpx
import pytest

from judge.base import C1Decision, guarded_c1, is_guard_stop
from judge.destination import CONTRACTS_PATH, lookup_contract, resolve_destination
from judge.http_adapter import HttpC1Router
from judge.rules import RuleC1Router
from orchestrator.policy import JST, check_eligibility, load_policy

A, B = "A-faq-format@1", "B-csv-codegen@1"
NOW = datetime(2026, 10, 20, 10, 0, tzinfo=JST)
A_INPUT = {"contract": A, "explanation": "expl-keihi-001@1", "options": {"文体": "です・ます調", "長さ": "400字以内"}}
B_INPUT = {"contract": B, "spec": "spec-csv-001@1", "options": {"言語": "Python 3.11"}, "free_text": "月ごとに合計して"}


def _write(tmp_path, mutate):
    data = json.loads(CONTRACTS_PATH.read_text(encoding="utf-8"))
    mutate(data)
    p = tmp_path / "contracts.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


# ---- 1. 契約の解決 ----

def test_resolve_exact_match():
    assert resolve_destination("A", A)[0] == "claude"
    assert resolve_destination("B", B)[0] == "codex"


@pytest.mark.parametrize("key", ["A-faq-format@2", "A-faq-format@0", "A-faq-format", "A-faq-format@1 ", "a-faq-format@1",
                                 "B-csv-codegen@2", "", None, 1])
def test_resolve_unknown_version_or_bad_key(key):
    dest, why = resolve_destination("A", key)
    assert dest is None and why


@pytest.mark.parametrize("route,key", [("B", A), ("A", B), ("local", A), ("human", B)])
def test_resolve_route_mismatch(route, key):
    assert resolve_destination(route, key)[0] is None


def test_lookup_disabled_contract(tmp_path):
    p = _write(tmp_path, lambda d: d[A].update(enabled=False))
    c, why = lookup_contract(A, p)
    assert c is None and "disabled" in why
    assert resolve_destination("A", A, p)[0] is None


@pytest.mark.parametrize("dests", [["claude", "codex"], [], ["codex", "codex"]])
def test_lookup_not_exactly_one_destination(tmp_path, dests):
    p = _write(tmp_path, lambda d: d[A].update(destinations=dests))
    c, why = lookup_contract(A, p)
    assert c is None
    assert resolve_destination("A", A, p)[0] is None


def test_lookup_missing_route(tmp_path):
    p = _write(tmp_path, lambda d: d[A].pop("route"))
    assert lookup_contract(A, p)[0] is None


def test_lookup_unreadable_config(tmp_path):
    assert lookup_contract(A, tmp_path / "missing.json")[0] is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    c, why = lookup_contract(A, bad)
    assert c is None and "unreadable" in why


def test_rules_c1_holds_on_bad_config(tmp_path, monkeypatch):
    import judge.rules as rules

    p = _write(tmp_path, lambda d: d[A].update(destinations=["claude", "codex"]))
    monkeypatch.setattr(rules, "lookup_contract", lambda key, path=p: lookup_contract(key, p))
    d = RuleC1Router().route("u_general", "FAQにして", A_INPUT)
    assert d.route == "human" and d.destination is None


# ---- guarded_c1 ----

class Stub:
    def __init__(self, route="A", dest="claude", truncated=False):
        self.d = C1Decision(route=route, destination=dest, reason="stub", model="m", revision="r",
                            question_version="q@1", truncated=truncated)

    def route(self, user, request, input):  # noqa: A002
        return self.d


def _g(router, inp, **kw):
    return guarded_c1(router, "u", "x", inp, 5.0, 4000, **kw)


def test_guard_passes_sole_destination():
    d = _g(Stub("A", "claude"), A_INPUT)
    assert (d.route, d.destination, d.model) == ("A", "claude", "m") and not is_guard_stop(d)
    d = _g(Stub("B", "codex"), B_INPUT)
    assert (d.route, d.destination) == ("B", "codex")


@pytest.mark.parametrize("stub,inp", [
    (Stub("A", "codex"), A_INPUT),                    # 逆の宛先
    (Stub("B", "claude"), B_INPUT),
    (Stub("A", None), A_INPUT),                       # 宛先欠落
    (Stub("A", "claude"), None),                      # 契約なし
    (Stub("A", "claude"), {"contract": "A-faq-format@2"}),  # 旧版・未知版
    (Stub("A", "claude"), B_INPUT),                   # 経路と契約の不整合
    (Stub("A", "claude", truncated=True), A_INPUT),
])
def test_guard_holds(stub, inp):
    d = _g(stub, inp)
    assert d.route == "human" and is_guard_stop(d) and d.destination is None


def test_guard_truncated_flag_and_original_recorded():
    d = _g(Stub("A", "claude", truncated=True), A_INPUT)
    assert d.truncated and "judge m rev=r q=q@1" in d.reason


def test_guard_bad_config_holds(tmp_path):
    p = _write(tmp_path, lambda d: d[A].update(destinations=["claude", "codex"]))
    d = _g(Stub("A", "claude"), A_INPUT, contracts_path=p)
    assert d.route == "human" and is_guard_stop(d)


def test_guard_non_external_route_passes_through():
    d = _g(Stub("local", None), None)
    assert d.route == "local" and not is_guard_stop(d)


# ---- HTTP 版 C1 の truncated ----

def _http(body):
    def h(req: httpx.Request):
        return httpx.Response(200, json=body)
    return HttpC1Router("http://judge.internal", "laya", transport=httpx.MockTransport(h))


def test_http_c1_truncated_reflected_and_held():
    body = {"model": "laya", "revision": "r1", "truncated": True,
            "answers": [{"id": "route", "answer": "A", "probs": {"A": 0.9}}]}
    d = _http(body).route("u", "x", {"contract": A})
    assert d.truncated is True and d.destination == "claude"
    g = guarded_c1(_http(body), "u", "x", A_INPUT, 5.0, 4000)
    assert g.route == "human" and g.truncated and is_guard_stop(g)


def test_http_c1_destination_from_contract_not_model():
    body = {"model": "laya", "revision": "r1",
            "answers": [{"id": "route", "answer": "B", "probs": {"B": 0.9}},
                        {"id": "destination", "answer": "claude", "probs": {"claude": 0.9}}]}
    d = _http(body).route("u", "x", {"contract": B, "destination_hint": "claude"})
    assert (d.route, d.destination, d.truncated) == ("B", "codex", False)


def test_http_c1_unknown_contract_has_no_destination_and_is_held():
    body = {"model": "laya", "revision": "r1", "answers": [{"id": "route", "answer": "A", "probs": {"A": 0.9}}]}
    d = _http(body).route("u", "x", {"contract": "A-faq-format@2"})
    assert d.destination is None
    g = guarded_c1(_http(body), "u", "x", {"contract": "A-faq-format@2"}, 5.0, 4000)
    assert g.route == "human"


# ---- 送信資格: 逆の宛先・旧版・宛先 2 件 ----

@pytest.mark.parametrize("user,inp,dest", [("u_general", A_INPUT, "codex"), ("u_manager", B_INPUT, "claude")])
def test_eligibility_rejects_reverse_destination(user, inp, dest):
    e = check_eligibility(load_policy(), user, inp, dest, NOW)
    assert not e.ok and "宛先" in e.reason


@pytest.mark.parametrize("user,inp,dest", [("u_general", A_INPUT, "claude"), ("u_manager", B_INPUT, "codex")])
def test_eligibility_accepts_sole_destination(user, inp, dest):
    assert check_eligibility(load_policy(), user, inp, dest, NOW).ok


@pytest.mark.parametrize("key", ["A-faq-format@2", "A-faq-format"])
def test_eligibility_rejects_old_version(key):
    e = check_eligibility(load_policy(), "u_general", {**A_INPUT, "contract": key}, "claude", NOW)
    assert not e.ok


def test_eligibility_rejects_multi_destination_contract():
    pol = load_policy()
    contracts = dict(pol.contracts)
    contracts[A] = replace(contracts[A], destinations=("claude", "codex"))
    pol2 = replace(pol, contracts=contracts)
    e = check_eligibility(pol2, "u_general", A_INPUT, "claude", NOW)
    assert not e.ok and "宛先" in e.reason


# ---- 評価ハーネス: 既定宛先で補完しない ----

def test_eval_build_payload_no_default_destination():
    from evalharness.run_v2_eval import build_payload

    case = {"input": {"contract": A, "explanation": "expl-keihi-001@1", "options": {}}}
    none = C1Decision(route="A", destination=None, reason="r", model="m", revision="r", question_version="q@1")
    assert build_payload(case, none) is None
    ok = C1Decision(route="A", destination="claude", reason="r", model="m", revision="r", question_version="q@1")
    assert build_payload(case, ok).destination == "claude"


def test_eval_cases_expected_destination_counts():
    from evalharness.run_v2_eval import DEFAULT_CASES, load_cases

    cases = load_cases(DEFAULT_CASES)
    dests = [c.get("expected_destination") for c in cases]
    assert (dests.count("claude"), dests.count("codex"), dests.count(None)) == (8, 18, 13)
    for c in cases:
        if c.get("expected_destination"):
            assert c["expected_route"] in ("A", "B")
            assert c["expected_destination"] == {"A": "claude", "B": "codex"}[c["input"]["contract"][0]]
