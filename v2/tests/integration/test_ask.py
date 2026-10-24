from __future__ import annotations

import inspect
import json

import pytest

from integration import ask as ask_mod
from integration.ask import ask
from integration.gateway_client import GatewayClient
from judge.base import C1Decision
from orchestrator.audit_store import AuditStore
from orchestrator.local_answer import Chunk
from orchestrator.policy import load_policy

POLICY = load_policy()
USER = "u_general"
QUESTION = "出張旅費の日当はいくらですか（架空規程の確認）"
CHUNK = Chunk(content="国内出張の日当は一日あたり二千円とする（架空規程 第5条）", source="ryohi.md", access_level=1,
              document_id="d_ryohi", chunk_index=0)
ANSWER = "国内出張の日当は一日二千円です [1]"


class StubC1:
    def __init__(self, route="local", destination=None, exc=None):
        self.route_value, self.destination, self.exc = route, destination, exc
        self.inputs = []

    def route(self, user, request, input):  # noqa: A002
        self.inputs.append(input)
        if self.exc:
            raise self.exc
        return C1Decision(route=self.route_value, destination=self.destination, reason="stub", model="stub",
                          revision="0", question_version="stub@1")


class FakeRetriever:
    def __init__(self, chunks=(CHUNK,)):
        self.chunks, self.calls = list(chunks), 0

    def retrieve(self, question, level):
        self.calls += 1
        return list(self.chunks)


class FakeGenerator:
    def __init__(self):
        self.calls = 0

    def answer(self, question, chunks):
        self.calls += 1
        return ANSWER


@pytest.fixture
def no_gateway(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("ローカル回答の経路で Gateway が呼ばれた")
    monkeypatch.setattr(GatewayClient, "prepare", boom)
    monkeypatch.setattr(GatewayClient, "commit", boom)


def test_ask_has_no_gateway_parameter():
    assert not any("gateway" in p.lower() for p in inspect.signature(ask).parameters)
    src = inspect.getsource(ask_mod)
    assert "gateway_client" not in src and "GatewayClient" not in src


def test_local_route_answers_without_gateway(no_gateway):
    c1 = StubC1("local")
    res = ask(c1, USER, QUESTION, POLICY, FakeRetriever(), FakeGenerator())
    assert res.run.outcome == "answered" and res.run.output_text == ANSWER
    assert res.sources == ("ryohi.md",)
    assert c1.inputs == [None]  # 自由質問は契約入力を持たない
    assert [e.stage for e in res.run.audit][:1] == ["C1"]


@pytest.mark.parametrize("route,dest,outcome", [
    ("code", None, "held"), ("human", None, "held"), ("reject", None, "rejected"),
])
def test_non_local_routes_do_not_search(route, dest, outcome, no_gateway):
    r, g = FakeRetriever(), FakeGenerator()
    res = ask(StubC1(route, dest), USER, QUESTION, POLICY, r, g)
    assert res.run.outcome == outcome and res.run.stopped_at == "C1"
    assert r.calls == 0 and g.calls == 0


@pytest.mark.parametrize("route", ["A", "B"])
def test_external_routes_from_free_question_are_held(route, no_gateway):
    # 契約入力が無いので guarded_c1 が human に倒す。どちらでも送らず検索もしない
    r = FakeRetriever()
    res = ask(StubC1(route, "claude"), USER, QUESTION, POLICY, r, FakeGenerator())
    assert res.run.outcome == "held" and res.run.stopped_at == "C1" and r.calls == 0


def test_c1_missing_or_failing_is_held(no_gateway):
    for c1 in (None, StubC1(exc=TimeoutError())):
        r = FakeRetriever()
        res = ask(c1, USER, QUESTION, POLICY, r, FakeGenerator())
        assert res.run.outcome == "held" and res.run.stopped_at == "C1" and r.calls == 0


def test_audit_persists_metadata_without_bodies(tmp_path, no_gateway):
    store = AuditStore(tmp_path / "audit.db")
    try:
        res = ask(StubC1("local"), USER, QUESTION, POLICY, FakeRetriever(), FakeGenerator(), audit_sink=store)
        assert res.run.audit_persisted
        (row,) = store.runs()
        assert row["outcome"] == "answered" and row["c1_route"] == "local"
        assert row["contract"] is None and row["destination"] is None
        assert row["response_visibility"] == "source_level_requester_only"
        assert row["output_chars"] == len(ANSWER)
        dumped = json.dumps([row, store.events(res.run.audit_run_id)], ensure_ascii=False, default=str)
        for body in (QUESTION, CHUNK.content, ANSWER):
            assert body not in dumped
        assert "d_ryohi" in dumped  # 根拠の document_id は残る
    finally:
        store.close()


def test_held_run_is_also_audited(tmp_path, no_gateway):
    store = AuditStore(tmp_path / "audit.db")
    try:
        res = ask(StubC1("local"), USER, QUESTION, POLICY, FakeRetriever([]), FakeGenerator(), audit_sink=store)
        (row,) = store.runs()
        assert row["outcome"] == "held" and row["stopped_at"] == "retrieve"
        assert QUESTION not in json.dumps(store.events(res.run.audit_run_id), ensure_ascii=False)
    finally:
        store.close()


# ---- code 経路 + decider（既定の無しでは従来どおり保留）

class _CodeVerdict:
    def __init__(self, kind, text="", used=(), record=None):
        self.kind, self.answer_text, self.used_chunks = kind, text, tuple(used)
        self.decision_record = record or {}


class _Decider:
    def __init__(self, verdict):
        self.verdict, self.calls = verdict, 0

    def decide(self, question, chunks):
        self.calls += 1
        return self.verdict


def test_ask_decider_default_is_none():
    assert inspect.signature(ask).parameters["decider"].default is None


def test_code_route_without_decider_is_still_held_at_c1(no_gateway):
    res = ask(StubC1("code"), USER, QUESTION, POLICY, FakeRetriever(), FakeGenerator())
    assert res.run.outcome == "held" and res.run.stopped_at == "C1"


def test_code_route_with_decider_answers_from_code(no_gateway):
    g, d = FakeGenerator(), _Decider(_CodeVerdict("decided", "結論: 必要", used=[CHUNK], record={"rule_id": "R-x", "outcome_key": "all_true"}))
    res = ask(StubC1("code"), USER, QUESTION, POLICY, FakeRetriever(), g, decider=d)
    assert res.run.outcome == "answered" and res.run.output_text == "結論: 必要"
    assert g.calls == 0 and d.calls == 1
    assert [e.stage for e in res.run.audit][:2] == ["C1", "retrieve"] and res.run.audit[0].result == "code"


def test_code_route_with_decider_held_stops_at_decide(no_gateway):
    g = FakeGenerator()
    res = ask(StubC1("code"), USER, QUESTION, POLICY, FakeRetriever(), g,
              decider=_Decider(_CodeVerdict("held", record={"hold_code": "facts:missing"})))
    assert res.run.outcome == "held" and res.run.stopped_at == "decide" and g.calls == 0


def test_local_route_without_decider_uses_model(no_gateway):
    # 10/24 変更: local でも decider を通す（決め事 3 節: 全質問が記録の検索を通る）。decider が無ければ従来どおりモデルが答える
    g = FakeGenerator()
    res = ask(StubC1("local"), USER, QUESTION, POLICY, FakeRetriever(), g)
    assert res.run.outcome == "answered" and res.run.output_text == ANSWER and g.calls == 1


def test_code_route_audit_has_no_bodies(tmp_path, no_gateway):
    store = AuditStore(tmp_path / "audit.db")
    try:
        text = "結論: 必要（総務部長の承認が要る）"
        res = ask(StubC1("code"), USER, QUESTION, POLICY, FakeRetriever(), FakeGenerator(), audit_sink=store,
                  decider=_Decider(_CodeVerdict("decided", text, used=[CHUNK], record={"rule_id": "R-x", "outcome_key": "all_true"})))
        (row,) = store.runs()
        assert row["outcome"] == "answered" and row["c1_route"] == "code"
        dumped = json.dumps([row, store.events(res.run.audit_run_id)], ensure_ascii=False, default=str)
        for body in (QUESTION, CHUNK.content, text):
            assert body not in dumped
        assert "R-x" in dumped
    finally:
        store.close()


# ---- local 経路でも decider を通す（全質問が記録の検索を通る）

def test_local_route_with_decider_decided_answers_from_code(no_gateway):
    g = FakeGenerator()
    d = _Decider(_CodeVerdict("decided", "結論: 必要", used=[CHUNK], record={"rule_id": "R-x", "outcome_key": "all_true"}))
    res = ask(StubC1("local"), USER, QUESTION, POLICY, FakeRetriever(), g, decider=d)
    assert res.run.outcome == "answered" and res.run.output_text == "結論: 必要"
    assert g.calls == 0 and d.calls == 1
    assert res.run.audit[0].stage == "C1" and res.run.audit[0].result == "local"


def test_local_route_with_decider_no_record_falls_back_to_model(no_gateway):
    g, d = FakeGenerator(), _Decider(_CodeVerdict("no_record"))
    res = ask(StubC1("local"), USER, QUESTION, POLICY, FakeRetriever(), g, decider=d)
    assert res.run.outcome == "answered" and res.run.output_text == ANSWER
    assert d.calls == 1 and g.calls == 1


def test_local_route_with_decider_held_stops_at_decide(no_gateway):
    g = FakeGenerator()
    res = ask(StubC1("local"), USER, QUESTION, POLICY, FakeRetriever(), g,
              decider=_Decider(_CodeVerdict("held", record={"hold_code": "facts:missing"})))
    assert res.run.outcome == "held" and res.run.stopped_at == "decide" and g.calls == 0
