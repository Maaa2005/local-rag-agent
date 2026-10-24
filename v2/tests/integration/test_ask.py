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


# ---- 最終の答えの記録（plan 7.4）: on_final は ask() 1 回につき 1 回、最後に呼ばれる

class _Boom:
    def __init__(self, exc=None, text=ANSWER):
        self.exc, self.text = exc, text

    def answer(self, question, chunks):
        if self.exc:
            raise self.exc
        return self.text


def _final(c1, *, decider=None, generator=None, retriever=None, user=USER):
    got = []
    res = ask(c1, user, QUESTION, POLICY, retriever or FakeRetriever(), generator or FakeGenerator(),
              decider=decider, on_final=got.append)
    assert len(got) == 1  # 1 回だけ
    return res, got[0]


def test_ask_on_final_default_is_none():
    assert inspect.signature(ask).parameters["on_final"].default is None
    ask(StubC1("local"), USER, QUESTION, POLICY, FakeRetriever(), FakeGenerator())  # 無くても動く


def test_on_final_decided_by_code(no_gateway):
    rec = {"rule_id": "R-x", "outcome_key": "all_true", "final_answer": None}
    res, f = _final(StubC1("local"), decider=_Decider(_CodeVerdict("decided", "結論: 必要", used=[CHUNK], record=rec)))
    assert f["route_final"] == "code" and f["final_answer"] == "結論: 必要" == res.run.output_text
    assert f["outcome"] == "answered" and f["stopped_at"] == "none" and f["rule_id"] == "R-x"
    assert rec["final_answer"] is None  # 渡した記録そのものは書き換えない


def test_on_final_held_by_decider(no_gateway):
    res, f = _final(StubC1("code"), decider=_Decider(_CodeVerdict("held", record={"hold_code": "facts:missing"})))
    assert f["route_final"] == "held" and f["final_answer"] is None
    assert f["outcome"] == "held" and f["stopped_at"] == "decide" and f["hold_code"] == "facts:missing"


def test_on_final_model_only_carries_the_models_actual_answer(no_gateway):
    g = _Boom(text="モデルが実際に答えた文 [1]")
    res, f = _final(StubC1("local"), decider=_Decider(_CodeVerdict("no_record", record={"route": "model_only", "final_answer": None})),
                    generator=g)
    assert f["route_final"] == "model_only" and f["final_answer"] == "モデルが実際に答えた文 [1]"
    assert f["route"] == "model_only" and f["outcome"] == "answered"


@pytest.mark.parametrize("exc,text", [(RuntimeError("gpu"), ANSWER), (None, "")])
def test_on_final_generation_failure_is_recorded_as_model_failed(no_gateway, exc, text):
    res, f = _final(StubC1("local"), decider=_Decider(_CodeVerdict("no_record", record={"route": "model_only"})),
                    generator=_Boom(exc, text))
    assert f["route_final"] == "model_failed" and f["final_answer"] is None
    assert f["outcome"] == "held" and f["stopped_at"] == "generate" and f["route"] == "model_only"


def test_on_final_no_decider(no_gateway):
    res, f = _final(StubC1("local"))
    assert f["route_final"] == "no_decider" and f["final_answer"] == ANSWER
    assert f["outcome"] == "answered" and "rule_id" not in f


def test_on_final_no_decider_generation_failure_is_other_but_still_recorded(no_gateway):
    res, f = _final(StubC1("local"), generator=_Boom(RuntimeError("gpu")))
    assert f["route_final"] == "other" and f["stopped_at"] == "generate" and f["final_answer"] is None


@pytest.mark.parametrize("c1,decider", [
    (None, None),
    (StubC1(exc=RuntimeError("down")), None),
    (StubC1("reject"), None),
    (StubC1("human"), _Decider(_CodeVerdict("no_record"))),
    (StubC1("code"), None),
    (StubC1("A", destination="x"), _Decider(_CodeVerdict("no_record"))),
])
def test_on_final_other_routes_stop_before_decider(no_gateway, c1, decider):
    res, f = _final(c1, decider=decider)
    assert f["route_final"] == "other" and f["final_answer"] is None and f["outcome"] != "answered"
    assert f["stopped_at"] == res.run.stopped_at
    if decider is not None:
        assert decider.calls == 0


def test_on_final_decider_given_but_no_evidence_is_other(no_gateway):
    d = _Decider(_CodeVerdict("no_record"))
    res, f = _final(StubC1("local"), decider=d, retriever=FakeRetriever(()))
    assert f["route_final"] == "other" and f["stopped_at"] == "retrieve" and d.calls == 0


def test_on_final_is_called_after_generation_and_audit(tmp_path, no_gateway):
    order = []

    class G:
        def answer(self, q, c):
            order.append("generate")
            return ANSWER
    store = AuditStore(tmp_path / "audit.db")
    try:
        ask(StubC1("local"), USER, QUESTION, POLICY, FakeRetriever(), G(), audit_sink=store,
            on_final=lambda r: order.append(("final", len(store.runs()))))
    finally:
        store.close()
    assert order == ["generate", ("final", 1)]


def test_on_final_log_is_not_in_audit_store(tmp_path, no_gateway):
    store = AuditStore(tmp_path / "audit.db")
    try:
        res, f = (lambda got: (ask(StubC1("local"), USER, QUESTION, POLICY, FakeRetriever(), FakeGenerator(),
                                   audit_sink=store, on_final=got.append), got[0]))([])
        dumped = json.dumps([store.runs(), store.events(res.run.audit_run_id)], ensure_ascii=False, default=str)
        assert ANSWER not in dumped and f["final_answer"] == ANSWER
    finally:
        store.close()
