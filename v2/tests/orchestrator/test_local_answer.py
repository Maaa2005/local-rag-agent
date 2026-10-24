from __future__ import annotations

import asyncio
import sys
import time
import types

import pytest

from orchestrator.local_answer import MAX_ANSWER_CHARS, Chunk, answer_locally
from orchestrator.policy import load_policy
from orchestrator.rag_backend import BackendGenerator, BackendRetriever, to_chunk

POLICY = load_policy()
GENERAL = "u_general"  # level 1
MANAGER = "u_manager"  # level 2
QUESTION = "経費精算の締め日はいつですか（架空社内規程の確認）"


class FakeRetriever:
    def __init__(self, chunks=(), exc: Exception | None = None) -> None:
        self.chunks = list(chunks)
        self.exc = exc
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, question, level):
        self.calls.append((question, level))
        if self.exc:
            raise self.exc
        return list(self.chunks)


class FakeGenerator:
    def __init__(self, text: str = "締め日は毎月25日です [1]", exc: Exception | None = None) -> None:
        self.text = text
        self.exc = exc
        self.seen: list[Chunk] = []

    def answer(self, question, chunks):
        self.seen = list(chunks)
        if self.exc:
            raise self.exc
        return self.text


def ch(level, doc="d1", content="経費精算は毎月25日締め（架空規程 第3条）", source="keihi.md"):
    return Chunk(content=content, source=source, access_level=level, document_id=doc, chunk_index=0)


def test_level_is_passed_to_retriever_before_search():
    r = FakeRetriever([ch(1)])
    answer_locally(MANAGER, QUESTION, POLICY, r, FakeGenerator())
    assert r.calls == [(QUESTION, 2)]


def test_answered_with_sources_and_visibility():
    res = answer_locally(GENERAL, QUESTION, POLICY, FakeRetriever([ch(1, source="keihi.md")]), FakeGenerator())
    run = res.run
    assert run.outcome == "answered" and run.stopped_at == "none"
    assert run.output_text == "締め日は毎月25日です [1]"
    assert res.sources == ("keihi.md",)
    assert run.can_view(GENERAL, 1)
    assert not run.can_view(MANAGER, 2)  # 依頼者本人だけ
    assert not run.can_view(GENERAL, None)


def test_higher_level_chunks_are_dropped_even_if_retriever_leaks_them():
    secret = ch(2, doc="d_secret", content="役員報酬の改定案（架空）は来期から適用")
    gen = FakeGenerator()
    res = answer_locally(GENERAL, QUESTION, POLICY, FakeRetriever([ch(1), secret]), gen)
    assert res.run.outcome == "answered"
    assert [c.document_id for c in gen.seen] == ["d1"]
    detail = next(e.detail for e in res.run.audit if e.stage == "retrieve")
    assert "dropped=1" in detail and "d_secret" not in detail
    assert res.run.min_view_level == 1


@pytest.mark.parametrize("level", [None, True, "1", 1.0])
def test_unknown_level_chunks_are_dropped(level):
    gen = FakeGenerator()
    res = answer_locally(MANAGER, QUESTION, POLICY, FakeRetriever([ch(level)]), gen)
    assert res.run.outcome == "held" and res.run.stopped_at == "retrieve"
    assert gen.seen == []


def test_min_view_level_is_max_source_level():
    res = answer_locally(MANAGER, QUESTION, POLICY, FakeRetriever([ch(1, "a"), ch(2, "b")]), FakeGenerator())
    assert res.run.min_view_level == 2
    assert res.run.can_view(MANAGER, 2) and not res.run.can_view(MANAGER, 1)


def test_unregistered_user_is_held_without_search():
    r = FakeRetriever([ch(1)])
    res = answer_locally("u_unknown", QUESTION, POLICY, r, FakeGenerator())
    assert res.run.outcome == "held" and res.run.stopped_at == "eligibility"
    assert r.calls == []


def test_retriever_failure_is_held():
    res = answer_locally(GENERAL, QUESTION, POLICY, FakeRetriever(exc=ConnectionError("qdrant down")), FakeGenerator())
    assert res.run.outcome == "held" and res.run.stopped_at == "retrieve"
    assert res.run.output_text is None


def test_zero_hits_is_held_without_generation():
    gen = FakeGenerator()
    res = answer_locally(GENERAL, QUESTION, POLICY, FakeRetriever([]), gen)
    assert res.run.outcome == "held" and res.run.stopped_at == "retrieve"
    assert gen.seen == []


@pytest.mark.parametrize("gen", [FakeGenerator(exc=TimeoutError()), FakeGenerator(text=""), FakeGenerator(text="  \n")])
def test_generation_failure_or_empty_is_held(gen):
    res = answer_locally(GENERAL, QUESTION, POLICY, FakeRetriever([ch(1)]), gen)
    assert res.run.outcome == "held" and res.run.stopped_at == "generate"
    assert res.run.output_text is None and not res.run.can_view(GENERAL, 1)


def test_long_answer_is_truncated():
    res = answer_locally(GENERAL, QUESTION, POLICY, FakeRetriever([ch(1)]), FakeGenerator("あ" * (MAX_ANSWER_CHARS + 10)))
    assert len(res.run.output_text) == MAX_ANSWER_CHARS
    assert any(e.stage == "generate" and "truncated=True" in e.detail for e in res.run.audit)


def test_bodies_cover_question_chunks_and_answer():
    c = ch(1)
    res = answer_locally(GENERAL, QUESTION, POLICY, FakeRetriever([c]), FakeGenerator())
    assert QUESTION in res.bodies and c.content in res.bodies and res.run.output_text in res.bodies


# ---- v1 バックエンドへの接続（偽の app モジュールで試す。実 Qdrant / vLLM は実機で確認） ----

@pytest.fixture
def fake_backend(monkeypatch):
    calls = {}

    async def retrieve(question, user_access_level):
        calls["retrieve"] = (question, user_access_level)
        calls.setdefault("loops", []).append(asyncio.get_running_loop())
        return [
            {"content": "本文A", "source_file": "/srv/watch/経理部/keihi.md", "score": 0.9, "access_level": 1,
             "document_id": "d1", "chunk_index": 3},
            {"content": "本文B", "source_file": "C:\\share\\人事\\jinji.md", "score": 0.5, "access_level": None,
             "document_id": "d2", "chunk_index": 0},
        ]

    async def stream_answer(question, chunks):
        calls["stream"] = (question, chunks)
        calls.setdefault("loops", []).append(asyncio.get_running_loop())
        for t in ["締め日は", "25日", " [1]"]:
            yield t

    r = types.ModuleType("fake_retriever")
    r.retrieve = retrieve
    g = types.ModuleType("fake_llm")
    g.stream_answer = stream_answer
    monkeypatch.setitem(sys.modules, "fake_retriever", r)
    monkeypatch.setitem(sys.modules, "fake_llm", g)
    return calls


def test_backend_retriever_maps_rows(fake_backend):
    got = BackendRetriever("fake_retriever").retrieve("q", 2)
    assert fake_backend["retrieve"] == ("q", 2)
    assert got[0] == Chunk(content="本文A", source="keihi.md", access_level=1, document_id="d1", chunk_index=3)
    assert got[1].source == "jinji.md" and got[1].access_level is None


def test_backend_generator_joins_stream_and_passes_basename_only(fake_backend):
    text = BackendGenerator("fake_llm").answer("q", [Chunk("本文A", "keihi.md", 1, "d1", 3)])
    assert text == "締め日は25日 [1]"
    _, rows = fake_backend["stream"]
    assert rows == [{"content": "本文A", "source_file": "keihi.md", "access_level": 1, "document_id": "d1",
                     "chunk_index": 3}]


def test_backend_end_to_end_drops_unknown_level(fake_backend):
    res = answer_locally(MANAGER, "q", POLICY, BackendRetriever("fake_retriever"), BackendGenerator("fake_llm"))
    assert res.run.outcome == "answered" and res.sources == ("keihi.md",)
    assert [r["document_id"] for r in fake_backend["stream"][1]] == ["d1"]


def test_backend_reuses_one_loop_across_questions(fake_backend):
    # v1 は AsyncQdrantClient を使い回すため、2 問目以降も同じループで動かないと失敗する
    r, g = BackendRetriever("fake_retriever"), BackendGenerator("fake_llm")
    for _ in range(2):
        assert answer_locally(MANAGER, "q", POLICY, r, g).run.outcome == "answered"
    loops = fake_backend["loops"]
    assert len(loops) == 4 and all(lp is loops[0] for lp in loops)


def test_backend_callable_from_running_loop(fake_backend):
    async def inside():
        return BackendRetriever("fake_retriever").retrieve("q", 1)
    assert [c.document_id for c in asyncio.run(inside())] == ["d1", "d2"]


def test_backend_timeout_is_held(monkeypatch):
    async def slow_retrieve(question, user_access_level):
        await asyncio.sleep(5)
        return []

    async def slow_stream(question, chunks):
        await asyncio.sleep(5)
        yield "x"

    r = types.ModuleType("slow_retriever")
    r.retrieve = slow_retrieve
    g = types.ModuleType("slow_llm")
    g.stream_answer = slow_stream
    monkeypatch.setitem(sys.modules, "slow_retriever", r)
    monkeypatch.setitem(sys.modules, "slow_llm", g)
    t0 = time.monotonic()
    res = answer_locally(GENERAL, "q", POLICY, BackendRetriever("slow_retriever", timeout_s=0.1), FakeGenerator())
    assert res.run.outcome == "held" and res.run.stopped_at == "retrieve" and "TimeoutError" in res.run.reason
    res = answer_locally(GENERAL, "q", POLICY, FakeRetriever([ch(1)]), BackendGenerator("slow_llm", timeout_s=0.1))
    assert res.run.outcome == "held" and res.run.stopped_at == "generate" and "TimeoutError" in res.run.reason
    assert time.monotonic() - t0 < 2


def test_backend_import_failure_is_held():
    res = answer_locally(GENERAL, "q", POLICY, BackendRetriever("no_such_module_xyz"), FakeGenerator())
    assert res.run.outcome == "held" and res.run.stopped_at == "retrieve"


@pytest.mark.parametrize("level,expected", [(True, None), ("2", None), (2, 2), (0, 0)])
def test_to_chunk_level(level, expected):
    assert to_chunk({"content": "x", "source_file": "a.md", "access_level": level}).access_level == expected


def test_to_chunk_missing_source():
    assert to_chunk({"content": "x"}).source == "不明"


# ---- decider（「コードが決める」方式）の差し込み。既定は無し（上のテストは decider なしの挙動）

class _Verdict:
    def __init__(self, kind, text="", used=(), record=None):
        self.kind, self.answer_text, self.used_chunks = kind, text, tuple(used)
        self.decision_record = record or {}


class FakeDecider:
    def __init__(self, verdict=None, exc: Exception | None = None):
        self.verdict, self.exc, self.calls = verdict, exc, []

    def decide(self, question, chunks):
        self.calls.append((question, list(chunks)))
        if self.exc:
            raise self.exc
        return self.verdict


def test_decider_default_is_none_and_signature_is_optional():
    import inspect
    p = inspect.signature(answer_locally).parameters["decider"]
    assert p.default is None and p.kind is inspect.Parameter.KEYWORD_ONLY


def test_decided_answer_skips_generator_and_uses_used_chunks():
    low, high = ch(1, doc="d1", source="a.md"), ch(2, doc="d2", source="b.md")
    v = _Verdict("decided", "【判定】\n結論: 必要", used=[high], record={"rule_id": "R-x", "outcome_key": "all_true"})
    g = FakeGenerator()
    res = answer_locally(MANAGER, QUESTION, POLICY, FakeRetriever([low, high]), g, decider=FakeDecider(v))
    assert res.run.outcome == "answered" and res.run.output_text == "【判定】\n結論: 必要"
    assert g.seen == []  # モデルは呼ばれない
    assert res.sources == ("b.md",) and res.run.min_view_level == 2
    assert res.run.visible_to == MANAGER and res.run.response_visibility == "source_level_requester_only"
    ev = {e.stage: e for e in res.run.audit}
    assert ev["decide"].result == "ok" and "rule=R-x" in ev["decide"].detail


def test_decider_sees_only_allowed_chunks():
    over = ch(2, doc="secret", source="s.md")
    d = FakeDecider(_Verdict("no_record"))
    answer_locally(GENERAL, QUESTION, POLICY, FakeRetriever([ch(1), over]), FakeGenerator(), decider=d)
    assert [c.document_id for c in d.calls[0][1]] == ["d1"]


def test_held_verdict_is_held_without_generator_and_carries_code_only():
    v = _Verdict("held", record={"hold_code": "facts:missing"})
    g = FakeGenerator()
    res = answer_locally(GENERAL, QUESTION, POLICY, FakeRetriever([ch(1)]), g, decider=FakeDecider(v))
    assert res.run.outcome == "held" and res.run.stopped_at == "decide"
    assert "facts:missing" in res.run.reason and g.seen == []


def test_no_record_falls_through_to_generator():
    g = FakeGenerator()
    res = answer_locally(GENERAL, QUESTION, POLICY, FakeRetriever([ch(1)]), g, decider=FakeDecider(_Verdict("no_record")))
    assert res.run.outcome == "answered" and res.run.output_text == "締め日は毎月25日です [1]"


@pytest.mark.parametrize("decider", [
    FakeDecider(exc=RuntimeError("boom")),
    FakeDecider(_Verdict("???")),
    FakeDecider(_Verdict("decided", "", used=[])),
    FakeDecider(_Verdict("decided", "結論: 必要", used=[])),  # 根拠なし
])
def test_decider_failure_or_unusable_verdict_is_held_not_generated(decider):
    g = FakeGenerator()
    res = answer_locally(GENERAL, QUESTION, POLICY, FakeRetriever([ch(1)]), g, decider=decider)
    assert res.run.outcome == "held" and res.run.stopped_at == "decide" and g.seen == []


def test_decider_not_called_without_evidence():
    d = FakeDecider(_Verdict("no_record"))
    res = answer_locally(GENERAL, QUESTION, POLICY, FakeRetriever([]), FakeGenerator(), decider=d)
    assert res.run.stopped_at == "retrieve" and d.calls == []


def test_decider_events_carry_no_body_text():
    secret = "98,000円の申請"
    v = _Verdict("decided", "結論: 必要 " + secret, used=[ch(1)], record={"rule_id": "R-x", "outcome_key": "all_true"})
    res = answer_locally(GENERAL, secret, POLICY, FakeRetriever([ch(1)]), FakeGenerator(), decider=FakeDecider(v))
    assert all(secret not in (e.detail or "") for e in res.run.audit)
    assert secret in res.bodies[-1]  # 本文は監査の除去用に bodies へ


# ---- 判断の記録（plan 7.4）: decider が走った時だけ LocalResult.decision_record に入る

def test_decision_record_field_defaults_to_none():
    import dataclasses
    from orchestrator.local_answer import LocalResult
    f = {x.name: x for x in dataclasses.fields(LocalResult)}
    assert f["decision_record"].default is None


def test_decision_record_filled_for_decided_held_and_no_record():
    rec = {"rule_id": "R-x", "outcome_key": "all_true", "hold_code": "facts:missing"}
    dec = answer_locally(GENERAL, QUESTION, POLICY, FakeRetriever([ch(1)]), FakeGenerator(),
                         decider=FakeDecider(_Verdict("decided", "結論: 必要", used=[ch(1)], record=rec)))
    held = answer_locally(GENERAL, QUESTION, POLICY, FakeRetriever([ch(1)]), FakeGenerator(),
                          decider=FakeDecider(_Verdict("held", record=rec)))
    nor = answer_locally(GENERAL, QUESTION, POLICY, FakeRetriever([ch(1)]), FakeGenerator(),
                         decider=FakeDecider(_Verdict("no_record", record={"route": "model_only"})))
    assert (dec.decision_record, dec.decider_kind) == (rec, "decided")
    assert (held.decision_record, held.decider_kind) == (rec, "held")
    assert (nor.decision_record, nor.decider_kind) == ({"route": "model_only"}, "no_record")
    assert nor.run.outcome == "answered"  # モデルが答えても記録は残る


def test_decision_record_survives_generation_failure_after_no_record():
    res = answer_locally(GENERAL, QUESTION, POLICY, FakeRetriever([ch(1)]), FakeGenerator(exc=RuntimeError("gpu")),
                         decider=FakeDecider(_Verdict("no_record", record={"route": "model_only"})))
    assert res.run.outcome == "held" and res.run.stopped_at == "generate"
    assert res.decision_record == {"route": "model_only"} and res.decider_kind == "no_record"


def test_decision_record_is_none_without_decider_or_before_it_runs():
    g = answer_locally(GENERAL, QUESTION, POLICY, FakeRetriever([ch(1)]), FakeGenerator())
    nochunk = answer_locally(GENERAL, QUESTION, POLICY, FakeRetriever([]), FakeGenerator(),
                             decider=FakeDecider(_Verdict("no_record")))
    assert g.decision_record is None and g.decider_kind is None
    assert nochunk.decision_record is None and nochunk.decider_kind is None


def test_decider_exception_leaves_no_record_and_marks_error():
    res = answer_locally(GENERAL, QUESTION, POLICY, FakeRetriever([ch(1)]), FakeGenerator(),
                         decider=FakeDecider(exc=RuntimeError("boom")))
    assert res.decision_record is None and res.decider_kind == "error"
