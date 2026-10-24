from __future__ import annotations

import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import pytest

from common.schemas import (
    C2Verdict,
    CommitRequest,
    FailureKind,
    PrepareRequest,
    PrepareResponse,
    SendState,
    StatusResponse,
    payload_digest,
)
from judge.base import GUARD_QUESTION_VERSION
from judge.base import C2Verdict as JudgeC2Verdict
from orchestrator.audit_store import AuditStore
from orchestrator.candidate import SendCandidate, build_candidate
from orchestrator.pipeline import GatewayNotReached, run_contract
from orchestrator.policy import DEFAULT_POLICY_DIR, JST, check_eligibility, load_policy, sha256_text

NOW = datetime(2026, 10, 20, 10, 0, tzinfo=JST)

A_INPUT = {
    "contract": "A-faq-format@1",
    "explanation": "expl-keihi-001@1",
    "options": {"文体": "です・ます調", "長さ": "400字以内"},
}
B_TEXT = "amount が負の行を除いて、月ごと・category ごとの合計を CSV で出力する関数を書いてください。"
B_INPUT = {
    "contract": "B-csv-codegen@1",
    "spec": "spec-csv-001@1",
    "options": {"言語": "Python 3.11"},
    "free_text": B_TEXT,
}


# ---- Fakes ----

class FakeGateway:
    def __init__(self, tamper_digest: bool = False, final: SendState = SendState.SUCCEEDED):
        self.prepared: list[PrepareRequest] = []
        self.commits: list[CommitRequest] = []
        self.tamper_digest = tamper_digest
        self.final = final

    @property
    def calls(self) -> int:
        return len(self.prepared) + len(self.commits)

    def prepare(self, req: PrepareRequest) -> PrepareResponse:
        self.prepared.append(req)
        d = payload_digest(req.payload)
        if self.tamper_digest:
            d = "0" * 64
        return PrepareResponse(request_id=req.request_id, digest=d, state=SendState.PREPARED)

    def commit(self, req: CommitRequest) -> StatusResponse:
        self.commits.append(req)
        failure = None if self.final == SendState.SUCCEEDED else FailureKind.unknown
        return StatusResponse(request_id=req.request_id, digest=req.expected_digest, state=self.final,
                              failure=failure, output_text="Q1..." if failure is None else None)

    def status(self, request_id: str) -> StatusResponse:
        raise NotImplementedError


class FakeC2:
    def __init__(self, decision: str = "allow", truncated: bool = False, delay: float = 0.0, exc: Exception | None = None,
                 question_version: str | None = "fake@1"):
        self.decision, self.truncated, self.delay, self.exc = decision, truncated, delay, exc
        # None なら質問の版を持たない（common.schemas の）C2Verdict を返す＝形式不正
        self.question_version = question_version
        self.seen = []

    def check(self, payload):
        self.seen.append(payload)
        if self.delay:
            time.sleep(self.delay)
        if self.exc:
            raise self.exc
        kw = dict(decision=self.decision, reason="fake", model="fake", revision="0", truncated=self.truncated)
        if self.question_version is None:
            return C2Verdict(**kw)
        return JudgeC2Verdict(**kw, question_version=self.question_version)


class FakeConfirmer:
    """mode: 'same' 候補の digest を返す / 'none' 未確認 / 'other' 別の digest を返す。"""

    def __init__(self, mode: str = "same", other: str | None = None):
        self.mode, self.other = mode, other
        self.seen: list[SendCandidate] = []

    def confirm(self, cand: SendCandidate) -> str | None:
        self.seen.append(cand)
        if self.mode == "same":
            return cand.digest
        if self.mode == "other":
            return self.other
        return None


def run(user="u_general", inp=None, dest="claude", now=NOW, c2=None, confirmer=None, gw=None, **kw):
    gw = gw or FakeGateway()
    res = run_contract(user, inp or A_INPUT, dest, now, c2 or FakeC2(), confirmer or FakeConfirmer(), gw, **kw)
    return res, gw


# ---- policy files ----

def test_policy_values_match_contract_examples():
    pol = load_policy()
    assert {u: pol.users[u].level for u in pol.users} == {"u_general": 1, "u_manager": 2, "u_exec": 3}
    assert "B-csv-codegen@1" not in pol.users["u_general"].contracts
    a = pol.approved["expl-keihi-001@1"]
    assert a.sha256 == sha256_text(a.body)
    assert a.expires.isoformat() == "2026-12-31" and a.source_level == 1
    assert pol.contracts["A-faq-format@1"].payload_chars == 2000
    assert pol.contracts["B-csv-codegen@1"].free_text_max == 200


def test_candidate_body_follows_template_and_is_frozen():
    pol = load_policy()
    e = check_eligibility(pol, "u_general", A_INPUT, "claude", NOW)
    c = build_candidate(user_id="u_general", contract=e.contract, approved=e.approved, inp=A_INPUT, destination="claude", now=NOW)
    sys_msg, user_msg = c.payload.messages
    assert sys_msg.role == "system" and sys_msg.content.startswith("あなたは社内向け FAQ の編集者です。")
    assert user_msg.content.startswith("文体: です・ます調\n長さ: 400字以内\n形式: 質問と回答の組を3〜5個\n\n説明文:\n")
    assert user_msg.content.endswith(e.approved.body)
    assert c.digest == payload_digest(c.payload)
    with pytest.raises(Exception):
        c.digest = "x"  # type: ignore[misc]


# ---- 成功 ----

def test_v001_contract_a_sent():
    res, gw = run()
    assert (res.outcome, res.stopped_at) == ("sent", "none")
    assert res.gateway_received and res.attempted and res.gateway_state == SendState.SUCCEEDED
    assert len(gw.prepared) == 1 and len(gw.commits) == 1
    assert gw.commits[0].expected_digest == res.digest == payload_digest(gw.prepared[0].payload)
    assert [e.stage for e in res.audit] == ["eligibility", "candidate", "C2", "confirm", "recheck", "gateway_received", "attempted", "result"]


def test_v011_contract_b_sent_codex():
    res, gw = run(user="u_manager", inp=B_INPUT, dest="codex")
    assert res.outcome == "sent"
    body = gw.prepared[0].payload.messages[1].content
    assert body.startswith("言語: Python 3.11\n仕様:\n入力: UTF-8 の CSV。") and body.endswith("依頼:\n" + B_TEXT)
    assert gw.prepared[0].payload.max_output_chars == 3000


def test_v002_bullet_option_allowed():
    res, _ = run(user="u_manager", inp={**A_INPUT, "options": {"文体": "箇条書き", "長さ": "200字以内"}})
    assert res.outcome == "sent"


# ---- 契約 A の拒否・保留 ----

def test_a_r1_lv3_source_rejected():
    inp = {"contract": "A-faq-format@1", "explanation": None, "source": "executive/M&A検討資料.md"}
    res, gw = run(inp=inp)
    assert (res.outcome, res.stopped_at) == ("rejected", "eligibility")
    assert gw.calls == 0 and not res.gateway_received


@pytest.fixture
def policy_copy(tmp_path: Path) -> Path:
    d = tmp_path / "policies"
    shutil.copytree(DEFAULT_POLICY_DIR, d)
    return d


def test_a_r2_tampered_text_rejected(policy_copy: Path):
    p = policy_copy / "approved" / "expl-keihi-001@1.json"
    a = json.loads(p.read_text(encoding="utf-8"))
    a["body"] = a["body"].replace("翌月10日", "翌月20日")
    p.write_text(json.dumps(a, ensure_ascii=False), encoding="utf-8")
    res, gw = run(policy=load_policy(policy_copy))
    assert (res.outcome, res.stopped_at) == ("rejected", "eligibility")
    assert "hash" in res.reason and gw.calls == 0


def test_a_r3_expired_rejected():
    res, gw = run(now=datetime(2027, 1, 1, 0, 0, tzinfo=JST))
    assert (res.outcome, res.stopped_at) == ("rejected", "eligibility")
    assert gw.calls == 0


def test_a_last_day_still_valid():
    res, _ = run(now=datetime(2026, 12, 31, 23, 59, tzinfo=JST))
    assert res.outcome == "sent"


def test_a_r4_c2_timeout_held():
    res, gw = run(c2=FakeC2(delay=0.5), c2_timeout_s=0.05)
    assert (res.outcome, res.stopped_at) == ("held", "C2")
    assert gw.calls == 0 and not res.gateway_received


def test_a_r5_options_changed_after_confirm_digest_mismatch():
    pol = load_policy()
    e = check_eligibility(pol, "u_general", A_INPUT, "claude", NOW)
    old_inp = {**A_INPUT, "options": {"文体": "です・ます調", "長さ": "200字以内"}}
    seen = build_candidate(user_id="u_general", contract=e.contract, approved=e.approved, inp=old_inp, destination="claude", now=NOW)
    res, gw = run(confirmer=FakeConfirmer("other", seen.digest))
    assert (res.outcome, res.stopped_at) == ("rejected", "digest")
    assert res.digest != seen.digest and gw.calls == 0


def test_a_rejects_free_text():
    res, gw = run(inp={**A_INPUT, "free_text": "社内規程の全文も足して"})
    assert (res.outcome, res.stopped_at) == ("rejected", "contract") and gw.calls == 0


@pytest.mark.parametrize("opts", [
    {"文体": "砕けた口調", "長さ": "400字以内"},
    {"文体": "です・ます調", "長さ": "5000字以内"},
    {"文体": "です・ます調"},
    {"文体": "です・ます調", "長さ": "400字以内", "追記": "M&A"},
])
def test_a_option_allowlist(opts):
    res, gw = run(inp={**A_INPUT, "options": opts})
    assert (res.outcome, res.stopped_at) == ("rejected", "contract") and gw.calls == 0


def test_destination_not_allowed():
    res, gw = run(dest="gemini")
    assert res.stopped_at == "eligibility" and gw.calls == 0


# ---- 契約 B の拒否・保留 ----

def test_b_r5_general_user_rejected():
    res, gw = run(user="u_general", inp=B_INPUT, dest="codex")
    assert (res.outcome, res.stopped_at) == ("rejected", "eligibility") and gw.calls == 0


def test_b_r6_free_text_over_200_rejected():
    res, gw = run(user="u_manager", inp={**B_INPUT, "free_text": "あ" * 201}, dest="codex")
    assert (res.outcome, res.stopped_at) == ("rejected", "contract") and gw.calls == 0


def test_b_free_text_200_ok():
    res, _ = run(user="u_manager", inp={**B_INPUT, "free_text": "あ" * 200}, dest="codex")
    assert res.outcome == "sent"


def test_b_r7_unconfirmed_held():
    res, gw = run(user="u_manager", inp=B_INPUT, dest="codex", confirmer=FakeConfirmer("none"))
    assert (res.outcome, res.stopped_at) == ("held", "confirm") and gw.calls == 0


# ---- C2 が allow 以外なら Gateway を呼ばない ----

@pytest.mark.parametrize("c2", [
    FakeC2("block"),
    FakeC2("hold"),
    FakeC2("allow", truncated=True),
    FakeC2(exc=TimeoutError("slow")),
    FakeC2(exc=RuntimeError("down")),
])
@pytest.mark.parametrize("user,inp", [("u_general", A_INPUT), ("u_manager", B_INPUT)])
def test_c2_non_allow_never_reaches_gateway(c2, user, inp):
    conf = FakeConfirmer()
    dest = "codex" if inp is B_INPUT else "claude"
    res, gw = run(user=user, inp=inp, dest=dest, c2=c2, confirmer=conf)
    assert (res.outcome, res.stopped_at) == ("held", "C2")
    assert gw.calls == 0 and not res.gateway_received and not res.attempted
    assert conf.seen == []


# ---- Gateway 側 ----

def test_gateway_digest_mismatch_stops_before_commit():
    res, gw = run(gw=FakeGateway(tamper_digest=True))
    assert (res.outcome, res.stopped_at) == ("rejected", "digest")
    assert res.gateway_received and not res.attempted and gw.commits == []


def test_gateway_unknown_not_reported_as_sent():
    res, _ = run(gw=FakeGateway(final=SendState.FAILED))
    assert res.outcome == "held" and res.attempted and res.gateway_state == SendState.FAILED


class RaisingPrepareGateway(FakeGateway):
    def __init__(self, exc: Exception):
        super().__init__()
        self.exc = exc

    def prepare(self, req: PrepareRequest) -> PrepareResponse:
        self.prepared.append(req)
        raise self.exc


def test_prepare_not_reached_marks_gateway_not_received():
    res, gw = run(gw=RaisingPrepareGateway(GatewayNotReached("connect refused")))
    assert (res.outcome, res.stopped_at) == ("held", "none")
    assert not res.gateway_received and not res.attempted and gw.commits == []


@pytest.mark.parametrize("exc", [TimeoutError("read timeout"), RuntimeError("HTTP 500")])
def test_prepare_unknown_failure_assumes_received(exc):
    """タイムアウト等は届いたか不明なので安全側（届いたかもしれない）に倒す。"""
    res, gw = run(gw=RaisingPrepareGateway(exc))
    assert (res.outcome, res.stopped_at) == ("held", "none")
    assert res.gateway_received and not res.attempted and gw.commits == []


# ---- response_visibility の読み込み ----

def test_policy_loads_response_visibility():
    pol = load_policy()
    assert pol.contracts["A-faq-format@1"].response_visibility == "source_level_requester_only"
    assert pol.contracts["B-csv-codegen@1"].response_visibility == "requester_only_no_exec"


@pytest.mark.parametrize("value", [None, "public"])
def test_policy_rejects_unknown_response_visibility(policy_copy: Path, value):
    p = policy_copy / "contracts.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    if value is None:
        del data["B-csv-codegen@1"]["response_visibility"]
    else:
        data["B-csv-codegen@1"]["response_visibility"] = value
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="response_visibility"):
        load_policy(policy_copy)


def test_rejected_run_has_no_visibility():
    res, gw = run(user="u_general", inp=B_INPUT, dest="codex")
    assert res.outcome == "rejected" and res.response_visibility is None and gw.calls == 0
    assert not res.can_view("u_general", 3)


# ---- C2 の応答が C2Verdict でない・形式不正でも held/C2 で止め、監査に残す ----

class RawC2:
    """check が任意の値をそのまま返す C2 実装。"""

    def __init__(self, raw):
        self.raw = raw
        self.calls = 0

    def check(self, payload):
        self.calls += 1
        return self.raw


class ExplodingVerdict:
    """属性アクセスのたびに例外を投げる応答。"""

    def __getattr__(self, name):
        raise RuntimeError(f"boom on {name}")


_VALID_DICT = dict(decision="allow", reason="dict", model="m", revision="1", question_version="q@1")


@pytest.mark.parametrize("raw", [
    pytest.param({k: v for k, v in _VALID_DICT.items() if k != "question_version"}, id="dict_without_question_version"),
    pytest.param({"decision": "maybe", "reason": "x", "model": "m", "revision": "1", "question_version": "q@1"},
                 id="dict_bad_decision"),
    pytest.param({"foo": "bar"}, id="dict_wrong_shape"),
    pytest.param(None, id="none"),
    pytest.param("allow", id="string"),
    pytest.param(C2Verdict(decision="allow", reason="no qv", model="m", revision="1"), id="verdict_without_question_version"),
    pytest.param(ExplodingVerdict(), id="attribute_access_raises"),
])
@pytest.mark.parametrize("user,inp,dest", [("u_general", A_INPUT, "claude"), ("u_manager", B_INPUT, "codex")])
def test_c2_malformed_response_held_and_persisted(tmp_path, raw, user, inp, dest):
    store = AuditStore(tmp_path / "audit.db")
    try:
        conf = FakeConfirmer()
        res, gw = run(user=user, inp=inp, dest=dest, c2=RawC2(raw), confirmer=conf, audit_sink=store)
        assert (res.outcome, res.stopped_at) == ("held", "C2")
        assert gw.calls == 0 and gw.prepared == [] and not res.gateway_received and not res.attempted
        assert conf.seen == []
        assert res.audit_persisted is True
        assert res.c2_verdict is not None and res.c2_verdict.decision == "hold"
        assert res.c2_verdict.question_version == GUARD_QUESTION_VERSION
        (row,) = store.runs()
        assert (row["outcome"], row["stopped_at"]) == ("held", "C2")
        assert row["c2_decision"] == "hold" and row["c2_question_version"] == GUARD_QUESTION_VERSION
    finally:
        store.close()


def test_c2_valid_dict_response_is_accepted():
    res, gw = run(c2=RawC2(dict(_VALID_DICT)))
    assert res.outcome == "sent" and len(gw.commits) == 1
    assert res.c2_verdict.question_version == "q@1"


@pytest.mark.parametrize("c2", [
    FakeC2(exc=RuntimeError("down")),
    FakeC2(delay=0.5),
    FakeC2("allow", truncated=True),
    FakeC2(question_version=None),
], ids=["exception", "timeout", "truncated", "no_question_version"])
def test_c2_failures_held_and_persisted(tmp_path, c2):
    store = AuditStore(tmp_path / "audit.db")
    try:
        res, gw = run(c2=c2, audit_sink=store, c2_timeout_s=0.05)
        assert (res.outcome, res.stopped_at) == ("held", "C2") and gw.calls == 0
        assert res.audit_persisted is True
        (row,) = store.runs()
        assert (row["outcome"], row["stopped_at"], row["c2_decision"]) == ("held", "C2", "hold")
    finally:
        store.close()


def test_c2_dict_allow_truncated_held_keeps_judge_question_version(tmp_path):
    store = AuditStore(tmp_path / "audit.db")
    try:
        res, gw = run(c2=RawC2({**_VALID_DICT, "truncated": True}), audit_sink=store)
        assert (res.outcome, res.stopped_at) == ("held", "C2") and gw.calls == 0
        assert res.c2_verdict.decision == "hold" and res.c2_verdict.truncated
        (row,) = store.runs()
        assert row["c2_decision"] == "hold" and row["c2_truncated"] == 1 and row["c2_question_version"] == "q@1"
    finally:
        store.close()


def test_c2_timeout_none_waits_without_timeout():
    # c2_timeout_s=None はタイムアウトなし（従来互換）。遅い C2 でも結果を待って使う
    res, gw = run(c2=FakeC2(delay=0.1), c2_timeout_s=None)
    assert res.outcome == "sent" and len(gw.commits) == 1
