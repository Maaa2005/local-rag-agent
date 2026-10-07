"""宛先固定（1 契約 1 宛先）の Gateway 側の検査。

- 社内 policies/contracts.json と gateway CONTRACTS の契約キー・版・宛先が一致する
- prepare は逆の宛先（A→codex、B→claude）と旧版・未知版を拒否する
- 旧方針（1 契約 2 宛先）で作られた PREPARED は commit で再検査され、送られない
- 障害時に別の API へ自動で切り替えない
"""
from __future__ import annotations

import json
import time
from dataclasses import replace

import pytest

from common.schemas import CommitRequest, FailureKind, Message, PrepareRequest, SendPayload, SendState
from gateway.adapters import FakeAdapter
from gateway.contracts import CONTRACTS
from gateway.service import ContractRevoked, GatewayService, InvalidRequest, UnknownContract
from gateway.store import SendStore
from orchestrator.policy import DEFAULT_POLICY_DIR, load_contracts

A, B = "A-faq-format@1", "B-csv-codegen@1"


def _req(rid: str, contract: str, dest: str) -> PrepareRequest:
    payload = SendPayload(
        contract=contract,
        destination=dest,
        messages=(Message(role="system", content="sys"), Message(role="user", content="hello")),
        max_output_chars=100,
    )
    return PrepareRequest(request_id=rid, payload=payload, expires_at=time.time() + 60)


@pytest.fixture
def adapters():
    return {"claude": FakeAdapter(reply="c"), "codex": FakeAdapter(reply="x")}


@pytest.fixture
def svc(tmp_path, adapters):
    return GatewayService(SendStore(tmp_path / "gw.db"), adapters)


# ---- 社内契約定義と Gateway レジストリの一致 ----

def test_policy_and_gateway_registry_match():
    policy = load_contracts(DEFAULT_POLICY_DIR / "contracts.json")
    raw = json.loads((DEFAULT_POLICY_DIR / "contracts.json").read_text(encoding="utf-8"))
    assert set(policy) == set(CONTRACTS)
    for key, c in policy.items():
        g = CONTRACTS[key]
        assert key == f'{raw[key]["id"]}@{raw[key]["version"]}'
        assert set(c.destinations) == set(g.destinations)
        assert len(c.destinations) == 1 and len(g.destinations) == 1
        assert c.enabled == g.enabled
        assert (c.payload_chars, c.output_chars) == (g.payload_chars, g.output_chars)
    assert policy[A].destinations == ("claude",) and policy[A].route == "A"
    assert policy[B].destinations == ("codex",) and policy[B].route == "B"


# ---- prepare: 逆の宛先・旧版を拒否 ----

@pytest.mark.parametrize("contract,dest", [(A, "codex"), (B, "claude")])
def test_prepare_rejects_reverse_destination(svc, adapters, contract, dest):
    with pytest.raises(InvalidRequest):
        svc.prepare(_req("req-rev-0001", contract, dest))
    assert adapters["claude"].received == [] and adapters["codex"].received == []


@pytest.mark.parametrize("contract,dest", [("A-faq-format@2", "claude"), ("B-csv-codegen@0", "codex"),
                                           ("A-faq-format", "claude")])
def test_prepare_rejects_old_or_unknown_version(svc, contract, dest):
    with pytest.raises(UnknownContract):
        svc.prepare(_req("req-old-0001", contract, dest))


@pytest.mark.parametrize("contract,dest", [(A, "claude"), (B, "codex")])
def test_prepare_accepts_sole_destination(svc, contract, dest):
    svc.prepare(_req("req-ok-00001", contract, dest))


def test_prepare_rejects_disabled_or_multi_destination(svc, monkeypatch):
    monkeypatch.setitem(CONTRACTS, A, replace(CONTRACTS[A], enabled=False))
    with pytest.raises(InvalidRequest):
        svc.prepare(_req("req-dis-0001", A, "claude"))
    monkeypatch.setitem(CONTRACTS, A, replace(CONTRACTS[A], enabled=True,
                                              destinations=frozenset({"claude", "codex"})))
    with pytest.raises(InvalidRequest):
        svc.prepare(_req("req-two-0001", A, "claude"))


# ---- 旧方針で作られた PREPARED は送らない ----

def test_old_policy_prepared_is_not_sent(svc, adapters, monkeypatch):
    # 旧方針（1 契約 2 宛先・宛先が許可集合に入っていれば可）の検査で codex 宛の PREPARED を作る
    import gateway.service as gs

    def legacy_check(contract, destination):
        spec = CONTRACTS.get(contract)
        if spec is None:
            raise UnknownContract("unknown contract")
        if destination not in {"claude", "codex"}:
            raise InvalidRequest("destination not allowed")
        return spec

    with monkeypatch.context() as m:
        m.setattr(gs, "check_contract_destination", legacy_check)
        old = svc.prepare(_req("req-legacy01", A, "codex"))
    assert svc.store.get(old.request_id).state == SendState.PREPARED
    # 現行（A→claude のみ）で commit
    with pytest.raises(ContractRevoked):
        svc.commit(CommitRequest(request_id=old.request_id, expected_digest=old.digest))
    assert adapters["codex"].received == [] and adapters["claude"].received == []
    assert svc.store.get(old.request_id).state == SendState.PREPARED


def test_disabled_after_prepare_is_not_sent(svc, adapters, monkeypatch):
    res = svc.prepare(_req("req-disab01", B, "codex"))
    monkeypatch.setitem(CONTRACTS, B, replace(CONTRACTS[B], enabled=False))
    with pytest.raises(ContractRevoked):
        svc.commit(CommitRequest(request_id=res.request_id, expected_digest=res.digest))
    assert adapters["codex"].received == []
    assert svc.store.get(res.request_id).state == SendState.PREPARED


def test_contract_revoked_is_invalid_request():
    assert issubclass(ContractRevoked, InvalidRequest)


# ---- 障害時に別 API へ切り替えない ----

@pytest.mark.parametrize("contract,dest,other,mode,kind", [
    (A, "claude", "codex", "not_sent", FailureKind.not_sent),
    (B, "codex", "claude", "not_sent", FailureKind.not_sent),
    (A, "claude", "codex", "unknown", FailureKind.unknown),
])
def test_no_failover_to_other_api(tmp_path, contract, dest, other, mode, kind):
    adapters = {dest: FakeAdapter(mode=mode), other: FakeAdapter()}
    svc = GatewayService(SendStore(tmp_path / "gw.db"), adapters)
    res = svc.prepare(_req("req-fail-001", contract, dest))
    st = svc.commit(CommitRequest(request_id=res.request_id, expected_digest=res.digest))
    assert st.state == SendState.FAILED and st.failure == kind
    assert len(adapters[dest].received) == 1 and adapters[other].received == []
    # 再 commit しても再送・切り替えしない
    st2 = svc.commit(CommitRequest(request_id=res.request_id, expected_digest=res.digest))
    assert st2.state == SendState.FAILED
    assert len(adapters[dest].received) == 1 and adapters[other].received == []
