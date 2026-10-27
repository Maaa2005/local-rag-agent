from __future__ import annotations

import time

from fastapi.testclient import TestClient

from common.schemas import Message, SendPayload, payload_digest
from gateway.adapters import FakeAdapter
from gateway.app import create_app
from gateway.service import GatewayService
from gateway.store import SendStore


def _body(rid="req-app-0001", text="hello"):
    payload = SendPayload(
        contract="A-faq-format@1",
        destination="claude",
        messages=(Message(role="user", content=text),),
        max_output_chars=50,
    )
    return payload, {
        "request_id": rid,
        "payload": payload.model_dump(mode="json"),
        "expires_at": time.time() + 60,
    }


def test_http_flow(tmp_path):
    fake = FakeAdapter(reply="out")
    svc = GatewayService(SendStore(tmp_path / "db"), {"claude": fake})
    with TestClient(create_app(svc)) as c:
        payload, body = _body()
        r = c.post("/prepare", json=body)
        assert r.status_code == 200
        d = r.json()["digest"]
        assert d == payload_digest(payload)
        r = c.post("/commit", json={"request_id": "req-app-0001", "expected_digest": "bad"})
        assert r.status_code == 409
        r = c.post("/commit", json={"request_id": "req-app-0001", "expected_digest": d})
        assert r.status_code == 200 and r.json()["state"] == "SUCCEEDED"
        assert c.get("/status/req-app-0001").json()["output_text"] == "out"
        assert c.get("/status/req-none-001").status_code == 404
        _, body2 = _body(text="other")
        assert c.post("/prepare", json=body2).status_code == 409
    assert len(fake.received) == 1


def test_http_rejects_extra_fields(tmp_path):
    svc = GatewayService(SendStore(tmp_path / "db"), {"claude": FakeAdapter()})
    with TestClient(create_app(svc)) as c:
        _, body = _body()
        body["payload"]["url"] = "https://evil.example"
        assert c.post("/prepare", json=body).status_code == 422
        _, body = _body()
        body["headers"] = {"X": "y"}
        assert c.post("/prepare", json=body).status_code == 422
        _, body = _body(text="x" * 2001)
        assert c.post("/prepare", json=body).status_code == 413


def test_startup_recovers(tmp_path):
    store = SendStore(tmp_path / "db")
    svc = GatewayService(store, {"claude": FakeAdapter()})
    from common.schemas import PrepareRequest

    payload, body = _body()
    svc.prepare(PrepareRequest.model_validate(body))
    store.claim_attempt("req-app-0001")
    with TestClient(create_app(svc)) as c:
        j = c.get("/status/req-app-0001").json()
        assert j["state"] == "FAILED" and j["failure"] == "unknown"
