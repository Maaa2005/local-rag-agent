"""GatewayService: prepare / commit / status / recover_on_startup。"""
from __future__ import annotations

import logging
import time
from typing import Callable, Mapping

from common.schemas import (
    CommitRequest,
    FailureKind,
    PrepareRequest,
    PrepareResponse,
    SendPayload,
    SendState,
    StatusResponse,
    payload_digest,
)
from gateway.adapters import Adapter, NotSentError, RejectedError
from gateway.store import SendRecord, SendStore

log = logging.getLogger("gateway")

# 契約種別ごとの送信本文合計の上限（文字数）
CONTRACT_LIMITS: dict[str, int] = {"A": 2000, "B": 1500}


class GatewayError(Exception):
    """Gateway が要求を拒否した。"""


class InvalidRequest(GatewayError):
    pass


class UnknownContract(InvalidRequest):
    pass


class PayloadTooLarge(InvalidRequest):
    pass


class Expired(GatewayError):
    pass


class Conflict(GatewayError):
    """同じ request_id で異なる内容。"""


class DigestMismatch(GatewayError):
    pass


class NotFound(GatewayError):
    pass


def contract_kind(contract: str) -> str:
    """"A-faq-format@1" → "A"。未知なら UnknownContract。"""
    head, sep, version = contract.partition("@")
    kind = head.split("-", 1)[0]
    if kind not in CONTRACT_LIMITS or not sep or not version or "-" not in head:
        raise UnknownContract("unknown contract")
    return kind


def body_chars(payload: SendPayload) -> int:
    return sum(len(m.content) for m in payload.messages)


class GatewayService:
    def __init__(
        self,
        store: SendStore,
        adapters: Mapping[str, Adapter],
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.store = store
        self.adapters = dict(adapters)
        self.clock = clock

    # ---- 検査 ----
    def _validate(self, req: PrepareRequest) -> None:
        p = req.payload
        kind = contract_kind(p.contract)
        if not p.messages:
            raise InvalidRequest("empty messages")
        if not any(m.role == "user" for m in p.messages):
            raise InvalidRequest("no user message")
        if body_chars(p) > CONTRACT_LIMITS[kind]:
            raise PayloadTooLarge(f"body exceeds {CONTRACT_LIMITS[kind]} chars for contract {kind}")
        if p.destination not in self.adapters:
            raise InvalidRequest("destination not available")
        if req.expires_at <= self.clock():
            raise Expired("already expired")

    @staticmethod
    def _status(rec: SendRecord) -> StatusResponse:
        return StatusResponse(
            request_id=rec.request_id,
            digest=rec.digest,
            state=rec.state,
            failure=rec.failure,
            output_text=rec.output_text,
        )

    # ---- API ----
    def prepare(self, req: PrepareRequest) -> PrepareResponse:
        digest = payload_digest(req.payload)
        existing = self.store.get(req.request_id)
        if existing is not None:
            if existing.digest != digest:
                raise Conflict("request_id already used with different content")
            return PrepareResponse(request_id=existing.request_id, digest=existing.digest, state=existing.state)
        self._validate(req)
        payload_json = req.payload.model_dump_json()
        inserted = self.store.insert_prepared(req.request_id, digest, payload_json, req.expires_at)
        if not inserted:  # 同時 prepare の競合
            rec = self.store.get(req.request_id)
            if rec is None or rec.digest != digest:
                raise Conflict("request_id already used with different content")
            return PrepareResponse(request_id=rec.request_id, digest=rec.digest, state=rec.state)
        return PrepareResponse(request_id=req.request_id, digest=digest, state=SendState.PREPARED)

    def commit(self, req: CommitRequest) -> StatusResponse:
        rec = self.store.get(req.request_id)
        if rec is None:
            raise NotFound("unknown request_id")
        if rec.digest != req.expected_digest:
            raise DigestMismatch("digest mismatch")
        if rec.state != SendState.PREPARED:
            return self._status(rec)
        if rec.expires_at <= self.clock():
            raise Expired("expired")
        payload = SendPayload.model_validate_json(rec.payload_json)
        # 保存内容の改変検出（DB 直接改変など）
        if payload_digest(payload) != rec.digest:
            raise DigestMismatch("stored payload digest mismatch")
        adapter = self.adapters.get(payload.destination)
        if adapter is None:
            raise InvalidRequest("destination not available")

        # 記録が成功したときだけ送る。失敗時は例外がそのまま上がりアダプタは呼ばれない。
        if not self.store.claim_attempt(rec.request_id):
            return self._status(self.store.get(rec.request_id))  # type: ignore[arg-type]

        try:
            out = adapter.send(payload)
        except NotSentError as e:
            log.warning("send not_sent request_id=%s kind=%s", rec.request_id, type(e).__name__)
            self.store.finish(rec.request_id, SendState.FAILED, FailureKind.not_sent)
        except RejectedError as e:
            log.warning("send rejected request_id=%s kind=%s", rec.request_id, type(e).__name__)
            self.store.finish(rec.request_id, SendState.FAILED, FailureKind.rejected)
        except Exception as e:  # noqa: BLE001 結果不明。自動再送しない
            log.warning("send unknown request_id=%s kind=%s", rec.request_id, type(e).__name__)
            self.store.finish(rec.request_id, SendState.FAILED, FailureKind.unknown)
        else:
            text = (out or "")[: payload.max_output_chars]
            self.store.finish(rec.request_id, SendState.SUCCEEDED, None, text)
        return self.status(rec.request_id)

    def status(self, request_id: str) -> StatusResponse:
        rec = self.store.get(request_id)
        if rec is None:
            raise NotFound("unknown request_id")
        return self._status(rec)

    def recover_on_startup(self) -> int:
        return self.store.fail_attempting_as_unknown()
