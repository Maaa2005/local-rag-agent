"""GatewayService: prepare / commit / status / recover_on_startup。"""
from __future__ import annotations

import logging
import time
from typing import Callable, Mapping

from pydantic import BaseModel

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
from gateway.contracts import ContractSpec, lookup
from gateway.store import PURGED_PAYLOAD, PurgeResult, SendRecord, SendStore, StoreError

log = logging.getLogger("gateway")

# commit 後の記録（finish）の再試行。DB 書き込みだけを短く繰り返す。外部送信は再送しない。
FINISH_ATTEMPTS = 3
FINISH_BACKOFF_SECONDS = 0.05


class TransitionEntry(BaseModel):
    from_state: SendState | None
    to_state: SendState
    failure: FailureKind | None = None
    at: float


class HistoryResponse(BaseModel):
    """状態遷移履歴。本文・出力は含めない。"""

    request_id: str
    transitions: list[TransitionEntry]


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


def contract_spec(contract: str) -> ContractSpec:
    """固定レジストリを完全一致で引く。未登録なら UnknownContract。"""
    spec = lookup(contract)
    if spec is None:
        raise UnknownContract("unknown contract")
    return spec


def body_chars(payload: SendPayload) -> int:
    return sum(len(m.content) for m in payload.messages)


class GatewayService:
    def __init__(
        self,
        store: SendStore,
        adapters: Mapping[str, Adapter],
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.store = store
        self.adapters = dict(adapters)
        self.clock = clock
        self.sleep = sleep

    # ---- 検査 ----
    def _validate(self, req: PrepareRequest) -> None:
        p = req.payload
        spec = contract_spec(p.contract)
        if p.destination not in spec.destinations:
            raise InvalidRequest("destination not allowed for contract")
        if not p.messages:
            raise InvalidRequest("empty messages")
        # 構造: system は先頭に最大 1 件、残りは user のみ（assistant は契約 A/B に不要）
        rest = p.messages[1:] if p.messages[0].role == "system" else p.messages
        if not rest:
            raise InvalidRequest("no user message")
        if any(m.role != "user" for m in rest):
            raise InvalidRequest("system message only allowed first")
        if body_chars(p) > spec.payload_chars:
            raise PayloadTooLarge(f"body exceeds {spec.payload_chars} chars for contract {spec.key}")
        if p.max_output_chars > spec.output_chars:
            raise InvalidRequest(f"max_output_chars exceeds {spec.output_chars} for contract {spec.key}")
        if p.destination not in self.adapters:
            raise InvalidRequest("destination not available")
        now = self.clock()
        if req.expires_at <= now:
            raise Expired("already expired")
        if req.expires_at > now + spec.max_ttl_seconds:
            raise InvalidRequest(f"expires_at too far (max ttl {spec.max_ttl_seconds:.0f}s)")

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
        if rec.payload_json == PURGED_PAYLOAD:  # 保持期限で本文が消えた行は送れない
            raise Expired("payload purged")
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
            out = adapter.send(payload, request_id=rec.request_id)
        except NotSentError as e:
            log.warning("send not_sent request_id=%s kind=%s", rec.request_id, type(e).__name__)
            ok = self._finish(rec.request_id, SendState.FAILED, FailureKind.not_sent)
        except RejectedError as e:
            log.warning("send rejected request_id=%s kind=%s", rec.request_id, type(e).__name__)
            ok = self._finish(rec.request_id, SendState.FAILED, FailureKind.rejected)
        except Exception as e:  # noqa: BLE001 結果不明。自動再送しない
            log.warning("send unknown request_id=%s kind=%s", rec.request_id, type(e).__name__)
            ok = self._finish(rec.request_id, SendState.FAILED, FailureKind.unknown)
        else:
            text = (out or "")[: payload.max_output_chars]
            ok = self._finish(rec.request_id, SendState.SUCCEEDED, None, text)
        if not ok:
            # 記録できなかった。送信結果は呼び出し側に確定させず、結果不明として返す。
            # DB 上は ATTEMPTING のまま残り、再起動時の recover_on_startup で FAILED/unknown になる。
            return StatusResponse(request_id=rec.request_id, digest=rec.digest,
                                  state=SendState.FAILED, failure=FailureKind.unknown)
        return self.status(rec.request_id)

    def _finish(self, request_id: str, state: SendState, failure: FailureKind | None,
                output_text: str | None = None) -> bool:
        """finish を DB 書き込みだけ短く再試行する。記録できたら True。

        rowcount != 1（ATTEMPTING でなかった等）はログのみで True を返す（再試行しても変わらない）。
        ログには request_id と例外型名だけを出す（本文・出力は出さない）。
        """
        for attempt in range(1, FINISH_ATTEMPTS + 1):
            try:
                updated = self.store.finish(request_id, state, failure, output_text)
            except StoreError as e:
                log.warning("finish failed request_id=%s attempt=%d kind=%s",
                            request_id, attempt, type(e.__cause__ or e).__name__)
                if attempt < FINISH_ATTEMPTS:
                    self.sleep(FINISH_BACKOFF_SECONDS * attempt)
                continue
            if updated != 1:
                log.error("finish rowcount=%s request_id=%s", updated, request_id)
            return True
        log.error("finish gave up request_id=%s", request_id)
        return False

    def status(self, request_id: str) -> StatusResponse:
        rec = self.store.get(request_id)
        if rec is None:
            raise NotFound("unknown request_id")
        return self._status(rec)

    def history(self, request_id: str) -> HistoryResponse:
        if self.store.get(request_id) is None:
            raise NotFound("unknown request_id")
        return HistoryResponse(
            request_id=request_id,
            transitions=[
                TransitionEntry(from_state=t.from_state, to_state=t.to_state, failure=t.failure, at=t.at)
                for t in self.store.history(request_id)
            ],
        )

    def recover_on_startup(self) -> int:
        return self.store.fail_attempting_as_unknown()

    def purge_expired(self, now: float | None = None) -> PurgeResult:
        """保持期限（本文 30 日・メタデータ 90 日）を過ぎた記録を消す。件数だけをログに出す。"""
        result = self.store.purge_expired(self.clock() if now is None else now)
        log.info("purge bodies=%d sends=%d history=%d",
                 result.bodies_purged, result.sends_deleted, result.history_deleted)
        return result
