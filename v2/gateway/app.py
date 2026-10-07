"""Gateway の HTTP 入口（FastAPI）。Unix ソケットで起動する。

環境変数:
  GATEWAY_DB            SQLite パス（既定 /data/gateway.db）
  GATEWAY_SOCKET        Unix ソケットパス（既定 /run/gateway/gateway.sock）
  GATEWAY_CLAUDE_MODEL  Claude のモデル名（未設定なら claude 宛先を無効）
  GATEWAY_CODEX_MODEL   Codex のモデル名（未設定なら codex 宛先を無効）
  GATEWAY_CLAUDE_KEY_FILE / GATEWAY_CODEX_KEY_FILE  キーのファイルパス（キー自体は環境変数に置かない）
  GATEWAY_PURGE_INTERVAL_SECONDS  保持期限削除の定期実行間隔（秒、既定 86400。不正値・0 以下は既定値、
                        60 未満は 60。解釈は gateway.purge.parse_purge_interval）

単一ワーカー前提: 起動時の recover_on_startup は ATTEMPTING の行をすべて FAILED/unknown にする。
複数ワーカー（別プロセス）で動かすと、後から起動したワーカーが他ワーカーの送信中の行を
不明扱いにしてしまう。そのため main() は uvicorn を workers=1 で固定して起動する。
並列度はワーカー内のスレッドプールで確保し、二重送信は claim_attempt の条件付き UPDATE で防ぐ。
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Path, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from common.schemas import (
    REQUEST_ID_PATTERN,
    CommitRequest,
    PrepareRequest,
    PrepareResponse,
    StatusResponse,
)
from gateway.service import (
    Conflict,
    DigestMismatch,
    Expired,
    GatewayError,
    GatewayService,
    HistoryResponse,
    NotFound,
    PayloadTooLarge,
)
from gateway.purge import purge_interval_from_env, purge_periodically
from gateway.store import StoreError


def _http_error(e: GatewayError) -> HTTPException:
    if isinstance(e, NotFound):
        code = 404
    elif isinstance(e, (Conflict, DigestMismatch)):
        code = 409
    elif isinstance(e, Expired):
        code = 410
    elif isinstance(e, PayloadTooLarge):
        code = 413
    else:
        code = 422
    return HTTPException(status_code=code, detail=type(e).__name__)


def create_app(service: GatewayService, purge_interval: float | None = None) -> FastAPI:
    """purge_interval: 定期 purge の間隔（秒）。None なら環境変数から読む。"""
    interval = purge_interval_from_env() if purge_interval is None else purge_interval
    if not interval > 0:
        raise ValueError("purge_interval must be positive")

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # 先に ATTEMPTING→unknown（updated_at=起動時刻）にしてから保持期限の削除を行う。
        service.recover_on_startup()
        try:
            service.purge_expired()
        except StoreError as e:
            # 削除に失敗しても送信記録の整合は崩れない（1 トランザクションで ROLLBACK）。起動は続ける。
            logging.getLogger("gateway").error(
                "startup purge failed kind=%s", type(e.__cause__ or e).__name__)
        # 常駐中の定期 purge。送信処理と同時に走ってよい（SendStore.purge_expired の docstring）。
        stop = asyncio.Event()
        task = asyncio.create_task(purge_periodically(lambda: service.purge_expired(), interval, stop))
        try:
            yield
        finally:
            stop.set()
            await task

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_req: Request, exc: RequestValidationError) -> JSONResponse:
        # 入力値（本文など）を応答に載せない。位置と種別だけ返す。
        detail = [{"loc": list(e.get("loc", ())), "type": e.get("type", "")} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": detail})

    # 外部送信はブロッキングなので def（スレッドプール）で処理する
    @app.post("/prepare", response_model=PrepareResponse)
    def prepare(req: PrepareRequest) -> PrepareResponse:
        try:
            return service.prepare(req)
        except GatewayError as e:
            raise _http_error(e) from None

    @app.post("/commit", response_model=StatusResponse)
    def commit(req: CommitRequest) -> StatusResponse:
        try:
            return service.commit(req)
        except GatewayError as e:
            raise _http_error(e) from None

    @app.get("/status/{request_id}", response_model=StatusResponse)
    def status(
        request_id: str = Path(min_length=1, max_length=64, pattern=REQUEST_ID_PATTERN),
    ) -> StatusResponse:
        try:
            return service.status(request_id)
        except GatewayError as e:
            raise _http_error(e) from None

    @app.get("/history/{request_id}", response_model=HistoryResponse)
    def history(
        request_id: str = Path(min_length=1, max_length=64, pattern=REQUEST_ID_PATTERN),
    ) -> HistoryResponse:
        try:
            return service.history(request_id)
        except GatewayError as e:
            raise _http_error(e) from None

    return app


def build_service_from_env() -> GatewayService:
    from gateway.adapters import ClaudeAdapter, CodexAdapter
    from gateway.store import SendStore

    adapters = {}
    if os.environ.get("GATEWAY_CLAUDE_MODEL"):
        adapters["claude"] = ClaudeAdapter(
            model=os.environ["GATEWAY_CLAUDE_MODEL"],
            key_path=os.environ.get("GATEWAY_CLAUDE_KEY_FILE", "/run/secrets/anthropic_api_key"),
        )
    if os.environ.get("GATEWAY_CODEX_MODEL"):
        adapters["codex"] = CodexAdapter(
            model=os.environ["GATEWAY_CODEX_MODEL"],
            key_path=os.environ.get("GATEWAY_CODEX_KEY_FILE", "/run/secrets/openai_api_key"),
        )
    store = SendStore(os.environ.get("GATEWAY_DB", "/data/gateway.db"))
    return GatewayService(store, adapters)


def main() -> None:
    import uvicorn

    sock = os.environ.get("GATEWAY_SOCKET", "/run/gateway/gateway.sock")
    # workers=1 固定（モジュール docstring の単一ワーカー前提を参照）
    uvicorn.run(create_app(build_service_from_env()), uds=sock, workers=1,
                log_level="info", access_log=False)


if __name__ == "__main__":
    main()
