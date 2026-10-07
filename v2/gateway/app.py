"""Gateway の HTTP 入口（FastAPI）。Unix ソケットで起動する。

環境変数:
  GATEWAY_DB            SQLite パス（既定 /data/gateway.db）
  GATEWAY_SOCKET        Unix ソケットパス（既定 /run/gateway/gateway.sock）
  GATEWAY_CLAUDE_MODEL  Claude のモデル名（未設定なら claude 宛先を無効）
  GATEWAY_CODEX_MODEL   Codex のモデル名（未設定なら codex 宛先を無効）
  GATEWAY_CLAUDE_KEY_FILE / GATEWAY_CODEX_KEY_FILE  キーのファイルパス（キー自体は環境変数に置かない）
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from common.schemas import CommitRequest, PrepareRequest, PrepareResponse, StatusResponse
from gateway.service import (
    Conflict,
    DigestMismatch,
    Expired,
    GatewayError,
    GatewayService,
    NotFound,
    PayloadTooLarge,
)


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


def create_app(service: GatewayService) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        service.recover_on_startup()
        yield

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

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
    def status(request_id: str) -> StatusResponse:
        try:
            return service.status(request_id)
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
    uvicorn.run(create_app(build_service_from_env()), uds=sock, log_level="info", access_log=False)


if __name__ == "__main__":
    main()
