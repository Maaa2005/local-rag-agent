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

外から `uvicorn --workers N` で起動されたり、同じ DB volume を複数コンテナで共有されたりしても
前提が崩れないよう、lifespan の開始時（recover_on_startup より前）に DB と同じディレクトリの
ロックファイル（<DB>.lock）へ fcntl.flock(LOCK_EX|LOCK_NB) を取る。取れなければ GatewayAlreadyRunning で
起動を失敗させる。ロックは lifespan の終了まで（= サーバプロセスの生存中）保持し、プロセスが落ちれば
カーネルが解放する。ロックファイル自体は消さない（消すと別プロセスが新しい inode を作ってロックが二重化する）。
flock はローカルファイルシステム（Docker の local volume を含む）前提。NFS 等の共有 FS では効かない。
"""
from __future__ import annotations

import asyncio
import fcntl
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


class GatewayAlreadyRunning(RuntimeError):
    """同じ DB を使う Gateway が既に動いている（単一ワーカー前提の違反）。"""


def lock_path_for(db_path: str) -> str:
    return f"{db_path}.lock"


def acquire_single_worker_lock(db_path: str) -> int:
    """<db_path>.lock に排他ロックを取り、保持中の fd を返す。取れなければ GatewayAlreadyRunning。

    解放は os.close(fd)（またはプロセス終了）。fd は O_CLOEXEC で exec 先に引き継がない。
    flock はファイル記述ごとのロックなので、同一プロセス内でも別に open すれば競合として検出される。
    """
    lock_path = lock_path_for(db_path)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise GatewayAlreadyRunning(
            f"Gateway の DB は別のプロセスが使用中です（ロック {lock_path} を取得できない）。"
            "Gateway は単一ワーカー前提のため、uvicorn --workers 2 以上や、同じ DB を共有する"
            "複数コンテナ／複数プロセスでは起動できません。"
        ) from None
    except BaseException:
        os.close(fd)
        raise
    return fd


def create_app(service: GatewayService, purge_interval: float | None = None) -> FastAPI:
    """purge_interval: 定期 purge の間隔（秒）。None なら環境変数から読む。"""
    interval = purge_interval_from_env() if purge_interval is None else purge_interval
    if not interval > 0:
        raise ValueError("purge_interval must be positive")

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # 単一ワーカー前提の強制（モジュール docstring）。recover_on_startup より前に取る。
        lock_fd = acquire_single_worker_lock(service.store.path)
        try:
            async with _serve(_app):
                yield
        finally:
            os.close(lock_fd)

    @asynccontextmanager
    async def _serve(_app: FastAPI):
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
