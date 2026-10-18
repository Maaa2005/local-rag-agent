"""orchestrator.pipeline.Gateway を満たす HTTP クライアント（Unix ソケット越し）。

- 自動再試行しない（httpx の transport retries=0、アプリ側でも再送しない）。
- commit がタイムアウトしたら GatewayTimeout を上げる。run_contract はそれを受けて
  status 照会で結果を確定させる（本文の再送はしない）。
- HTTP エラーは GatewayHTTPError（status_code と detail=Gateway 側の例外名）に写す。
"""
from __future__ import annotations

import httpx

from common.schemas import CommitRequest, PrepareRequest, PrepareResponse, StatusResponse
from orchestrator.pipeline import GatewayNotReached


class GatewayClientError(Exception):
    """Gateway との通信で結果を受け取れなかった。"""


class GatewayTimeout(GatewayClientError):
    """応答待ちで時間切れ。送信されたかどうかは status で確かめる。"""


class GatewayUnavailable(GatewayClientError, GatewayNotReached):
    """接続できなかった（要求は Gateway に届いていない）。"""


class GatewayHTTPError(GatewayClientError):
    """Gateway が要求を拒否した（4xx/5xx）。"""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"HTTP {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class GatewayClient:
    def __init__(
        self,
        socket_path: str,
        *,
        timeout_s: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            transport=transport or httpx.HTTPTransport(uds=socket_path, retries=0),
            base_url="http://gateway",
            timeout=timeout_s,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "GatewayClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        try:
            r = self._client.request(method, path, json=body)
        except httpx.TimeoutException as e:
            raise GatewayTimeout(f"{method} {path}: {type(e).__name__}") from e
        except httpx.ConnectError as e:
            raise GatewayUnavailable(f"{method} {path}: {type(e).__name__}") from e
        except httpx.HTTPError as e:
            raise GatewayClientError(f"{method} {path}: {type(e).__name__}") from e
        if r.status_code >= 400:
            try:
                detail = str(r.json().get("detail", ""))
            except ValueError:
                detail = ""
            raise GatewayHTTPError(r.status_code, detail or r.reason_phrase)
        return r.json()

    def prepare(self, req: PrepareRequest) -> PrepareResponse:
        return PrepareResponse.model_validate(self._call("POST", "/prepare", req.model_dump(mode="json")))

    def commit(self, req: CommitRequest) -> StatusResponse:
        return StatusResponse.model_validate(self._call("POST", "/commit", req.model_dump(mode="json")))

    def status(self, request_id: str) -> StatusResponse:
        return StatusResponse.model_validate(self._call("GET", f"/status/{request_id}"))
