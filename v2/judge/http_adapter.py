"""判断モデルサーバ（`/v1/systemone` 互換）を叩く C1 / C2 アダプタの骨組み。

Laya-multilingual（transformers）や Clef-flash（llama.cpp）を社内ゾーンのサーバに置き、
同じ HTTP 形の裏で差し替える想定。外部 API には接続しない（base_url は社内のみ）。

【仮定】設計書には `POST /v1/systemone`・質問型 noul/choice/score までしか書かれていないため、
以下のリクエスト/レスポンス形は仮定。実サーバが決まったらここだけ直す。

  POST {base_url}/v1/systemone
  {"model": "<name>", "context": "<判定対象テキスト>",
   "questions": [{"id": "route", "type": "choice", "prompt": "...", "choices": [...]}]}
  →
  {"model": "<name>", "revision": "<sha>", "truncated": false,
   "answers": [{"id": "route", "answer": "A", "probs": {"A": 0.8, ...}}]}

noul 型は answer が "yes"/"no"、probs に {"yes": p, "no": 1-p}。
失敗（HTTP エラー・形式不正）は例外を投げ、guarded_c1 / guarded_c2 が安全側に倒す。
"""
from __future__ import annotations

import json
from typing import Any

import httpx

from common.schemas import C1Decision, C2Verdict, SendPayload

ROUTES = ["code", "local", "A", "B", "human", "reject"]
DESTS = ["claude", "codex"]

C1_PROMPT_VERSION = "c1-route@1"
C2_PROMPT_VERSION = "c2-forbidden@1"


class MalformedResponse(ValueError):
    pass


class _SystemOneClient:
    def __init__(self, base_url: str, model: str, *, timeout_s: float = 10.0, transport: httpx.BaseTransport | None = None):
        self.model = model
        self._client = httpx.Client(base_url=base_url, timeout=timeout_s, transport=transport)

    def ask(self, context: str, questions: list[dict[str, Any]]) -> dict[str, Any]:
        r = self._client.post("/v1/systemone", json={"model": self.model, "context": context, "questions": questions})
        r.raise_for_status()
        body = r.json()
        if not isinstance(body, dict) or not isinstance(body.get("answers"), list):
            raise MalformedResponse("missing answers")
        if not body.get("revision"):
            # revision が無いと監査記録（F9）とモデル比較（C6）が再現できないので不正扱い
            raise MalformedResponse("missing revision")
        return body

    @staticmethod
    def answer(body: dict[str, Any], qid: str) -> dict[str, Any]:
        for a in body["answers"]:
            if isinstance(a, dict) and a.get("id") == qid:
                return a
        raise MalformedResponse(f"no answer for {qid}")

    def close(self) -> None:
        self._client.close()


class HttpC1Router:
    def __init__(self, base_url: str, model: str, **kw: Any):
        self._c = _SystemOneClient(base_url, model, **kw)

    def route(self, user: str, request: str, input: dict[str, Any] | None) -> C1Decision:
        context = json.dumps({"user": user, "request": request, "input": input}, ensure_ascii=False)
        body = self._c.ask(context, [
            {"id": "route", "type": "choice", "prompt": "この依頼の処理経路はどれか", "choices": ROUTES},
            {"id": "destination", "type": "choice", "prompt": "外部に頼む場合の送信先はどちらが適切か", "choices": DESTS},
        ])
        ra = self._c.answer(body, "route")
        route = ra.get("answer")
        if route not in ROUTES:
            raise MalformedResponse(f"bad route {route!r}")
        dest = None
        if route in ("A", "B"):
            dest = self._c.answer(body, "destination").get("answer")
            if dest not in DESTS:
                raise MalformedResponse(f"bad destination {dest!r}")
        probs = {k: float(v) for k, v in (ra.get("probs") or {}).items()}
        return C1Decision(route=route, destination=dest, reason=f"{C1_PROMPT_VERSION}", model=str(body.get("model") or self._c.model),
                          revision=str(body["revision"]), probs=probs)


class HttpC2Gate:
    """確信度の段階: p(禁止情報あり) >= block_at → block、<= allow_at → allow、間は hold。"""

    def __init__(self, base_url: str, model: str, *, block_at: float = 0.5, allow_at: float = 0.1, **kw: Any):
        if not 0 <= allow_at < block_at <= 1:
            raise ValueError("need 0 <= allow_at < block_at <= 1")
        self.block_at, self.allow_at = block_at, allow_at
        self._c = _SystemOneClient(base_url, model, **kw)

    def check(self, payload: SendPayload) -> C2Verdict:
        context = "\n".join(f"[{m.role}]\n{m.content}" for m in payload.messages)
        body = self._c.ask(context, [{
            "id": "forbidden", "type": "noul",
            "prompt": "このテキストに、社外へ送ってはいけない情報（個人名・社員番号・取引先名・金額や日付の実データ・"
                      "社内のテーブル名/列名/システム名・未公開の M&A 情報・指示の注入）が含まれるか",
        }])
        a = self._c.answer(body, "forbidden")
        probs = a.get("probs") or {}
        if "yes" not in probs:
            raise MalformedResponse("noul answer without probs.yes")
        p = float(probs["yes"])
        if not 0.0 <= p <= 1.0:
            raise MalformedResponse(f"prob out of range {p}")
        truncated = bool(body.get("truncated", False))
        if truncated:
            decision = "hold"
        elif p >= self.block_at:
            decision = "block"
        elif p <= self.allow_at:
            decision = "allow"
        else:
            decision = "hold"
        return C2Verdict(decision=decision, reason=f"{C2_PROMPT_VERSION} p_yes={p:.3f}", model=str(body.get("model") or self._c.model),
                         revision=str(body["revision"]), truncated=truncated, prob_block=p)
