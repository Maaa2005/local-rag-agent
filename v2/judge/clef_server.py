"""Clef-flash（llama-server の /v1/systemone）を `/v1/systemone`（docs/redesign-v2.md「判断モデルの HTTP I/F」）の形で出す変換サーバ。

llama-server の /v1/systemone は質問の定義（type / instructions / criteria）も答え（noul は P(true)・choice は
probabilities）も Laya と同じ形だが、I/F とは違う（答えが質問 id の辞書・revision が無い）。この shim は要求を
llama-server に渡して答えを I/F の形に詰め替えるだけで、判定はしない（閾値・経路の決定は C1 / C2 アダプタ側）。
写し方は laya-shim@1 と同じだが、コードは共有しない（片方を直したときに、もう片方の版が黙って変わらないように）。

  起動（社内ゾーンのみ。llama-server は 127.0.0.1 に置き、この shim だけが叩く）:
    llama-server -m <gguf> --host 127.0.0.1 --port 8081 -c 8192 -b 8192 -ub 8192 -np 1 -ngl 999 \\
      --alias clef-flash --no-webui --offline
    JUDGE_UPSTREAM=http://127.0.0.1:8081 JUDGE_MODEL_PATH=<gguf> JUDGE_MODEL_SHA256=<sha256> \\
      JUDGE_MODEL_REVISION=<HF の repo@sha/ファイル名> JUDGE_LLAMA_BUILD=<b番号-commit> \\
      python -m judge.clef_server --host 127.0.0.1 --port 8080

写し方:
  - 1 要求 1 問だけ受ける。Clef は 1 要求の質問をまとめて 1 つのプロンプトで判断するので、質問を足すと
    他の質問の答えまで変わる。評価は 1 問ずつ測るので、測っていない組み合わせでは答えない
  - choice: criteria を {ラベル: descriptions[ラベル] か None} にして渡す
  - noul: Clef の noul は P(true)。probs = {"yes": P(true), "no": 1 - P(true)}
  - score: 未対応（422）。今の C1 / C2 は使わない
  - answer: Clef の choice ではなく、返した probs の最大から出し直す。同点は選択肢の順で先のもの
    （noul の 0.5 ちょうどは yes = 禁止情報あり側）
  - truncated: Clef は切り捨てない（プロンプト全体が --ubatch-size に収まらないと llama-server がエラーを返す）。
    エラーは 500 にするので、アダプタが human / hold に倒す。usage に truncated が載ったときだけ true
  - revision: "<モデルの版>;sha256=<先頭 16 桁>;llama.cpp=<build>;n_ctx=<n>+clef-shim@1"。
    写し方を変えたら SHIM_VERSION を上げる（監査で区別するため）

版の確かめ方（確かめられない版では起動しない・答えない）:
  - 起動時: llama-server の /props で読み込んだファイル・build・文脈長・slot 数・alias を照合し、
    ファイルの sha256 を JUDGE_MODEL_SHA256 と照合する
  - 要求ごと: 本文を送る前に /props をもう一度照合する（評価の途中で llama-server が別の設定で
    起動し直されても、違う版の答えを混ぜない）。重みの sha256 は起動時だけ
  - --ubatch-size は /props に出ないので照合できない。小さいと長い入力がエラー（500 → 保留）になるだけで、
    答えの値は変わらない
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

SHIM_VERSION = "clef-shim@1"

# 受け付ける大きさの上限。判定対象の長さは guard（max_input_chars）が先に切るので、ここは壊れた要求を止める柵
MAX_CONTEXT_CHARS = 50_000
MAX_QUESTIONS = 1
MAX_CHOICES = 64
MAX_TEXT_CHARS = 2_000

DEFAULT_UPSTREAM = "http://127.0.0.1:8081"
DEFAULT_N_CTX = 8192
DEFAULT_UPSTREAM_TIMEOUT_S = 50.0
# 名前で解決するとホスト設定で外へ向けられるので、IP の直書きだけ受ける
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1"})


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=1, max_length=64)
    type: Literal["noul", "choice", "score"]
    prompt: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    choices: list[str] | None = None
    descriptions: dict[str, str] | None = None
    min: int | None = None
    max: int | None = None


class SystemOneRequest(BaseModel):
    # 知らない欄は捨てずに 422。意味の分からない指定を黙って無視して答えると、頼まれていない判定になる
    model_config = ConfigDict(extra="forbid", strict=True)

    model: str = Field(min_length=1)
    context: str = Field(min_length=1, max_length=MAX_CONTEXT_CHARS)
    questions: list[Question] = Field(min_length=1, max_length=MAX_QUESTIONS)


class BadQuestion(ValueError):
    pass


class BadModelOutput(ValueError):
    pass


@dataclass(frozen=True)
class Upstream:
    """llama-server がこの値で動いているときだけ答える。"""
    model_path: str
    build: str
    n_ctx: int
    alias: str


def to_clef(q: Question) -> dict[str, Any]:
    """I/F の質問 1 問を Clef の質問定義にする。受けられない質問は BadQuestion。"""
    if q.type == "score":
        raise BadQuestion(f"{q.id}: score is not supported by {SHIM_VERSION}")
    if q.min is not None or q.max is not None:
        raise BadQuestion(f"{q.id}: min/max are only for score")
    if q.type == "noul":
        if q.choices is not None or q.descriptions is not None:
            raise BadQuestion(f"{q.id}: noul takes no choices/descriptions")
        return {"type": "noul", "instructions": q.prompt}
    choices = q.choices or []
    if not 2 <= len(choices) <= MAX_CHOICES:
        raise BadQuestion(f"{q.id}: choice needs 2..{MAX_CHOICES} choices")
    if len(set(choices)) != len(choices) or any(not c or len(c) > MAX_TEXT_CHARS for c in choices):
        raise BadQuestion(f"{q.id}: choices must be unique non-empty labels")
    desc = q.descriptions or {}
    if not set(desc) <= set(choices):
        raise BadQuestion(f"{q.id}: descriptions for unknown choices {sorted(set(desc) - set(choices))}")
    if any(not v or len(v) > MAX_TEXT_CHARS for v in desc.values()):
        raise BadQuestion(f"{q.id}: descriptions must be non-empty")
    return {"type": "choice", "instructions": q.prompt, "criteria": {c: desc.get(c) for c in choices}}


def _prob(v: Any, where: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0.0 <= v <= 1.0:
        raise BadModelOutput(f"{where}: probability {v!r} out of range")
    return float(v)


def from_clef(q: Question, raw: Any) -> dict[str, Any]:
    """Clef の答え 1 問分を I/F の answer にする。形が想定と違えば BadModelOutput。"""
    if not isinstance(raw, dict) or raw.get("type") != q.type:
        raise BadModelOutput(f"{q.id}: unexpected answer shape")
    if q.type == "noul":
        p = _prob(raw.get("noul"), q.id)
        return {"id": q.id, "answer": "yes" if p >= 0.5 else "no", "probs": {"yes": p, "no": round(1.0 - p, 6)}}
    got = raw.get("probabilities")
    choices = q.choices or []
    if not isinstance(got, dict) or set(got) != set(choices):
        raise BadModelOutput(f"{q.id}: probabilities keys differ from choices")
    probs = {c: _prob(got[c], q.id) for c in choices}
    # max は最初の最大を返すので、同点は選択肢の順で先のもの
    return {"id": q.id, "answer": max(choices, key=lambda c: probs[c]), "probs": probs}


def _input_tokens(usage: Any) -> int:
    n = usage.get("input_tokens") if isinstance(usage, dict) else None
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise BadModelOutput("usage.input_tokens missing")
    return n


def props_mismatch(props: Any, up: Upstream) -> list[str]:
    """/props と期待値の違う欄の名前。空なら一致。"""
    if not isinstance(props, dict):
        return ["props"]
    gen = props.get("default_generation_settings")
    path = props.get("model_path")
    got = {
        "model_path": os.path.realpath(path) if isinstance(path, str) and path else None,
        "build_info": props.get("build_info"),
        "n_ctx": gen.get("n_ctx") if isinstance(gen, dict) else None,
        "total_slots": props.get("total_slots"),
        "model_alias": props.get("model_alias"),
    }
    want = {"model_path": os.path.realpath(up.model_path), "build_info": up.build, "n_ctx": up.n_ctx,
            "total_slots": 1, "model_alias": up.alias}
    return [k for k in want if got[k] != want[k]]


def upstream_client(url: str, *, timeout_s: float = DEFAULT_UPSTREAM_TIMEOUT_S,
                    transport: httpx.BaseTransport | None = None) -> httpx.Client:
    """llama-server への接続。同じ機械の llama-server 以外へは判定対象を流さない。"""
    u = httpx.URL(url)
    if u.scheme != "http" or u.host not in LOOPBACK_HOSTS or u.userinfo or u.path not in ("", "/") or u.query:
        raise SystemExit(f"JUDGE_UPSTREAM must be http://127.0.0.1:<port> or http://[::1]:<port>, got {url!r}")
    # trust_env=False: 環境変数のプロキシ経由で外へ出さない。follow_redirects=False: 転送先へ本文を流さない
    return httpx.Client(base_url=url, timeout=timeout_s, follow_redirects=False, trust_env=False, transport=transport)


def _get_props(client: httpx.Client) -> Any:
    r = client.get("/props")
    r.raise_for_status()
    return r.json()


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_upstream(client: httpx.Client, up: Upstream, *, model_sha256: str) -> None:
    """起動時の照合。違えば SystemExit（違う版で判定を始めない）。"""
    try:
        props = _get_props(client)
    except (httpx.HTTPError, ValueError) as e:
        raise SystemExit(f"cannot read upstream /props: {type(e).__name__}") from None
    bad = props_mismatch(props, up)
    if bad:
        raise SystemExit(f"upstream differs from the expected setup: {bad}")
    got = file_sha256(up.model_path)
    if got != model_sha256.lower():
        raise SystemExit(f"model file sha256 {got} != JUDGE_MODEL_SHA256")


def runtime_revision(revision: str, *, model_sha256: str, up: Upstream) -> str:
    return f"{revision};sha256={model_sha256.lower()[:16]};llama.cpp={up.build};n_ctx={up.n_ctx}"


def create_app(upstream: httpx.Client, up: Upstream, *, model_name: str, revision: str, log: Any = None) -> FastAPI:
    if not revision:
        raise ValueError("revision is required")
    served_revision = f"{revision}+{SHIM_VERSION}"
    lock = threading.Lock()  # 推論は 1 本ずつ（llama-server も slot 1 つ）
    log = log if log is not None else sys.stderr
    app = FastAPI(title="clef /v1/systemone shim", docs_url=None, redoc_url=None, openapi_url=None)

    def emit(event: dict[str, Any]) -> None:
        print(json.dumps(event), file=log, flush=True)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"model": model_name, "revision": served_revision}

    @app.post("/v1/systemone")
    def systemone(req: SystemOneRequest) -> dict[str, Any]:
        if req.model != model_name:
            # 別のモデル名で頼まれた判定を、このモデルが答えたことにしない
            raise HTTPException(422, f"this server serves {model_name!r}")
        try:
            clef_qs = {q.id: to_clef(q) for q in req.questions}
        except BadQuestion as e:
            raise HTTPException(422, str(e)) from None
        ids = list(clef_qs)
        with lock:
            try:
                bad = props_mismatch(_get_props(upstream), up)
            except (httpx.HTTPError, ValueError) as e:
                emit({"event": "upstream_unreachable", "error": type(e).__name__})
                raise HTTPException(500, "inference failed") from None
            if bad:
                emit({"event": "upstream_changed", "fields": bad})
                raise HTTPException(500, "upstream setup changed")
            t0 = time.perf_counter()
            try:
                r = upstream.post("/v1/systemone", json={"state": req.context, "questions": clef_qs})
            except httpx.TimeoutException as e:
                emit({"event": "inference_timeout", "error": type(e).__name__,
                      "infer_s": round(time.perf_counter() - t0, 4)})
                raise HTTPException(504, "inference timed out") from None
            except httpx.HTTPError as e:
                # 例外の文面は判定対象を含み得るので、応答にもログにも型名だけ出す
                emit({"event": "inference_error", "error": type(e).__name__})
                raise HTTPException(500, "inference failed") from None
            infer_s = time.perf_counter() - t0
        if r.status_code != 200:
            # llama-server のエラー本文は入力の一部を含み得るので、状態コードだけ残す
            emit({"event": "inference_error", "status": r.status_code, "infer_s": round(infer_s, 4)})
            raise HTTPException(500, "inference failed")
        try:
            out = r.json()
            if not isinstance(out, dict) or out.get("model") != up.alias:
                raise BadModelOutput("unexpected model")
            answers = [from_clef(q, out["answers"].get(q.id)) for q in req.questions]
            usage = out["usage"]
            input_tokens = _input_tokens(usage)
        except (BadModelOutput, KeyError, TypeError, AttributeError, ValueError) as e:
            emit({"event": "bad_model_output", "error": type(e).__name__})
            raise HTTPException(500, "unexpected model output") from None
        truncated = bool(usage.get("truncated"))
        # 本文は残さず、件ごとの突き合わせ用に本文のハッシュ先頭とトークン数だけを出す
        emit({"event": "answered", "ids": ids, "ctx": hashlib.sha256(req.context.encode()).hexdigest()[:12],
              "chars": len(req.context), "input_tokens": input_tokens, "truncated": truncated,
              "infer_s": round(infer_s, 4)})
        return {"model": model_name, "revision": served_revision, "truncated": truncated, "answers": answers}

    return app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args(argv)
    env = os.environ
    need = {k: env.get(k, "") for k in ("JUDGE_MODEL_PATH", "JUDGE_MODEL_SHA256", "JUDGE_MODEL_REVISION", "JUDGE_LLAMA_BUILD")}
    missing = [k for k, v in need.items() if not v]
    if missing:
        raise SystemExit(f"required: {missing}")
    up = Upstream(model_path=need["JUDGE_MODEL_PATH"], build=need["JUDGE_LLAMA_BUILD"],
                  n_ctx=int(env.get("JUDGE_N_CTX", DEFAULT_N_CTX)), alias=env.get("JUDGE_UPSTREAM_MODEL", "clef-flash"))
    client = upstream_client(env.get("JUDGE_UPSTREAM", DEFAULT_UPSTREAM),
                             timeout_s=float(env.get("JUDGE_UPSTREAM_TIMEOUT", DEFAULT_UPSTREAM_TIMEOUT_S)))
    verify_upstream(client, up, model_sha256=need["JUDGE_MODEL_SHA256"])
    app = create_app(client, up, model_name=env.get("JUDGE_MODEL_NAME", "clef-flash"),
                     revision=runtime_revision(need["JUDGE_MODEL_REVISION"], model_sha256=need["JUDGE_MODEL_SHA256"], up=up))
    import uvicorn

    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
