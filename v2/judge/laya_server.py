"""Laya-multilingual を `/v1/systemone`（docs/redesign-v2.md「判断モデルの HTTP I/F」）の形で出す変換サーバ。

laya 付属の laya-serve は応答の形が I/F と違う（答えが質問 id の辞書・revision が無い・truncated が
usage の中）ため使わず、laya の Agent.system_one を直接呼んで I/F の形に詰め替える。ここでは判定しない
（閾値・経路の決定は C1 / C2 アダプタ側）。決めるのは「何を Laya に聞き、答えをどう I/F に写すか」だけ。

  起動（社内ゾーンのみ。HF から取るのは初回だけで、以降は HF_HUB_OFFLINE=1）:
    JUDGE_MODEL=convaiinnovations/laya-multilingual JUDGE_MODEL_REVISION=<sha> JUDGE_DEVICE=mps \\
      python -m judge.laya_server --host 127.0.0.1 --port 8080

写し方:
  - choice: criteria を {ラベル: descriptions[ラベル] か None} にして渡す。Laya は「ラベル: 説明」を選択肢として読む
  - noul: Laya の noul は P(true)。probs = {"yes": P(true), "no": 1 - P(true)}
  - score: 未対応（422）。今の C1 / C2 は使わない。使うときに段階の説明の渡し方を決める
  - answer: Laya の choice ではなく、返した probs の最大から出し直す。Laya は確率を小数 4 桁に丸めて返すので
    丸めで同点になり得る。同点は選択肢の順で先のもの（noul の 0.5 ちょうどは yes = 禁止情報あり側）
  - truncated: usage.truncated・truncated_questions が空でない・選択肢の畳み込み（usage.options）・usage が
    読めない、のどれかなら true。本文か選択肢の一部を見ずに出した答えなので、アダプタが human / hold に倒す
  - revision: "<モデルの版>+laya-shim@1"。写し方を変えたら SHIM_VERSION を上げる（監査で区別するため）

モデルの版の確かめ方（確かめられない版では起動しない）:
  - HF の id: JUDGE_MODEL_REVISION の sha で固定して取り、読み込んだ版（agent.revision）と一致を確認
  - ローカルのパス: JUDGE_MODEL_SHA256（{チェックポイント内の相対パス: sha256} の JSON）で重みを照合し、
    JUDGE_MODEL_REVISION をその重みの版として返す
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
from typing import Any, Literal, Protocol

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

SHIM_VERSION = "laya-shim@1"

# 受け付ける大きさの上限。判定対象の長さは guard（max_input_chars）が先に切るので、ここは壊れた要求を止める柵
MAX_CONTEXT_CHARS = 50_000
MAX_QUESTIONS = 16
MAX_CHOICES = 64
MAX_TEXT_CHARS = 2_000

# 学習時の max_len に合わせる。encoder は 8192 位置まで読めるが、1024 を超える長さでの
# 判定の質は確かめていない。超えた入力は Laya が切り捨てを返し、shim が保留（C2）/ 人へ（C1）にする
DEFAULT_MAX_LEN = 1024


class LayaAgent(Protocol):
    def system_one(self, state: Any, questions: dict[str, Any], **kw: Any) -> dict[str, Any]: ...


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


def to_laya(q: Question) -> dict[str, Any]:
    """I/F の質問 1 問を Laya の質問定義にする。受けられない質問は BadQuestion。"""
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


def from_laya(q: Question, raw: Any) -> dict[str, Any]:
    """Laya の答え 1 問分を I/F の answer にする。形が想定と違えば BadModelOutput。"""
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


def is_truncated(usage: Any) -> bool:
    if not isinstance(usage, dict):
        return True
    # options は head 予算に選択肢が収まらず、同じトークン幅に畳まれた選択肢があるときだけ載る
    return bool(usage.get("truncated")) or bool(usage.get("truncated_questions")) or "options" in usage


def create_app(agent: LayaAgent, *, model_name: str, revision: str, max_len: int = DEFAULT_MAX_LEN,
               head_max_len: int | None = None, log: Any = None) -> FastAPI:
    if not revision:
        raise ValueError("revision is required")
    served_revision = f"{revision}+{SHIM_VERSION}"
    lock = threading.Lock()  # 推論は 1 本ずつ（mps / CPU のモデルを同時に叩かない）
    log = log if log is not None else sys.stderr
    app = FastAPI(title="laya /v1/systemone shim", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"model": model_name, "revision": served_revision}

    @app.post("/v1/systemone")
    def systemone(req: SystemOneRequest) -> dict[str, Any]:
        if req.model != model_name:
            # 別のモデル名で頼まれた判定を、このモデルが答えたことにしない
            raise HTTPException(422, f"this server serves {model_name!r}")
        ids = [q.id for q in req.questions]
        if len(set(ids)) != len(ids):
            raise HTTPException(422, "duplicate question id")
        try:
            laya_qs = {q.id: to_laya(q) for q in req.questions}
        except BadQuestion as e:
            raise HTTPException(422, str(e)) from None
        kw: dict[str, Any] = {"max_len": max_len}
        if head_max_len is not None:
            kw["head_max_len"] = head_max_len
        t0 = time.perf_counter()
        try:
            with lock:
                out = agent.system_one(req.context, laya_qs, **kw)
        except Exception as e:
            # 例外の文面は判定対象を含み得るので、応答にもログにも型名だけ出す
            print(json.dumps({"event": "inference_error", "error": type(e).__name__}), file=log, flush=True)
            raise HTTPException(500, "inference failed") from None
        infer_s = time.perf_counter() - t0
        try:
            answers_raw = out["answers"]
            answers = [from_laya(q, answers_raw.get(q.id)) for q in req.questions]
        except (BadModelOutput, KeyError, TypeError, AttributeError) as e:
            print(json.dumps({"event": "bad_model_output", "error": type(e).__name__}), file=log, flush=True)
            raise HTTPException(500, "unexpected model output") from None
        usage = out.get("usage")
        truncated = is_truncated(usage)
        u = usage if isinstance(usage, dict) else {}
        # 本文は残さず、件ごとの突き合わせ用に本文のハッシュ先頭とトークン数だけを出す（長文ケースの切り分け用）
        print(json.dumps({"event": "answered", "ids": ids,
                          "ctx": hashlib.sha256(req.context.encode()).hexdigest()[:12],
                          "chars": len(req.context), "state_tokens": u.get("state_tokens"),
                          "state_tokens_dropped": u.get("state_tokens_dropped"),
                          "input_tokens": u.get("input_tokens"), "truncated": truncated,
                          "infer_s": round(infer_s, 4)}), file=log, flush=True)
        return {"model": model_name, "revision": served_revision, "truncated": truncated, "answers": answers}

    return app


def load_agent(model: str, revision: str, *, device: str, sha256: dict[str, str] | None) -> LayaAgent:
    """版を確かめて Laya を読む。確かめられないときは SystemExit（違う版で判定を始めない）。"""
    import laya  # torch を含むので、判断サーバを動かすときだけ読む

    if not revision:
        raise SystemExit("JUDGE_MODEL_REVISION is required")
    if os.path.isdir(model):
        if not sha256:
            raise SystemExit("a local JUDGE_MODEL needs JUDGE_MODEL_SHA256 to verify the weights")
        return laya.load(model, device=device, expected_sha256=sha256)
    agent = laya.load(model, device=device, revision=revision, expected_sha256=sha256)
    got = getattr(agent, "revision", None)
    if got != revision:
        raise SystemExit(f"loaded revision {got!r} != JUDGE_MODEL_REVISION {revision!r}")
    return agent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args(argv)
    env = os.environ
    model = env.get("JUDGE_MODEL") or env.get("JUDGE_MODEL_PATH")
    if not model:
        raise SystemExit("JUDGE_MODEL (HF id) or JUDGE_MODEL_PATH (local dir) is required")
    sha = json.loads(env["JUDGE_MODEL_SHA256"]) if env.get("JUDGE_MODEL_SHA256") else None
    revision = env.get("JUDGE_MODEL_REVISION", "")
    agent = load_agent(model, revision, device=env.get("JUDGE_DEVICE", "cpu"), sha256=sha)
    head = env.get("JUDGE_HEAD_MAX_LEN")
    app = create_app(agent, model_name=env.get("JUDGE_MODEL_NAME", "laya-multilingual"), revision=revision,
                     max_len=int(env.get("JUDGE_MAX_LEN", DEFAULT_MAX_LEN)),
                     head_max_len=int(head) if head else None)
    import uvicorn

    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
