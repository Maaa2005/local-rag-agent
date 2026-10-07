"""判断モデル層（C1 振り分け・C2 送信ゲート）の共通 I/F と安全側ラッパ。

設計書 N4「判定の失敗・タイムアウト・形式不正・入力切り捨ては送らない側に倒す」を
ここで一か所に集める。実装（rules / http）は Protocol を満たすだけでよく、
失敗時の扱いは guarded_c1 / guarded_c2 が必ず安全側に変換する。

質問の版（設計書 F9）: 判断モデル層が返す C1Decision / C2Verdict は、common.schemas の同名型を
継承して question_version（判断モデルへ投げた質問セットの版）を必須にしたもの。
common.schemas は Gateway と共有しており判断モデル層の項目を持ち込まないため、ここで拡張する。
pipeline 側の型注釈は common.schemas のままで、こちらのインスタンスはそのまま渡せる。
質問の版が無い判定は revision 欠落と同じく形式不正として安全側に倒す（どの質問への答えか
監査で再現できないため）。
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field, ValidationError

from common import schemas
from common.schemas import SendPayload

GUARD_MODEL = "guard"
GUARD_REVISION = "v1"
# guard 自身が判定したとき（入力超過・タイムアウト・例外・形式不正）の「質問の版」。
# 判断モデルに質問していない（または答えを使っていない）ので、guard の固定規則の版を記録する
GUARD_QUESTION_VERSION = "guard-failsafe@1"


class C1Decision(schemas.C1Decision):
    question_version: str = Field(min_length=1)


class C2Verdict(schemas.C2Verdict):
    question_version: str = Field(min_length=1)


def question_version(label: str, spec: Any) -> str:
    """質問セットの版。人が付けた版ラベル＋実際の質問定義の短いハッシュ。

    ラベルを上げ忘れて質問文だけ変えても、ハッシュが変わるので監査で区別できる。
    """
    h = hashlib.sha256(json.dumps(spec, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return f"{label}#{h[:12]}"


def _coerce(model: type[BaseModel], raw: Any) -> Any:
    """判定結果を判断モデル層の型に揃える。common.schemas の型（質問の版なし）は検証し直す。"""
    if isinstance(raw, model):
        return raw
    if isinstance(raw, BaseModel):
        raw = raw.model_dump()
    return model.model_validate(raw)


@runtime_checkable
class C1Router(Protocol):
    def route(self, user: str, request: str, input: dict[str, Any] | None) -> C1Decision: ...


@runtime_checkable
class C2Gate(Protocol):
    def check(self, payload: SendPayload) -> C2Verdict: ...


def payload_chars(payload: SendPayload) -> int:
    """C2 が検査すべき全テキスト（全メッセージ本文）の文字数。"""
    return sum(len(m.content) for m in payload.messages)


def _hold(reason: str, *, truncated: bool = False, model: str = GUARD_MODEL, revision: str = GUARD_REVISION,
          question_version: str = GUARD_QUESTION_VERSION) -> C2Verdict:
    return C2Verdict(decision="hold", reason=reason, model=model, revision=revision, truncated=truncated,
                     question_version=question_version)


def _run_with_timeout(fn, timeout_s: float):
    # 遅延した判定スレッドは待たずに切り捨てる（結果は使わない）
    ex = cf.ThreadPoolExecutor(max_workers=1)
    try:
        fut = ex.submit(fn)
        return fut.result(timeout=timeout_s)
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def guarded_c2(gate: C2Gate, payload: SendPayload, timeout_s: float, max_input_chars: int) -> C2Verdict:
    """C2 を呼び、失敗はすべて hold に倒す。allow を返すのは正常完了かつ全文検査済みのときだけ。"""
    n = payload_chars(payload)
    if n > max_input_chars:
        # 全文を検査できないので判定モデルに渡す前に保留する（設計書: 入力切り捨て→保留）
        return _hold(f"input too long: {n} > {max_input_chars} chars", truncated=True)
    try:
        raw = _run_with_timeout(lambda: gate.check(payload), timeout_s)
    except cf.TimeoutError:
        return _hold(f"C2 timeout after {timeout_s}s")
    except Exception as e:  # noqa: BLE001  判定の失敗はすべて hold
        return _hold(f"C2 error: {type(e).__name__}: {e}")
    try:
        verdict = _coerce(C2Verdict, raw)
    except (ValidationError, TypeError) as e:
        return _hold(f"C2 malformed verdict: {type(e).__name__}")
    if verdict.truncated and verdict.decision == "allow":
        # 判定したモデル・revision・質問の版はそのまま残す（どの質問で切り捨てが起きたか分かるように）
        return _hold("C2 reported truncated input", truncated=True, model=verdict.model, revision=verdict.revision,
                     question_version=verdict.question_version)
    return verdict


def _human(reason: str) -> C1Decision:
    return C1Decision(route="human", destination=None, reason=reason, model=GUARD_MODEL, revision=GUARD_REVISION,
                      question_version=GUARD_QUESTION_VERSION)


def guarded_c1(
    router: C1Router, user: str, request: str, input: dict[str, Any] | None, timeout_s: float, max_input_chars: int
) -> C1Decision:
    """C1 を呼び、失敗は route='human'（人に回す・外部へ送らない）に倒す。"""
    n = len(request) + (len(str(input)) if input else 0)
    if n > max_input_chars:
        return _human(f"input too long: {n} > {max_input_chars} chars")
    try:
        raw = _run_with_timeout(lambda: router.route(user, request, input), timeout_s)
    except cf.TimeoutError:
        return _human(f"C1 timeout after {timeout_s}s")
    except Exception as e:  # noqa: BLE001
        return _human(f"C1 error: {type(e).__name__}: {e}")
    try:
        dec = _coerce(C1Decision, raw)
    except (ValidationError, TypeError) as e:
        return _human(f"C1 malformed decision: {type(e).__name__}")
    if dec.route in ("A", "B") and dec.destination is None:
        # 外部経路なのに宛先が無いのは形式不正扱い
        return _human("C1 external route without destination")
    return dec
