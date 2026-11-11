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
import contextvars
import hashlib
import json
import threading
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

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
    # 判断モデルが入力を切り捨てて判定したか。True の判定は guarded_c1 が human に倒す
    truncated: bool = False
    # 判断サーバへの要求の番号（X-Request-Id）。判断サーバのログと突き合わせる。guard が止めた判定でも、
    # 要求を送っていれば残す。モデルに聞いていない判定だけ None
    judge_request_id: str | None = None


class C2Verdict(schemas.C2Verdict):
    question_version: str = Field(min_length=1)
    # 判断モデルに送った context の sha256（docs/v2-c2-eval-plan.md 6 章）。モデルに送っていない判定は None
    context_sha256: str | None = None
    # 判断サーバへの要求の番号（X-Request-Id）。guard が止めた判定でも、要求を送っていれば残す。
    # モデルに聞いていない判定だけ None。C2-FT は部品の番号を送った順にカンマで並べる（judge.ft_gate）
    judge_request_id: str | None = None


# 判定の呼び出しの間に送った要求の番号の受け皿。例外・時間切れでは応答が手元に無いので、番号はアダプタが
# 送る前にここへ入れる（送った要求はサーバのログに残るので、止めた判定からも突き合わせられるようにする）。
# 重ねて張れる（guard の中で FT が部品ごとに張る）。別スレッドへは引き継がれないので、判定を動かすスレッドの中で張る
_SENT_REQUEST_IDS: contextvars.ContextVar[tuple[list[str], ...]] = contextvars.ContextVar(
    "judge_sent_request_ids", default=())


class RequestAfterStop(RuntimeError):
    """guard が判定を打ち切った後の要求。送らずに、見捨てられたスレッドの判定を終わらせる。"""


class _GuardSent(list):
    """guard の受け皿。時間切れで閉じたら番号を受けない。

    時間切れでもスレッドは止められないので、FT の次の部品や 2 段構えの後段へ進もうとする。結果を返した後に
    送った要求の番号はどこにも残せないため、送る前に断る。番号の記録と閉じる操作を同じ錠で分けるので、
    送った要求の番号は必ず閉じる時の中身に入っている（閉じる直前に記録して送った要求は、応答を待たず番号だけ残る）。
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._closed = False

    def append(self, request_id: str) -> None:
        with self._lock:
            if self._closed:
                raise RequestAfterStop("the guard already stopped this judgement")
            super().append(request_id)

    def close(self) -> None:
        with self._lock:
            self._closed = True


def note_request_id(request_id: str) -> None:
    """送る前に呼ぶ。guard が打ち切った後なら RequestAfterStop（送らない）。"""
    for sink in _SENT_REQUEST_IDS.get():  # 外側から順。guard が断れば内側の受け皿にも残らない
        sink.append(request_id)


def collecting_request_ids(sent: list[str], fn: Callable[[], Any]) -> Any:
    """fn の間に送った要求の番号を sent に集める。"""
    token = _SENT_REQUEST_IDS.set((*_SENT_REQUEST_IDS.get(), sent))
    try:
        return fn()
    finally:
        _SENT_REQUEST_IDS.reset(token)


def _stop_request_id(sent: Sequence[str], answered: str | None) -> tuple[str | None, str]:
    """止めた判定に残す (番号, 理由欄の末尾)。2 段構えと同じく後の番号を残し、前の番号は `; first req=` にする。"""
    rid = answered or (sent[-1] if sent else None)
    return rid, "".join(f"; first req={s}" for s in sent if rid is not None and s not in rid)


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
          question_version: str = GUARD_QUESTION_VERSION, sent: Sequence[str] = (),
          answered: str | None = None) -> C2Verdict:
    rid, tail = _stop_request_id(sent, answered)
    return C2Verdict(decision="hold", reason=reason + tail, model=model, revision=revision, truncated=truncated,
                     question_version=question_version, judge_request_id=rid)


def _run_with_timeout(fn, timeout_s: float):
    # 遅延した判定スレッドは待たずに切り捨てる（結果は使わない）
    ex = cf.ThreadPoolExecutor(max_workers=1)
    try:
        fut = ex.submit(fn)
        return fut.result(timeout=timeout_s)
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def _call(fn, timeout_s: float | None):
    """timeout_s=None はタイムアウトなし（呼び出したスレッドでそのまま待つ）。"""
    return fn() if timeout_s is None else _run_with_timeout(fn, timeout_s)


def check_c2_verdict(raw: Any, *, sent: Sequence[str] = ()) -> C2Verdict:
    """C2 実装の応答を判断モデル層の C2Verdict に揃え、使えない応答は hold に倒す。

    形式不正（dict の不足・None・文字列・質問の版の欠落など）と、全文を見ずに出した allow は hold。
    sent は判定の間に送った要求の番号（止めた判定にも残す）。
    """
    try:
        verdict = _coerce(C2Verdict, raw)
    except (ValidationError, TypeError) as e:
        return _hold(f"C2 malformed verdict: {type(e).__name__}", sent=sent)
    if verdict.truncated and verdict.decision == "allow":
        # 判定したモデル・revision・質問の版はそのまま残す（どの質問で切り捨てが起きたか分かるように）
        return _hold("C2 reported truncated input", truncated=True, model=verdict.model, revision=verdict.revision,
                     question_version=verdict.question_version, sent=sent, answered=verdict.judge_request_id)
    return verdict


def call_c2(gate: C2Gate, payload: SendPayload, timeout_s: float | None) -> C2Verdict:
    """C2 を呼び、例外・タイムアウト・形式不正・切り捨てを hold に倒す（入力長の上限は掛けない）。"""
    sent = _GuardSent()
    try:
        raw = _call(lambda: collecting_request_ids(sent, lambda: gate.check(payload)), timeout_s)
    except cf.TimeoutError:
        sent.close()  # 番号を読む前に閉じる。これより後の要求は送られない
        return _hold(f"C2 timeout after {timeout_s}s", sent=sent)
    except Exception as e:  # noqa: BLE001  判定の失敗はすべて hold
        return _hold(f"C2 error: {type(e).__name__}: {e}", sent=sent)
    return check_c2_verdict(raw, sent=sent)


def guarded_c2(gate: C2Gate, payload: SendPayload, timeout_s: float, max_input_chars: int) -> C2Verdict:
    """C2 を呼び、失敗はすべて hold に倒す。allow を返すのは正常完了かつ全文検査済みのときだけ。"""
    n = payload_chars(payload)
    if n > max_input_chars:
        # 全文を検査できないので判定モデルに渡す前に保留する（設計書: 入力切り捨て→保留）
        return _hold(f"input too long: {n} > {max_input_chars} chars", truncated=True)
    return call_c2(gate, payload, timeout_s)


def _human(reason: str, *, truncated: bool = False, sent: Sequence[str] = (), answered: str | None = None) -> C1Decision:
    rid, tail = _stop_request_id(sent, answered)
    return C1Decision(route="human", destination=None, reason=reason + tail, model=GUARD_MODEL, revision=GUARD_REVISION,
                      question_version=GUARD_QUESTION_VERSION, truncated=truncated, judge_request_id=rid)


def is_guard_stop(decision: Any) -> bool:
    """guarded_c1 が失敗・不正・契約不整合で止めた判定か（＝保留扱い）。"""
    return getattr(decision, "model", None) == GUARD_MODEL


def guarded_c1(
    router: C1Router, user: str, request: str, input: dict[str, Any] | None, timeout_s: float, max_input_chars: int,
    *, contracts_path: Any = None,
) -> C1Decision:
    """C1 を呼び、失敗は route='human'（人に回す・外部へ送らない）に倒す。

    外部経路（A/B）の判定は、入力の契約キーを社内の契約定義から完全一致で引き直し、
    経路が契約と一致し、宛先がその契約の唯一の許可宛先と一致するときだけ通す。
    未知の版・無効契約・経路と契約の不整合・宛先の欠落や不一致・設定異常・切り捨ては human。
    さらに契約入力が契約の許容キー外を含む（元資料 source など）／入力種別キー（explanation / spec）が
    欠落・空・非文字列なら、判定モデルの答えに関わらず human（judge.destination.contract_input_problem）。
    """
    from judge.destination import CONTRACTS_PATH, contract_input_problem, lookup_contract, resolve_destination

    n = len(request) + (len(str(input)) if input else 0)
    if n > max_input_chars:
        return _human(f"input too long: {n} > {max_input_chars} chars", truncated=True)
    sent = _GuardSent()
    try:
        raw = _run_with_timeout(lambda: collecting_request_ids(sent, lambda: router.route(user, request, input)), timeout_s)
    except cf.TimeoutError:
        sent.close()  # 番号を読む前に閉じる。これより後の要求は送られない
        return _human(f"C1 timeout after {timeout_s}s", sent=sent)
    except Exception as e:  # noqa: BLE001
        return _human(f"C1 error: {type(e).__name__}: {e}", sent=sent)
    try:
        dec = _coerce(C1Decision, raw)
    except (ValidationError, TypeError) as e:
        return _human(f"C1 malformed decision: {type(e).__name__}", sent=sent)

    def stop(reason: str, *, truncated: bool = False) -> C1Decision:
        return _human(reason, truncated=truncated, sent=sent, answered=dec.judge_request_id)

    # guard が止めた判定は model=guard で返す（呼び出し側が「保留」と「モデルが他経路を選んだ」を
    # 区別できるように）。元の判定のモデル・版・質問の版は理由欄に残す
    judged = f" [judge {dec.model} rev={dec.revision} q={dec.question_version} route={dec.route} dest={dec.destination}]"
    if dec.truncated:
        # 全文を見ずに出した判定は使わない（判定したモデル・版は残す）
        return stop("C1 reported truncated input" + judged, truncated=True)
    if dec.route in ("A", "B"):
        if dec.destination is None:
            # 外部経路なのに宛先が無いのは形式不正扱い
            return stop("C1 external route without destination" + judged)
        key = (input or {}).get("contract")
        expected, why = resolve_destination(dec.route, key, contracts_path or CONTRACTS_PATH)
        if expected is None:
            return stop(f"C1 external route not backed by contract: {why}" + judged)
        if dec.destination != expected:
            return stop(f"C1 destination {dec.destination} is not the sole destination {expected} of {key}" + judged)
        # 契約入力の中身も判定モデルと独立に照合する。許容外キー（元資料 source を含む）や入力種別の
        # 欠落・空・非文字列があれば、判定モデルが外部経路を選んでも通さない
        contract, why = lookup_contract(key, contracts_path or CONTRACTS_PATH)
        if contract is None:
            return stop(f"C1 external route not backed by contract: {why}" + judged)
        problem = contract_input_problem(contract, input)
        if problem is not None:
            return stop(f"C1 external route with unusable contract input: {problem}" + judged)
    return dec
