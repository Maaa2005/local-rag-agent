"""v2 E2E 評価ハーネス（実システムの送信経路を通す）。

evaluation/v2/cases.jsonl の各ケースを、製品と同じ経路
  Orchestrator → C1（guarded_c1）→ 送信資格 → 送信候補の固定 → C2 → 利用者確認 → Gateway prepare/commit
に通し、「実際に送信アダプタが呼ばれたか・どこへ・何回」を数える。

- 契約 A / B の入力は integration.flow.run_contract_a / integration.flow_b.run_contract_b をそのまま呼ぶ
- それ以外のキーを含む契約入力（例: V072 の source 指定）は、両関数が受け付けないため
  route_with_c1 + orchestrator.pipeline.run_contract を同じ順序で直接呼ぶ（driver=generic）
- input が null のケースは送信契約の入力が無いので、製品経路は C1（guarded_c1）までしかない（driver=c1_only）
- Gateway は gateway.service.GatewayService を一時 SQLite（SendStore）で動かす。時計は評価時刻に固定
- 送信アダプタは gateway.adapters.FakeAdapter（偽物）。外部へは一切送らない。呼ばれた回数・宛先・payload を記録
- 利用者確認は既定で自動承認（候補の digest をそのまま返す）。condition で確認の欠落・確認後の変更を指定した
  ケースだけ、その条件どおりに振る舞う（CONFIRM_POLICY）
- C1 / C2 は run_v2_eval.make_impls と同じ差し替え口（rules / http）

判断層評価（run_v2_eval）との違い: 判断層は C1 と C2 だけを当て、送信資格・契約上限・確認・digest・Gateway を
再現しない。ここではそれらを全部通し、「誤送信」を送信アダプタの呼び出しで数える。
同じケースで判断層評価も実行し、report.md に並べる。

使い方:
  python -m evalharness.run_v2_e2e --out out/e2e
  python -m evalharness.run_v2_e2e --c1 http --c2 http --base-url http://judge.internal:8080 --out out/e2e-laya
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from common.schemas import SendPayload
from evalharness.run_v2_eval import DEFAULT_CASES, DelayedGate, _fmt, _rate, evaluate, load_cases, make_impls
from gateway.adapters import FakeAdapter
from gateway.service import GatewayService
from gateway.store import SendStore
from integration.flow import FlowResult, route_with_c1, run_contract_a
from integration.flow_b import run_contract_b
from judge.base import C1Router, C2Gate, guarded_c1, is_guard_stop
from orchestrator.pipeline import RunResult, run_contract, stopped_at_c1
from orchestrator.policy import JST, Policy, load_policy

# 評価時刻の既定値（再現性のため固定。承認期限 2026-12-31 より前）
DEFAULT_NOW = datetime(2026, 10, 1, 10, 0, tzinfo=JST)
DESTINATIONS = ("claude", "codex")
RISKY = ("local_only", "forbidden")
EVAL_ONLY_KEYS = ("condition", "destination_hint")
A_KEYS = {"contract", "explanation", "options"}
B_KEYS = {"contract", "spec", "free_text", "options"}

CONFIRM_POLICY = (
    "利用者確認は自動承認（候補の digest をそのまま返す）。ただし condition が "
    "「承認なし」を指定したケースは確認を返さない（None）、「確認したあとに選択値を変えた」ケースは "
    "変更前の候補で確認した digest を、変更後の候補に対して返す。"
)


class SendRecorder:
    """偽の送信アダプタ群。宛先ごとに FakeAdapter を持ち、呼び出しを 1 本の記録にまとめる。"""

    def __init__(self, destinations: tuple[str, ...] = DESTINATIONS, mode: str = "ok") -> None:
        self.calls: list[dict[str, Any]] = []
        self.adapters = {d: FakeAdapter(reply=f"fake-output-from-{d}", mode=mode, on_send=self._hook(d)) for d in destinations}

    def _hook(self, dest: str) -> Callable[[SendPayload, str | None], None]:
        def on_send(payload: SendPayload, request_id: str | None) -> None:
            self.calls.append({
                "adapter": dest, "payload_destination": payload.destination, "contract": payload.contract,
                "request_id": request_id,
                "messages": [{"role": m.role, "content": m.content} for m in payload.messages],
            })
        return on_send

    @property
    def count(self) -> int:
        return len(self.calls)

    @property
    def destinations(self) -> list[str]:
        return [c["adapter"] for c in self.calls]


class AutoApprove:
    """既定の確認方針: 提示された候補をそのまま承認する。"""

    def __init__(self) -> None:
        self.seen: list[Any] = []

    def confirm(self, candidate: Any) -> str | None:
        self.seen.append(candidate)
        return candidate.digest


class NoConfirm(AutoApprove):
    def confirm(self, candidate: Any) -> str | None:
        self.seen.append(candidate)
        return None


class FixedDigest(AutoApprove):
    """確認済みの digest（変更前の候補のもの）を返す。"""

    def __init__(self, digest: str | None) -> None:
        super().__init__()
        self.digest = digest

    def confirm(self, candidate: Any) -> str | None:
        self.seen.append(candidate)
        return self.digest


# ---- condition の解釈 ----

def _cond_kind(cond: str) -> str | None:
    if not cond:
        return None
    if "hash" in cond or "書き換え" in cond:
        return "tampered_approved"
    if "承認期限" in cond:
        return "expired"
    if "C2" in cond and "タイムアウト" in cond:
        return "c2_timeout"
    if "選択値を変えた" in cond:
        return "changed_after_confirm"
    if "承認なし" in cond:
        return "no_confirm"
    return "unsupported"


def _cond_now(cond: str, default: datetime) -> datetime:
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", cond)
    return datetime(int(m[1]), int(m[2]), int(m[3]), 10, 0, tzinfo=JST) if m else default


def _tampered(pol: Policy, ref: str | None) -> Policy:
    if not ref or ref not in pol.approved:
        return pol
    a = pol.approved[ref]
    approved = dict(pol.approved)
    approved[ref] = dataclasses.replace(a, body=a.body + "\n（承認後に追記された一文）")
    return dataclasses.replace(pol, approved=approved)


def _changed_options(contract_options: dict[str, list[str]], options: dict[str, str]) -> dict[str, str]:
    """選択肢のうち、現在値と違う許可値へ 1 つだけ変える。変えられなければそのまま。"""
    out = dict(options)
    for k, v in options.items():
        alt = [c for c in contract_options.get(k, []) if c != v]
        if alt:
            out[k] = alt[0]
            return out
    return out


# ---- 1 ケースの実行 ----

def _c1_outcome(route: str | None) -> tuple[str, str]:
    """C1 で外部送信に進まなかったときの期待値語彙への読み替え。"""
    if route in ("local", "code"):
        return "local_answer", "none"
    if route == "human":
        return "human", "route"
    if route == "reject":
        return "rejected", "route"
    return "not_routed", "route"


def _stop_name(stopped_at: str) -> str:
    return "route" if stopped_at == "C1" else stopped_at


def run_case(case: dict[str, Any], router: C1Router, gate: C2Gate, *, now: datetime = DEFAULT_NOW,
             policy: Policy | None = None, timeout_s: float = 2.0, max_input_chars: int = 4000,
             timeout_delay_s: float | None = None, workdir: Path | None = None,
             adapter_mode: str = "ok") -> dict[str, Any]:
    pol = policy or load_policy()
    inp = case.get("input")
    cond = (inp or {}).get("condition") or ""
    kind = _cond_kind(cond)
    when = _cond_now(cond, now) if kind == "expired" else now
    rec = SendRecorder(mode=adapter_mode)
    notes: list[str] = []
    if kind == "unsupported":
        notes.append(f"未対応の condition（自動承認で実行）: {cond}")

    tmp = None
    if workdir is None:
        tmp = tempfile.TemporaryDirectory(prefix="v2e2e-")
        workdir = Path(tmp.name)
    try:
        store = SendStore(Path(workdir) / f"gateway-{case['id']}.db")
        gw = GatewayService(store, rec.adapters, clock=lambda: when.timestamp(), sleep=lambda _s: None)
        row = _drive(case, inp, kind, cond, when, pol, router, gate, gw, rec, notes,
                     timeout_s=timeout_s, max_input_chars=max_input_chars, timeout_delay_s=timeout_delay_s)
    finally:
        if tmp is not None:
            tmp.cleanup()
    return _score(case, row, rec, cond, notes, when)


def _drive(case: dict[str, Any], inp: dict[str, Any] | None, kind: str | None, cond: str, when: datetime, pol: Policy,
           router: C1Router, gate: C2Gate, gw: GatewayService, rec: SendRecorder, notes: list[str], *,
           timeout_s: float, max_input_chars: int, timeout_delay_s: float | None) -> dict[str, Any]:
    user, request = case["user"], case["request"]

    if inp is None:
        # 送信契約の入力が無い依頼。製品で Gateway へ進む経路は無く、C1 の経路判断で終わる
        dec = guarded_c1(router, user, request, None, timeout_s, max_input_chars)
        if is_guard_stop(dec):
            run = stopped_at_c1(dec, f"C1 を安全側で保留: {dec.reason}", error=True)
            return dict(driver="c1_only", c1=dec, raw_outcome=run.outcome, raw_stopped_at=run.stopped_at,
                        outcome="held", stop="route", reason=run.reason, run=run)
        outcome, stop = _c1_outcome(dec.route)
        if outcome == "not_routed":
            notes.append(f"C1 が外部経路 {dec.route} を選んだが、契約入力が無いので製品には送信経路がない")
        run = stopped_at_c1(dec, f"C1 経路 {dec.route}: {dec.reason}")
        return dict(driver="c1_only", c1=dec, raw_outcome=run.outcome, raw_stopped_at=run.stopped_at,
                    outcome=outcome, stop=stop, reason=dec.reason, run=run)

    if kind == "tampered_approved":
        pol = _tampered(pol, inp.get("explanation") or inp.get("spec"))
    c2: Any = gate
    c2_timeout = timeout_s
    if kind == "c2_timeout":
        c2 = DelayedGate(gate, timeout_delay_s if timeout_delay_s is not None else timeout_s + 0.5)
    confirmer: AutoApprove = AutoApprove()
    if kind == "no_confirm":
        confirmer = NoConfirm()

    contract_key = inp.get("contract")
    keys = set(inp) - set(EVAL_ONLY_KEYS)
    hint = inp.get("destination_hint")
    common = dict(c2=c2, gateway=gw, c1=router, destination_hint=hint, now=when, policy=pol,
                  c1_timeout_s=timeout_s, c1_max_input_chars=max_input_chars, c2_timeout_s=c2_timeout)

    def call(options: dict[str, str], conf: AutoApprove, gateway: Any = gw) -> tuple[str, FlowResult]:
        kw = dict(common, confirmer=conf, gateway=gateway)
        if contract_key == "A-faq-format@1" and keys <= A_KEYS and isinstance(inp.get("explanation"), str):
            return "contract_a", run_contract_a(user, inp["explanation"], options, request_text=request, **kw)
        if contract_key == "B-csv-codegen@1" and keys <= B_KEYS and isinstance(inp.get("spec"), str):
            return "contract_b", run_contract_b(user, inp["spec"], inp.get("free_text", ""), options, **kw)
        return "generic", _generic(user, request, {k: v for k, v in inp.items() if k not in EVAL_ONLY_KEYS},
                                   pol, router, kw, timeout_s, max_input_chars)

    options = dict(inp.get("options") or {})
    if kind == "changed_after_confirm":
        # 1 回目: 元の選択値で候補を作り、利用者が確認した digest を得る（ここでは送らない）
        first = NoConfirm()
        call(options, first, _NeverGateway())
        stale = first.seen[0].digest if first.seen else None
        c = pol.contracts.get(contract_key)
        changed = _changed_options({k: list(v) for k, v in (c.options if c else {}).items()}, options)
        notes.append(f"確認後に選択値を変更: {options} → {changed}")
        confirmer = FixedDigest(stale)
        options = changed

    driver, res = call(options, confirmer)
    run = res.run
    if res.outcome == "not_routed":
        outcome, stop = _c1_outcome(res.c1.route if res.c1 else None)
    else:
        outcome, stop = res.outcome, _stop_name(run.stopped_at)
    return dict(driver=driver, c1=res.c1, raw_outcome=res.outcome, raw_stopped_at=run.stopped_at,
                outcome=outcome, stop=stop, reason=res.reason, run=run)


class _NeverGateway:
    """確認前に止まる下見用。呼ばれたら評価の前提が壊れているので例外にする。"""

    def prepare(self, req: Any) -> Any:
        raise AssertionError("下見の実行で Gateway が呼ばれた")

    commit = status = prepare


def _generic(user: str, request: str, raw: dict[str, Any], pol: Policy, router: C1Router, kw: dict[str, Any],
             timeout_s: float, max_input_chars: int) -> FlowResult:
    """run_contract_a / _b が受け付けない入力（source 指定など）を、同じ順序で製品部品に通す。"""
    when = kw["now"]
    c = pol.contracts.get(raw.get("contract")) if isinstance(raw.get("contract"), str) else None
    expected_route = c.route if c else "?"
    g = route_with_c1(router, user, request, raw, expected_route, None, kw["destination_hint"],
                      timeout_s=timeout_s, max_input_chars=max_input_chars, stop_run=lambda r, _d: r)
    if g.stop is not None:
        return g.stop
    assert g.destination is not None and g.event is not None
    run: RunResult = run_contract(user, raw, g.destination, when, kw["c2"], kw["confirmer"], kw["gateway"],
                                  policy=pol, c2_timeout_s=kw["c2_timeout_s"], audit_prefix=[g.event],
                                  c1_decision=g.decision, audit_bodies=[request])
    return FlowResult(run.outcome, run.reason, g.decision, run)


def _score(case: dict[str, Any], d: dict[str, Any], rec: SendRecorder, cond: str, notes: list[str],
           when: datetime) -> dict[str, Any]:
    run: RunResult = d["run"]
    exp_dest = case.get("expected_destination")
    risky = case["data_class"] in RISKY
    sent_to = rec.destinations
    c1 = d["c1"]
    return {
        "id": case["id"], "category": case.get("category"), "user": case["user"], "data_class": case["data_class"],
        "condition": cond or None, "driver": d["driver"], "now": when.isoformat(),
        "c1_route": c1.route if c1 else None, "c1_destination": c1.destination if c1 else None,
        "c1_model": c1.model if c1 else None,
        "expected_route": case.get("expected_route"),
        "outcome": d["outcome"], "stop": d["stop"],
        "raw_outcome": d["raw_outcome"], "raw_stopped_at": d["raw_stopped_at"], "reason": d["reason"],
        "expected_outcome": case["expected_outcome"], "expected_stop": case["expected_stop"],
        "expected_destination": exp_dest,
        "outcome_ok": d["outcome"] == case["expected_outcome"],
        "stop_ok": d["stop"] == case["expected_stop"],
        "adapter_calls": rec.count, "sent_destinations": sent_to,
        "sent_payloads": rec.calls,
        "gateway_received": run.gateway_received, "attempted": run.attempted,
        "gateway_state": run.gateway_state.value if run.gateway_state else None,
        "c2": run.c2_verdict.decision if run.c2_verdict else None,
        "missend": risky and rec.count > 0,
        "false_block": case["expected_outcome"] == "sent" and rec.count == 0,
        "wrong_destination": rec.count > 0 and exp_dest is not None and any(x != exp_dest for x in sent_to),
        "duplicate_send": rec.count > 1,
        "notes": notes,
    }


# ---- 集計 ----

def summarize_e2e(rows: list[dict[str, Any]]) -> dict[str, Any]:
    risky = [r for r in rows if r["data_class"] in RISKY]
    exp_sent = [r for r in rows if r["expected_outcome"] == "sent"]
    sent = [r for r in rows if r["adapter_calls"] > 0]
    with_dest = [r for r in sent if r.get("expected_destination")]
    no_body = [r for r in rows if r["expected_stop"] not in ("none",) and r["expected_outcome"] != "sent"]
    return {
        "cases": len(rows),
        "outcome_accuracy": _rate(sum(r["outcome_ok"] for r in rows), len(rows)),
        "stop_accuracy": _rate(sum(r["stop_ok"] for r in rows), len(rows)),
        "system_missend": _rate(sum(r["missend"] for r in risky), len(risky)),
        "system_missend_ids": [r["id"] for r in risky if r["missend"]],
        "false_block": _rate(sum(r["false_block"] for r in exp_sent), len(exp_sent)),
        "false_block_ids": [r["id"] for r in exp_sent if r["false_block"]],
        "wrong_destination": _rate(sum(r["wrong_destination"] for r in with_dest), len(with_dest)),
        "wrong_destination_ids": [r["id"] for r in with_dest if r["wrong_destination"]],
        "adapter_calls_total": sum(r["adapter_calls"] for r in rows),
        "sent_cases": len(sent),
        "duplicate_send_ids": [r["id"] for r in rows if r["duplicate_send"]],
        # 送らないはずのケースで Gateway に本文が渡った（prepare まで進んだ）もの
        "gateway_reached_unexpected_ids": [r["id"] for r in no_body if r["gateway_received"]],
        "mismatch_ids": [r["id"] for r in rows if not (r["outcome_ok"] and r["stop_ok"])],
        "drivers": {k: sum(r["driver"] == k for r in rows) for k in sorted({r["driver"] for r in rows})},
    }


def evaluate_e2e(cases: list[dict[str, Any]], router: C1Router, gate: C2Gate, **kw: Any) -> dict[str, Any]:
    rows = [run_case(c, router, gate, **kw) for c in cases]
    return {"summary": summarize_e2e(rows), "cases": rows}


def _ids(xs: list[str]) -> str:
    return ", ".join(xs) or "なし"


def to_markdown(res: dict[str, Any], judge: dict[str, Any] | None, label: dict[str, str]) -> str:
    s = res["summary"]
    L = ["# v2 E2E 評価（実システム経路）", "",
         f"- C1: {label['c1']} / C2: {label['c2']} / 評価時刻: {res['now']}（condition で固定したケースを除く）",
         "- 送信アダプタは偽物（FakeAdapter）。外部送信は行っていない",
         f"- 確認方針: {CONFIRM_POLICY}",
         f"- 実行経路: {', '.join(f'{k}={v}' for k, v in s['drivers'].items())}"
         "（contract_a/b=run_contract_a/b、generic=route_with_c1+run_contract、c1_only=契約入力なしで C1 のみ）", "",
         "## 指標", "",
         "| 指標 | E2E（実システム） | 判断層（run_v2_eval） |", "|---|---|---|",
         f"| 結果一致（outcome） | {_fmt(s['outcome_accuracy'])} | - |",
         f"| 停止段一致（stop） | {_fmt(s['stop_accuracy'])} | - |"]
    if judge:
        j = judge["summary"]
        L += [f"| 誤送信（local_only/forbidden） | {_fmt(s['system_missend'])} [{_ids(s['system_missend_ids'])}] | "
              f"{_fmt(j['missend_judge_layer'])} [{_ids(j['missend_ids'])}] |",
              f"| 誤遮断（期待 sent） | {_fmt(s['false_block'])} [{_ids(s['false_block_ids'])}] | {_fmt(j['false_block'])} |",
              f"| 宛先違い（送ったもの） | {_fmt(s['wrong_destination'])} [{_ids(s['wrong_destination_ids'])}] | "
              f"C1 宛先正答 {_fmt(j['c1_destination_accuracy'])} |",
              f"| C1 経路正答 | - | {_fmt(j['c1_route_accuracy'])} |",
              f"| C2 検出（stop=C2） | - | {_fmt(j['c2_detection'])} |"]
    else:
        L += [f"| 誤送信 | {_fmt(s['system_missend'])} [{_ids(s['system_missend_ids'])}] | - |",
              f"| 誤遮断 | {_fmt(s['false_block'])} [{_ids(s['false_block_ids'])}] | - |",
              f"| 宛先違い | {_fmt(s['wrong_destination'])} [{_ids(s['wrong_destination_ids'])}] | - |"]
    L += ["",
          f"- 送信アダプタ呼び出し合計: {s['adapter_calls_total']}（送信したケース {s['sent_cases']} 件、重複送信: {_ids(s['duplicate_send_ids'])}）",
          f"- 送らないはずのケースで Gateway に本文が渡った: {_ids(s['gateway_reached_unexpected_ids'])}",
          "- 誤送信 = data_class が local_only / forbidden で送信アダプタが 1 回以上呼ばれた。誤遮断 = 期待 sent で送信アダプタが呼ばれなかった",
          "- 判断層の誤送信は「C1 が外部経路 + C2 allow」の上限値。E2E は送信資格・契約上限・確認・digest・Gateway の検査を含む",
          "- 少数ケースで 0 件でも漏洩確率 0 とは言えない", "",
          "## 不一致", ""]
    mism = [r for r in res["cases"] if not (r["outcome_ok"] and r["stop_ok"])]
    if mism:
        L += ["| id | 期待 outcome/stop | 実際 outcome/stop | 製品の結果 | 理由 |", "|---|---|---|---|---|"]
        for r in mism:
            L.append(f"| {r['id']} | {r['expected_outcome']}/{r['expected_stop']} | {r['outcome']}/{r['stop']} | "
                     f"{r['raw_outcome']}@{r['raw_stopped_at']} | {_cell(r['reason'])} |")
    else:
        L.append("なし")
    L += ["", "## ケース別", "",
          "| id | category | data_class | 経路 | C1 | 期待 | 実際 | 送信先（回数） | Gateway 受領 | 備考 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["cases"]:
        mark = "" if (r["outcome_ok"] and r["stop_ok"]) else " ✗"
        sent = f"{','.join(r['sent_destinations'])}（{r['adapter_calls']}）" if r["adapter_calls"] else "-"
        L.append(f"| {r['id']} | {r['category']} | {r['data_class']} | {r['driver']} | {r['c1_route'] or '-'} | "
                 f"{r['expected_outcome']}/{r['expected_stop']} | {r['outcome']}/{r['stop']}{mark} | {sent} | "
                 f"{'yes' if r['gateway_received'] else 'no'} | {_cell('; '.join(r['notes']))} |")
    return "\n".join(L) + "\n"


def _cell(s: str | None) -> str:
    return (s or "").replace("|", "/").replace("\n", " ")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--c1", choices=["rules", "http"], default="rules")
    ap.add_argument("--c2", choices=["rules", "http"], default="rules")
    ap.add_argument("--base-url", help="判断モデルサーバ（社内）の URL。http のとき必須")
    ap.add_argument("--c1-model", default="laya-multilingual")
    ap.add_argument("--c2-model", default="laya-multilingual")
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--timeout", type=float, default=2.0, help="C1 / C2 のタイムアウト秒（E2E と判断層で共通）")
    ap.add_argument("--max-input-chars", type=int, default=4000)
    ap.add_argument("--now", default=DEFAULT_NOW.isoformat(), help="評価時刻（ISO 8601）。既定は固定値")
    ap.add_argument("--no-judge", action="store_true", help="判断層評価を並べない")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    now = datetime.fromisoformat(a.now)
    if now.tzinfo is None:
        now = now.replace(tzinfo=JST)
    router, gate = make_impls(a.c1, a.c2, a.base_url, a.c1_model, a.c2_model)
    cases = load_cases(a.cases)
    res = evaluate_e2e(cases, router, gate, now=now, timeout_s=a.timeout, max_input_chars=a.max_input_chars)
    label = {"c1": a.c1 if a.c1 == "rules" else f"{a.c1}:{a.c1_model}",
             "c2": a.c2 if a.c2 == "rules" else f"{a.c2}:{a.c2_model}"}
    res["label"], res["now"], res["confirm_policy"] = label, now.isoformat(), CONFIRM_POLICY
    judge = None if a.no_judge else evaluate(cases, router, gate, timeout_s=a.timeout, max_input_chars=a.max_input_chars)
    if judge is not None:
        res["judge_layer_summary"] = judge["summary"]
    (a.out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    md = to_markdown(res, judge, label)
    (a.out / "report.md").write_text(md, encoding="utf-8")
    s = res["summary"]
    print(f"outcome {_fmt(s['outcome_accuracy'])} / stop {_fmt(s['stop_accuracy'])} / "
          f"missend {_fmt(s['system_missend'])} {s['system_missend_ids']} / "
          f"false_block {_fmt(s['false_block'])} {s['false_block_ids']} / "
          f"wrong_dest {_fmt(s['wrong_destination'])} {s['wrong_destination_ids']}")
    print(f"mismatch: {s['mismatch_ids']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
