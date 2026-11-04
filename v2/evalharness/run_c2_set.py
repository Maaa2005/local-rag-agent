"""出所で分ける C2 を、評価の決まり（docs/v2-c2-eval-rules.md）の集合 1 組で採点する。Issue #17。

集合の置き場（開発用は evaluation/v2/c2_dev120）:
    bodies/<ID>@<版>.json  承認済み本文と承認の記録（acknowledged・allowed_kinds・forbidden_kinds・checked_by・reviewed_by）
    cases.jsonl            1 行 1 問（id・group・side・label・contract・approved_ref・free_text・options）

本番の policies は変えない。写しに本文と表と登録簿を足して検査を作る。登録の手順（judge.c2_registry）を通らない本文は
登録せず、その本文の問は保留に数える（誤通過 0 の線では止めた側。許可の問なら誤停止）。
流し方（v2 で）:
    python -m evalharness.run_c2_set ../evaluation/v2/c2_dev120 --out <結果.json>
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from judge import c2_registry as reg
from judge.provenance_gate import ProvenanceC2Gate
from orchestrator.candidate import ContractViolation, build_candidate
from orchestrator.policy import DEFAULT_POLICY_DIR, Policy, load_policy, sha256_text

NOW = datetime(2026, 10, 1, tzinfo=timezone.utc)
USER = "u_manager"
# 7 節の 2 の「止まった理由」。verdict の文の頭で分ける
REASON_CLASSES = (
    ("values", "values not approved"),
    ("kind", "request kind unknown"),
    ("registry", ("not registered", "registration", "request values differ", "reviewer", "sha256")),
    ("shape", ("provenance", "contract", "approved part", "letters outside")),
    ("combination", ("combination",)),
    ("content", ("request matched",)),
)


def reason_class(reason: str) -> str:
    for name, marks in REASON_CLASSES:
        for mark in (marks,) if isinstance(marks, str) else marks:
            if mark in reason:
                return name
    return "other"


def build_policy_dir(set_dir: Path, policy_dir: Path, sealed: bool = False) -> dict[str, str]:
    """写しに本文・表・登録簿を足す。登録を断られた本文は {ref: 理由} で返す（登録簿には入れない）。

    sealed なら fixture: の確認者を止める（評価の決まり 5 節。開発用の架空本文だけに使う確認者）。
    """
    shutil.copytree(DEFAULT_POLICY_DIR, policy_dir)
    records = {}
    for p in sorted((set_dir / "bodies").glob("*.json")):
        a = json.loads(p.read_text(encoding="utf-8"))
        if sealed and any(str(a.get(k, "")).strip().casefold().startswith("fixture:") for k in ("reviewed_by", "checked_by")):
            raise SystemExit(f"{p.name}: 封印用の本文に fixture: の確認者は使えない")
        a["sha256"] = sha256_text(a["body"])
        (policy_dir / "approved" / p.name).write_text(json.dumps(a, ensure_ascii=False), encoding="utf-8")
        records[f'{a["id"]}@{a["version"]}'] = a

    tables_path = policy_dir / "c2_tables.json"
    tables = json.loads(tables_path.read_text(encoding="utf-8"))
    for ref, a in records.items():
        row: dict[str, Any] = {k: "allow" for k in a.get("allowed_kinds", [])}
        row.update({k: {"decision": "block", "reason": f"承認の記録で {ref} に禁じた種類"} for k in a.get("forbidden_kinds", [])})
        tables["combinations"][ref] = row
    tables_path.write_text(json.dumps(tables, ensure_ascii=False), encoding="utf-8")

    policy = load_policy(policy_dir)
    entries = list(reg.load_registry(policy_dir / "c2_registry.json").entries_list())
    refused: dict[str, str] = {}
    for ref, a in records.items():
        approved = policy.approved[ref]
        rows = []
        try:
            for ckey in approved.contracts:
                contract = policy.contracts[ckey]
                for dest in contract.destinations:
                    if dest in approved.destinations:
                        rows.append(reg.register(approved, contract, dest, acknowledged=a.get("acknowledged", []),
                                                 checked_by=a["checked_by"], reviewed_by=a["reviewed_by"]))
        except reg.RegistrationRefused as e:
            refused[ref] = str(e)
            continue
        entries += rows
    reg.write_registry(policy_dir / "c2_registry.json", entries)
    return refused


def judge_case(case: Mapping[str, Any], policy: Policy, gate: ProvenanceC2Gate, refused: Mapping[str, str]) -> dict[str, Any]:
    ref = case["approved_ref"]
    contract = policy.contracts[case["contract"]]
    if ref in refused:
        decision, reason = "hold", f"registration refused: {refused[ref]}"
    else:
        options = case.get("options") or {k: v[0] for k, v in contract.options.items()}
        try:
            cand = build_candidate(user_id=USER, contract=contract, approved=policy.approved[ref],
                                   inp={"options": options, "free_text": case.get("free_text")},
                                   destination=contract.destinations[0], now=NOW)
        except ContractViolation as e:
            decision, reason = "hold", f"contract violation: {e}"
        else:
            verdict = gate.check_candidate(cand)
            decision, reason = verdict.decision, verdict.reason
    return {"id": case["id"], "group": case["group"], "side": case["side"], "label": case["label"], "pair_id": case.get("pair_id"),
            "decision": decision, "reason_class": reason_class(reason), "reason": reason}


def tally(rows: list[dict[str, Any]]) -> dict[str, Any]:
    missed = [r["id"] for r in rows if r["label"] == "block" and r["decision"] == "allow"]
    stopped = [r["id"] for r in rows if r["label"] == "allow" and r["decision"] != "allow"]
    return {"n": len(rows), "false_pass": len(missed), "false_stop": len(stopped),
            "false_pass_ids": missed, "false_stop_ids": stopped}


def run_set(set_dir: Path, sealed: bool = False) -> dict[str, Any]:
    cases = [json.loads(line) for line in (set_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    with tempfile.TemporaryDirectory() as tmp:
        policy_dir = Path(tmp) / "policies"
        refused = build_policy_dir(set_dir, policy_dir, sealed)
        policy = load_policy(policy_dir)
        gate = ProvenanceC2Gate(policy=policy, registry_path=policy_dir / "c2_registry.json",
                                tables_path=policy_dir / "c2_tables.json")
        rows = [judge_case(c, policy, gate, refused) for c in cases]
    return {"question_version": gate.question_version, "refused": refused, "all": tally(rows),
            "by_side": {s: tally([r for r in rows if r["side"] == s]) for s in ("request", "body")},
            "by_group": {g: tally([r for r in rows if r["group"] == g]) for g in sorted({r["group"] for r in rows})},
            "cases": rows}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("set_dir", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--sealed", action="store_true", help="封印用: fixture: の確認者を止める")
    a = p.parse_args(argv)
    res = run_set(a.set_dir, sealed=a.sealed)
    if a.out:
        a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("all", "by_side", "by_group")}, ensure_ascii=False))
    print("登録を断られた本文:", len(res["refused"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
