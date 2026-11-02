"""承認済み本文の登録簿（Issue #17「ID とハッシュの許可表」）。

送信時の C2（judge.provenance_gate）は、承認済み本文の中身を手がかり語で検査し直さない。代わりに
この登録簿で「同じ ID・版・本文（sha256）が、この契約・用途・宛先で登録時の検査を通ったか」を引く。
ハッシュは同じ本文であることの証明で、安全な本文であることの証明ではない。そこで登録時に内容を
検査し、その結果を用途・宛先・版（ID@版と契約の版）に結びつけて残す。

登録時の検査（registration_hits）は、入口で依頼文と同じ正規化（NFKC・書式文字の除去）をしてから、
今の RuleC2Gate の規則に連絡先（メール・電話・内線）を足した規則を当てる。当たった項目は、承認者が
一つずつ承認（acknowledged）しない限り登録しない。当たっていない項目の承認も受け付けない。
検査の規則を変えると REGISTRATION_CHECK_VERSION が変わり、古い版で登録した本文は送信時に保留になる
（登録し直すまで送らない）。

登録簿は policies/c2_registry.json。作り直しと照合:
  python -m judge.c2_registry --write   # policies/approved/ から作り直す（当たりがあれば止まる）
  python -m judge.c2_registry --check   # ファイルが今の本文・今の検査と一致するか
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from judge.base import question_version
from judge.normalize import normalize_text
from judge.rules import _C2_PATTERNS
from orchestrator.policy import DEFAULT_POLICY_DIR, ApprovedText, Contract, Policy, load_policy, sha256_text

DEFAULT_REGISTRY_PATH = DEFAULT_POLICY_DIR / "c2_registry.json"
FORMAT = "c2-registry@1"

_REGISTRATION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    *_C2_PATTERNS,
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("phone", re.compile(r"(内線\s*\d{3,}|(?<!\d)\d{2,4}-\d{2,4}-\d{3,4}(?!\d))")),
]
REGISTRATION_CHECK_VERSION = question_version(
    "c2-registration@1", {"normalize": "judge.normalize.normalize_text@1",
                          "patterns": [(n, p.pattern, p.flags) for n, p in _REGISTRATION_PATTERNS]})

_ENTRY_KEYS = ("ref", "sha256", "contract", "purpose", "destination", "check_version", "hits", "acknowledged",
               "checked_by", "result")


class RegistrationRefused(ValueError):
    """登録時の検査を通らない、または承認の範囲外の登録。"""


def registration_hits(body: str) -> list[str]:
    text = normalize_text(body)
    return [name for name, pat in _REGISTRATION_PATTERNS if pat.search(text)]


def register(approved: ApprovedText, contract: Contract, destination: str, *, acknowledged: Sequence[str] = (),
             checked_by: str) -> dict[str, Any]:
    """1 つの使い道（契約・用途・宛先）について登録時の検査をし、登録簿の 1 行を返す。

    承認の範囲外（契約・用途・宛先が承認済み本文または契約と合わない）、本文が承認時の sha256 と
    違う、当たった項目が承認されていない、のどれかなら RegistrationRefused。
    """
    if sha256_text(approved.body) != approved.sha256:
        raise RegistrationRefused(f"{approved.key}: body does not match its approved sha256")
    if contract.key not in approved.contracts:
        raise RegistrationRefused(f"{approved.key}: contract {contract.key} is not in its approval")
    if approved.kind != contract.input_kind:
        raise RegistrationRefused(f"{approved.key}: kind {approved.kind} is not the contract input {contract.input_kind}")
    if approved.purpose != contract.purpose:
        raise RegistrationRefused(f"{approved.key}: purpose {approved.purpose} differs from {contract.purpose}")
    if destination not in approved.destinations or destination not in contract.destinations:
        raise RegistrationRefused(f"{approved.key}: destination {destination} is outside the approval or contract")
    hits = registration_hits(approved.body)
    ack = sorted(set(acknowledged))
    extra = [a for a in ack if a not in hits]
    if extra:
        raise RegistrationRefused(f"{approved.key}: acknowledged items were not hit: {extra}")
    missing = [h for h in hits if h not in ack]
    if missing:
        raise RegistrationRefused(f"{approved.key}: registration check hit {missing} without acknowledgement")
    return {"ref": approved.key, "sha256": approved.sha256, "contract": contract.key, "purpose": contract.purpose,
            "destination": destination, "check_version": REGISTRATION_CHECK_VERSION, "hits": hits,
            "acknowledged": ack, "checked_by": checked_by, "result": "pass"}


def build_entries(policy: Policy, *, checked_by: str,
                  acknowledged: Mapping[str, Sequence[str]] | None = None) -> list[dict[str, Any]]:
    """policies/ の承認済み本文のすべての使い道（承認と契約の両方が許す契約×宛先）を登録する。"""
    out = []
    for a in sorted(policy.approved.values(), key=lambda x: x.key):
        for ck in a.contracts:
            c = policy.contracts.get(ck)
            if c is None:
                continue
            for d in c.destinations:
                if d in a.destinations:
                    out.append(register(a, c, d, acknowledged=(acknowledged or {}).get(a.key, ()),
                                        checked_by=checked_by))
    return out


def _dump(entries: Iterable[Mapping[str, Any]]) -> str:
    rows = [{k: e[k] for k in _ENTRY_KEYS} for e in entries]
    rows.sort(key=lambda e: (e["ref"], e["contract"], e["destination"]))
    return json.dumps({"format": FORMAT, "entries": rows}, ensure_ascii=False, indent=2) + "\n"


def write_registry(path: Path | str, entries: Iterable[Mapping[str, Any]]) -> None:
    Path(path).write_bytes(_dump(entries).encode("utf-8"))


@dataclass(frozen=True)
class Registry:
    entries: tuple[Mapping[str, Any], ...]
    sha256: str  # 登録簿の中身の sha256（質問の版に入れる）

    def find(self, ref: str, contract: str, purpose: str, destination: str) -> Mapping[str, Any] | None:
        for e in self.entries:
            if (e["ref"], e["contract"], e["purpose"], e["destination"]) == (ref, contract, purpose, destination):
                return e
        return None

    def entries_list(self) -> list[dict[str, Any]]:
        return [dict(e) for e in self.entries]


def load_registry(path: Path | str = DEFAULT_REGISTRY_PATH) -> Registry:
    raw = Path(path).read_bytes()
    doc = json.loads(raw.decode("utf-8"))
    if doc.get("format") != FORMAT:
        raise ValueError(f"registry format {doc.get('format')!r} != {FORMAT}")
    entries = []
    for e in doc["entries"]:
        if set(e) != set(_ENTRY_KEYS):
            raise ValueError(f"registry entry keys {sorted(e)} != {sorted(_ENTRY_KEYS)}")
        entries.append(e)
    seen = {(e["ref"], e["contract"], e["purpose"], e["destination"]) for e in entries}
    if len(seen) != len(entries):
        raise ValueError("registry has duplicate uses")
    # 改行コードや並びの違いで版が変わらないよう、正規化した書き出しのハッシュを使う
    return Registry(tuple(entries), hashlib.sha256(_dump(entries).encode("utf-8")).hexdigest())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true", help="policies/approved/ から登録簿を作り直す")
    g.add_argument("--check", action="store_true", help="登録簿が今の本文・今の検査と一致するか")
    ap.add_argument("--policy-dir", type=Path, default=DEFAULT_POLICY_DIR)
    ap.add_argument("--registry", type=Path, default=None)
    ap.add_argument("--checked-by", default="registry-build")
    a = ap.parse_args(argv)
    path = a.registry or a.policy_dir / "c2_registry.json"
    pol = load_policy(a.policy_dir)
    old = load_registry(path) if path.exists() else None
    # 承認者の承認（acknowledged）は今の登録簿から引き継ぐ。本文が変わった物は引き継がない
    ack = {e["ref"]: e["acknowledged"] for e in (old.entries if old else ())
           if e["ref"] in pol.approved and pol.approved[e["ref"]].sha256 == e["sha256"]}
    try:
        entries = build_entries(pol, checked_by=a.checked_by, acknowledged=ack)
    except RegistrationRefused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 1
    if a.write:
        write_registry(path, entries)
        print(f"wrote {len(entries)} entries to {path}")
        return 0
    if old is None:
        print(f"missing: {path}", file=sys.stderr)
        return 1
    want = json.loads(_dump(entries))["entries"]
    have = json.loads(_dump(old.entries))["entries"]
    strip = lambda rows: [{k: v for k, v in r.items() if k != "checked_by"} for r in rows]  # noqa: E731
    if strip(want) != strip(have):
        print("registry differs from the current approved texts / registration check", file=sys.stderr)
        return 1
    print(f"ok: {len(have)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
