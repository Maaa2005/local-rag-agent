"""承認済み本文の登録簿（Issue #17「ID とハッシュの許可表」）。

送信時の C2（judge.provenance_gate）は、承認済み本文の中身を手がかり語で検査し直さない。代わりに
この登録簿で「同じ ID・版・本文（sha256）が、この契約・用途・宛先で登録時の検査を通ったか」を引く。
ハッシュは同じ本文であることの証明で、安全な本文であることの証明ではない。そこで登録時に内容を
検査し、その結果を用途・宛先・版（ID@版と契約の版）に結びつけて残す。

登録時の検査（registration_hits）は、送信時に依頼文へ当てる手がかり語・文字種（judge.request_cues）と、
今の RuleC2Gate の規則（金額は閾値の形でも当たる）の和。契約 A は送信内容がほぼ承認済み本文だけなので、
ここが唯一の内容検査になる。当たった項目は、承認者が一つずつ承認（acknowledged）しない限り登録しない。
当たっていない項目の承認も受け付けない。検査の規則を変えると REGISTRATION_CHECK_VERSION が変わり、
古い版で登録した本文は送信時に保留になる（登録し直すまで送らない）。

登録簿は policies/c2_registry.json。署名は無いので、読み込み時に行の形と「当たり＝承認」を確かめる。
作り直しと照合:
  python -m judge.c2_registry --write [--acknowledge ID@版=項目,項目 --checked-by 承認者]
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
from judge.request_cues import CUES_VERSION, cue_hits, unexpected_letters
from judge.rules import _C2_PATTERNS
from orchestrator.policy import DEFAULT_POLICY_DIR, ApprovedText, Contract, Policy, load_policy, sha256_text

DEFAULT_REGISTRY_PATH = DEFAULT_POLICY_DIR / "c2_registry.json"
FORMAT = "c2-registry@1"

REGISTRATION_CHECK_SPEC = {
    "request_cues": CUES_VERSION,  # 正規化の版を含む
    "rule_patterns": [(n, p.pattern, p.flags) for n, p in _C2_PATTERNS],
    "unexpected_script": "judge.request_cues.unexpected_letters",
}
REGISTRATION_CHECK_VERSION = question_version("c2-registration@2", REGISTRATION_CHECK_SPEC)

_ENTRY_KEYS = ("ref", "sha256", "contract", "purpose", "destination", "check_version", "hits", "acknowledged",
               "checked_by", "result")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class RegistrationRefused(ValueError):
    """登録時の検査を通らない、または承認の範囲外の登録。"""


def registration_hits(body: str) -> list[str]:
    n = normalize_text(body)
    hits = [name for name, pat in _C2_PATTERNS if pat.search(n)] + cue_hits(body)
    if unexpected_letters(body):
        hits.append("unexpected_script")
    return list(dict.fromkeys(hits))


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


def build_entries(policy: Policy, *, checked_by: str, acknowledged: Mapping[str, Sequence[str]] | None = None,
                  previous: Iterable[Mapping[str, Any]] = ()) -> list[dict[str, Any]]:
    """policies/ の承認済み本文のすべての使い道（承認と契約の両方が許す契約×宛先）を登録する。

    previous（今の登録簿）に、同じ使い道・本文・当たり・承認の行があれば、その checked_by を引き継ぐ。
    """
    def same(e: Mapping[str, Any]) -> tuple[Any, ...]:
        return (e["ref"], e["contract"], e["destination"], e["sha256"], tuple(e["hits"]), tuple(e["acknowledged"]))

    prev = {same(e): e["checked_by"] for e in previous}
    out = []
    for a in sorted(policy.approved.values(), key=lambda x: x.key):
        for ck in a.contracts:
            c = policy.contracts.get(ck)
            if c is None:
                continue
            for d in c.destinations:
                if d in a.destinations:
                    e = register(a, c, d, acknowledged=(acknowledged or {}).get(a.key, ()), checked_by=checked_by)
                    e["checked_by"] = prev.get(same(e), checked_by)
                    out.append(e)
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


def _row_problem(e: Any) -> str | None:
    if not isinstance(e, dict) or set(e) != set(_ENTRY_KEYS):
        return f"keys {sorted(e) if isinstance(e, dict) else type(e).__name__} != {sorted(_ENTRY_KEYS)}"
    for k in ("ref", "contract", "purpose", "destination", "check_version", "checked_by", "result"):
        if not isinstance(e[k], str) or not e[k]:
            return f"{k} must be a non-empty string"
    if not _SHA256.fullmatch(e["sha256"] if isinstance(e["sha256"], str) else ""):
        return "sha256 must be 64 lowercase hex digits"
    for k in ("hits", "acknowledged"):
        if not isinstance(e[k], list) or not all(isinstance(x, str) for x in e[k]):
            return f"{k} must be a list of strings"
    if set(e["hits"]) != set(e["acknowledged"]):
        return f"hits {e['hits']} and acknowledged {e['acknowledged']} differ"
    if e["result"] != "pass":
        return f"result {e['result']!r} is not pass"
    return None


def load_registry(path: Path | str = DEFAULT_REGISTRY_PATH) -> Registry:
    doc = json.loads(Path(path).read_bytes().decode("utf-8"))
    if not isinstance(doc, dict) or doc.get("format") != FORMAT:
        raise ValueError(f"registry format != {FORMAT}")
    if not isinstance(doc.get("entries"), list):
        raise ValueError("registry entries must be a list")
    for i, e in enumerate(doc["entries"]):
        why = _row_problem(e)
        if why is not None:
            raise ValueError(f"registry entry {i}: {why}")
    entries = doc["entries"]
    if len({(e["ref"], e["contract"], e["purpose"], e["destination"]) for e in entries}) != len(entries):
        raise ValueError("registry has duplicate uses")
    # 改行コードや並びの違いで版が変わらないよう、正規化した書き出しのハッシュを使う
    return Registry(tuple(entries), hashlib.sha256(_dump(entries).encode("utf-8")).hexdigest())


def _parse_ack(items: Sequence[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for s in items:
        ref, sep, names = s.partition("=")
        if not sep or not ref:
            raise SystemExit(f"--acknowledge needs ID@版=項目,項目: {s!r}")
        out[ref] = [n for n in names.split(",") if n]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true", help="policies/approved/ から登録簿を作り直す")
    g.add_argument("--check", action="store_true", help="登録簿が今の本文・今の検査と一致するか")
    ap.add_argument("--policy-dir", type=Path, default=DEFAULT_POLICY_DIR)
    ap.add_argument("--registry", type=Path, default=None)
    ap.add_argument("--acknowledge", action="append", default=[],
                    help="承認者が承知の上で残す当たり。ID@版=項目,項目（--write のとき）")
    ap.add_argument("--checked-by", default="registry-build", help="新しく登録・承認した行に残す確認者")
    a = ap.parse_args(argv)
    path = a.registry or a.policy_dir / "c2_registry.json"
    pol = load_policy(a.policy_dir)
    old = load_registry(path) if path.exists() else None
    # 承認（acknowledged）は今の登録簿から引き継ぐ。本文が変わった物は引き継がない。--acknowledge が優先
    ack = {e["ref"]: e["acknowledged"] for e in (old.entries if old else ())
           if e["ref"] in pol.approved and pol.approved[e["ref"]].sha256 == e["sha256"]}
    if a.acknowledge:
        if not a.write:
            raise SystemExit("--acknowledge is only for --write")
        ack.update(_parse_ack(a.acknowledge))
    try:
        entries = build_entries(pol, checked_by=a.checked_by, acknowledged=ack, previous=old.entries if old else ())
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
    if _dump(entries) != _dump(old.entries):
        print("registry differs from the current approved texts / registration check", file=sys.stderr)
        return 1
    print(f"ok: {len(old.entries)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
