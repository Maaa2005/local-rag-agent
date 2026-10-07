"""送信資格の検査。policies/ の利用者・契約・承認済みテキストを読み、送ってよいかを決める。

送信資格 = 入力に使う全資料の閲覧権限 ∧ 契約の利用権限 ∧ 各入力の当該用途・宛先への外部利用許可
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

DEFAULT_POLICY_DIR = Path(__file__).resolve().parent.parent / "policies"
JST = timezone(timedelta(hours=9))

# 元資料パスの先頭ディレクトリ → 閲覧レベル（v1 の access_level）
SOURCE_DIR_LEVEL = {"general": 1, "manager": 2, "executive": 3}


@dataclass(frozen=True)
class User:
    id: str
    level: int
    contracts: tuple[str, ...]
    free_text_approval: tuple[str, ...]


@dataclass(frozen=True)
class Contract:
    key: str  # "A-faq-format@1"
    enabled: bool
    purpose: str
    input_kind: str  # "explanation" | "spec"
    options: Mapping[str, tuple[str, ...]]
    free_text_max: int | None
    payload_chars: int
    output_chars: int
    destinations: tuple[str, ...]
    template: str


@dataclass(frozen=True)
class ApprovedText:
    key: str  # "expl-keihi-001@1"
    kind: str
    body: str
    sha256: str
    approver: str
    purpose: str
    destinations: tuple[str, ...]
    expires: date
    contracts: tuple[str, ...]
    source: str | None
    source_level: int


@dataclass(frozen=True)
class Policy:
    users: Mapping[str, User]
    contracts: Mapping[str, Contract]
    approved: Mapping[str, ApprovedText]


@dataclass(frozen=True)
class Eligibility:
    ok: bool
    reason: str
    user: User | None = None
    contract: Contract | None = None
    approved: ApprovedText | None = None


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_policy(policy_dir: Path | str = DEFAULT_POLICY_DIR) -> Policy:
    d = Path(policy_dir)
    users = {
        uid: User(uid, int(u["level"]), tuple(u["contracts"]), tuple(u["free_text_approval"]))
        for uid, u in _load_json(d / "users.json").items()
    }
    contracts = {}
    for key, c in _load_json(d / "contracts.json").items():
        ft = c.get("free_text")
        contracts[key] = Contract(
            key=key,
            enabled=bool(c["enabled"]),
            purpose=c["purpose"],
            input_kind=c["input"],
            options={k: tuple(v) for k, v in c["options"].items()},
            free_text_max=int(ft["max_chars"]) if ft else None,
            payload_chars=int(c["limits"]["payload_chars"]),
            output_chars=int(c["limits"]["output_chars"]),
            destinations=tuple(c["destinations"]),
            template=c["template"],
        )
    approved = {}
    for p in sorted((d / "approved").glob("*.json")):
        a = _load_json(p)
        key = f'{a["id"]}@{a["version"]}'
        approved[key] = ApprovedText(
            key=key,
            kind=a["kind"],
            body=a["body"],
            sha256=a["sha256"],
            approver=a["approver"],
            purpose=a["purpose"],
            destinations=tuple(a["destinations"]),
            expires=date.fromisoformat(a["expires"]),
            contracts=tuple(a["contracts"]),
            source=a.get("source"),
            source_level=int(a["source_level"]),
        )
    return Policy(users=users, contracts=contracts, approved=approved)


def _source_level(source: str) -> int | None:
    head = source.replace("\\", "/").split("/", 1)[0]
    return SOURCE_DIR_LEVEL.get(head)


def _deny(reason: str, **kw: Any) -> Eligibility:
    return Eligibility(False, reason, **kw)


def check_eligibility(
    policy: Policy,
    user_id: str,
    inp: Mapping[str, Any],
    destination: str,
    now: datetime,
) -> Eligibility:
    """送信資格を検査する。now は aware datetime（naive は JST とみなす）。"""
    user = policy.users.get(user_id)
    if user is None:
        return _deny(f"未登録の利用者: {user_id}")
    contract_key = inp.get("contract")
    contract = policy.contracts.get(contract_key) if isinstance(contract_key, str) else None
    if contract is None:
        return _deny(f"未登録の契約: {contract_key}", user=user)
    if not contract.enabled:
        return _deny(f"契約が無効: {contract.key}", user=user)
    if contract.key not in user.contracts:
        return _deny(f"{user.id} は契約 {contract.key} の利用権限がない", user=user)
    if destination not in contract.destinations:
        return _deny(f"宛先 {destination} は契約 {contract.key} で許可されていない", user=user)

    # 元資料を直接指定した入力は承認対象外（承認済みテキストだけを送れる）
    source = inp.get("source")
    if source:
        lvl = _source_level(str(source))
        if lvl is None or lvl > user.level:
            return _deny(f"元資料 {source} の閲覧権限がない（利用者 Lv{user.level}）", user=user)
        return _deny(f"元資料 {source} は送信用の事前承認がない", user=user)

    ref = inp.get(contract.input_kind)
    if not isinstance(ref, str) or not ref:
        return _deny(f"契約 {contract.key} の入力 {contract.input_kind} に事前承認済みテキストの指定がない", user=user)
    approved = policy.approved.get(ref)
    if approved is None:
        return _deny(f"事前承認済みテキストが見つからない: {ref}", user=user)
    base = dict(user=user, contract=contract, approved=approved)
    if approved.kind != contract.input_kind or contract.key not in approved.contracts:
        return _deny(f"{ref} は契約 {contract.key} での利用が承認されていない", **base)
    if approved.source_level > user.level:
        return _deny(f"{ref} の元資料（Lv{approved.source_level}）の閲覧権限がない", **base)
    if sha256_text(approved.body) != approved.sha256:
        return _deny(f"{ref} の本文が承認時と一致しない（hash 不一致）。再承認が必要", **base)
    if destination not in approved.destinations:
        return _deny(f"{ref} は宛先 {destination} への外部利用が承認されていない", **base)
    if approved.purpose != contract.purpose:
        return _deny(f"{ref} の承認用途（{approved.purpose}）が契約の用途と異なる", **base)
    local_now = (now if now.tzinfo else now.replace(tzinfo=JST)).astimezone(JST)
    if local_now.date() > approved.expires:
        return _deny(f"{ref} の承認期限（{approved.expires.isoformat()}）が切れている", **base)
    if contract.free_text_max is not None and contract.key not in user.free_text_approval:
        return _deny(f"{user.id} は契約 {contract.key} の依頼文を外部利用承認できない", **base)
    return Eligibility(True, "ok", **base)
