"""C1 の宛先解決。契約キーを社内の契約定義（policies/contracts.json）から完全一致で引き、
その契約の唯一の許可宛先を返す。

設計書: 初版では、C1 が処理経路を判断し、外部経路については対応する有効な契約版の唯一の
許可宛先を返す。宛先の推論・代替選択は行わない。
- 宛先を判断モデルに質問しない。destination_hint（利用者の希望）は参考情報で宛先を変えない。
- 前方一致（別の版への読み替え）や「読めなければ両方」の代替はしない。
- 設定異常・未知の版・無効契約・経路と契約の不整合・宛先が 1 件でない場合は解決しない
  （呼び出し側が human / hold に倒す）。
Orchestrator の送信資格の検査も同じ orchestrator.policy.load_contracts で同じファイルを読む。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from orchestrator.policy import DEFAULT_POLICY_DIR, Contract, load_contracts

CONTRACTS_PATH = DEFAULT_POLICY_DIR / "contracts.json"
EXTERNAL_ROUTES = ("A", "B")


def lookup_contract(contract_key: Any, path: Path | str = CONTRACTS_PATH) -> tuple[Contract | None, str]:
    """契約キーを完全一致で引く。使えない契約なら (None, 理由)。"""
    if not isinstance(contract_key, str) or not contract_key:
        return None, f"contract key missing: {contract_key!r}"
    try:
        contracts = load_contracts(path)
    except Exception as e:  # noqa: BLE001  設定が読めない＝設定異常として保留
        return None, f"contract config unreadable: {type(e).__name__}"
    c = contracts.get(contract_key)
    if c is None:
        return None, f"unknown contract version: {contract_key}"
    if not c.enabled:
        return None, f"contract disabled: {contract_key}"
    if c.route not in EXTERNAL_ROUTES:
        return None, f"contract {contract_key} has no external route: {c.route!r}"
    if len(c.destinations) != 1:
        return None, f"contract {contract_key} must have exactly one destination: {list(c.destinations)}"
    return c, "ok"


def resolve_destination(route: str, contract_key: Any, path: Path | str = CONTRACTS_PATH) -> tuple[str | None, str]:
    """外部経路 route と契約キーから唯一の宛先を返す。解決できなければ (None, 理由)。"""
    c, why = lookup_contract(contract_key, path)
    if c is None:
        return None, why
    if route != c.route:
        return None, f"route {route} does not match contract {contract_key} (route {c.route})"
    return c.destinations[0], f"contract {contract_key} sole destination"
