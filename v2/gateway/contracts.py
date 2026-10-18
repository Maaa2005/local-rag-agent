"""Gateway 側の固定契約レジストリ（コード内定数）。

値は policies/contracts.json（Orchestrator 側の契約定義）の limits / destinations / enabled と揃える
（一致は tests/gateway/test_destination_fixed.py で自動検査）。
初版は 1 契約 1 宛先（A=claude、B=codex）。宛先が 1 件でない・無効な契約は prepare も commit も通さない。
Gateway は社内ゾーンの設定ファイルを読まず、この定数だけを正とする。
キーは完全一致で引く（"A-faq-format@1" など）。
"""
from __future__ import annotations

from dataclasses import dataclass

from common.schemas import MAX_MESSAGES  # noqa: F401  messages 件数上限（schemas と共通）

# 出力文字数 → トークン上限の換算係数（日本語 1 字 ≒ 1〜2 トークンを見込んだ安全側）
TOKENS_PER_OUTPUT_CHAR = 2


@dataclass(frozen=True)
class ContractSpec:
    key: str
    payload_chars: int          # 送信本文（全 message の content 合計）の上限
    output_chars: int           # max_output_chars の上限
    destinations: frozenset[str]
    max_ttl_seconds: float      # expires_at - now の上限
    enabled: bool = True        # False なら prepare を拒否し、既存の PREPARED も commit で送らない

    @property
    def max_output_tokens(self) -> int:
        return self.output_chars * TOKENS_PER_OUTPUT_CHAR


CONTRACTS: dict[str, ContractSpec] = {
    "A-faq-format@1": ContractSpec(
        key="A-faq-format@1",
        payload_chars=2000,
        output_chars=1500,
        destinations=frozenset({"claude"}),
        max_ttl_seconds=15 * 60,
    ),
    "B-csv-codegen@1": ContractSpec(
        key="B-csv-codegen@1",
        payload_chars=1500,
        output_chars=3000,
        destinations=frozenset({"codex"}),
        max_ttl_seconds=15 * 60,
    ),
}



def lookup(contract: str) -> ContractSpec | None:
    return CONTRACTS.get(contract)
