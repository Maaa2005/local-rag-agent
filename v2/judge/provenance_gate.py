"""出所で分ける C2（Issue #17・docs/research/2026-11-02/judge.md の 1 番）。

送信物を「承認済み本文（ID とハッシュ）」と「依頼文」に分けて検査する。判定はすべてコードで、
モデルは使わない（Shieldstral などの比較は送信の判定に入れない。使うなら再審議）。
既定の C2 は今も judge.rules.RuleC2Gate（integration.flow.default_c2）で、これは選んだときだけ使う。

出所（区分）は組み立て側（orchestrator.candidate._render）が付ける。ゲートは本文中の見出しを
解析せず、SendCandidate.rendered の区分だけを使う。判定の順:
1. 区分の形（ずれたら保留）: 区分から復元した内容が送信内容と一致する／契約のテンプレートどおりの
   並びで、固定文が一字一句同じ／選択値が契約の許可リストにある／候補の approved_ref・sha256 と一致する
2. 承認済み本文: 登録簿（judge.c2_registry）に、同じ ID@版・契約・用途・宛先の登録がある（無ければ保留）。
   本文の sha256 が登録と違えば遮断。登録時の検査の版が今と違えば保留（登録し直すまで送らない）
3. 依頼文だけに手がかり語の検査（正規化してから）。当たれば遮断
4. 依頼の種類（表の request_kinds）を決め、承認済み本文 ID × 種類 を表の combinations で引く。
   禁止は遮断、表に無い組み合わせと種類不明は保留
遮断が 1 つでもあれば遮断、なければ保留が 1 つでもあれば保留、どちらも無ければ許可。

SendCandidate を受け取る check_candidate が本体。C2Gate の I/F（check(payload)）だけでは出所が
分からないので保留にする。送信手順（orchestrator.pipeline）は check_candidate を持つゲートに
候補を束ねて渡す（pipeline.bind_candidate）。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from common.provenance import RenderedMessage, mismatch
from common.schemas import SendPayload
from judge.base import C2Verdict, question_version
from judge.c2_registry import DEFAULT_REGISTRY_PATH, REGISTRATION_CHECK_VERSION, Registry, load_registry
from judge.normalize import normalize_text
from orchestrator.policy import DEFAULT_POLICY_DIR, Contract, Policy, load_policy, sha256_text

MODEL = "provenance-rules"
REVISION = "provenance-v1"
DEFAULT_TABLES_PATH = DEFAULT_POLICY_DIR / "c2_tables.json"
TABLES_FORMAT = "c2-tables@1"

# 依頼文だけに掛ける手がかり語（正規化した後の文字に当てる）。契約 B の「禁止する情報」
# （docs/v2-contract-examples.md: 個人名・社員番号、取引先名、金額・日付の実データ、社内のテーブル名）、
# 未公開の M&A、指示の注入、送信審査への働きかけ、区分の見出しの偽装、契約の制約を外す依頼。
# 金額・日付は「〜円を超える」「〜以降」のような処理条件（契約 B が送ってよいとする閾値）には当てない
_NOT_COND_AMOUNT = r"(?!\s*(を?超|以上|以下|未満|より(大き|小さ|多|少|高|安)))"
_NOT_COND_DATE = r"(?!\s*日?\s*(以降|以前|から|まで|より(前|後)))"
REQUEST_CUES: list[tuple[str, re.Pattern[str]]] = [
    ("employee_id", re.compile(r"(?<![A-Za-z0-9])E\d{3,}")),
    ("person_name", re.compile(r"[一-龥]{1,4}(さん|くん|氏|(?<![仕同模多各異])様)")),
    ("counterparty", re.compile(r"([A-Z]社|株式会社|有限会社|合同会社|\(株\)|\(有\))")),
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("phone", re.compile(r"(内線\s*\d{3,}|(?<!\d)\d{2,4}-\d{2,4}-\d{3,4}(?!\d))")),
    ("amount", re.compile(r"((?<![\d,.])\d[\d,.]*\s*(億|万|千)?円|[億万]円)" + _NOT_COND_AMOUNT)),
    ("real_date", re.compile(r"(?<!\d)\d{4}\s*[-/年]\s*\d{1,2}\s*[-/月]\s*\d{1,2}(?!\d)" + _NOT_COND_DATE)),
    ("table_name", re.compile(r"(?<![\w.])[A-Za-z_][A-Za-z0-9_]*\.(?!(?:csv|py|txt|json|md|xlsx?)\b)[A-Za-z_][A-Za-z0-9_]*")),
    ("m_and_a", re.compile(r"(M&A|買収|合併|TOB|出資比率|株式(を|の)?(段階)?(取得|譲渡|売却|買い増し|買い付け)|段階取得|"
                           r"資本(業務)?提携|事業譲渡|経営統合|子会社化)", re.I)),
    ("injection", re.compile(r"(前の指示(は|を)?無視|指示(は|を)無視|指示を(すべて|全部)?忘れ|これまでの(指示|命令)|"
                             r"ignore (all )?(the )?previous|システムプロンプト|system prompt|全文を出力)", re.I)),
    ("gate_override", re.compile(r"(送信(ゲート|前の(審査|検査|確認))|審査の対象外|(検査|審査|判定)(を|の|は)?(対象外|通過済|免除|不要|省略|スキップ)|"
                                 r"(安全|許可|問題(が)?ない|問題なし)と(判定|判断|みな)|判定する側|判定(者|役|モデル)(へ|に)の?指示|"
                                 r"(答え|回答|判定)は「?(はい|いいえ|yes|no|allow|許可)」?と(して|する)|確認済みです|承認済みです)", re.I)),
    ("forged_heading", re.compile(r"^\s*(-{3,}\s*$|区分\s*:|仕様\s*:|依頼\s*:|説明文\s*:|(system|user|assistant)\s*:|\[(system|user)\])",
                                  re.I | re.M)),
    ("constraint_override", re.compile(r"(条件|制約|仕様|ルール)(は|を)?(すべて|全部)?(外し|無視し|守らなく|破っ|緩め|気にしなく)")),
]


def normalize_request(text: str) -> str:
    return normalize_text(text)


def request_cue_hits(text: str) -> list[str]:
    t = normalize_request(text)
    return [name for name, pat in REQUEST_CUES if pat.search(t)]


@dataclass(frozen=True)
class Tables:
    kinds: Mapping[str, Any]  # 契約キー → 固定の種類（str）または [(種類, 正規表現)]
    combinations: Mapping[str, Mapping[str, tuple[str, str]]]  # ID@版 → 種類 → (allow/block, 理由)
    canonical: str  # 版のハッシュに使う正規化 JSON

    def request_kinds(self, contract: str, request: str | None) -> list[str]:
        k = self.kinds.get(contract)
        if k is None:
            return []
        if isinstance(k, str):
            return [k] if request is None else []
        if request is None:
            return []
        t = normalize_request(request)
        return [name for name, pat in k if pat.search(t)]


def load_tables(path: Path | str = DEFAULT_TABLES_PATH) -> Tables:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("format") != TABLES_FORMAT:
        raise ValueError(f"tables format {doc.get('format')!r} != {TABLES_FORMAT}")
    kinds: dict[str, Any] = {}
    for ck, spec in doc["request_kinds"].items():
        if set(spec) == {"fixed"} and isinstance(spec["fixed"], str) and spec["fixed"]:
            kinds[ck] = spec["fixed"]
        elif set(spec) == {"rules"} and isinstance(spec["rules"], dict) and spec["rules"]:
            try:
                kinds[ck] = [(n, re.compile(p)) for n, p in spec["rules"].items()]
            except re.error as e:
                raise ValueError(f"request_kinds {ck}: bad pattern: {e}") from e
        else:
            raise ValueError(f"request_kinds {ck}: exactly one of fixed (str) or rules (object) is required")
    combos: dict[str, dict[str, tuple[str, str]]] = {}
    for ref, row in doc["combinations"].items():
        combos[ref] = {}
        for kind, v in row.items():
            decision, reason = (v, "") if isinstance(v, str) else (v.get("decision"), v.get("reason", ""))
            if decision not in ("allow", "block") or not isinstance(reason, str):
                raise ValueError(f"combinations {ref} x {kind}: decision must be allow or block, got {v!r}")
            combos[ref][kind] = (decision, reason)
    canonical = json.dumps({"request_kinds": doc["request_kinds"], "combinations": doc["combinations"]},
                           ensure_ascii=False, sort_keys=True)
    return Tables(kinds, combos, canonical)


class ProvenanceC2Gate:
    def __init__(self, policy: Policy | None = None, registry_path: Path | str = DEFAULT_REGISTRY_PATH,
                 tables_path: Path | str = DEFAULT_TABLES_PATH):
        self.policy = policy or load_policy()
        self.registry: Registry = load_registry(registry_path)
        self.tables = load_tables(tables_path)
        self.question_version = question_version("provenance-c2@1", {
            "normalize": "judge.normalize.normalize_text@1",
            "cues": [(n, p.pattern, p.flags) for n, p in REQUEST_CUES],
            "registration_check": REGISTRATION_CHECK_VERSION,
            "registry_sha256": self.registry.sha256,
            "tables": self.tables.canonical,
        })

    def _verdict(self, decision: str, reasons: Sequence[str]) -> C2Verdict:
        prob = {"allow": 0.0, "block": 1.0}.get(decision)
        return C2Verdict(decision=decision, reason="; ".join(reasons), model=MODEL, revision=REVISION,
                         prob_block=prob, question_version=self.question_version)

    def check(self, payload: SendPayload) -> C2Verdict:
        return self._verdict("hold", ["no provenance: provenance gate needs the send candidate (check_candidate)"])

    def check_candidate(self, cand: Any) -> C2Verdict:
        """cand は orchestrator.candidate.SendCandidate（payload・rendered・approved_ref・approved_sha256）。"""
        payload: SendPayload = cand.payload
        rendered: Sequence[RenderedMessage] = cand.rendered
        why = mismatch(rendered, payload.messages)
        if why is not None:
            return self._verdict("hold", [f"provenance mismatch: {why}"])
        contract = self.policy.contracts.get(payload.contract)
        if contract is None or cand.contract != payload.contract:
            return self._verdict("hold", [f"contract {payload.contract} unknown or differs from candidate"])
        why = _canonical_problem(contract, rendered)
        if why is not None:
            return self._verdict("hold", [why])

        parts = [p for m in rendered for p in m.parts]
        approved = [p for p in parts if p.source == "approved"]
        requests = [p.text for p in parts if p.source == "free_text"]
        blocks: list[str] = []
        holds: list[str] = []
        refs: list[str] = []
        for p in approved:
            if p.ref is None:
                return self._verdict("hold", ["approved part without ref"])
            sha = sha256_text(p.text)
            if p.ref != cand.approved_ref or sha != cand.approved_sha256:
                return self._verdict("hold", [f"approved part {p.ref} differs from candidate approved_ref/sha256"])
            refs.append(p.ref)
            e = self.registry.find(p.ref, contract.key, contract.purpose, payload.destination)
            if e is None:
                holds.append(f"{p.ref} not registered for {contract.key}/{payload.destination}")
                continue
            if e["sha256"] != sha:
                blocks.append(f"{p.ref} body sha256 differs from registry")
            elif e["result"] != "pass":
                holds.append(f"{p.ref} registration result {e['result']}")
            elif e["check_version"] != REGISTRATION_CHECK_VERSION:
                holds.append(f"{p.ref} registration check outdated ({e['check_version']})")

        for text in requests:
            hits = request_cue_hits(text)
            if hits:
                blocks.append("request matched: " + ",".join(hits))

        request = "\n".join(requests) if requests else None
        kinds = self.tables.request_kinds(contract.key, request)
        if not kinds:
            holds.append(f"request kind unknown for {contract.key}")
        for ref in refs:
            row = self.tables.combinations.get(ref, {})
            for kind in kinds:
                if kind not in row:
                    holds.append(f"combination {ref} x {kind} not in table")
                elif row[kind][0] == "block":
                    blocks.append(f"combination {ref} x {kind} forbidden" + (f": {row[kind][1]}" if row[kind][1] else ""))

        if blocks:
            return self._verdict("block", blocks)
        if holds:
            return self._verdict("hold", holds)
        return self._verdict("allow", [f"registered {','.join(refs)}; kinds {','.join(kinds)}; no request cue"])


def _canonical_problem(contract: Contract, rendered: Sequence[RenderedMessage]) -> str | None:
    """区分の並びが契約のテンプレートどおりか。固定文は一字一句、選択値は許可リストで確かめる。"""
    from orchestrator.candidate import ContractViolation, _render

    stub_opts = {k: f"\x00option:{k}" for k in contract.options}
    stub_free = "\x00free_text" if contract.free_text_max is not None else None
    try:
        stub = _render(contract, SimpleNamespace(body="\x00approved", key="\x00ref"), stub_opts, stub_free)
    except (ContractViolation, KeyError) as e:
        return f"not canonical: template of {contract.key} cannot be rendered ({e})"
    if len(stub) != len(rendered):
        return f"not canonical: {len(rendered)} messages where template has {len(stub)}"
    for i, (sm, rm) in enumerate(zip(stub, rendered)):
        if sm.role != rm.role or len(sm.parts) != len(rm.parts):
            return f"not canonical: message {i} shape differs from template"
        for sp, rp in zip(sm.parts, rm.parts):
            if sp.source != rp.source:
                return f"not canonical: message {i} has {rp.source} where template has {sp.source}"
            if sp.source in ("system", "template") and sp.text != rp.text:
                return f"not canonical: message {i} fixed {sp.source} text differs from template"
            if sp.source == "option":
                key = sp.text.removeprefix("\x00option:")
                if rp.text not in contract.options.get(key, ()):
                    return f"option {key} value is not in the contract allowlist"
    return None
