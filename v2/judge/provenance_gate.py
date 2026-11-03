"""出所で分ける C2（Issue #17・docs/research/2026-11-02/judge.md の 1 番）。

送信物を「承認済み本文（ID とハッシュ）」と「依頼文」に分けて検査する。判定はすべてコードで、
モデルは使わない（Shieldstral などの比較は送信の判定に入れない。使うなら再審議）。
既定の C2 は今も judge.rules.RuleC2Gate（integration.flow.default_c2）で、これは選んだときだけ使う。

出所（区分）は組み立て側（orchestrator.candidate._render）が付ける。ゲートは本文中の見出しを
解析せず、SendCandidate.rendered の区分だけを使う。判定の順:
1. 区分の形（ずれたら保留）: 区分から復元した内容が送信内容と一致する／契約のテンプレートどおりの
   並びで、固定文が一字一句同じ／選択値が契約の許可リストにある／添付（common.provenance.attachments）が
   候補の approved_ref・sha256 と一致する
2. 承認済み本文: 登録簿（judge.c2_registry）に、同じ ID@版・契約・用途・宛先の登録が無ければ保留。
   本文の sha256 が登録と違えば遮断。登録時の検査の版が今と違えば保留（登録し直すまで送らない）
3. 依頼文だけに手がかり語の検査（judge.request_cues。正規化した写しと空白を除いた写し）。当たれば遮断。
   かな・漢字・ラテン文字以外の文字があれば保留
4. 依頼文の自由な値（judge.request_values）: 数字・漢数字の値・英字の語・引用符の中身が、契約の語彙と
   書き方（表の request_values）か承認済み本文の承認の記録（登録簿に写した request_values）に無ければ保留
   （本人決定 11/3 の 1）。承認の記録の値が登録簿に写した物と違えば保留（登録し直すまで送らない）
5. 依頼の種類: 依頼文を区切り（。!?;、改行と、英字に挟まれていない ,）に分け、区切りごとに表の request_kinds
   で種類を決める（正規化した写しと空白を除いた写しの両方に当てる）。種類にもあいさつなどの中立の文にも
   当たらない区切りが 1 つでもあれば、種類不明として保留。依頼文の無い契約は表の固定の種類。
   承認済み本文 ID × 種類 を表の combinations で引き、禁止は遮断、表に無い組み合わせは保留
遮断が 1 つでもあれば遮断、なければ保留が 1 つでもあれば保留、どちらも無ければ許可。

SendCandidate を受け取る check_candidate が本体。C2Gate の I/F（check(payload)）だけでは出所が
分からないので保留にする。送信手順（orchestrator.pipeline）は check_candidate を持つゲートに
候補を束ねて渡す（pipeline.bind_candidate）。

登録簿・表・方針はゲートを作ったときに読む。登録の取り消しや表の変更は、ゲートを作り直すまで効かない。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from common.provenance import RenderedMessage, attachments, mismatch
from common.schemas import SendPayload
from judge.base import C2Verdict, question_version
from judge.c2_registry import DEFAULT_REGISTRY_PATH, REGISTRATION_CHECK_VERSION, Registry, load_registry
from judge.normalize import compact_text, normalize_text
from judge.request_cues import CUES_VERSION, REQUEST_CUES, cue_hits, unexpected_letters
from judge.request_values import VALUES_VERSION, ValueRules, compile_value_rules, unapproved_values
from orchestrator.policy import DEFAULT_POLICY_DIR, Contract, Policy, load_policy

MODEL = "provenance-rules"
REVISION = "provenance-v3"
DEFAULT_TABLES_PATH = DEFAULT_POLICY_DIR / "c2_tables.json"
TABLES_FORMAT = "c2-tables@1"
# 区切り: 句点・感嘆符・疑問符・セミコロン・改行・読点と、英字に挟まれていない ,（month,category は区切らない）
_SENTENCE_END = re.compile(r"(?:[。!?;\n、]|,(?![ ]*[A-Za-z0-9_])|(?<![A-Za-z0-9_]),)+")
# 種類の無い区切りでも、助詞で終わる修飾（「category ごとに、」「amount が負の行は、」）は後ろの区切りに結ぶ。
# 動詞のて形（「読み込んで、」）で終わる区切りは結ばない（付け足しの依頼になるため）。Codex 11/3
_MODIFIER_END = re.compile(r"(ごとに|ごとの|別に|単位で|について|に対して|に関して|場合は?|なら(ば)?|では|には|からは|までは|"
                           r"ときは|際は|は|に|から|まで|より|で|を|も|と)$")
_TE_FORM = re.compile(r"(て|[んい]で)$")

__all__ = ["MODEL", "REQUEST_CUES", "DEFAULT_TABLES_PATH", "ProvenanceC2Gate", "Tables", "load_tables",
           "normalize_request", "request_cue_hits"]


def normalize_request(text: str) -> str:
    return normalize_text(text)


def request_cue_hits(text: str) -> list[str]:
    return cue_hits(text)


@dataclass(frozen=True)
class _KindRules:
    rules: tuple[tuple[str, re.Pattern[str]], ...]
    neutral: re.Pattern[str] | None


@dataclass(frozen=True)
class Tables:
    kinds: Mapping[str, str | _KindRules]  # 契約キー → 固定の種類 または 依頼文の規則
    combinations: Mapping[str, Mapping[str, tuple[str, str]]]  # ID@版 → 種類 → (allow/block, 理由)
    canonical: str  # 版のハッシュに使う正規化 JSON
    values: Mapping[str, ValueRules] = None  # type: ignore[assignment]  # 契約キー → 依頼文の自由な値の承認

    def request_kinds(self, contract: str, request: str | None) -> tuple[list[str], list[int]]:
        """(当たった種類, 種類の決まらない文の番号)。契約が表に無い・依頼文の有無が合わないときは ([], [])。"""
        k = self.kinds.get(contract)
        if k is None or isinstance(k, str) != (request is None):
            return [], []
        if isinstance(k, str):
            return [k], []
        kinds: list[str] = []
        unknown: list[int] = []
        sentences = [s.strip(" 、,") for s in _SENTENCE_END.split(normalize_request(request))]
        clauses = [x for x in sentences if x]
        pending: int | None = None  # 後ろの区切りに結ぶのを待っている修飾の区切り
        for i, s in enumerate(clauses):
            c = compact_text(s)  # 「r e q u e s t s」のように語を空白で割る言い換え用
            hit = [name for name, pat in k.rules if pat.search(s) or pat.search(c)]
            if hit:
                kinds.extend(h for h in hit if h not in kinds)
                pending = None
            elif k.neutral is not None and (k.neutral.fullmatch(s) or k.neutral.fullmatch(c)):
                continue
            elif i < len(clauses) - 1 and _MODIFIER_END.search(s) and not _TE_FORM.search(s):
                pending = i if pending is None else pending
            else:
                unknown.append(i)
        if pending is not None:
            unknown.append(pending)
        return kinds, sorted(unknown)


def _compile(where: str, pattern: Any) -> re.Pattern[str]:
    if not isinstance(pattern, str):
        raise ValueError(f"{where}: pattern must be a string")
    try:
        pat = re.compile(pattern, re.I)
    except re.error as e:
        raise ValueError(f"{where}: bad pattern: {e}") from e
    if pat.search(""):
        raise ValueError(f"{where}: pattern matches the empty text")
    return pat


def load_tables(path: Path | str = DEFAULT_TABLES_PATH) -> Tables:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("format") != TABLES_FORMAT:
        raise ValueError(f"tables format {doc.get('format') if isinstance(doc, dict) else doc!r} != {TABLES_FORMAT}")
    if not isinstance(doc.get("request_kinds"), dict) or not isinstance(doc.get("combinations"), dict):
        raise ValueError("tables need request_kinds and combinations objects")
    kinds: dict[str, str | _KindRules] = {}
    for ck, spec in doc["request_kinds"].items():
        if not isinstance(spec, dict):
            raise ValueError(f"request_kinds {ck}: must be an object")
        if set(spec) == {"fixed"} and isinstance(spec["fixed"], str) and spec["fixed"]:
            kinds[ck] = spec["fixed"]
        elif set(spec) in ({"rules"}, {"rules", "neutral"}) and isinstance(spec["rules"], dict) and spec["rules"]:
            rules = tuple((n, _compile(f"request_kinds {ck}.{n}", p)) for n, p in spec["rules"].items())
            neutral = _compile(f"request_kinds {ck}.neutral", spec["neutral"]) if "neutral" in spec else None
            kinds[ck] = _KindRules(rules, neutral)
        else:
            raise ValueError(f"request_kinds {ck}: exactly one of fixed (str) or rules (object, optional neutral)")
    combos: dict[str, dict[str, tuple[str, str]]] = {}
    for ref, row in doc["combinations"].items():
        if not isinstance(row, dict):
            raise ValueError(f"combinations {ref}: must be an object")
        combos[ref] = {}
        for kind, v in row.items():
            decision, reason = (v, "") if isinstance(v, str) else (
                (v.get("decision"), v.get("reason", "")) if isinstance(v, dict) else (None, ""))
            if decision not in ("allow", "block") or not isinstance(reason, str):
                raise ValueError(f"combinations {ref} x {kind}: decision must be allow or block, got {v!r}")
            combos[ref][kind] = (decision, reason)
    rv = doc.get("request_values", {})
    if not isinstance(rv, dict):
        raise ValueError("request_values must be an object")
    values = {ck: compile_value_rules(f"request_values {ck}", spec) for ck, spec in rv.items()}
    canonical = json.dumps({"request_kinds": doc["request_kinds"], "combinations": doc["combinations"],
                            "request_values": rv}, ensure_ascii=False, sort_keys=True)
    return Tables(kinds, combos, canonical, values)


class ProvenanceC2Gate:
    def __init__(self, policy: Policy | None = None, registry_path: Path | str = DEFAULT_REGISTRY_PATH,
                 tables_path: Path | str = DEFAULT_TABLES_PATH):
        self.policy = policy or load_policy()
        self.registry: Registry = load_registry(registry_path)
        self.tables = load_tables(tables_path)
        self._qv_spec = {
            "cues": CUES_VERSION,  # 正規化・手がかり語・文字種
            "canonical": "template parts via orchestrator.candidate._render@1",
            "sentences": _SENTENCE_END.pattern,
            "modifier": [_MODIFIER_END.pattern, _TE_FORM.pattern],
            "registration_check": REGISTRATION_CHECK_VERSION,
            "registry_sha256": self.registry.sha256,
            "tables": self.tables.canonical,
            "request_values": VALUES_VERSION,  # 自由な値の切り出し
            # 承認者も版に入れる（全文確認者が承認者と別人かをこの承認者で見るため）
            "approved_records": {k: [a.approver, list(a.request_identifiers), [t for t, _ in a.request_values]]
                                 for k, a in sorted(self.policy.approved.items())},
        }
        self.question_version = question_version("provenance-c2@3", self._qv_spec)

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

        blocks: list[str] = []
        holds: list[str] = []
        refs: list[str] = []
        body_ids: list[str] = []
        body_values: list[str] = []
        for ref, sha in attachments(rendered):
            if ref is None:
                return self._verdict("hold", ["approved part without ref"])
            if ref != cand.approved_ref or sha != cand.approved_sha256:
                return self._verdict("hold", [f"approved part {ref} differs from candidate approved_ref/sha256"])
            refs.append(ref)
            e = self.registry.find(ref, contract.key, contract.purpose, payload.destination)
            if e is None:
                holds.append(f"{ref} not registered for {contract.key}/{payload.destination}")
            elif e["sha256"] != sha:
                blocks.append(f"{ref} body sha256 differs from registry")
            elif e["check_version"] != REGISTRATION_CHECK_VERSION:
                holds.append(f"{ref} registration check outdated ({e['check_version']})")
            if e is not None:
                # 依頼文に書いてよい本文ごとの値は、登録簿に写した（全文確認を通った）物を使う。
                # 承認の記録（policies/approved）の値が後から変わっていれば、登録し直すまで保留
                rv = e["request_values"]
                a = self.policy.approved.get(ref)
                if a is not None and rv != {"identifiers": list(a.request_identifiers),
                                            "values": [t for t, _ in a.request_values]}:
                    holds.append(f"{ref} request values differ from the registry (register again)")
                if a is not None and e["reviewed_by"].strip() == a.approver.strip():
                    holds.append(f"{ref} full-text reviewer is the approver (needs another person)")
                body_ids += rv["identifiers"]
                body_values += rv["values"]

        requests = [p.text for m in rendered for p in m.parts if p.source == "free_text"]
        for text in requests:
            hits = cue_hits(text)
            if hits:
                blocks.append("request matched: " + ",".join(hits))
            odd = unexpected_letters(text)
            if odd:
                holds.append("request has letters outside the expected scripts: " + ",".join(odd[:5]))
            rules = (self.tables.values or {}).get(contract.key)
            if rules is None:
                holds.append(f"no request value rules for {contract.key}")
            else:
                bad = unapproved_values(text, rules, body_ids, body_values)
                if bad:
                    holds.append("request has values not approved in advance: " + ",".join(dict.fromkeys(bad[:8])))

        kinds, unknown = self.tables.request_kinds(contract.key, "\n".join(requests) if requests else None)
        if not kinds or unknown:
            where = f" (sentences {unknown})" if unknown else ""
            holds.append(f"request kind unknown for {contract.key}{where}")
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
    from orchestrator.candidate import _render

    stub_opts = {k: f"\x00option:{k}" for k in contract.options}
    stub_free = "\x00free_text" if contract.free_text_max is not None else None
    try:
        stub = _render(contract, SimpleNamespace(body="\x00approved", key="\x00ref"), stub_opts, stub_free)
    except Exception as e:  # noqa: BLE001  テンプレートが組めない・組み立ての変更で属性が足りない等はすべて保留
        return f"not canonical: template of {contract.key} cannot be rendered ({type(e).__name__}: {e})"
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
