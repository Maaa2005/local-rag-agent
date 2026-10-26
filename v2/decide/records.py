"""確認済みの構造化記録（JSON）の読み込みと検証（F7 の「記録が規程に合っているか」の機械検査）。

記録は規程の本文（文書）だけを見て人が作る。この module はその記録を
- 構造: 知らない欄・「または」・入れ子・3 つを超える条件・知らない比べ方は拒否する
- 文書との照合（記録の doc_version が文書の SHA-256 と一致する時だけ）:
    * 引用は、示した行に一字一句そのまま在る
    * 条件の境目の語（10万円・8割・14日・翌月10日・完全な日付）が引用の中に在る
      （単位のない量の事実は unit: null と明示する。境目の数は単位なし「5以上」でも、単位付きでもよい）
    * 境目の直後の言い回し（超える・以上・未満・過ぎた・まで 等。質問側と同じ表で読む）が、条件の比べ方と一致する
      （同じ境目が引用に何度も出るなら、全部の言い回しが一致していなければならない）
    * 境目が問の中の別の日付（type: fact）なら、引用の中のその日付の名前（label）の直後の言い回し（より前・以後 等）を
      条件の比べ方と照らす。名前が引用に無ければ拒否。成り立つ側・成り立たない側の両方が書かれた引用は名前付き境目と同じ検査
  で確かめる。不一致は RecordError（黙って読み込まない）。
版が違う（SHA 不一致・文書なし）記録は構造だけ検査して stale として読み込み、engine が「版」で保留する。

この検査で分かるのは「境目の数・比べ方の向きが引用と食い違う」「relation の表が許さない側に結論がある」ことまで。
結論側の取り違え（最初から逆に書かれた記録）、対象の取り違え、手がかりの語の取り違えは、この検査では検出できない。
それを止めるのは採点（作問役）と確認済みの表（verified.json。load_records の require_verified）で、
表が保証するのは「確認済みの物から変わったら止まる」ことだけ（計画 7.2）。
.jsonl は開かない。
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

from . import parse

MAX_CONDITIONS = 3
OPS = (">", ">=", "<", "<=")
BASES = ("explicit", "unknown")  # 否定で推論した側（inferred_negation）は使わない（計画 7.1）
# relation（原文の文の型）ごとに、結論（unknown 以外）を持てる側（計画 7.1 の表）
RELATIONS: dict[str, tuple[str, ...]] = {
    "if_then": ("all_true",),
    "if_then_not": ("all_true",),
    "only_if": ("any_false",),
    "if_else": ("all_true", "any_false"),
}
VERIFIED_FILE = "verified.json"
FACT_KINDS = ("money", "date", "quantity", "ratio")
MATTERS_FOR = ("all_true", "any_false", "both")


class RecordError(ValueError):
    pass


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Evidence:
    quote: str
    line: int


@dataclass(frozen=True)
class Fact:
    role: str
    label: str
    kind: str  # money / date / quantity / ratio
    unit: str | None  # 円 / None / 年・か月・日・km … / %
    cues: tuple[str, ...]
    percent_cues: tuple[str, ...] = ()
    numerator: str | None = None
    denominator: str | None = None


@dataclass(frozen=True)
class Threshold:
    type: str  # number / date / offset_days / month_day / fact
    value: Any = None  # Fraction（number）か date（date）
    base: str | None = None  # offset_days / month_day / fact の基準になる事実の role（fact は問の中の別の日付そのもの）
    days: int | None = None
    count_from: str | None = None
    month_offset: int | None = None
    day: int | None = None


@dataclass(frozen=True)
class Condition:
    fact: str
    op: str
    threshold: Threshold
    evidence: Evidence


@dataclass(frozen=True)
class Exclusion:
    terms: tuple[str, ...]
    reason: str
    evidence: Evidence


@dataclass(frozen=True)
class ExceptionRule:
    description: str
    cues: tuple[str, ...]
    evidence: Evidence


@dataclass(frozen=True)
class OtherRequirement:
    description: str
    cues: tuple[str, ...]
    assume_met_if_unstated: bool
    matters_for: str
    evidence: Evidence


@dataclass(frozen=True)
class Outcome:
    value: str  # options のどれか、または "unknown"
    basis: str  # explicit / unknown
    evidence: Evidence | None


@dataclass(frozen=True)
class Record:
    rule_id: str
    doc_path: str
    doc_version: str
    description: str
    supported: bool
    scope_reason: str | None
    groups: tuple[tuple[str, ...], ...]
    excludes: tuple[Exclusion, ...]
    facts: tuple[Fact, ...]
    conditions: tuple[Condition, ...]
    exceptions: tuple[ExceptionRule, ...]
    other_requirements: tuple[OtherRequirement, ...]
    all_true: Outcome | None
    any_false: Outcome | None
    options: tuple[str, ...]
    stale: bool = False  # 文書の版が記録の doc_version と違う（または文書が無い）
    relation: str | None = None  # if_then / if_then_not / only_if / if_else（supported なら必須）

    def fact(self, role: str) -> Fact:
        for f in self.facts:
            if f.role == role:
                return f
        raise KeyError(role)


class DocSource(Protocol):
    def read_text(self, doc_path: str) -> str | None: ...


# ---------------------------------------------------------------- 構造の検査

def _obj(d: Any, allowed: dict[str, bool], ctx: str) -> dict[str, Any]:
    """allowed: {キー: 必須か}。知らないキー・足りない必須キーは拒否。"""
    if not isinstance(d, dict):
        raise RecordError(f"{ctx}: オブジェクトでない")
    unknown = set(d) - set(allowed)
    if unknown:
        raise RecordError(f"{ctx}: 知らない欄 {sorted(unknown)}（「または」・入れ子は扱わない）")
    missing = [k for k, req in allowed.items() if req and k not in d]
    if missing:
        raise RecordError(f"{ctx}: 欄が無い {missing}")
    return d


def _str(v: Any, ctx: str, *, empty_ok: bool = False) -> str:
    if not isinstance(v, str) or (not empty_ok and not v.strip()):
        raise RecordError(f"{ctx}: 文字列でない／空")
    return v


def _strs(v: Any, ctx: str, *, min_len: int = 1) -> tuple[str, ...]:
    if not isinstance(v, list) or len(v) < min_len or not all(isinstance(x, str) and x for x in v):
        raise RecordError(f"{ctx}: 文字列の一覧でない／空")
    return tuple(v)


def _evidence(d: Any, ctx: str) -> Evidence:
    d = _obj(d, {"quote": True, "line": True}, ctx)
    line = d["line"]
    if not isinstance(line, int) or isinstance(line, bool) or line < 1:
        raise RecordError(f"{ctx}: line は 1 以上の整数")
    return Evidence(_str(d["quote"], f"{ctx}.quote"), line)


def _threshold(d: Any, ctx: str) -> Threshold:
    if not isinstance(d, dict) or "type" not in d:
        raise RecordError(f"{ctx}: type が無い")
    t = d["type"]
    if t == "number":
        d = _obj(d, {"type": True, "value": True}, ctx)
        v = d["value"]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise RecordError(f"{ctx}: value は数")
        return Threshold("number", value=Fraction(str(v)))
    if t == "date":
        d = _obj(d, {"type": True, "value": True}, ctx)
        try:
            return Threshold("date", value=date.fromisoformat(_str(d["value"], f"{ctx}.value")))
        except ValueError as e:
            raise RecordError(f"{ctx}: 日付が読めない") from e
    if t == "offset_days":
        d = _obj(d, {"type": True, "base": True, "days": True, "count_from": True}, ctx)
        if not isinstance(d["days"], int) or isinstance(d["days"], bool) or d["days"] < 1:
            raise RecordError(f"{ctx}: days は 1 以上の整数")
        if d["count_from"] != "next_day":
            raise RecordError(f"{ctx}: count_from は next_day だけ")
        return Threshold("offset_days", base=_str(d["base"], f"{ctx}.base"), days=d["days"], count_from="next_day")
    if t == "month_day":
        d = _obj(d, {"type": True, "base": True, "month_offset": True, "day": True}, ctx)
        mo, day = d["month_offset"], d["day"]
        if not isinstance(mo, int) or isinstance(mo, bool) or not 0 <= mo <= 2:
            raise RecordError(f"{ctx}: month_offset は 0〜2")
        if not isinstance(day, int) or isinstance(day, bool) or not 1 <= day <= 31:
            raise RecordError(f"{ctx}: day は 1〜31")
        return Threshold("month_day", base=_str(d["base"], f"{ctx}.base"), month_offset=mo, day=day)
    if t == "fact":
        d = _obj(d, {"type": True, "base": True}, ctx)
        return Threshold("fact", base=_str(d["base"], f"{ctx}.base"))
    raise RecordError(f"{ctx}: 知らない境目の型 {t!r}")


def _fact(d: Any, ctx: str) -> Fact:
    d = _obj(d, {"role": True, "label": True, "kind": True, "unit": False, "cues": False,
                 "percent_cues": False, "numerator": False, "denominator": False}, ctx)
    kind = d["kind"]
    if kind not in FACT_KINDS:
        raise RecordError(f"{ctx}: 知らない kind {kind!r}")
    unit = d.get("unit")
    if kind == "money":
        unit = "円"
    elif kind == "ratio":
        unit = "%"
    elif kind == "date":
        unit = None
    elif "unit" in d and (unit is None or unit == ""):
        unit = ""  # 単位のない量（「評価が5以上」の 5 など）。書き忘れと区別するため unit: null を明示した時だけ
    elif not isinstance(unit, str) or not unit:
        raise RecordError(f"{ctx}: quantity には unit が要る（単位のない量は \"unit\": null と明示する）")
    if kind == "ratio":
        pc = _strs(d.get("percent_cues", []), f"{ctx}.percent_cues", min_len=0)
        num, den = d.get("numerator"), d.get("denominator")
        if (num is None) != (den is None):
            raise RecordError(f"{ctx}: numerator と denominator は両方か、どちらも無しか")
        if not pc and num is None:
            raise RecordError(f"{ctx}: ratio には percent_cues か numerator/denominator が要る")
        return Fact(_str(d["role"], f"{ctx}.role"), _str(d["label"], f"{ctx}.label"), kind, unit,
                    _strs(d.get("cues", []), f"{ctx}.cues", min_len=0), pc, num, den)
    if "percent_cues" in d or "numerator" in d or "denominator" in d:
        raise RecordError(f"{ctx}: percent_cues / numerator / denominator は ratio だけ")
    return Fact(_str(d["role"], f"{ctx}.role"), _str(d["label"], f"{ctx}.label"), kind, unit,
                _strs(d.get("cues"), f"{ctx}.cues"))


def _outcome(d: Any, ctx: str, options: tuple[str, ...]) -> Outcome:
    d = _obj(d, {"value": True, "basis": True, "evidence": False}, ctx)
    basis = d["basis"]
    if basis not in BASES:
        raise RecordError(f"{ctx}: 知らない basis {basis!r}")
    value = _str(d["value"], f"{ctx}.value")
    if basis == "unknown":
        if value != "unknown":
            raise RecordError(f"{ctx}: basis が unknown なら value も unknown")
        return Outcome("unknown", "unknown", _evidence(d["evidence"], f"{ctx}.evidence") if "evidence" in d else None)
    if value not in options:
        raise RecordError(f"{ctx}: value {value!r} が options に無い")
    if "evidence" not in d:
        raise RecordError(f"{ctx}: evidence が要る")
    return Outcome(value, basis, _evidence(d["evidence"], f"{ctx}.evidence"))


def parse_record(data: Any) -> Record:
    """構造だけを検査して Record にする（文書との照合は validate_against_document）。"""
    top = _obj(data, {
        "rule_id": True, "doc_path": True, "doc_version": True, "description": True, "supported": True,
        "scope_reason": False, "relation": False, "subject": True, "facts": True, "conditions": True, "exceptions": False,
        "other_requirements": False, "outcomes": True, "options": True}, "record")
    rule_id = _str(top["rule_id"], "rule_id")
    ctx = rule_id
    supported = top["supported"]
    if not isinstance(supported, bool):
        raise RecordError(f"{ctx}: supported は true/false")
    subject = _obj(top["subject"], {"groups": True, "exclude": False}, f"{ctx}.subject")
    groups = subject["groups"]
    if not isinstance(groups, list) or not 1 <= len(groups) <= 4:
        raise RecordError(f"{ctx}.subject.groups: 1〜4 組")
    groups_t = tuple(_strs(g, f"{ctx}.subject.groups[{i}]") for i, g in enumerate(groups))
    excludes = tuple(
        Exclusion(*(lambda e: (_strs(e["terms"], f"{ctx}.exclude[{i}].terms"), _str(e["reason"], f"{ctx}.exclude[{i}].reason"),
                               _evidence(e["evidence"], f"{ctx}.exclude[{i}].evidence")))(
            _obj(e, {"terms": True, "reason": True, "evidence": True}, f"{ctx}.exclude[{i}]")))
        for i, e in enumerate(subject.get("exclude", [])))
    options = _strs(top["options"], f"{ctx}.options")
    if len(options) != 2 or options[0] == options[1]:
        raise RecordError(f"{ctx}.options: 異なる 2 つ")
    facts_raw = top["facts"]
    if not isinstance(facts_raw, list):
        raise RecordError(f"{ctx}.facts: 一覧でない")
    facts = tuple(_fact(f, f"{ctx}.facts[{i}]") for i, f in enumerate(facts_raw))
    roles = [f.role for f in facts]
    if len(set(roles)) != len(roles):
        raise RecordError(f"{ctx}.facts: role が重複")
    by_role = {f.role: f for f in facts}
    for f in facts:
        if f.kind == "ratio" and f.numerator is not None:
            for r in (f.numerator, f.denominator):
                if r not in by_role or by_role[r].kind != "quantity":
                    raise RecordError(f"{ctx}.facts[{f.role}]: numerator/denominator は quantity の role")
            if by_role[f.numerator].unit != by_role[f.denominator].unit:  # type: ignore[index]
                raise RecordError(f"{ctx}.facts[{f.role}]: 分子と分母の単位が違う")
    conds_raw = top["conditions"]
    if not isinstance(conds_raw, list):
        raise RecordError(f"{ctx}.conditions: 一覧でない")
    if len(conds_raw) > MAX_CONDITIONS:
        raise RecordError(f"{ctx}.conditions: {MAX_CONDITIONS} 件まで（「かつ」だけ。「または」・入れ子は扱わない）")
    if supported and not conds_raw:
        raise RecordError(f"{ctx}.conditions: 対応する記録には条件が要る")
    conds = []
    for i, c in enumerate(conds_raw):
        c = _obj(c, {"fact": True, "op": True, "threshold": True, "evidence": True}, f"{ctx}.conditions[{i}]")
        if c["op"] not in OPS:
            raise RecordError(f"{ctx}.conditions[{i}]: 知らない比べ方 {c['op']!r}")
        if c["fact"] not in by_role:
            raise RecordError(f"{ctx}.conditions[{i}]: 知らない事実 {c['fact']!r}")
        th = _threshold(c["threshold"], f"{ctx}.conditions[{i}].threshold")
        f = by_role[c["fact"]]
        if th.type == "fact":
            cc = f"{ctx}.conditions[{i}]"
            b = by_role.get(th.base)  # type: ignore[arg-type]
            if b is None:
                raise RecordError(f"{cc}: fact の境目の base {th.base!r} が facts に無い")
            if th.base == c["fact"]:
                raise RecordError(f"{cc}: fact の境目の base は比べる事実と別の role にする（自分自身とは比べられない）")
            if f.kind != "date" or b.kind != "date":
                raise RecordError(f"{cc}: fact の境目は日付（date）の事実どうしだけ。日付以外は「前・後・以前・以後」の"
                                  "比べ方が決まらず、単位や換算も合わないので使えない")
            if f.label == b.label:
                raise RecordError(f"{cc}: 比べる事実と base の label が同じで、引用の中で見分けられない")
        if th.type == "number" and f.kind == "date":
            raise RecordError(f"{ctx}.conditions[{i}]: 日付の事実に数の境目は使えない")
        if th.type != "number" and f.kind != "date":
            raise RecordError(f"{ctx}.conditions[{i}]: 日付でない事実に日付の境目は使えない")
        if th.base is not None and (th.base not in by_role or by_role[th.base].kind != "date"):
            raise RecordError(f"{ctx}.conditions[{i}]: base は date の事実")
        conds.append(Condition(c["fact"], c["op"], th, _evidence(c["evidence"], f"{ctx}.conditions[{i}].evidence")))
    excs = []
    for i, e in enumerate(top.get("exceptions", [])):
        e = _obj(e, {"description": True, "cues": True, "evidence": True}, f"{ctx}.exceptions[{i}]")
        excs.append(ExceptionRule(_str(e["description"], "description"), _strs(e["cues"], "cues"),
                                  _evidence(e["evidence"], f"{ctx}.exceptions[{i}].evidence")))
    others = []
    for i, o in enumerate(top.get("other_requirements", [])):
        o = _obj(o, {"description": True, "cues": True, "assume_met_if_unstated": True, "matters_for": True,
                     "evidence": True}, f"{ctx}.other_requirements[{i}]")
        if not isinstance(o["assume_met_if_unstated"], bool) or o["matters_for"] not in MATTERS_FOR:
            raise RecordError(f"{ctx}.other_requirements[{i}]: assume_met_if_unstated / matters_for が不正")
        others.append(OtherRequirement(_str(o["description"], "description"), _strs(o["cues"], "cues"),
                                       o["assume_met_if_unstated"], o["matters_for"],
                                       _evidence(o["evidence"], f"{ctx}.other_requirements[{i}].evidence")))
    outs = _obj(top["outcomes"], {"all_true": False, "any_false": False}, f"{ctx}.outcomes")
    if supported and set(outs) != {"all_true", "any_false"}:
        raise RecordError(f"{ctx}.outcomes: all_true と any_false の両方が要る（導けない側は basis unknown）")
    relation = top.get("relation")
    if relation is None:
        if supported:
            raise RecordError(f"{ctx}: relation が要る（{' / '.join(RELATIONS)}）")
    elif relation not in RELATIONS:
        raise RecordError(f"{ctx}: 知らない relation {relation!r}")
    scope_reason = top.get("scope_reason")
    if not supported and not (isinstance(scope_reason, str) and scope_reason.strip()):
        raise RecordError(f"{ctx}: 対応範囲外の記録には scope_reason が要る")
    all_true = _outcome(outs["all_true"], f"{ctx}.outcomes.all_true", options) if "all_true" in outs else None
    any_false = _outcome(outs["any_false"], f"{ctx}.outcomes.any_false", options) if "any_false" in outs else None
    for name, out in (("all_true", all_true), ("any_false", any_false)):
        if out is None or out.value == "unknown":
            continue
        if relation is None:
            raise RecordError(f"{ctx}.outcomes.{name}: 結論を持つには relation が要る")
        if name not in RELATIONS[relation]:
            raise RecordError(f"{ctx}.outcomes.{name}: relation {relation} はこの側に結論を持てない（unknown にする）")
    return Record(
        rule_id=rule_id,
        doc_path=_str(top["doc_path"], "doc_path"),
        doc_version=_str(top["doc_version"], "doc_version"),
        description=_str(top["description"], "description"),
        supported=supported, scope_reason=scope_reason, groups=groups_t, excludes=excludes, facts=facts,
        conditions=tuple(conds), exceptions=tuple(excs), other_requirements=tuple(others),
        all_true=all_true, any_false=any_false, options=options, relation=relation)


# ---------------------------------------------------------------- 文書との照合（F7）

def _check_quote(ev: Evidence, lines: list[str], ctx: str) -> None:
    if ev.line > len(lines):
        raise RecordError(f"{ctx}: {ev.line} 行目が文書に無い")
    if ev.quote not in lines[ev.line - 1]:
        raise RecordError(f"{ctx}: 引用が文書の {ev.line} 行目にそのまま無い")


def _all_agree(ops: list[str | None], want: str, ctx: str) -> None:
    if not ops:
        raise RecordError(f"{ctx}: 引用に境目の語が無い")
    bad = [o for o in ops if o != want]
    if bad:
        raise RecordError(f"{ctx}: 境目の直後の言い回しが比べ方 {want!r} と一致しない（引用の読み: {ops}）")


def _check_condition(rec: Record, c: Condition, ctx: str) -> None:
    f = rec.fact(c.fact)
    th, q = c.threshold, c.evidence.quote
    if th.type == "number" and f.kind == "quantity" and not f.unit:
        # 単位のない量: 境目の数に単位が付いていれば（「5点以上」）拒否。単位なしなら直後の比べ方の語で向きを確かめる（計画 7.3）
        toks = [(end, v, u) for end, v, u in parse.quote_number_tokens(q) if v == th.value]
        united = [u for _, _, u in toks if u is not None]
        if united:
            raise RecordError(f"{ctx}: 単位のない量の境目 {parse.fmt_number(th.value)} に単位が付いている"
                              f"（引用の「{parse.fmt_number(th.value)}{'' if united[0] == parse.UNKNOWN_UNIT else united[0]}…」）。"
                              "単位の付いた量として記録する")
        _all_agree([parse.phrase_op(q, end, "num") for end, _, _ in toks], c.op,
                   f"{ctx}: 境目 {parse.fmt_number(th.value)}（単位なし）")
    elif th.type == "number":
        kind = f.unit or ""
        toks = [(end, v) for end, v in parse.quote_tokens(q, kind) if v == th.value]
        _all_agree([parse.phrase_op(q, end, "num") for end, _ in toks], c.op, f"{ctx}: 境目 {parse.fmt_number(th.value)}{kind}")
    elif th.type == "date":
        toks = [(end, v) for end, v in parse.quote_tokens(q, "date") if v == th.value]
        _all_agree([parse.phrase_op(q, end, "date") for end, _ in toks], c.op, f"{ctx}: 境目 {th.value.isoformat()}")
    elif th.type == "month_day":
        toks = [t for t in parse.month_day_tokens(q) if t[1] == th.month_offset and t[3] == "day" and t[2] == th.day]
        _check_boundary(rec, c, q, [(t[0], False) for t in toks], f"{ctx}: 境目 翌月{th.day}日")
    elif th.type == "fact":
        _check_fact_boundary(rec, c, q, f"{ctx}: 境目 {rec.fact(th.base).label}")  # type: ignore[arg-type]
    else:  # offset_days: 「翌日から数えてN日」か「翌日を数え始めとしてN日目」（後者は名前を付けて比べる形だけ）
        nq = parse._normalize(q)
        starts = parse.start_count_tokens(q)
        if "翌日から数えて" not in nq and not starts:
            raise RecordError(f"{ctx}: count_from=next_day なのに引用に「翌日から数えて」（または「翌日を数え始めとしてN日目」）が無い")
        ends = [(t[0], False) for t in parse.day_count_tokens(q) if t[1] == th.days] if "翌日から数えて" in nq else []
        ends += [(e, True) for e, n in starts if n == th.days]
        _check_boundary(rec, c, q, ends, f"{ctx}: 境目 {th.days}日")


_COMPLEMENT = {">": "<=", "<=": ">", ">=": "<", "<": ">="}


def _check_boundary(rec: Record, c: Condition, q: str, ends: list[tuple[int, bool]], label: str) -> None:
    """日数・翌月N日の境目（ends: [(式の終わり位置, 名前付きだけを受けるか)]）の直後の読みが、条件の比べ方と一致するか。

    直後が比べ方の語（「を過ぎた」）なら今まで通り。直後が名付け（「を期限日とし」）なら、後に出る名前の参照で比べ方を取る。
    """
    if not ends:
        raise RecordError(f"{label}: 引用に境目の語が無い")
    direct: list[str | None] = []
    named: list[list[tuple[str | None, str]]] = []
    for end, named_only in ends:
        refs = parse.named_boundary_refs(q, end)
        if refs is not None:
            named.append(refs)
        elif named_only:
            raise RecordError(f"{label}: 「翌日を数え始めとして」は境目に名前を付けて（「…日目を期限日とし」）比べる形だけ受ける")
        else:
            direct.append(parse.phrase_op(q, end, "date"))
    if direct:
        _all_agree(direct, c.op, label)
    for refs in named:
        _check_named(rec, c, refs, label)


def _check_fact_boundary(rec: Record, c: Condition, q: str, label: str) -> None:
    """問の中の別の日付（base の事実）と比べる境目。引用の中の base の名前（label）の直後の言い回しを日付の表で読み、
    条件の比べ方と照らす。名前が出る所が複数なら、成り立つ側と成り立たない側の両方が書かれた引用として _check_named と同じ検査をする。
    """
    text = parse._normalize(q)
    name = parse._normalize(rec.fact(c.fact).label)
    base = parse._normalize(rec.fact(c.threshold.base).label)  # type: ignore[arg-type]
    # 比べる事実の名前が base の名前を含む（「登録申請日」と「申請日」）時は、その中の出現を base の出現と数えない
    masked = [m.span() for m in re.finditer(re.escape(name), text)] if base != name and base in name else []
    hits = [m.span() for m in re.finditer(re.escape(base), text)
            if not any(a <= m.start() < b for a, b in masked)]
    if not hits:
        raise RecordError(f"{label}: 引用に base の名前「{rec.fact(c.threshold.base).label}」が無い"  # type: ignore[arg-type]
                          "（比べ方の言い回しは名前の直後で読む）")
    refs: list[tuple[str | None, str]] = []
    for i, (_, end) in enumerate(hits):
        stop = [x for x in (text.find(ch, end) for ch in "。\n") if x != -1]
        limit = min(stop) if stop else len(text)
        if i + 1 < len(hits):
            limit = min(limit, hits[i + 1][0])
        refs.append((parse.phrase_op(q, end, "date"), text[end:limit]))
    _check_named(rec, c, refs, label)


def _check_named(rec: Record, c: Condition, refs: list[tuple[str | None, str]], label: str) -> None:
    if not refs:
        raise RecordError(f"{label}: 境目に名前を付けているが、名前で比べる言い回しが引用に無い（名前が定義されていない／参照されていない）")
    ops = [op for op, _ in refs]
    if any(op is None for op in ops):
        raise RecordError(f"{label}: 名前の直後に比べ方の語が無い（または名前が定義より前に出ている）")
    want, comp = c.op, _COMPLEMENT[c.op]
    if len(ops) == 1:
        if ops[0] != want:
            raise RecordError(f"{label}: 名前の参照の比べ方が {want!r} と一致しない（引用の読み: {ops}）")
        return
    if any(op not in (want, comp) for op in ops):
        raise RecordError(f"{label}: 名前の参照の比べ方が {want!r} とその反対の側のどちらでもない（引用の読み: {ops}）")
    if set(ops) != {want, comp}:
        raise RecordError(f"{label}: 同じ名前の参照が全部同じ比べ方で、成り立つ側と成り立たない側の言い方が矛盾している（引用の読み: {ops}）")
    # 両側が書かれた引用: 参照の後ろの文に結論の言い方（options）が在れば、記録の結論の側と合っていなければならない。
    # （在る事だけを確かめる。options の語が無い文は判断しない）
    for op, seg in refs:
        hit = [o for o in rec.options if o in seg]
        hit = [o for o in hit if not any(o != o2 and o in o2 for o2 in hit)]
        if len(hit) != 1:
            continue
        if op == comp:
            out = rec.any_false
        elif len(rec.conditions) == 1:
            out = rec.all_true  # 条件が 1 つの時だけ「この条件が成り立つ = all_true」
        else:
            continue
        if out is not None and out.value != "unknown" and out.value != hit[0]:
            raise RecordError(f"{label}: 引用で比べ方 {op!r} の側は「{hit[0]}」なのに、記録はその側の結論を「{out.value}」にしている")


def validate_against_document(rec: Record, text: str) -> None:
    """記録が文書（rec.doc_version と同じ版）に合っているか。不一致は RecordError。"""
    lines = text.split("\n")
    ctx = rec.rule_id
    for i, e in enumerate(rec.excludes):
        _check_quote(e.evidence, lines, f"{ctx}.exclude[{i}]")
    for i, c in enumerate(rec.conditions):
        _check_quote(c.evidence, lines, f"{ctx}.conditions[{i}]")
        _check_condition(rec, c, f"{ctx}.conditions[{i}]")
    for i, e in enumerate(rec.exceptions):
        _check_quote(e.evidence, lines, f"{ctx}.exceptions[{i}]")
    for i, o in enumerate(rec.other_requirements):
        _check_quote(o.evidence, lines, f"{ctx}.other_requirements[{i}]")
    for name, out in (("all_true", rec.all_true), ("any_false", rec.any_false)):
        if out is not None and out.evidence is not None:
            _check_quote(out.evidence, lines, f"{ctx}.outcomes.{name}")


def build_record(data: Any, source: DocSource | None) -> Record:
    """JSON 相当の dict から Record を作る。文書が在り版が一致すれば文書と照合し、そうでなければ stale。"""
    rec = parse_record(data)
    text = source.read_text(rec.doc_path) if source is not None else None
    if text is not None and sha256_text(text) == rec.doc_version:
        validate_against_document(rec, text)
        return rec
    return Record(**{**rec.__dict__, "stale": True})


def record_sha256(data: dict) -> str:
    """記録（JSON を読んだ dict）の中身の SHA-256。キーの順・空白に依らない。"""
    blob = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


_HEX64 = re.compile(r"[0-9a-f]{64}")


def _is_sha(v: Any) -> bool:
    return isinstance(v, str) and _HEX64.fullmatch(v) is not None


def load_verified_table(path: Path | str) -> dict[str, str]:
    """確認済みの表（verified.json）を読む。形が違えば RecordError。戻り値は {rule_id: 記録の SHA-256}。"""
    p = Path(path)
    if not p.is_file():
        raise RecordError(f"確認済みの表 {p.name} が無い（評価の設定では必須）")
    try:
        table = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise RecordError(f"{p.name}: JSON が読めない") from e
    if not isinstance(table, dict):
        raise RecordError(f"{p.name}: オブジェクトでない")
    if not _is_sha(table.get("grade_report_sha256")):
        raise RecordError(f"{p.name}: grade_report_sha256 が SHA-256（64 桁の 16 進）でない")
    if not isinstance(table.get("graded_by"), str) or not table["graded_by"].strip():
        raise RecordError(f"{p.name}: graded_by が無い")
    recs = table.get("records")
    if not isinstance(recs, dict) or not all(isinstance(k, str) and _is_sha(v) for k, v in recs.items()):
        raise RecordError(f"{p.name}: records が {{rule_id: SHA-256}} の形でない")
    return dict(recs)


def load_records(directory: Path | str, source: DocSource | None, *, verified: Path | str | None = None,
                 require_verified: bool = False) -> list[Record]:
    """directory の *.json（記録）を読む。.jsonl など他の拡張子は見ない。verified.json は記録ではない。

    require_verified=True（評価の設定）: 確認済みの表（verified。省略時は directory/verified.json）が無い、
    表に無い記録、中身の SHA-256 が表と違う記録、記録の無い表の項目があれば、記録を作る前に RecordError。
    """
    d = Path(directory)
    table_path = Path(verified) if verified is not None else d / VERIFIED_FILE
    skip = {VERIFIED_FILE}
    if require_verified:
        if table_path.parent.resolve() == d.resolve():
            skip.add(table_path.name)
    raw: list[tuple[Path, Any]] = []
    for p in sorted(d.glob("*.json")):
        if p.name in skip:
            continue
        try:
            raw.append((p, json.loads(p.read_text(encoding="utf-8"))))
        except json.JSONDecodeError as e:
            raise RecordError(f"{p.name}: JSON が読めない") from e
    if require_verified:
        table = load_verified_table(table_path)
        seen: set[str] = set()
        for p, data in raw:
            rid = data.get("rule_id") if isinstance(data, dict) else None
            if not isinstance(rid, str) or not rid:
                raise RecordError(f"{p.name}: rule_id が読めない（確認済みの表と照合できない）")
            if rid not in table:
                raise RecordError(f"{p.name}: 記録 {rid} は確認済みの表に無い")
            if record_sha256(data) != table[rid]:
                raise RecordError(f"{p.name}: 記録 {rid} の中身が確認済みの表と違う（確認後に変更された）")
            seen.add(rid)
        gone = sorted(set(table) - seen)
        if gone:
            raise RecordError(f"確認済みの表にあるが記録が無い: {', '.join(gone)}")
    out = []
    for p, data in raw:
        try:
            out.append(build_record(data, source))
        except RecordError as e:
            raise RecordError(f"{p.name}: {e}") from e
    ids = [r.rule_id for r in out]
    if len(set(ids)) != len(ids):
        raise RecordError("rule_id が重複している")
    return out
