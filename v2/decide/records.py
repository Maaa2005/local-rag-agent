"""確認済みの構造化記録（JSON）の読み込みと検証（F7 の「記録が規程に合っているか」の機械検査）。

記録は規程の本文（文書）だけを見て人が作る。この module はその記録を
- 構造: 知らない欄・「または」・入れ子・3 つを超える条件・知らない比べ方は拒否する
- 文書との照合（記録の doc_version が文書の SHA-256 と一致する時だけ）:
    * 引用は、示した行に一字一句そのまま在る
    * 条件の境目の語（10万円・8割・14日・翌月10日・完全な日付）が引用の中に在る
      （単位のない量の事実は unit: null と明示する。境目の数は単位なし「5以上」でも、単位付きでもよい）
    * 境目の直後の言い回し（超える・以上・未満・過ぎた・まで 等。質問側と同じ表で読む）が、条件の比べ方と一致する
      （同じ境目が引用に何度も出るなら、全部の言い回しが一致していなければならない）
  で確かめる。不一致は RecordError（黙って読み込まない）。
版が違う（SHA 不一致・文書なし）記録は構造だけ検査して stale として読み込み、engine が「版」で保留する。

この検査で分かるのは「境目の数・比べ方の向きが引用と食い違う」ことまで。
結論側（成り立つ時・成り立たない時にどちらの結論を出すか）の取り違え、対象の取り違え、手がかりの語の取り違えは検出できない。
.jsonl は開かない。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

from . import parse

MAX_CONDITIONS = 3
OPS = (">", ">=", "<", "<=")
BASES = ("explicit", "inferred_negation", "unknown")
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
    type: str  # number / date / offset_days / month_day
    value: Any = None  # Fraction（number）か date（date）
    base: str | None = None
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
    basis: str  # explicit / inferred_negation / unknown
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
        "scope_reason": False, "subject": True, "facts": True, "conditions": True, "exceptions": False,
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
    scope_reason = top.get("scope_reason")
    if not supported and not (isinstance(scope_reason, str) and scope_reason.strip()):
        raise RecordError(f"{ctx}: 対応範囲外の記録には scope_reason が要る")
    return Record(
        rule_id=rule_id,
        doc_path=_str(top["doc_path"], "doc_path"),
        doc_version=_str(top["doc_version"], "doc_version"),
        description=_str(top["description"], "description"),
        supported=supported, scope_reason=scope_reason, groups=groups_t, excludes=excludes, facts=facts,
        conditions=tuple(conds), exceptions=tuple(excs), other_requirements=tuple(others),
        all_true=_outcome(outs["all_true"], f"{ctx}.outcomes.all_true", options) if "all_true" in outs else None,
        any_false=_outcome(outs["any_false"], f"{ctx}.outcomes.any_false", options) if "any_false" in outs else None,
        options=options)


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
        # 単位のない量: 境目の数は単位なし、または単位付きで書かれていてよい。直後（単位があれば単位の後）の比べ方の語で向きを確かめる
        toks = [(end, v) for end, v in parse.quote_number_tokens(q) if v == th.value]
        _all_agree([parse.phrase_op(q, end, "num") for end, _ in toks], c.op,
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
        _all_agree([parse.phrase_op(q, t[0], "date") for t in toks], c.op, f"{ctx}: 境目 翌月{th.day}日")
    else:  # offset_days
        if "翌日から数えて" not in parse._normalize(q):
            raise RecordError(f"{ctx}: count_from=next_day なのに引用に「翌日から数えて」が無い")
        toks = [t for t in parse.day_count_tokens(q) if t[1] == th.days]
        _all_agree([parse.phrase_op(q, t[0], "date") for t in toks], c.op, f"{ctx}: 境目 {th.days}日")


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


def load_records(directory: Path | str, source: DocSource | None) -> list[Record]:
    """directory の *.json（記録）を読む。.jsonl など他の拡張子は見ない。"""
    out = []
    for p in sorted(Path(directory).glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise RecordError(f"{p.name}: JSON が読めない") from e
        try:
            out.append(build_record(data, source))
        except RecordError as e:
            raise RecordError(f"{p.name}: {e}") from e
    ids = [r.rule_id for r in out]
    if len(set(ids)) != len(ids):
        raise RecordError("rule_id が重複している")
    return out
