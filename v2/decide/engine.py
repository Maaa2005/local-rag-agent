"""決めるのはコード（F7・設計 B）: 記録の選択 → 事実の取り出し → 計算 → 結論の選択。

モデルは使わない。結果は次の 3 種類のどれか（kind で区別する）:
- Decided:  答えの文（render が組み立てた型）と判断の記録、使った根拠
- Held:     保留。段（version / subject / facts / exception / scope / outcome / other_requirements）と理由
- NoRecord: 対象の記録が 0 件。「数の規程ではない」としてモデルの通常の回答へ回す（判断の記録に「モデルだけの回答」と残す）

この module は .jsonl を開かない。文書は DocLookup（document_id → (doc_path, sha256)）経由で、
規程の本文の読み取りは FileDocLookup（evaluation/corpus・corpus_nx_dev のように「渡した dir だけ」）が行う。
検索結果（chunk）は content / document_id / access_level / source を持つ物として扱う（orchestrator は import しない）。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from . import parse, render
from .records import Condition, Fact, Record


class DocLookup(Protocol):
    def lookup(self, document_id: str | None) -> tuple[str, str] | None:
        """document_id -> (doc_path, 本文の SHA-256)。知らない文書は None。"""
        ...


@dataclass(frozen=True)
class Decided:
    answer_text: str
    decision_record: dict[str, Any]
    used_chunks: tuple[Any, ...]
    kind: str = field(default="decided", init=False)


@dataclass(frozen=True)
class Held:
    stage: str
    reason: str
    decision_record: dict[str, Any]
    kind: str = field(default="held", init=False)


@dataclass(frozen=True)
class NoRecord:
    decision_record: dict[str, Any]
    kind: str = field(default="no_record", init=False)


Result = Decided | Held | NoRecord


class _Hold(Exception):
    def __init__(self, stage: str, code: str, reason: str):
        super().__init__(reason)
        self.stage, self.code, self.reason = stage, code, reason


# ---------------------------------------------------------------- 文書の置き場（ファイル）

class FileDocLookup:
    """渡した dir（evaluation/corpus など）にある .md だけを引く。/watched/<sub>/<f>.md か <sub>/<f>.md を受ける。"""

    def __init__(self, dirs: Sequence[Path | str]):
        self.dirs = [Path(d).resolve() for d in dirs]

    def _resolve(self, ref: str | None) -> tuple[str, Path] | None:
        if not isinstance(ref, str):
            return None
        s = ref.replace("\\", "/")
        if s.startswith("/watched/"):
            s = s[len("/watched/"):]
        if s.startswith("/") or not s.endswith(".md"):
            return None
        parts = s.split("/")
        if len(parts) != 2 or any(p in ("", ".", "..") for p in parts):
            return None
        for d in self.dirs:
            p = (d / parts[0] / parts[1]).resolve()
            if d in p.parents and p.is_file():
                return "/watched/" + s, p
        return None

    def lookup(self, document_id: str | None) -> tuple[str, str] | None:
        r = self._resolve(document_id)
        if r is None:
            return None
        return r[0], hashlib.sha256(r[1].read_bytes()).hexdigest()

    def read_text(self, doc_path: str) -> str | None:
        r = self._resolve(doc_path)
        return None if r is None else r[1].read_text(encoding="utf-8")


# ---------------------------------------------------------------- 質問の読み取り

_OPTION_RE = re.compile(r"結論\s*[:：]\s*([^」\n]+?)\s*」")
_NEGATIVE_MARKS = ("ない", "なし", "無い", "無し", "紛失", "忘れ", "未", "分から", "わから", "不明", "まだ")


def _n(s: str) -> str:
    return parse._normalize(s)


_NUM_COMMA = re.compile(r"(?<=\d)[,，](?=\d{3})")


def _clause_bounds(text: str, pos: int) -> tuple[int, int]:
    text = _NUM_COMMA.sub("0", text)  # 98,000 の桁区切りは節の切れ目ではない（長さは変えない）
    a = max((text.rfind(c, 0, pos) for c in parse.CLAUSE_BREAKS), default=-1) + 1
    ends = [i for i in (text.find(c, pos) for c in parse.CLAUSE_BREAKS) if i != -1]
    return a, (min(ends) if ends else len(text))


def _term_hit(text: str, terms: Sequence[str]) -> str | None:
    for t in terms:
        if _n(t) in text:
            return t
    return None


# ---------------------------------------------------------------- 事実の取り出し

@dataclass
class _Val:
    role: str
    value: Any
    kind: str  # 'date' か単位
    sources: list[dict[str, Any]]


def _fact_unit_kind(f: Fact) -> str:
    return "date" if f.kind == "date" else (f.unit or "")


def _eligible(f: Fact, c: parse.Cand) -> str | None:
    """候補 c が事実 f の値になりうるか。'ok' / 'mismatch'（同じ時間の家族だが換算できない）/ None（別物）。"""
    if f.kind == "date":
        return "ok" if c.kind == "date" else None
    if f.kind == "money":
        return "ok" if c.kind == "円" else None
    if f.kind == "ratio":
        return "ok" if c.kind == "%" else None
    if c.kind == "date" or c.kind == "円":
        return None
    if f.unit == c.kind or parse.convert(Fraction(1), c.kind, f.unit or "") is not None:
        return "ok"
    if parse.family(c.kind) == "time" and parse.family(f.unit or "") == "time":
        return "mismatch"
    return None


def _cues_of(f: Fact) -> tuple[str, ...]:
    return f.percent_cues if f.kind == "ratio" else f.cues


def _assign(ex: parse.Extracted, active: list[Fact]) -> dict[int, Fact]:
    """各候補（index）の役割を決める。決められない物は結果に入れない。矛盾は _Hold。"""
    text = ex.text
    cands = ex.cands
    assigned: dict[int, Fact] = {}
    for i, c in enumerate(cands):
        opts = [(f, _eligible(f, c)) for f in active]
        opts = [(f, e) for f, e in opts if e is not None]
        if not opts:
            continue
        cs, ce = _clause_bounds(text, c.start)
        ce = max(ce, c.end)
        nxt = cands[i + 1].start if i + 1 < len(cands) else len(text)
        wend = min(ce, nxt)

        def hits(lo: int, hi: int, after: bool) -> list[tuple[int, Fact]]:
            seg, out = text[lo:hi], []
            for f, e in opts:
                for cue in _cues_of(f):
                    k = _n(cue)
                    j = seg.find(k)
                    while j != -1:
                        if e != "mismatch":  # 換算できない時間の単位には手がかりを付けない
                            out.append((j if after else j + len(k), f))
                        j = seg.find(k, j + 1)
            return out

        h = hits(c.end, wend, True)
        pick: set[str] | None = None
        by: dict[str, Fact] = {}
        if h:  # 直後の語を先に見る（文の中の最初の手がかり）
            first = min(p for p, _ in h)
            pick = {f.role for p, f in h if p == first}
            by = {f.role: f for _, f in h}
        else:  # 無ければ直前の最も近い手がかり
            h = hits(cs, c.start, False)
            if h:
                last = max(p for p, _ in h)
                pick = {f.role for p, f in h if p == last}
                by = {f.role: f for _, f in h}
        if pick is None:
            continue  # 手がかりが付かない候補。単位のずれは下の「割り当てのない候補」の検査で扱う
        if len(pick) > 1:
            raise _Hold("facts", "role_ambiguous", f"値の役割が 2 つに読める（{c.text}）")
        assigned[i] = by[next(iter(pick))]
    return assigned


def _to_unit(c: parse.Cand, f: Fact) -> Any:
    if f.kind in ("date", "money", "ratio"):
        return c.value
    v = parse.convert(c.value, c.kind, f.unit or "")
    return v


def _extract_facts(rec: Record, ex: parse.Extracted, needed: set[str]) -> dict[str, _Val]:
    by = {f.role: f for f in rec.facts}
    # 比率の分子・分母も、比率が要るなら探す
    want = set(needed)
    for r in list(want):
        f = by[r]
        if f.kind == "ratio" and f.numerator:
            want |= {f.numerator, f.denominator}  # type: ignore[arg-type]
    active = [f for f in rec.facts if f.role in want]
    assigned = _assign(ex, active)
    vals: dict[str, _Val] = {}
    for i, f in assigned.items():
        c = ex.cands[i]
        v = _to_unit(c, f)
        src = {"text": c.text, "start": c.start, "end": c.end}
        cur = vals.get(f.role)
        if cur is None:
            vals[f.role] = _Val(f.role, v, _fact_unit_kind(f), [src])
        elif cur.value == v:
            cur.sources.append(src)
        else:
            raise _Hold("facts", "conflict",
                        f"同じ役割「{f.label}」に違う値が 2 つ以上ある（{parse.fmt_value(cur.kind, cur.value)} と {c.text}）")
    # 割り当てのない候補: 適格な事実が同じ値を持っている時だけ許す
    for i, c in enumerate(ex.cands):
        if i in assigned:
            continue
        opts = [f for f in active if _eligible(f, c) == "ok"]
        if not opts:
            if any(_eligible(f, c) == "mismatch" for f in active):
                raise _Hold("facts", "unit_mismatch", f"単位が規程と換算できない（{c.text}）")
            continue
        ok = False
        for f in opts:
            cur = vals.get(f.role)
            if cur is not None and _to_unit(c, f) == cur.value:
                ok = True
        if not ok:
            raise _Hold("facts", "role_undecided", f"値の役割が決まらない（{c.text}）")
    # 比率
    for r in needed:
        f = by[r]
        if f.kind != "ratio":
            continue
        pct = vals.get(r)
        comp = None
        if f.numerator:
            n, d = vals.get(f.numerator), vals.get(f.denominator)  # type: ignore[arg-type]
            if n is not None and d is not None:
                if d.value == 0:
                    raise _Hold("facts", "denominator_zero", "分母が 0")
                if n.value > d.value:
                    raise _Hold("facts", "numerator_gt_denominator", "分子が分母より大きい")
                comp = n.value / d.value * 100
            elif (n is None) != (d is None) and pct is None:
                raise _Hold("facts", "missing", f"「{f.label}」の分子と分母の片方しか分からない")
        if pct is not None and comp is not None and abs(pct.value - comp) > Fraction(1, 2):
            raise _Hold("facts", "conflict", f"百分率（{parse.fmt_number(pct.value)}%）と分子・分母から出した割合が合わない")
        if comp is not None:
            srcs = vals[f.numerator].sources + vals[f.denominator].sources  # type: ignore[index,arg-type]
            vals[r] = _Val(r, comp, "%", srcs + (pct.sources if pct else []))
        elif pct is None:
            raise _Hold("facts", "missing", f"「{f.label}」が質問から分からない")
    for r in needed:
        if r not in vals:
            raise _Hold("facts", "missing", f"「{by[r].label}」が質問から分からない")
    return vals


# ---------------------------------------------------------------- 計算

_MONTH_WORD = {0: "当月", 1: "翌月", 2: "翌々月"}


def _threshold_value(c: Condition, f: Fact, vals: dict[str, _Val]) -> tuple[Any, str]:
    """境目の値と、答えの文に書く説明。"""
    th = c.threshold
    if th.type == "number":
        return th.value, _fmt_fact_value(f, th.value)
    if th.type == "date":
        return th.value, parse.fmt_date(th.value)
    base = vals[th.base].value  # type: ignore[index]
    if th.type == "offset_days":
        d = base + timedelta(days=th.days)  # type: ignore[arg-type]
        return d, f"{parse.fmt_date(d)}（{parse.fmt_date(base)}の翌日から数えて{th.days}日）"
    y, m = parse.add_months(base, th.month_offset)  # type: ignore[arg-type]
    d = parse._to_date(y, m, th.day)  # type: ignore[arg-type]
    if d is None:
        raise _Hold("facts", "invalid_date", f"{y}年{m}月に{th.day}日が無い")
    return d, f"{parse.fmt_date(d)}（{parse.fmt_date(base)}の{_MONTH_WORD[th.month_offset]}{th.day}日）"  # type: ignore[index]


def _fmt_fact_value(f: Fact, v: Any) -> str:
    if f.kind == "date":
        return parse.fmt_date(v)
    return parse.fmt_value(f.unit or "", v)


# ---------------------------------------------------------------- 本体

class DecisionEngine:
    def __init__(self, records: Sequence[Record], lookup: DocLookup,
                 on_record: Callable[[dict[str, Any]], None] | None = None):
        self.records = list(records)
        self.lookup = lookup
        self.on_record = on_record

    # 判断の記録は答えを作ったその場で 1 回だけ作り、on_record に渡す
    def decide(self, question: str, chunks: Sequence[Any]) -> Result:
        dr: dict[str, Any] = {
            "route": "code", "rule_id": None, "doc_path": None, "doc_version": None, "evidence": [],
            "extracted": [], "computation": [], "conclusion": None, "outcome_key": None, "outcome_basis": None,
            "stage": None, "reason": None, "hold_code": None, "final_answer": None, "correct": None,
            "excluded": [], "candidates": [],
        }
        try:
            res = self._decide(question, chunks, dr)
        except _Hold as h:
            dr.update(stage=h.stage, reason=h.reason, hold_code=f"{h.stage}:{h.code}", route="held")
            res = Held(h.stage, h.reason, dr)
        except Exception as e:  # noqa: BLE001  予期しない失敗も保留（黙ってモデルに戻さない）
            dr.update(stage="engine", reason=f"内部エラー: {type(e).__name__}", hold_code="engine:error", route="held")
            res = Held("engine", dr["reason"], dr)
        if self.on_record is not None:
            self.on_record(dr)
        return res

    def _decide(self, question: str, chunks: Sequence[Any], dr: dict[str, Any]) -> Result:
        # 1. 検索で取れた文書に結び付いた記録だけが候補
        docs: dict[str, tuple[str, list[Any]]] = {}
        for ch in chunks:
            hit = self.lookup.lookup(getattr(ch, "document_id", None))
            if hit is None:
                continue
            docs.setdefault(hit[0], (hit[1], []))[1].append(ch)
        cands = [r for r in self.records if r.doc_path in docs]
        dr["candidates"] = [r.rule_id for r in cands]
        text = parse.clean_question(question)

        # 2. 対象の判定
        matched: list[tuple[int, Record]] = []
        for r in cands:
            ex_hit = next((e for e in r.excludes if _term_hit(text, e.terms)), None)
            if ex_hit is not None:
                dr["excluded"].append({"rule_id": r.rule_id, "reason": ex_hit.reason})
                continue
            if all(_term_hit(text, g) for g in r.groups):
                matched.append((len(r.groups), r))
        if not matched:
            dr.update(route="model_only", reason="対象の記録が 0 件。数の規程ではないとして、モデルだけの回答に回す",
                      stage="subject", hold_code=None)
            return NoRecord(dr)
        top = max(n for n, _ in matched)
        winners = [r for n, r in matched if n == top]
        if len(winners) > 1:
            raise _Hold("subject", "tie", "対象の記録が 2 件以上で決められない: " + "、".join(r.rule_id for r in winners))
        rec = winners[0]
        dr.update(rule_id=rec.rule_id, doc_path=rec.doc_path, doc_version=rec.doc_version)

        # 3. 版
        sha_now, doc_chunks = docs[rec.doc_path]
        if rec.stale or sha_now != rec.doc_version:
            raise _Hold("version", "stale", "規程が更新され、記録が未確認（文書の版が記録と違う）")

        # 4. 対応範囲
        if not rec.supported:
            raise _Hold("scope", "unsupported", rec.scope_reason or "対応範囲外")
        has_date = any(f.kind == "date" for f in rec.facts)
        ex = parse.extract(question)
        if has_date and ex.business_days:
            raise _Hold("scope", "business_day", "営業日で数える期限は対応範囲外（休業日の表を持たない）")

        # 5. 例外
        for e in rec.exceptions:
            cue = _term_hit(text, e.cues)
            if cue:
                raise _Hold("exception", "cue", f"例外に当たりうる事情（{cue}）が述べられている: {e.description}")

        # 6. 質問が触れない前提
        for o in rec.other_requirements:
            cue = _term_hit(text, o.cues)
            if cue:
                a, b = _clause_bounds(text, text.find(_n(cue)))
                if any(m in text[a:b] for m in _NEGATIVE_MARKS):
                    raise _Hold("other_requirements", "unmet", f"{o.description}が満たされていない／分からないと読める")
            elif not o.assume_met_if_unstated:
                raise _Hold("other_requirements", "unstated", f"{o.description}について質問が触れていない")

        # 7. 事実
        if has_date and ex.unresolved_date:
            raise _Hold("facts", "unresolved_date", "年の分からない日付がある")
        needed: set[str] = set()
        for c in rec.conditions:
            needed.add(c.fact)
            if c.threshold.base:
                needed.add(c.threshold.base)
        vals = _extract_facts(rec, ex, needed)
        by = {f.role: f for f in rec.facts}
        dr["extracted"] = [
            {"role": v.role, "label": by[v.role].label, "value": _fmt_fact_value(by[v.role], v.value),
             "sources": v.sources} for v in vals.values()]

        # 8. 計算
        results = []
        comp = []
        for c in rec.conditions:
            f = by[c.fact]
            tv, tdesc = _threshold_value(c, f, vals)
            lhs = vals[c.fact].value
            ok = parse.holds(lhs, c.op, tv)
            results.append(ok)
            comp.append({"fact": c.fact, "label": f.label, "value": _fmt_fact_value(f, lhs), "op": c.op,
                         "threshold": tdesc,
                         "result": ok, "quote": c.evidence.quote, "line": c.evidence.line})
        dr["computation"] = comp
        dr["evidence"] = [{"quote": c.evidence.quote, "line": c.evidence.line} for c in rec.conditions]

        # 9. 結論
        key = "all_true" if all(results) else "any_false"
        out = rec.all_true if key == "all_true" else rec.any_false
        dr["outcome_key"] = key
        if out is None or out.basis == "unknown" or out.value == "unknown":
            dr["outcome_basis"] = "unknown"
            raise _Hold("outcome", "unknown", "規程から導けない側の結論のため保留（機械的に反対の結論にしない）")
        dr["outcome_basis"] = out.basis
        asked = [o.strip() for o in _OPTION_RE.findall(question)]
        if asked and out.value not in asked:
            raise _Hold("outcome", "option_mismatch", "質問が求める結論の言い方に、記録の結論が無い")
        dr["conclusion"] = out.value

        # 10. 答えの文
        assume = [o for o in rec.other_requirements
                  if o.assume_met_if_unstated and o.matters_for in (key, "both")]
        text_out = render.render(rec, dr, out, key, assume)
        if text_out.strip().split("\n")[-1] != f"結論: {out.value}":
            raise _Hold("outcome", "render_mismatch", "答えの文の最後の行が選んだ結論と合わない")
        dr["final_answer"] = text_out
        dr["stage"] = "decided"
        return Decided(text_out, dr, tuple(doc_chunks))
