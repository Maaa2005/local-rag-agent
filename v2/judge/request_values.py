"""依頼文の自由な値を、前もって承認した値・書き方に限る（Issue #17・本人決定 11/3 の 1）。

協議（#17・11/2）で、手がかり語（拒否リスト）を足すだけでは「1850万円以上の役員」のような閾値の形の
実データや、Codex レビューの言い換え（区切り無しの電話・EMP1024・hr . salary など）を止めきれないと
分かった。そこで依頼文から自由な値を切り出し、承認した物だけを通す（OWASP の許可リスト型の入力検査）。

自由な値（正規化した写しの上で切り出す）:
- 数字の並び（10・3.11・5,000）
- 漢数字の値（2 字以上続く物・〇 を含む物: 一〇二四・〇九〇…、1 字でも金額・日付・人数などの単位が続く物:
  五百円・三名・十月）
- 元号の年（令和元年・平成5年）
- 英字の語（[A-Za-z_][A-Za-z0-9_]*。_ で区切った各部分を照合する: total_amount は total と amount）
- 引用符の中身（「」『』""“”''‘’）。中身が承認した値そのもの、または承認した語・値だけなら可。
  中の語や数字は引用符ごとに判定し、別に数えない

承認の出どころ:
- 契約ごと（policies/c2_tables.json の request_values）: 一般の英字の語（Python・pandas・集計の語）と、
  一般の書き方（forms: 上位N件・N行・0/1・UTF-8 など。金額・日付は入れない）。書き方に当たった範囲の
  数字と英字の語は可
- 承認済み本文ごと（policies/approved/*.json の request_values・登録簿に写す）: 仕様の列名などの英字の語と、
  承認した値そのもの（例: 10万円）。値は出現の範囲にまるごと入る数字だけを許す（210万円 は 10万円 で通らない）

カタカナ語・ひらがな・漢字の語は対象にしない（技術のカタカナ語を前もって並べきれず誤停止が増えるため）。
敬称の無い漢字の人名・カタカナの社名は、この検査でも手がかり語でも拾えない残る危険として #17 に書く。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from judge.base import question_version
from judge.normalize import normalize_text

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUMBER = re.compile(r"(?<![A-Za-z0-9_])\d+(?:[.,]\d+)*")
_KANJI_DIGITS = "〇一二三四五六七八九十百千万億兆壱弐参拾"
_KANJI_NUMBER = re.compile(rf"[{_KANJI_DIGITS}]*〇[{_KANJI_DIGITS}]*|[{_KANJI_DIGITS}]{{2,}}|"
                           rf"[{_KANJI_DIGITS}](?=\s*(円|ドル|ユーロ|年|月|日|名|人|歳|割|%))")
_ERA_YEAR = re.compile(r"(令和|平成|昭和|大正|明治)\s*(元|\d+|[〇一二三四五六七八九十]+)\s*年")
_QUOTED = re.compile(r"「([^」]*)」|『([^』]*)』|\"([^\"]*)\"|“([^”]*)”|'([^']*)'|‘([^’]*)’")

# 切り出しの版（ゲートの question_version に入れる。切り出しを変えると版が変わる）
VALUES_SPEC = {"word": _WORD.pattern, "number": _NUMBER.pattern, "kanji_number": _KANJI_NUMBER.pattern,
               "era_year": _ERA_YEAR.pattern, "quoted": _QUOTED.pattern, "word_parts": "whole word, else split on _",
               "quoted_rest": r"[^\s.,:;/・-]", "approved_value": "whole occurrence covers the token"}
VALUES_VERSION = question_version("request-values@1", VALUES_SPEC)


@dataclass(frozen=True)
class ValueRules:
    """契約ごとの承認（英字の語は小文字にしてある）。"""
    identifiers: frozenset[str]
    forms: tuple[tuple[str, re.Pattern[str]], ...]


def compile_value_rules(where: str, spec: object) -> ValueRules:
    if not isinstance(spec, dict) or set(spec) != {"identifiers", "forms"}:
        raise ValueError(f"{where}: needs exactly identifiers and forms")
    ids, forms = spec["identifiers"], spec["forms"]
    if not isinstance(ids, list) or not all(isinstance(x, str) and _WORD.fullmatch(x) for x in ids):
        raise ValueError(f"{where}.identifiers: must be a list of ASCII words")
    if not isinstance(forms, dict):
        raise ValueError(f"{where}.forms: must be an object of name -> pattern")
    compiled = []
    for name, p in forms.items():
        if not isinstance(p, str):
            raise ValueError(f"{where}.forms.{name}: pattern must be a string")
        try:
            pat = re.compile(p, re.I)
        except re.error as e:
            raise ValueError(f"{where}.forms.{name}: bad pattern: {e}") from e
        if pat.search(""):
            raise ValueError(f"{where}.forms.{name}: pattern matches the empty text")
        compiled.append((name, pat))
    return ValueRules(frozenset(x.lower() for x in ids), tuple(compiled))


def _occurrences(text: str, needle: str) -> Iterable[tuple[int, int]]:
    i = text.find(needle)
    while needle and i >= 0:
        yield i, i + len(needle)
        i = text.find(needle, i + 1)


def _inside(span: tuple[int, int], covered: Sequence[tuple[int, int]]) -> bool:
    return any(s <= span[0] and span[1] <= e for s, e in covered)


def _word_ok(word: str, known: frozenset[str]) -> bool:
    """語全体が一覧にあれば可（to_datetime・drop_duplicates）。無ければ _ で区切った各部分を見る。"""
    if word.lower() in known:
        return True
    parts = [p for p in word.lower().split("_") if p]
    return bool(parts) and all(p in known for p in parts)


def _check(n: str, rules: ValueRules, known: frozenset[str], values: Sequence[str],
           skip: Sequence[tuple[int, int]] = ()) -> list[tuple[int, str]]:
    """n の中の承認されていない値（位置, '種類:値'）。skip の範囲（引用符）の中は見ない。"""
    covered = [m.span() for _, pat in rules.forms for m in pat.finditer(n)]
    covered += [sp for v in values for sp in _occurrences(n, v)]
    bad: list[tuple[int, str]] = []
    for kind, pat in (("number", _NUMBER), ("kanji_number", _KANJI_NUMBER), ("era_year", _ERA_YEAR), ("word", _WORD)):
        for m in pat.finditer(n):
            if _inside(m.span(), skip) or _inside(m.span(), covered):
                continue
            if kind == "word" and _word_ok(m.group(0), known):
                continue
            bad.append((m.start(), f"{kind}:{m.group(0)}"))
    return bad


def unapproved_values(text: str, rules: ValueRules, body_identifiers: Iterable[str],
                      body_values: Iterable[str]) -> list[str]:
    """依頼文 text（生の文字列）の、承認されていない自由な値（'種類:値' の列。出てきた順・重複あり）。"""
    n = normalize_text(text)
    known = rules.identifiers | frozenset(x.lower() for x in body_identifiers)
    values = [v for v in (normalize_text(x) for x in body_values) if v]
    quoted: list[tuple[int, int]] = []
    bad: list[tuple[int, str]] = []
    for m in _QUOTED.finditer(n):
        quoted.append(m.span())
        inner = next(g for g in m.groups() if g is not None)
        if not inner.strip() or inner.strip() in values:
            continue
        # 中身が承認した語・値と区切りの記号だけなら可。日本語の語などが残れば引用符ごと保留
        rest = _WORD.sub("", _NUMBER.sub("", inner))
        for v in sorted(values, key=len, reverse=True):
            rest = rest.replace(v, "")
        if _check(inner, rules, known, values) or re.search(r"[^\s.,:;/・-]", rest):
            bad.append((m.start(), f"quoted:{inner}"))
    bad += _check(n, rules, known, values, skip=quoted)
    return [item for _, item in sorted(bad)]
