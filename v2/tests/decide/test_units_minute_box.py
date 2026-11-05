"""単位「分」と「箱」: 時間↔分の換算、数える単位としての箱、「分」を含む別の言葉との取り違えが無い事。

規程の文書は、この試験の中で作った架空の 1 本。.jsonl・開発用の問は開かない。
"""
from __future__ import annotations

import copy
from fractions import Fraction

import pytest

from decide import parse
from decide.engine import DecisionEngine
from decide.records import RecordError, build_record, sha256_text

LINES = [
    "# 設備利用規程",                                                                          # 1
    "",                                                                                        # 2
    "第2会議室は、人数が8人未満かつ利用時間が80分以下の場合は利用可、人数が8人以上又は利用時間が80分超の場合は利用不可とする。",  # 3
    "備品庫への搬入は、4箱以下は受入可、4箱超は受入不可とする。",                              # 4
]
DOC = "\n".join(LINES) + "\n"
PATH = "/watched/general/設備利用規程.md"
OPT_USE = "最後の行には「結論: 利用可」または「結論: 利用不可」のどちらかだけを書いてください。"
OPT_BOX = "最後の行には「結論: 受入可」または「結論: 受入不可」のどちらかだけを書いてください。"


def ev(line, quote):
    assert quote in LINES[line - 1]
    return {"quote": quote, "line": line}


class Docs:
    def lookup(self, document_id):
        return (PATH, sha256_text(DOC)) if document_id == PATH else None

    def read_text(self, doc_path):
        return DOC if doc_path == PATH else None


class Ch:
    def __init__(self):
        self.document_id = PATH
        self.content = "x"
        self.access_level = 1
        self.source = "設備利用規程.md"


ROOM = {
    "rule_id": "R-room", "doc_path": PATH, "doc_version": sha256_text(DOC), "description": "会議室", "supported": True,
    "relation": "if_else", "subject": {"groups": [["会議室"], ["利用"]]},
    "facts": [{"role": "n", "label": "人数", "kind": "quantity", "unit": "人", "cues": ["人数"]},
              {"role": "t", "label": "利用時間", "kind": "quantity", "unit": "分", "cues": ["利用時間"]}],
    "conditions": [{"fact": "n", "op": "<", "threshold": {"type": "number", "value": 8}, "evidence": ev(3, "人数が8人未満")},
                   {"fact": "t", "op": "<=", "threshold": {"type": "number", "value": 80}, "evidence": ev(3, "利用時間が80分以下")}],
    "outcomes": {"all_true": {"value": "利用可", "basis": "explicit", "evidence": ev(3, "人数が8人未満かつ利用時間が80分以下の場合は利用可")},
                 "any_false": {"value": "利用不可", "basis": "explicit", "evidence": ev(3, "人数が8人以上又は利用時間が80分超の場合は利用不可")}},
    "options": ["利用可", "利用不可"],
}
BOX = {
    "rule_id": "R-box", "doc_path": PATH, "doc_version": sha256_text(DOC), "description": "搬入", "supported": True,
    "relation": "if_else", "subject": {"groups": [["搬入"], ["受入"]]},
    "facts": [{"role": "b", "label": "箱数", "kind": "quantity", "unit": "箱", "cues": ["搬入"]}],
    "conditions": [{"fact": "b", "op": "<=", "threshold": {"type": "number", "value": 4}, "evidence": ev(4, "4箱以下は受入可")}],
    "outcomes": {"all_true": {"value": "受入可", "basis": "explicit", "evidence": ev(4, "4箱以下は受入可")},
                 "any_false": {"value": "受入不可", "basis": "explicit", "evidence": ev(4, "4箱超は受入不可")}},
    "options": ["受入可", "受入不可"],
}


def decide(q, *datas):
    docs = Docs()
    return DecisionEngine([build_record(copy.deepcopy(d), docs) for d in datas], docs).decide(q, [Ch()])


def last(res):
    assert res.kind == "decided", (res.kind, res.decision_record)
    return res.answer_text.strip().split("\n")[-1]


def room_q(people, time):
    return f"第2会議室を使います。人数は{people}、利用時間は{time}です。利用できますか？" + OPT_USE


# ---------------------------------------------------------------- 分

@pytest.mark.parametrize("people,time,want", [
    ("7人", "79分", "利用可"), ("7人", "80分", "利用可"), ("7人", "81分", "利用不可"),
    ("7人", "30分間", "利用可"),
    ("7人", "1時間", "利用可"),          # 60 分
    ("7人", "2時間", "利用不可"),        # 120 分
    ("8人", "30分", "利用不可"), ("7人", "30分", "利用可"),
])
def test_minute_boundary_and_hour_conversion(people, time, want):
    assert last(decide(room_q(people, time), ROOM)) == f"結論: {want}"


def test_hour_and_a_half_is_ninety_minutes_not_one_hour():
    # 「1時間半」を 1 時間（60 分）と読むと 80 分以下で「利用可」になってしまう。90 分なので不可
    assert last(decide(room_q("7人", "1時間半"), ROOM)) == "結論: 利用不可"
    assert parse.extract("1時間半").cands[0].value == Fraction(3, 2)


def test_hour_plus_minute_is_held_as_conflict():
    res = decide(room_q("7人", "1時間20分"), ROOM)
    assert res.kind == "held"  # 「1時間」と「20分」は同じ役割に違う値（2年11か月と同じ扱い）


def test_minute_against_day_fact_is_unit_mismatch():
    res = decide(room_q("7人", "2日"), ROOM)
    assert res.kind == "held" and res.decision_record["hold_code"] == "facts:unit_mismatch"


def kinds(q):
    return [(c.kind, c.value) for c in parse.extract(q).cands]


def test_minute_convert_table():
    assert parse.convert(Fraction(2), "時間", "分") == Fraction(120)
    assert parse.convert(Fraction(90), "分", "時間") == Fraction(3, 2)
    assert parse.convert(Fraction(1), "日", "分") is None


@pytest.mark.parametrize("text", [
    "4月分の請求", "5分の1が対象", "十分な広さ", "半分を使う", "一部分だけ", "分かる範囲", "分からない点", "分担する",
    "10分から開始",  # 「分から」は数字の後でも単位にしない（保留側に倒す）
])
def test_minute_does_not_collide_with_other_words(text):
    assert [k for k, _ in kinds(text) if k in ("分", "時間")] == []


def test_minute_unit_reads_and_clock_time_is_not_duration():
    assert kinds("利用は90分です") == [("分", Fraction(90))]
    assert kinds("30分間利用") == [("分", Fraction(30))]
    assert kinds("9時30分に開始") == []  # 時刻は長さではない
    assert kinds("3割5分") == [("%", Fraction(35))]  # 「5分」を二重に読まない


def test_minute_collision_in_a_question_does_not_decide_wrongly():
    # 「4月分」の「分」を単位に読まない: 時間の候補が無いので保留（誤って 4 分と読んで判定しない）
    res = decide("第2会議室の4月分の利用を申し込みます。人数は7人です。利用できますか？" + OPT_USE, ROOM)
    assert res.kind == "held"


# ---------------------------------------------------------------- 箱

@pytest.mark.parametrize("n,want", [("3箱", "受入可"), ("4箱", "受入可"), ("5箱", "受入不可"), ("12箱", "受入不可")])
def test_box_boundary(n, want):
    q = f"備品庫への搬入が{n}あります。受入できますか？" + OPT_BOX
    assert last(decide(q, BOX)) == f"結論: {want}"


def test_box_is_a_counted_unit_and_bare_word_is_not():
    assert kinds("6箱を搬入") == [("箱", Fraction(6))]
    assert kinds("資料箱を搬入") == []
    assert kinds("資料箱に6個入れる") == [("個", Fraction(6))]


def test_box_unit_does_not_match_other_counted_unit():
    res = decide("備品庫への搬入が5個あります。受入できますか？" + OPT_BOX, BOX)
    assert res.kind == "held"  # 個は箱ではない


def test_box_and_minute_records_load_against_their_document():
    docs = Docs()
    assert build_record(copy.deepcopy(BOX), docs) and build_record(copy.deepcopy(ROOM), docs)


def test_reversed_op_or_wrong_unit_rejected():
    docs = Docs()
    room = copy.deepcopy(ROOM)
    room["conditions"][1]["op"] = ">"  # 「80分以下」に >
    with pytest.raises(RecordError):
        build_record(room, docs)
    box = copy.deepcopy(BOX)
    box["conditions"][0]["op"] = ">"  # 「4箱以下」に >
    with pytest.raises(RecordError):
        build_record(box, docs)
    box = copy.deepcopy(BOX)
    box["facts"][0]["unit"] = "個"  # 引用には 4個 が無い
    with pytest.raises(RecordError):
        build_record(box, docs)
    room = copy.deepcopy(ROOM)
    room["facts"][1]["unit"] = "時間"  # 引用には 80時間 が無い
    with pytest.raises(RecordError):
        build_record(room, docs)
