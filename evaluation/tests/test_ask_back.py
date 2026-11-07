"""聞き返しの仕組み (evaluation/gpu_local/ask_back.py) のテスト。

確かめること:
- 足りない値を聞けば決まり、本人も分からなければ保留になる
- 読み違いは確かめる形だけが見つけ、認められれば誤ったまま残る
- 無関係な枠は聞かない。読んでいれば聞く数が減る
- 数は別の式で手計算した値と合う（やり取り・確かめる項目・打ち込み）
"""
from __future__ import annotations

import random
from decimal import Decimal

from evaluation.gpu_local import ask_back as ab

rc = ab.rc

_RAW = {
    "format": "rule-card@1", "id": "t", "status": "draft", "drafted_by": "t", "confirmed_by": None,
    "doc": {"source": "試験規程", "section": "第1条", "quote": "勤続3年以上かつ所定の届出があれば支給する。"},
    "target": "試験", "match": {"conclusions": ["支給", "不支給"], "keywords": ["試験"]},
    "slots": {
        "勤続年数": {"type": "number", "unit": "年", "describe": "勤続年数"},
        "届出": {"type": "bool", "describe": "届出の有無"},
        "懲戒": {"type": "bool", "describe": "懲戒の有無"},
        "部署人数": {"type": "number", "unit": "人", "describe": "判定に使わない数"},
    },
    "derive": {},
    "constraints": [],
    "conditions": [
        {"name": "勤続3年以上", "left": "勤続年数", "op": ">=", "right": 3, "quote": "勤続3年以上"},
        {"name": "届出あり", "left": "届出", "op": "==", "right": "あり", "quote": "所定の届出"},
    ],
    "exceptions": [{"name": "懲戒あり", "left": "懲戒", "op": "==", "right": "あり", "quote": "懲戒",
                    "conclusion": "不支給", "if_unknown": "hold"}],
    "conclusion": {"then": "支給", "else": "不支給"},
}


def _card() -> rc.Card:
    return rc.parse_card(_RAW)


def _d(x: float) -> list[Decimal]:
    return [Decimal(str(x))]


TRUTH = {"勤続年数": _d(5), "届出": ["あり"], "懲戒": ["なし"]}


def _user(card, truth=None, accept=0.0, err=0.0, skip=0, seed="s"):
    return ab.User(card, dict(TRUTH if truth is None else truth), accept, err, skip, random.Random(seed))


def test_no_read_asks_until_decided():
    card = _card()
    r = ab.dialog(card, {}, _user(card), "読まない選ぶ")
    assert r["conclusion"] == "支給"
    assert r["turns"] == r["typed"] == 3  # 3 つとも答えないと「支給」にならない
    assert r["items"] == 0
    assert "部署人数" not in r["asked"]


def test_failing_condition_stops_early():
    card = _card()
    r = ab.dialog(card, {}, _user(card, {**TRUTH, "届出": ["なし"]}), "読まない選ぶ")
    assert r["conclusion"] == "不支給"
    assert r["turns"] == 1  # 届出が無ければ、例外も同じ結論なので残りは聞かなくても決まる


def test_unknown_to_user_holds():
    card = _card()
    r = ab.dialog(card, {"勤続年数": _d(5), "懲戒": ["なし"]}, _user(card, {**TRUTH, "届出": None}), "本命")
    assert r["conclusion"] is None


def test_unknown_but_irrelevant_still_decides():
    """勤続が足りないなら、届出を本人が分からなくても結論は決まる。"""
    card = _card()
    r = ab.dialog(card, {"勤続年数": _d(1)}, _user(card, {**TRUTH, "勤続年数": _d(1), "届出": None}), "本命")
    assert r["conclusion"] == "不支給"


def test_baseline_never_asks():
    card = _card()
    r = ab.dialog(card, {"勤続年数": _d(5)}, _user(card), "基準")
    assert r["conclusion"] is None
    assert r["turns"] == r["typed"] == 0


def test_check_fixes_misread_with_honest_user():
    card = _card()
    wrong = {"勤続年数": _d(1)}  # 本当は 5 年
    assert ab.dialog(card, wrong, _user(card), "聞く")["conclusion"] == "不支給"  # 確かめないと誤る
    r = ab.dialog(card, wrong, _user(card), "本命")
    assert r["conclusion"] == "支給"
    assert r["typed"] >= 1 and r["items"] >= 1


def test_check_confirms_right_read_without_typing():
    card = _card()
    r = ab.dialog(card, {"勤続年数": _d(1)}, _user(card, {**TRUTH, "勤続年数": _d(1)}), "本命")
    assert r["conclusion"] == "不支給"
    assert (r["turns"], r["items"], r["typed"]) == (1, 1, 0)


def test_certificate_is_minimal():
    """1 年なら勤続だけで不支給。届出の読みは確かめなくてよい。"""
    card = _card()
    cand = {n: None for n in card.slots} | {"勤続年数": _d(1), "届出": ["あり"], "懲戒": ["なし"]}
    assert ab.certificate(card, cand, "不支給", set()) == ["勤続年数"]


def test_swayed_user_keeps_misread():
    card = _card()
    wrong = {"勤続年数": _d(1)}
    r = ab.dialog(card, wrong, _user(card, accept=1.0), "本命")
    assert r["conclusion"] == "不支給"
    assert r["typed"] == 0


def test_skipping_user_accepts_large_screen_only():
    card = _card()
    cand = {"勤続年数": _d(9), "届出": ["あり"], "懲戒": ["あり"]}  # 本当は懲戒なしなのに、あり
    truth = {**TRUTH, "勤続年数": _d(9), "懲戒": ["なし"]}
    # 例外の読みだけで不支給なので画面は 1 項目。上限 3 以下なら見て直す
    assert ab.dialog(card, cand, _user(card, truth, accept=0.0, skip=3), "本命")["conclusion"] == "支給"
    # 3 項目の画面は上限 2 を超えるので、全部そのまま認める
    wrong_yes = {"勤続年数": _d(9), "届出": ["あり"], "懲戒": ["なし"]}
    truth2 = {**TRUTH, "勤続年数": _d(1)}
    r = ab.dialog(card, wrong_yes, _user(card, truth2, skip=2), "本命")
    assert r["conclusion"] == "支給" and r["items"] == 3 and r["typed"] == 0
    assert ab.dialog(card, wrong_yes, _user(card, truth2), "本命")["conclusion"] == "不支給"


def test_range_read_is_confirmed_when_truth_inside():
    assert ab._same(_d(3) + _d(7), _d(5))
    assert not ab._same(_d(3) + _d(7), _d(9))


def test_wrong_answer_is_a_different_value():
    card = _card()
    u = _user(card)
    assert u._wrong("届出", ["あり"]) == ["なし"]
    assert u._wrong("勤続年数", _d(4)) in (_d(6.0), _d(2.0))


def test_order_policy_asks_in_slot_order():
    card = _card()
    assert ab.next_slot(card, {n: None for n in card.slots}, set(), "order") == "勤続年数"


def test_missing_truth_is_recorded_not_guessed():
    card = _card()
    r = ab.dialog(card, {}, _user(card, {}), "読まない選ぶ")
    assert r["conclusion"] is None
    assert r["no_truth"]


def test_derived_read_is_dropped_when_inputs_are_read():
    raw = {**_RAW, "slots": {
        "開始": {"type": "time", "describe": "開始"}, "終了": {"type": "time", "describe": "終了"},
        "時間": {"type": "number", "unit": "分", "describe": "時間"}},
        "derive": {"時間": {"minutes_between": ["開始", "終了"]}},
        "conditions": [{"name": "90分以上", "left": "時間", "op": ">=", "right": 90, "quote": "所定の届出"}],
        "exceptions": []}
    card = rc.parse_card(raw)
    first = {"開始": [rc.parse_time("09:00")], "終了": [rc.parse_time("11:00")], "時間": _d(30)}  # 読みが矛盾
    u = ab.User(card, {"開始": first["開始"], "終了": first["終了"]})
    assert ab.dialog(card, first, u, "基準")["conclusion"] == "支給"  # 計算した 120 分が勝つ
