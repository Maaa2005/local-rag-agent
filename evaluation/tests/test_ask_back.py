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


def _tax_card():
    raw = {**_RAW, "slots": {
        "税抜き": {"type": "number", "unit": "円", "describe": "税抜き"},
        "税込み": {"type": "number", "unit": "円", "describe": "税込み"},
        "認定日": {"type": "date", "describe": "認定の日"}},
        "derive": {"税込み": {"mul": ["税抜き", 1.1]}},
        "conditions": [{"name": "5000円以下", "left": "税込み", "op": "<=", "right": 5000, "quote": "所定の届出"}],
        "exceptions": []}
    return raw


def test_asked_input_falls_back_to_the_form_the_user_has():
    card = rc.parse_card(_tax_card())
    u = ab.User(card, {"税込み": _d(4400)})
    r = ab.dialog(card, {}, u, "読まない選ぶ")
    assert r["conclusion"] == "支給"
    assert r["turns"] == r["typed"] == 1
    assert not r["no_truth"]


def test_not_applicable_amount_is_no_value_not_a_crash():
    card = _card()
    assert ab.parse_truth(card, {"勤続年数": ["なし"], "懲戒": ["なし"]}) == {"勤続年数": None, "懲戒": ["なし"]}


def _leave_card():
    """700 点以上か、復職後 12 か月以内なら 600 点以上。"""
    return rc.parse_card({**_RAW, "slots": {
        "点数": {"type": "number", "unit": "点", "describe": "点数"},
        "復職日": {"type": "date", "describe": "復職日"},
        "締切日": {"type": "date", "describe": "締切日"},
        "復職後月数": {"type": "number", "unit": "か月", "describe": "復職後月数"}},
        "derive": {"復職後月数": {"months_between": ["復職日", "締切日"]}},
        "conditions": [{"name": "点数", "quote": "所定の届出", "any": [
            {"name": "700", "left": "点数", "op": ">=", "right": 700},
            {"name": "特例", "all": [{"name": "12", "left": "復職後月数", "op": "<=", "right": 12},
                                    {"name": "600", "left": "点数", "op": ">=", "right": 600}]}]}],
        "exceptions": []})


def test_not_applicable_means_the_comparison_fails_not_unknown():
    card = _leave_card()
    truth = ab.parse_truth(card, {"点数": [650], "復職日": ["なし"], "復職後月数": ["なし"], "締切日": ["2027-04-30"]})
    assert truth["復職日"] is None and truth["復職後月数"] == [ab.o.BIG]
    r = ab.dialog(card, {}, ab.User(card, truth), "読まない選ぶ")
    assert r["conclusion"] == "不支給"
    assert not r["no_truth"]


def test_not_applicable_corrects_a_zero_read():
    """「復職していない」を 0 か月と読むと特例が効く。確かめて直れば誤らない。"""
    card = _leave_card()
    truth = ab.parse_truth(card, {"点数": [650], "復職日": ["なし"], "復職後月数": ["なし"], "締切日": ["2027-04-30"]})
    first = {"点数": _d(650), "復職後月数": _d(0)}
    assert ab.dialog(card, first, ab.User(card, dict(truth)), "聞く")["conclusion"] == "支給"
    assert ab.dialog(card, first, ab.User(card, dict(truth)), "本命")["conclusion"] == "不支給"


def test_unknown_input_with_unknown_target_stays_unknown():
    card = _leave_card()
    truth = ab.parse_truth(card, {"点数": [650], "復職日": None, "復職後月数": None, "締切日": ["2027-04-30"]})
    assert ab.dialog(card, {}, ab.User(card, truth), "読まない選ぶ")["conclusion"] is None


def _claim_card():
    """懲戒は本人が言った時だけ当たる例外。触れていなければ今の読みは「なし」にする。"""
    raw = {**_RAW, "exceptions": [{**_RAW["exceptions"][0], "if_unknown": "ignore"}],
           "claim_only": {"懲戒": {"basis": "本人が言った時だけ"}}}
    return rc.parse_card(raw)


def test_untouched_is_asked_only_in_the_new_main_arm():
    card = _claim_card()
    q = "試験です。勤続5年で届出もあります。\n最後の行には「結論: 支給」または「結論: 不支給」のどちらかだけを書いてください。"
    reading = {"勤続年数": {"status": "値", "quote": "勤続5年", "value": 5},
               "届出": {"status": "値", "quote": "届出もあります", "value": "あり"},
               "懲戒": {"status": "触れない", "quote": ""}}
    plain = ab.o.to_candidates(card, q, reading)
    assert plain["懲戒"] == ["なし"]
    assert ab.first_read(card, q, reading, "旧本命")["懲戒"] == ["なし"]
    assert ab.first_read(card, q, reading, "本命")["懲戒"] is None
    assert ab.first_read(card, q, reading, "本命")["勤続年数"] == plain["勤続年数"]


def test_new_main_arm_does_not_show_untouched_to_a_swayed_user():
    """旧本命は「なし」を認めて誤り、本命は直接聞くので誤らない。"""
    card = _card()
    truth = {**TRUTH, "懲戒": ["あり"]}
    shown = {"勤続年数": _d(5), "届出": ["あり"], "懲戒": ["なし"]}
    assert ab.dialog(card, shown, _user(card, truth, accept=1.0), "旧本命")["conclusion"] == "支給"
    hidden = {"勤続年数": _d(5), "届出": ["あり"]}
    assert ab.dialog(card, hidden, _user(card, truth, accept=1.0), "本命")["conclusion"] == "不支給"


def test_scene_diff_keeps_pairs_together_and_signs():
    scenes = [{"id": "a", "pair_id": "p"}, {"id": "b", "pair_id": "p"}, {"id": "c", "pair_id": None}]
    d = ab.scene_diff({"a": 1.0, "b": 1.0}, {"c": 1.0}, scenes, n=500)
    assert d["diff"] == 1.0
    assert d["lo"] <= d["diff"] <= d["hi"]


def _res(**main):
    base = {"correct": 0, "wrong": 0, "hold_wrong": 0, "held": 18, "stopped": 0, "per_rule_min": 0, "per_rule": {},
            "pairs": 0, "turns_med": 0, "turns_max": 0, "items_mean": 0, "typed_sum": 10, "no_truth": 0,
            "wrong_by_scene": {}}
    good = {**base, "correct": 66, "per_rule_min": 10, "turns_med": 3, "turns_max": 6, "typed_sum": 5} | main
    return {"本命/正直": good, "基準/正直": base, "本命/流される": base, "基準/流される": base, "読まない選ぶ/正直": base}


RAW_OK = [{"conclusion": None, "should_hold": True, "seconds": 15.0, "calls": 1}]


def test_judge_passes_at_exact_lines():
    assert all(ab.judge_one(_res(), [{"id": "a"}], RAW_OK)["lines"].values())


def test_judge_fails_each_line():
    scenes = [{"id": "a"}]
    def line(res, key, raw=RAW_OK):
        return ab.judge_one(res, scenes, raw)["lines"][key]
    assert not line(_res(correct=64), "1 正しい決定 65/72・各規程 10/12")
    assert not line(_res(per_rule_min=9), "1 正しい決定 65/72・各規程 10/12")
    assert not line(_res(wrong=1), "2 誤った断定 0")
    assert not line(_res(held=17, hold_wrong=1), "3 本人も分からない 18/18")
    assert not line(_res(turns_max=7), "5 やり取り 中央値3・最大6")
    assert not line(_res(turns_med=3.5), "5 やり取り 中央値3・最大6")
    assert not line(_res(typed_sum=6), "6 打ち込みが読まない形の半分以下")
    r = _res()
    r["本命/流される"] = {**r["本命/流される"], "wrong_by_scene": {"a": 0.1}}
    assert not line(r, "4 流される時の誤りが基準以下")
    slow = [{"conclusion": None, "should_hold": True, "seconds": 15.1, "calls": 1}]
    assert not line(_res(), "7 読みの時間 p95 15秒", slow)
