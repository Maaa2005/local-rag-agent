"""境目が「問の中の別の日付」（threshold.type == "fact"）の記録の検査と判定。

規程の例: 「実技講座の事前登録は、登録申請日が講座開始日より前で、かつ登録申請日時点の勤続期間が6ヶ月以上の場合に登録可とする。
登録申請日が講座開始日以後、又は勤続期間が6ヶ月未満の場合は登録不可とする。」
規程の文書・記録はこの試験の中で作った架空の物（decide/records/ には置かない）。.jsonl・開発用の問は開かない。
"""
from __future__ import annotations

import copy

import pytest

from decide.engine import DecisionEngine
from decide.records import RecordError, build_record, sha256_text

LINES = [
    "# 講座規程",                                                                                    # 1
    "",                                                                                              # 2
    "実技講座の事前登録は、登録申請日が講座開始日より前で、かつ登録申請日時点の勤続期間が6ヶ月以上の場合に登録可とする。"
    "登録申請日が講座開始日以後、又は勤続期間が6ヶ月未満の場合は登録不可とする。",                      # 3
    "実技講座の事前登録は、登録申請日が講座開始日以前の場合に登録可とする。",                           # 4  以前（<=）
    "実技講座の事前登録は、講座開始日が登録申請日より前の場合に登録可とする。",                         # 5  基準の名前の直後に言い回しが無い（向きが逆の書き方）
    "実技講座の事前登録は、登録申請日が開始日より前の場合に登録可とする。",                             # 6  名前が違う
    "実技講座の事前登録は、登録申請日が講座開始日より前の場合に登録可、講座開始日より前の場合に登録不可とする。",  # 7  同じ比べ方が両側
    "実技講座の事前登録は、登録申請日が講座開始日より前の場合に登録不可、講座開始日以後の場合に登録可とする。",  # 8  結論の側が逆
]
DOC = "\n".join(LINES) + "\n"
PATH = "/watched/general/講座規程.md"
OPT = "最後の行には「結論: 登録可」または「結論: 登録不可」のどちらかだけを書いてください。"

FULL = LINES[2]
ONE_SIDE = "登録申請日が講座開始日より前"

F_APPLY = {"role": "apply", "label": "登録申請日", "kind": "date", "cues": ["申請"]}
F_START = {"role": "start", "label": "講座開始日", "kind": "date", "cues": ["開始"]}
F_TENURE = {"role": "tenure", "label": "勤続期間", "kind": "quantity", "unit": "か月", "cues": ["勤続"]}


class Docs:
    def lookup(self, document_id):
        return (PATH, sha256_text(DOC)) if document_id == PATH else None

    def read_text(self, doc_path):
        return DOC if doc_path == PATH else None


class Ch:
    def __init__(self, document_id):
        self.document_id = document_id
        self.content = "x"
        self.access_level = 1
        self.source = "講座規程.md"


CHUNKS = [Ch(PATH)]


def ev(quote, line=3):
    return {"quote": quote, "line": line}


def rec(*, op="<", quote=FULL, line=3, base="start", fact="apply", facts=None, tenure=True,
        a="登録可", b="登録不可", relation="if_else"):
    """例の記録。tenure=False なら勤続の条件を付けず、条件は日付どうしの比べだけ（1 条件）。"""
    q = ev(quote, line)
    conds = [{"fact": fact, "op": op, "threshold": {"type": "fact", "base": base}, "evidence": q}]
    if tenure:
        conds.append({"fact": "tenure", "op": ">=", "threshold": {"type": "number", "value": 6},
                      "evidence": ev("勤続期間が6ヶ月以上")})
    return {"rule_id": "R-course", "doc_path": PATH, "doc_version": sha256_text(DOC), "description": "実技講座の事前登録",
            "supported": True, "relation": relation, "subject": {"groups": [["講座"], ["登録"]]},
            "facts": facts if facts is not None else [F_APPLY, F_START, F_TENURE],
            "conditions": conds,
            "outcomes": {"all_true": {"value": a, "basis": "explicit", "evidence": ev(quote, line)},
                         "any_false": {"value": b, "basis": "explicit", "evidence": ev(quote, line)}},
            "options": ["登録可", "登録不可"]}


def decide(q, data=None):
    docs = Docs()
    return DecisionEngine([build_record(copy.deepcopy(data or rec()), docs)], docs).decide(q, CHUNKS)


def last(res):
    assert res.kind == "decided", (res.kind, getattr(res, "reason", None), res.decision_record)
    return res.answer_text.strip().split("\n")[-1]


def q_apply_first(apply_, start, tenure="8ヶ月"):
    return (f"勤続{tenure}の社員が、{apply_}に実技講座の事前登録を申請しました。講座は{start}に開始します。"
            f"事前登録できますか？" + OPT)


def q_start_first(apply_, start, tenure="8ヶ月"):
    return (f"実技講座は{start}に開始します。勤続{tenure}の社員が、{apply_}に事前登録を申請しました。"
            f"事前登録できますか？" + OPT)


# ---------------------------------------------------------------- 読み込み（通る形）

@pytest.mark.parametrize("quote", [FULL, ONE_SIDE, "登録申請日が講座開始日より前で、かつ登録申請日時点の勤続期間が6ヶ月以上の場合に登録可とする"])
def test_loads_with_full_or_one_sided_quote(quote):
    assert build_record(rec(quote=quote), Docs()).conditions[0].threshold.type == "fact"


def test_loads_other_ops_and_phrases():
    # 「以前」は <=。「以後」は >=（片側の引用）。
    assert build_record(rec(op="<=", quote="登録申請日が講座開始日以前", line=4, tenure=False), Docs())
    assert build_record(rec(op=">=", quote="登録申請日が講座開始日以後", tenure=False), Docs())


def test_after_phrases_read_against_ops():
    doc_lines = ["# 規程", "登録申請日が講座開始日より後の場合に登録可とする。", "登録申請日が講座開始日以降の場合に登録可とする。"]
    text = "\n".join(doc_lines) + "\n"

    class D:
        def read_text(self, p):
            return text

    for line, op in ((2, ">"), (3, ">=")):
        d = rec(op=op, quote=doc_lines[line - 1].split("の場合")[0], line=line, tenure=False)
        d["doc_version"] = sha256_text(text)
        assert build_record(d, D())
        d = rec(op="<" if op == ">" else "<=", quote=doc_lines[line - 1].split("の場合")[0], line=line, tenure=False)
        d["doc_version"] = sha256_text(text)
        with pytest.raises(RecordError, match="一致しない"):
            build_record(d, D())


# ---------------------------------------------------------------- 判定（申請日と開始日の前後）

@pytest.mark.parametrize("apply_,want", [
    ("2026年11月9日", "登録可"),      # 開始日の 1 日前
    ("2026年11月10日", "登録不可"),   # 同じ日
    ("2026年11月11日", "登録不可"),   # 1 日後
])
def test_apply_vs_start_boundary(apply_, want):
    assert last(decide(q_apply_first(apply_, "2026年11月10日"))) == f"結論: {want}"


def test_tenure_short_is_not_allowed():
    assert last(decide(q_apply_first("2026年11月9日", "2026年11月10日", tenure="5ヶ月"))) == "結論: 登録不可"
    assert last(decide(q_apply_first("2026年11月9日", "2026年11月10日", tenure="6ヶ月"))) == "結論: 登録可"


@pytest.mark.parametrize("apply_,want", [("2026年11月9日", "登録可"), ("2026年11月10日", "登録不可"), ("2026年11月11日", "登録不可")])
def test_sentence_order_swapped_gives_same_conclusion(apply_, want):
    assert last(decide(q_start_first(apply_, "2026年11月10日"))) == f"結論: {want}"
    assert last(decide(q_apply_first(apply_, "2026年11月10日"))) == f"結論: {want}"


def test_roles_not_swapped_when_dates_reversed():
    # 申請日の方が後ろの日付、開始日の方が前の日付（早い日付が開始日）。取り違えると登録可になる。
    assert last(decide(q_start_first("2026年11月20日", "2026年11月10日"))) == "結論: 登録不可"
    assert last(decide(q_apply_first("2026年11月20日", "2026年11月10日"))) == "結論: 登録不可"
    assert last(decide(q_start_first("2026年11月10日", "2026年11月20日"))) == "結論: 登録可"


def test_year_boundary():
    assert last(decide(q_apply_first("2026年12月31日", "2027年1月1日"))) == "結論: 登録可"
    assert last(decide(q_apply_first("2027年1月1日", "2027年1月1日"))) == "結論: 登録不可"


def test_le_op_includes_same_day():
    d = rec(op="<=", quote="登録申請日が講座開始日以前", line=4, tenure=False)
    assert last(decide(q_apply_first("2026年11月10日", "2026年11月10日"), d)) == "結論: 登録可"
    assert last(decide(q_apply_first("2026年11月11日", "2026年11月10日"), d)) == "結論: 登録不可"


# ---------------------------------------------------------------- 開始日が質問に無い → 保留（事実の段）

def test_missing_base_date_is_held_at_facts():
    res = decide("勤続8ヶ月の社員が、2026年11月9日に実技講座の事前登録を申請しました。事前登録できますか？" + OPT)
    assert res.kind == "held" and res.stage == "facts", (res.kind, res.decision_record)
    assert "講座開始日" in res.reason
    assert res.decision_record["hold_code"] == "facts:missing"


def test_missing_apply_date_is_held_at_facts():
    res = decide("勤続8ヶ月の社員が実技講座の事前登録を申請しました。講座は2026年11月10日に開始します。事前登録できますか？" + OPT)
    assert res.kind == "held" and res.stage == "facts"


# ---------------------------------------------------------------- 答えの文の計算の行

def test_computation_line_has_same_form_as_other_types():
    res = decide(q_apply_first("2026年11月9日", "2026年11月10日"))
    lines = res.answer_text.split("\n")
    i = lines.index("計算:")
    assert lines[i + 1] == "- 登録申請日 2026年11月9日 < 2026年11月10日（講座開始日） → 成り立つ"
    assert lines[i + 2] == "- 勤続期間 8か月 ≥ 6か月 → 成り立つ"
    assert res.decision_record["computation"][0]["threshold"] == "2026年11月10日（講座開始日）"
    assert any(l.startswith("- 講座開始日: 2026年11月10日（質問の「2026年11月10日」）") for l in lines)


# ---------------------------------------------------------------- 読み込みで拒む例

def reject(data):
    docs = Docs()
    with pytest.raises(RecordError) as e:
        build_record(copy.deepcopy(data), docs)
    return str(e.value)


def test_reject_base_role_missing():
    assert "facts に無い" in reject(rec(base="nope"))


def test_reject_base_is_not_a_date():
    msg = reject(rec(base="tenure"))
    assert "日付" in msg and "fact" in msg


def test_reject_compared_fact_is_not_a_date():
    d = rec(fact="tenure", base="start", tenure=False)
    d["conditions"][0]["evidence"] = ev("勤続期間が講座開始日より前")
    assert "日付" in reject(d)


def test_reject_base_is_itself():
    assert "自分自身" in reject(rec(base="apply"))


def test_reject_same_label_for_two_roles():
    twin = {**F_START, "label": "登録申請日"}
    assert "label" in reject(rec(facts=[F_APPLY, twin, F_TENURE]))


def test_reject_extra_or_missing_keys():
    d = rec()
    d["conditions"][0]["threshold"]["days"] = 3
    assert "知らない欄" in reject(d)
    d = rec()
    del d["conditions"][0]["threshold"]["base"]
    assert "欄が無い" in reject(d)


@pytest.mark.parametrize("op", [">", ">=", "<="])
def test_reject_op_disagrees_with_quote_phrase(op):
    assert "一致しない" in reject(rec(op=op, quote=ONE_SIDE, tenure=False))  # 「より前」は < だけ


def test_reject_quote_phrase_after_is_greater():
    assert "一致しない" in reject(rec(op="<", quote="登録申請日が講座開始日以後", tenure=False))


def test_reject_quote_without_base_name():
    # 引用に「講座開始日」が無い（名前が違う）。比べ方の言い回しはあっても拒否
    msg = reject(rec(quote="登録申請日が開始日より前", line=6, tenure=False))
    assert "base の名前「講座開始日」が無い" in msg


def test_reject_base_name_without_phrase_right_after():
    # 向きが逆の書き方（基準の名前の直後に比べ方の語が無い）
    assert "比べ方の語が無い" in reject(rec(quote="講座開始日が登録申請日より前", line=5, tenure=False))


def test_reject_when_base_name_only_inside_compared_facts_label():
    # 比べる事実の名前「登録申請日」の中の「申請日」は、基準の名前として数えない
    f_short = {**F_START, "role": "start", "label": "申請日"}
    msg = reject(rec(facts=[F_APPLY, f_short, F_TENURE], quote="登録申請日が講座開始日より前", tenure=False))
    assert "base の名前「申請日」が無い" in msg


def test_reject_both_sides_same_phrase_or_unrelated_pair():
    assert "矛盾" in reject(rec(quote=LINES[6], line=7, tenure=False))  # 「より前」が両側
    # 補集合でない組（より前 と 以前）。引用は本文の連続した一続きでなければならないので、本文にその形の行を足した版で確かめる
    text = DOC + "登録申請日が講座開始日より前の場合に登録可、講座開始日以前の場合に登録不可とする。\n"

    class D:
        def read_text(self, p):
            return text

    d = rec(quote="登録申請日が講座開始日より前の場合に登録可、講座開始日以前の場合に登録不可", line=9, tenure=False)
    d["doc_version"] = sha256_text(text)
    with pytest.raises(RecordError, match="どちらでもない"):
        build_record(d, D())


def test_reject_both_sides_with_reversed_conclusion():
    # 引用は「より前 → 登録不可」なのに、記録は「より前（成り立つ側）→ 登録可」
    assert "結論" in reject(rec(quote=LINES[7], line=8, tenure=False))
    # 比べ方も >= にして結論の側を合わせれば通る（「以後 → 登録可」が成り立つ側）
    assert build_record(rec(op=">=", quote=LINES[7], line=8, tenure=False, a="登録可", b="登録不可"), Docs())


def test_both_sides_quote_conclusion_side_checked_for_full_sentence():
    # 例の文の両側: 「以後 … 登録不可」は成り立たない側（any_false）の結論。記録が any_false を登録可にしていたら拒否
    assert "結論" in reject(rec(a="登録可", b="登録可"))
    assert "結論" in reject(rec(a="登録不可", b="登録可"))
    # 条件が 2 つの時、成り立つ側（all_true）は「この条件だけ」では決まらないので検査しない（成り立たない側だけ検査する）
    assert build_record(rec(a="登録不可", b="登録不可"), Docs())
    # 条件が 1 つなら、成り立つ側の結論も引用の「より前 … 登録可」と照らす
    assert "結論" in reject(rec(a="登録不可", b="登録不可", tenure=False))
    assert build_record(rec(tenure=False), Docs())


def test_stale_record_loads_structure_only():
    # 版が違う（文書が無い）記録は構造だけ検査して読み込む。構造の誤りは拒否
    d = rec()
    d["doc_version"] = "0" * 64
    assert build_record(d, Docs()).stale
    d = rec(base="nope")
    d["doc_version"] = "0" * 64
    with pytest.raises(RecordError):
        build_record(d, Docs())
