"""手がかりの語を含む節の読み取り（decide/polarity.py）の試験。

否定の語を一覧に足して通すのではなく、活用・接頭辞・範囲の言い回し・良い意味の否定・二重否定・「分からない」・
否定の付き先を、それぞれ別の形で確かめる。読めない形が黙って「満たしている」にならない事も固定する。
"""
from __future__ import annotations

import pytest

from decide import parse, polarity
from decide.polarity import EVENT, MET, REQUIREMENT, UNKNOWN, UNMET

CUE = ["領収書"]


def read(text, cues=CUE, mode=REQUIREMENT):
    r = polarity.read_cues(parse.clean_question(text), cues, mode)
    assert r is not None, text
    return r


def status(text, cues=CUE, mode=REQUIREMENT):
    return read(text, cues, mode).status


def test_no_cue_reads_none():
    assert polarity.read_cues(parse.clean_question("申請します。"), CUE) is None


def test_clause_bounds_ignores_thousand_separator_comma():
    t = "領収書は98,000円です。"
    a, b = polarity.clause_bounds(t, t.index("領収書"))
    assert t[a:b] == "領収書は98,000円です"


# ---- 3 件の不具合の文

def test_bug_cases():
    assert status("領収書はありません。") == UNMET
    assert status("領収書の添付は問題ない。") == MET
    assert status("受講報告書は受講から14日未満で提出済みです。", ["受講報告書"]) == MET


# ---- 否定の活用（どれも「満たしていない」）

@pytest.mark.parametrize("text", [
    "領収書はない。", "領収書がなく、困っています。", "領収書はなかった。", "領収書はありません。",
    "領収書はありませんでした。", "領収書を出せず困っています。", "領収書を出せずに申請しました。",
    "領収書は出せぬ。", "領収書は無い。", "領収書は無し。", "領収書はなし。", "領収書は持っていない。",
    "領収書は持っていません。", "領収書は添付していませんでした。", "領収書は添付しなかった。",
    "領収書がなくて申請しました。", "領収書は添付しないで出します。",
])
def test_negation_conjugations_are_unmet(text):
    assert status(text) == UNMET, text


# ---- 接頭辞（未・不・無）

@pytest.mark.parametrize("text,want", [
    ("領収書は未提出です。", UNMET), ("領収書は不備があります。", UNMET), ("領収書は未添付です。", UNMET),
    ("領収書は無記入です。", UNMET), ("領収書は無効と言われました。", UNMET),
    ("領収書は未確認です。", UNKNOWN), ("領収書は未定です。", UNKNOWN),
    ("領収書は無料で発行されました。", MET), ("領収書は未来の日付で発行されました。", MET),
    ("領収書は不動産会社の発行です。", MET),
])
def test_prefix_words(text, want):
    assert status(text) == want, text


# ---- 範囲の言い回し（否定ではない）

@pytest.mark.parametrize("text", [
    "領収書は購入から14日未満で提出済みです。", "領収書は3万円未満の分を提出済みです。",
    "領収書は購入から14日を超えない範囲で提出済みです。", "領収書は5万円以上のものを添付済みです。",
])
def test_range_phrases_are_not_negation(text):
    assert status(text) == MET, text


def test_range_word_does_not_hide_a_real_negation():
    assert status("購入から14日未満の領収書はありません。") == UNMET
    assert status("領収書は14日未満ですが、ありません。") == UNMET


def test_range_spans_come_from_parse_tables():
    t = parse.clean_question("購入から14日未満で提出済み")
    assert parse.range_spans(t) and t[slice(*parse.range_spans(t)[0])].startswith("未満")


# ---- 語の途中の「ない」「ず」「ぬ」は否定ではない

@pytest.mark.parametrize("text", [
    "領収書は内部の案内に沿って提出済みです。", "領収書は少ない枚数ですが提出済みです。",
    "領収書は必ず提出済みです。", "領収書はまず提出済みです。", "領収書はずっと保管済みです。",
    "領収書は危ないと聞いて提出済みです。", "領収書は内緒にして提出済みです。",
])
def test_mid_word_is_not_negation(text):
    assert status(text) == MET, text


# ---- 良い意味の否定

@pytest.mark.parametrize("text", [
    "領収書の添付は問題ない。", "領収書の添付は問題なし。", "領収書は支障ない。", "領収書は差し支えない。",
    "領収書は構わない。", "領収書は心配ない。", "領収書は間違いない。", "領収書の添付に問題はありません。",
    "領収書は不備がありません。", "領収書は紛失していない。", "領収書は忘れていません。",
])
def test_good_negations_are_met(text):
    assert status(text) == MET, text


# ---- 二重否定・分からない・確認中は「分からない」

@pytest.mark.parametrize("text", [
    "領収書がないわけではない。", "領収書はなくはない。", "領収書はないとは言えない。",
    "領収書があるかどうか分からない。", "領収書は分からない。", "領収書が分からなかった。", "領収書はわかりません。",
    "領収書は不明です。", "領収書は未定です。", "領収書は確認中です。", "領収書があるのかないのか不明。",
    "領収書はまだ分かっていない。", "領収書は添付しなければならないのか。", "領収書はないでしょうか。",
])
def test_unknown_readings(text):
    assert status(text) == UNKNOWN, text


def test_ambiguous_double_negation_is_never_met():
    for text in ("領収書がないことはない。", "領収書はなくもない。", "領収書がないとは限らない。"):
        assert status(text) != MET, text


# ---- 紛失・忘れ・欠く

@pytest.mark.parametrize("text", [
    "領収書を紛失しました。", "領収書を忘れました。", "領収書を失念しました。", "領収書が漏れています。",
    "領収書を欠いたまま申請します。", "領収書をなくしました。",
])
def test_loss_words_are_unmet(text):
    assert status(text) == UNMET, text


# ---- 否定の付き先（同じ節の別の語に付く否定は、手がかりの否定ではない）

@pytest.mark.parametrize("text,want", [
    ("領収書は添付済みだが承認はまだない。", MET),
    ("領収書は添付済みで、承認はありません。", MET),
    ("領収書は添付済みですが承認は取れていません。", MET),
    ("承認はないが領収書は添付済みです。", MET),
    ("承認はなく領収書は添付済みです。", MET),
    ("領収書も承認もない。", UNMET),
    ("領収書と承認はありません。", UNMET),
    ("領収書は、ありません。", UNMET),
    ("領収書の添付はなく、承認は済んでいます。", UNMET),
])
def test_negation_attaches_to_its_own_word(text, want):
    assert status(text) == want, text


def test_two_negations_on_the_cue_are_not_met():
    assert status("領収書もないし承認もありません") != MET


# ---- 例外の事情（EVENT）: 当たる／当たらない／分からない

def test_event_idiom_is_not_a_negation():
    r = read("やむを得ない事情がありました。", ["やむを得"], EVENT)
    assert r.status == MET  # 「やむを得ない」の「ない」は否定ではない


@pytest.mark.parametrize("text,want", [
    ("災害による遅延がありました。", MET),
    ("災害による遅延ではありません。", UNMET),
    ("災害ではなく寝坊です。", UNMET),
    ("災害は関係ない。", UNMET),
    ("災害かどうか分からない。", UNKNOWN),
    ("災害だったと思います。", MET),
    ("災害を理由にはしていません。", UNMET),
])
def test_event_readings(text, want):
    assert status(text, ["災害"], EVENT) == want, text


def test_event_negated_loss_words_stay_the_event():
    # 事情の文の中身の「紛失」は、事情が無い事を意味しない
    assert status("災害で書類を紛失しました。", ["災害"], EVENT) == MET


def test_event_idiom_negated_is_unmet():
    assert status("やむを得ない事情はありませんでした。", ["やむを得"], EVENT) == UNMET


# ---- まとめ方

def test_multiple_mentions_take_the_worse_reading():
    assert status("領収書は添付済みです。領収書を紛失しました。") == UNMET
    assert status("災害ではありません。別に災害がありました。", ["災害"], EVENT) == MET


def test_reading_carries_cue_clause_and_why():
    r = read("購入しました。領収書はありません。")
    assert r.cue == "領収書" and r.clause == "領収書はありません" and "ません" in r.why


# ---- 述語が読点の先にある書き方は、読めない形のまま「満たしている」にしない

@pytest.mark.parametrize("text,want", [
    ("領収書を、紛失しました。", UNMET),
    ("領収書については、ありません。", UNMET),
    ("領収書は、残念ながら、ありません。", UNMET),
    ("領収書は、現在、ありません。", UNMET),
    ("領収書は、確認中です。", UNKNOWN),
    ("領収書は、提出済みです。", MET),
    ("領収書については、問題ありません。", MET),
])
def test_predicate_after_comma(text, want):
    assert status(text) == want, text


@pytest.mark.parametrize("text,want", [
    ("領収書は14日未満ですが、ありません。", UNMET),  # 主題を省いた否定は手がかりの述語
    ("領収書は添付済みですが、承認はありません。", MET),                            # 自分の主題を持つ否定は別の語の物
    ("領収書は添付済みですが、承認がありません。", MET),
    ("領収書は添付済みで、問題ありません。", MET),
])
def test_omitted_topic_carries_over_the_connector(text, want):
    assert status(text) == want, text


# ---- 事情（EVENT）: 事情そのものに付く否定だけが「当たらない」。原因の「で」を挟んだ別の述語の否定は事情が述べられている

@pytest.mark.parametrize("text,cue", [
    ("やむを得ない事情で期限までに申請できませんでした。", "やむを得"),
    ("災害の影響で申請が間に合わなかった。", "災害"),
    ("災害で電車が動かず遅れました。", "災害"),
    ("災害で遅れて申請できなかった。", "災害"),
    ("災害があり、申請できませんでした。", "災害"),
])
def test_event_negation_on_another_predicate_keeps_the_event(text, cue):
    assert status(text, [cue], EVENT) == MET, text


@pytest.mark.parametrize("text", [
    "災害による遅延ではありません。", "災害は関係ありません。", "災害がなかった。", "災害を理由にしていない。",
    "災害に当たる事情ではありません。",
])
def test_event_negation_on_the_event_itself(text):
    assert status(text, ["災害"], EVENT) == UNMET, text
