"""規程カード方式（app.services.rule_cards）と、数の比べの安い腕（境界の対比例・コード形式・数の正規化）の試験。

Issue #16。使う問は N の 12 問（evaluation/v2/numeric_boundary.jsonl）と開発用 24 問（numeric_dev24.jsonl）だけ。
確かめ用 48 問・封の問は使わない。カードは evaluation/v2/rule_cards/（規程の文から下書きした draft）。

小型モデルはこの PC で動かせないので、抽出役は「質問の文から正しく値と引用を拾う読み手」（_GoldReader）で
置き換える。ここで確かめるのは、カード・検査・判定のコードが正しい抽出から正しい結論を出すことと、
間違った抽出・カードでは保留・読み込み停止になること（停止試験）。モデルの抽出の当たり方は測っていない。
"""
from __future__ import annotations

import asyncio
import json
import re
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import settings
from app.services import llm, numeric_check, rule_cards
from app.services.rule_cards import CardError, decide, load_cards, parse_card, select_card, validate_extraction

ROOT = Path(__file__).resolve().parents[2]
CARDS_DIR = ROOT / "evaluation" / "v2" / "rule_cards"
QUESTIONS = [json.loads(line) for f in ("numeric_boundary.jsonl", "numeric_dev24.jsonl")
             for line in (ROOT / "evaluation" / "v2" / f).read_text(encoding="utf-8").splitlines() if line.strip()]
CARDS = load_cards(CARDS_DIR, allow_draft=True)
BY_ID = {c.id: c for c in CARDS}

# 問の rule_id（評価用の印）→ カード。試験の答え合わせだけに使い、実行時のコードには渡さない
RULE_TO_CARD = {
    "R1-expense-dept-head": "keihi-buchou-shounin", "R2-expense-deadline": "keihi-teishutsu-kigen",
    "R3-paid-leave-attendance": "shugyo-yukyu-fuyo", "R4-exec-bonus-achievement": "yakuin-gyoseki-shoyo",
    "D1": "bihin-soumubuchou", "D2": "fukuri-jutaku-teate", "D3": "fukuri-kenshin-kaizen", "D4": "fukuri-tsukin-teate",
    "D5": "bihin-shinsei-kigen", "D6": "bihin-eizoku-hyosho", "D7": "fukuri-ningen-dock", "D8": "fukuri-ikuji-shienkin",
}

_D = r"(\d{4}年\d{1,2}月\d{1,2}日)"
# カードごとの「正しい読み手」: (項目, 正規表現)。group(0) が引用、group(1) が値
GOLD = {
    "keihi-buchou-shounin": [("申請額", r"([\d,]+)円の申請")],
    "keihi-teishutsu-kigen": [("発生日", _D + "に発生した"), ("提出日", _D + "に経費精算システムから提出")],
    "shugyo-yukyu-fuyo": [("勤続月数", r"入社から(\d+)ヶ月が経過"), ("全労働日数", r"全労働日(\d+)日"),
                          ("出勤日数", r"出勤したのは(\d+)日")],
    "yakuin-gyoseki-shoyo": [("営業利益達成率", r"営業利益目標の達成率は(\d+)%"), ("ROE達成率", r"ROE目標の達成率も(\d+)%")],
    "bihin-soumubuchou": [("1件の金額", r"1件([\d,]+)円")],
    "bihin-shinsei-kigen": [("購入日", _D + "に購入した"), ("申請日", r"購入申請を" + _D)],
    "bihin-eizoku-hyosho": [("入社日", r"入社日は" + _D)],
    "fukuri-jutaku-teate": [("勤続年数", r"勤続年数はちょうど(\d+)年")],
    "fukuri-kenshin-kaizen": [("受診率", r"受診率は(\d+)%")],
    "fukuri-tsukin-teate": [("通勤距離", r"通勤距離は([\d.]+)km")],
    "fukuri-ningen-dock": [("受診日", r"人間ドックを" + _D)],
    "fukuri-ikuji-shienkin": [("誕生日", r"子どもが" + _D), ("申請日", r"申請を" + _D)],
}


def _value(slot: rule_cards.Slot, s: str):
    if slot.type == "date":
        y, m, d = map(int, re.findall(r"\d+", s))
        return date(y, m, d).isoformat()
    v = Decimal(s.replace(",", ""))
    return int(v) if v == v.to_integral() else float(v)


def gold_extraction(card: rule_cards.Card, question: str) -> dict:
    out = {name: {"value": None, "quote": None} for name in card.slots}
    for name, pat in GOLD[card.id]:
        m = re.search(pat, question)
        assert m, (card.id, name, question)
        out[name] = {"value": _value(card.slots[name], m.group(1)), "quote": m.group(0)}
    return out


class FakeExtractor:
    """抽出役の代わり。fn(card_id, question) の結果を返す（例外も投げられる）。"""

    def __init__(self, fn):
        self.fn = fn
        self.calls: list[tuple[list, dict]] = []

    async def extract(self, messages, schema):
        self.calls.append((messages, schema))
        return self.fn(messages, schema)


def run(coro):
    return asyncio.run(coro)


# ===== カードの読み込みと選び方 =====

def test_all_cards_load_and_are_drafts():
    assert len(CARDS) == 12 and all(c.status == "draft" for c in CARDS)
    # 実行時は確認済みだけ。下書きしかないので何も使わない
    assert load_cards(CARDS_DIR) == []


def test_card_quotes_are_in_the_regulation_documents():
    # カードの引用は規程の文書の文そのもの（下書きでも規程の外の文を混ぜない）
    roots = [ROOT / "evaluation" / "corpus", ROOT / "evaluation" / "v2" / "corpus_nx_dev"]
    for c in CARDS:
        texts = [(r / c.doc["source"]).read_text(encoding="utf-8") for r in roots if (r / c.doc["source"]).exists()]
        assert texts and any(c.doc["quote"] in t for t in texts), c.id


@pytest.mark.parametrize("q", QUESTIONS, ids=lambda q: q["id"])
def test_each_question_selects_its_card(q):
    card = select_card(CARDS, q["question"])
    assert card is not None and card.id == RULE_TO_CARD[q["rule_id"]]


def test_select_card_needs_exactly_one_match():
    q = QUESTIONS[0]["question"]
    assert select_card([], q) is None
    twin = parse_card(dict(BY_ID["keihi-buchou-shounin"].raw, id="twin"))
    assert select_card([BY_ID["keihi-buchou-shounin"], twin], q) is None
    assert select_card(CARDS, "経費精算で98,000円の申請をします。部門長の承認は要りますか？") is None  # 選択肢が無い


# ===== 正しい抽出 → コードの結論（N 12 問 + 開発用 24 問） =====

@pytest.mark.parametrize("q", QUESTIONS, ids=lambda q: q["id"])
def test_gold_extraction_gives_expected_conclusion(q):
    card = select_card(CARDS, q["question"])
    text, d = run(rule_cards.answer_with_card(card, q["question"],
                                              FakeExtractor(lambda m, s: gold_extraction(card, q["question"]))))
    assert d.conclusion == q["expected_conclusion"], (d.hold_reason, d.conditions)
    assert text.splitlines()[-1] == f"結論: {q['expected_conclusion']}"


def test_n09_ratio_uses_role_names_not_the_first_number():
    # N09: 全労働日 100 日のうち出勤 90 日。出勤率 0.9 ≥ 0.8 で付与（分子と分母の取り違えをしない）
    q = next(x for x in QUESTIONS if x["id"] == "N09")["question"]
    card = BY_ID["shugyo-yukyu-fuyo"]
    values, problems = validate_extraction(card, q, gold_extraction(card, q))
    assert problems == [] and values["出勤日数"] == 90 and values["全労働日数"] == 100
    assert decide(card, values).conclusion == "付与される"


# ===== 停止試験: 間違った抽出は保留 =====

N07 = next(x for x in QUESTIONS if x["id"] == "N07")["question"]
YUKYU = BY_ID["shugyo-yukyu-fuyo"]


def _with(**change):
    ex = gold_extraction(YUKYU, N07)
    for k, v in change.items():
        ex[k] = v
    return ex


@pytest.mark.parametrize("bad,why", [
    (_with(出勤日数={"value": 80, "quote": "出勤したのは79日"}), "not the number in its quote"),
    (_with(出勤日数={"value": 79, "quote": "出勤したのは79日間ずっと"}), "not in the question"),
    (_with(出勤日数={"value": 79, "quote": None}), "value without quote"),
    (_with(出勤日数={"value": "79", "quote": "出勤したのは79日"}), "not a number"),
    (_with(出勤日数={"value": 79}), "needs value and quote"),
    (_with(全労働日数={"value": 100, "quote": "出勤したのは79日"}), "not the number in its quote"),
    ("出勤日数は79日", "not an object"),
])
def test_wrong_extraction_is_held(bad, why):
    text, d = run(rule_cards.answer_with_card(YUKYU, N07, FakeExtractor(lambda m, s: bad)))
    assert d.conclusion is None and why in d.hold_reason
    assert text.splitlines()[-1] == "結論: 保留"


def test_invalid_value_in_an_unused_slot_does_not_hold():
    """N09: 出勤日数が書かれていれば欠勤日数は使わない。計算で埋めた欠勤日数は使わず、保留の理由にもしない。"""
    q = next(x for x in QUESTIONS if x["id"] == "N09")["question"]
    card = next(c for c in CARDS if c.id == "shugyo-yukyu-fuyo")
    raw = gold_extraction(card, q)
    raw["欠勤日数"] = {"value": 10, "quote": raw["出勤日数"]["quote"]}
    values, problems = rule_cards.validate_extraction(card, q, raw)
    assert problems == [] and values["欠勤日数"] is None
    # 使う枠が検査に通らない時は保留のまま
    raw["全労働日数"] = {"value": 999, "quote": raw["全労働日数"]["quote"]}
    assert rule_cards.validate_extraction(card, q, raw)[1]
    # 出勤日数が書かれていなければ欠勤日数を使うので保留のまま
    raw2 = gold_extraction(card, q)
    raw2["出勤日数"] = {"value": None, "quote": None}
    raw2["欠勤日数"] = {"value": 10, "quote": raw2["全労働日数"]["quote"]}
    assert rule_cards.validate_extraction(card, q, raw2)[1]


def test_swapped_roles_violate_the_constraint():
    # 出勤日数と全労働日数を取り違えた（どちらも引用は質問にある）→ 出勤日数 ≤ 全労働日数 に反して保留
    bad = _with(出勤日数={"value": 100, "quote": "全労働日100日"}, 全労働日数={"value": 79, "quote": "出勤したのは79日"})
    values, problems = validate_extraction(YUKYU, N07, bad)
    assert problems == []
    d = decide(YUKYU, values)
    assert d.conclusion is None and "筋が通らない" in d.hold_reason


def test_date_must_match_its_quote():
    q = next(x for x in QUESTIONS if x["id"] == "N04")["question"]
    card = BY_ID["keihi-teishutsu-kigen"]
    ex = gold_extraction(card, q)
    ex["提出日"] = {"value": "2026-10-11", "quote": ex["提出日"]["quote"]}
    _, problems = validate_extraction(card, q, ex)
    assert problems and "date" in problems[0]


def test_extractor_failure_is_held():
    def boom(m, s):
        raise TimeoutError("slow")
    text, d = run(rule_cards.answer_with_card(YUKYU, N07, FakeExtractor(boom)))
    assert d.conclusion is None and "抽出に失敗" in d.hold_reason and text.endswith("結論: 保留")


# ===== 不明の扱い（成立 / 不成立 / 不明） =====

def test_unknown_fact_still_decides_when_another_condition_fails():
    # 勤続の期間が書かれていなくても、出勤率が 8 割未満なら「付与されない」と決まる
    ex = _with(勤続月数={"value": None, "quote": None})
    values, problems = validate_extraction(YUKYU, N07, ex)
    d = decide(YUKYU, values)
    assert problems == [] and d.conclusion == "付与されない"


def test_unknown_fact_holds_when_it_could_change_the_answer():
    q = next(x for x in QUESTIONS if x["id"] == "N09")["question"]
    ex = gold_extraction(YUKYU, q)
    ex["勤続月数"] = {"value": None, "quote": None}
    values, _ = validate_extraction(YUKYU, q, ex)
    d = decide(YUKYU, values)
    assert d.conclusion is None and "勤続6ヶ月を経過した" in d.hold_reason


def test_derived_quantity_from_absences():
    q = "入社から6ヶ月が経過しました。この期間の全労働日は120日で、欠勤は30日でした。年次有給休暇は付与されますか？"
    ex = {"勤続月数": {"value": 6, "quote": "入社から6ヶ月が経過"}, "全労働日数": {"value": 120, "quote": "全労働日は120日"},
          "欠勤日数": {"value": 30, "quote": "欠勤は30日"}, "出勤日数": {"value": None, "quote": None}}
    values, problems = validate_extraction(YUKYU, q, ex)
    d = decide(YUKYU, values)
    assert problems == [] and d.values["出勤日数"] == 90 and d.conclusion == "付与されない"  # 90/120 = 0.75


def test_kanji_and_wide_numbers_in_the_question_are_normalized():
    q = "経費精算で十万五千円の申請をします。部門長の承認も必要ですか？最後の行には「結論: 必要」または「結論: 不要」のどちらかだけを書いてください。"
    card = BY_ID["keihi-buchou-shounin"]
    ex = {"申請額": {"value": 105000, "quote": "十万五千円の申請"}}
    values, problems = validate_extraction(card, q, ex)
    assert problems == [] and decide(card, values).conclusion == "必要"


# ===== 例外 =====

def _card_with_exception(if_unknown: str):
    raw = dict(BY_ID["keihi-buchou-shounin"].raw)
    raw["slots"] = dict(raw["slots"], 災害時={"type": "bool", "describe": "災害時の緊急の申請か"})
    raw["exceptions"] = [{"name": "災害時の緊急申請", "left": "災害時", "op": "==", "right": "あり",
                          "conclusion": "不要", "if_unknown": if_unknown}]
    return parse_card(raw)


def test_exception_overrides_and_unknown_follows_card_setting():
    c = _card_with_exception("hold")
    assert decide(c, {"申請額": Decimal(150000), "災害時": "あり"}).conclusion == "不要"
    assert decide(c, {"申請額": Decimal(150000), "災害時": "なし"}).conclusion == "必要"
    assert decide(c, {"申請額": Decimal(150000), "災害時": None}).conclusion is None
    assert decide(_card_with_exception("ignore"), {"申請額": Decimal(150000), "災害時": None}).conclusion == "必要"


# ===== 停止試験: おかしなカードは読み込みで止める =====

@pytest.mark.parametrize("change,why", [
    ({"format": "rule-card@0"}, "format"),
    ({"status": "confirmed"}, "confirmed_by"),
    ({"status": "maybe"}, "status"),
    ({"conclusion": {"then": "不要", "else": "必要"}}, "conclusion"),
    ({"conditions": [{"name": "x", "left": "申請金額", "op": ">", "right": 1}]}, "unknown slot"),
    ({"conditions": [{"name": "x", "left": "申請額", "op": "=>", "right": 1}]}, "unknown op"),
    ({"conditions": [{"name": "x", "left": {"pow": ["申請額", 2]}, "op": ">", "right": 1}]}, "unknown function"),
    ({"conditions": [{"name": "x", "left": {"sub": ["申請額"]}, "op": ">", "right": 1}]}, "takes 2"),
    ({"conditions": []}, "no conditions"),
    ({"slots": {"申請額": {"type": "money", "describe": "x"}}}, "slot"),
    ({"match": {"conclusions": ["必要", "必要"]}}, "two different"),
    ({"doc": {"source": "general/経費精算規程.md"}}, "quote"),
])
def test_malformed_card_is_rejected(change, why):
    raw = dict(BY_ID["keihi-buchou-shounin"].raw, **change)
    with pytest.raises(CardError, match=why):
        parse_card(raw)


def test_duplicate_card_ids_are_rejected(tmp_path):
    for name in ("a.json", "b.json"):
        (tmp_path / name).write_text(json.dumps(BY_ID["keihi-buchou-shounin"].raw, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(CardError, match="duplicate"):
        load_cards(tmp_path, allow_draft=True)


def test_wrong_card_holds_instead_of_answering():
    # カードの式が日付と数を比べる（カードの誤り）→ 例外にせず保留
    raw = dict(BY_ID["keihi-buchou-shounin"].raw,
               conditions=[{"name": "x", "left": "申請額", "op": ">", "right": {"date": "2026-01-01"}}])
    d = decide(parse_card(raw), {"申請額": Decimal(1)})
    assert d.conclusion is None and "計算できない" in d.hold_reason


# ===== 抽出の形と抽出役 =====

def test_extraction_schema_and_prompt():
    s = rule_cards.extraction_schema(YUKYU)
    assert set(s["required"]) == set(YUKYU.slots) and s["additionalProperties"] is False
    assert s["properties"]["出勤日数"]["properties"]["value"]["type"] == ["number", "null"]
    msgs = rule_cards.extraction_messages(YUKYU, N07)
    assert "最後の行" not in msgs[1]["content"] and "出勤日数" in msgs[0]["content"]


def test_openai_compat_extractor_sends_json_schema_at_temperature_zero():
    seen = {}

    class Completions:
        async def create(self, **kw):
            seen.update(kw)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"申請額": null}'))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    ex = rule_cards.OpenAICompatExtractor("http://x/v1", "small", client=client)
    out = run(ex.extract([{"role": "user", "content": "q"}], {"type": "object"}))
    assert out == {"申請額": None}
    assert seen["temperature"] == 0 and seen["model"] == "small"
    assert seen["response_format"]["type"] == "json_schema"
    assert seen["response_format"]["json_schema"]["schema"] == {"type": "object"}


# ===== 答えの組み立てにつないだとき =====

def _collect(agen):
    async def go():
        return [t async for t in agen]
    return run(go())


def test_stream_answer_uses_card_and_skips_the_model(monkeypatch):
    q = next(x for x in QUESTIONS if x["id"] == "N09")
    card = BY_ID["shugyo-yukyu-fuyo"]
    monkeypatch.setattr(settings, "rule_cards_enabled", True)
    monkeypatch.setattr(settings, "rule_cards_dir", str(CARDS_DIR))
    monkeypatch.setattr(settings, "rule_cards_allow_draft", True)
    llm.set_rule_card_extractor(FakeExtractor(lambda m, s: gold_extraction(card, q["question"])))
    called = []

    async def no_model(name):
        called.append(name)
        raise AssertionError("the model must not be called")
    monkeypatch.setattr(llm, "_resolve_provider", no_model)
    try:
        out = "".join(_collect(llm.stream_answer(q["question"], [])))
    finally:
        llm.set_rule_card_extractor(None)
    assert out.splitlines()[-1] == "結論: 付与される" and called == []


def test_stream_answer_falls_back_when_no_card_matches(monkeypatch):
    monkeypatch.setattr(settings, "rule_cards_enabled", True)
    monkeypatch.setattr(settings, "rule_cards_dir", str(CARDS_DIR))
    monkeypatch.setattr(settings, "rule_cards_allow_draft", True)

    class P:
        async def stream(self, messages):
            yield "モデルの答え"

    async def provider(name):
        return P()
    monkeypatch.setattr(llm, "_resolve_provider", provider)
    assert "".join(_collect(llm.stream_answer("在宅勤務の申請方法は？", []))) == "モデルの答え"


def test_cards_off_by_default():
    assert settings.rule_cards_enabled is False and settings.rule_cards_allow_draft is False


# ===== 安い腕: 境界の対比例・コード形式・数の正規化 =====

def _check(question: str, rule: str):
    return numeric_check.check(question, [rule])


def test_contrast_lines_for_money_percent_and_dates():
    r = _check("1件98,000円の備品を買います。", "1件あたり10万円を超える備品は承認を要する。")
    assert numeric_check.contrast_lines(r)[1:] == ["- 99,999円 なら当てはまらない", "- 100,000円 なら当てはまらない",
                                                    "- 100,001円 なら当てはまる"]
    r = _check("受診率は1.8%でした。", "受診率が2%以下の部署は報告する。")
    assert numeric_check.contrast_lines(r)[1:] == ["- 1.9% なら当てはまる", "- 2% なら当てはまる", "- 2.1% なら当てはまらない"]
    assert numeric_check.contrast_lines(numeric_check.CheckResult(hold_reason="x")) == []


def test_code_form_is_not_executed_text():
    r = _check("1件98,000円の備品を買います。", "1件あたり10万円を超える備品は承認を要する。")
    code = numeric_check.code_form(r)
    assert "境目 = 100000" in code and "値 = 98000" in code and "当てはまる = 値 > 境目  # → False" in code
    assert numeric_check.code_form(numeric_check.CheckResult(hold_reason="x")) is None


def test_arms_are_added_to_the_calculation_block_only_when_enabled(monkeypatch):
    chunks = [{"content": "1件あたり10万円を超える備品は承認を要する。", "source_file": "a.md"}]
    q = "1件98,000円の備品を買います。承認は要りますか？"
    monkeypatch.setattr(settings, "numeric_check_enabled", True)
    base = llm._build_messages(q, chunks)[1]["content"]
    assert "境目の前後" not in base and "コードにすると" not in base
    monkeypatch.setattr(settings, "numeric_contrast_enabled", True)
    monkeypatch.setattr(settings, "numeric_codeform_enabled", True)
    both = llm._build_messages(q, chunks)[1]["content"]
    assert "境目の前後" in both and "当てはまる = 値 > 境目" in both


def test_normalize_arm_rewrites_question_and_context(monkeypatch):
    chunks = [{"content": "1件あたり十万円を超える備品は承認を要する。", "source_file": "a.md"}]
    q = "１件９万８千円の備品を買います。承認は要りますか？"
    off = llm._build_messages(q, chunks)[1]["content"]
    monkeypatch.setattr(settings, "numeric_normalize_enabled", True)
    on = llm._build_messages(q, chunks)[1]["content"]
    assert "十万円" in off and "100,000円" in on and "98,000円" in on


# ===== 時刻・min/max・月末・夜またぎ・入れ子の条件・例外の結論が同じ時（rule-card@1 の足した部品） =====

from app.services.rule_cards import Clock  # noqa: E402


def _mk(slots, conditions, *, derive=None, exceptions=None, combine="all", conclusions=("対象", "対象外"), **over):
    """試験用の小さなカード。"""
    raw = {"format": "rule-card@1", "id": "t-card", "status": "draft", "drafted_by": "x", "confirmed_by": None,
           "doc": {"source": "架空規程.md", "section": "第1条", "quote": "架空の定め。"}, "target": "架空",
           "match": {"conclusions": list(conclusions), "keywords": ["架空"]}, "slots": slots, "conditions": conditions,
           "combine": combine, "conclusion": {"then": conclusions[0], "else": conclusions[1]}}
    if derive:
        raw["derive"] = derive
    if exceptions:
        raw["exceptions"] = exceptions
    raw.update(over)
    return parse_card(raw)


def _leaf(name, left, op, right, **kw):
    return {"name": name, "left": left, "op": op, "right": right, **kw}


TIME_SLOT = {"退勤": {"type": "time", "describe": "退勤した時刻"}}
D = Decimal


def _after_22():
    return _mk(TIME_SLOT, [_leaf("22時以降", "退勤", ">=", {"time": "22:00"})])


def test_time_slot_extracts_and_compares_in_minutes():
    c = _after_22()
    q = "昨日は22時ちょうどに退勤しました。深夜手当の対象ですか？"
    values, problems = validate_extraction(c, q, {"退勤": {"value": "22:00", "quote": "22時ちょうどに退勤"}})
    assert problems == [] and values["退勤"] == Clock(22 * 60)
    assert decide(c, values).conclusion == "対象"  # 以降は >=（ちょうどを含む）
    assert decide(c, {"退勤": Clock(21 * 60 + 59)}).conclusion == "対象外"
    assert decide(c, {"退勤": None}).conclusion is None
    s = rule_cards.extraction_schema(c)["properties"]["退勤"]["properties"]["value"]
    assert s["type"] == ["string", "null"] and re.fullmatch(s["pattern"], "23:59") and not re.fullmatch(s["pattern"], "24:00")
    assert "HH:MM" in rule_cards.extraction_messages(c, q)[0]["content"]


@pytest.mark.parametrize("quote,value", [
    ("22時に退勤", "22:00"), ("22時30分に退勤", "22:30"), ("22:30に退勤", "22:30"), ("午後10時に退勤", "22:00"),
    ("午前9時に退勤", "09:00"), ("正午に退勤", "12:00"), ("午後10時30分に退勤", "22:30"), ("22時半に退勤", "22:30"),
    ("０時に退勤", "00:00"), ("２２：３０に退勤", "22:30"),
])
def test_time_readings(quote, value):
    values, problems = validate_extraction(_after_22(), quote + "しました。", {"退勤": {"value": value, "quote": quote}})
    assert problems == [] and values["退勤"] == rule_cards.parse_time(value)


@pytest.mark.parametrize("quote,value", [
    ("22時30分に退勤", "22:00"),        # 引用の時刻と違う
    ("22時に退勤", "10:00"),            # 午前午後の取り違え
    ("午後10時に退勤", "10:00"),
    ("午前9時に退勤", "21:00"),
    ("24時に退勤", "24:00"),            # 24:00 は無効
    ("24時に退勤", "00:00"),            # 24時を 0 時と読み替えない
    ("午前12時に退勤", "00:00"),        # 午前12時は読まない（曖昧）
    ("3時間働いて退勤", "03:00"),       # 「3時間」は時刻ではない
    ("22時に退勤", "22:00:00"), ("22時に退勤", "2200"), ("22時に退勤", 22), ("22時に退勤", "25:00"),
])
def test_time_value_must_match_its_quote(quote, value):
    _, problems = validate_extraction(_after_22(), quote + "しました。", {"退勤": {"value": value, "quote": quote}})
    assert problems and "time" in problems[0]


def test_time_card_shape_is_checked():
    with pytest.raises(CardError, match="bad time"):
        _mk(TIME_SLOT, [_leaf("x", "退勤", ">=", {"time": "24:00"})])
    for bad in ("9:00", "22:60", "22時", 2200):
        with pytest.raises(CardError, match="bad time"):
            _mk(TIME_SLOT, [_leaf("x", "退勤", ">=", {"time": bad})])
    # 時刻と数を比べるカードの誤りは保留（例外にしない）
    d = decide(_mk(TIME_SLOT, [_leaf("x", "退勤", ">=", 22)]), {"退勤": Clock(1320)})
    assert d.conclusion is None and "計算できない" in d.hold_reason
    assert "22:30" in rule_cards.render_answer(_after_22(), decide(_after_22(), {"退勤": Clock(22 * 60 + 30)}))


def test_min_and_max():
    slots = {"請求額": {"type": "number", "unit": "円", "describe": "請求額"},
             "実費": {"type": "number", "unit": "円", "describe": "実費"}}
    c = _mk(slots, [_leaf("上限以内", "請求額", "<=", {"min": ["実費", 8000]})])
    assert decide(c, {"請求額": D(8000), "実費": D(9000)}).conclusion == "対象"   # 小さい方は 8000
    assert decide(c, {"請求額": D(8001), "実費": D(9000)}).conclusion == "対象外"
    assert decide(c, {"請求額": D(6001), "実費": D(6000)}).conclusion == "対象外"   # 小さい方は実費
    assert decide(c, {"請求額": D(6000), "実費": None}).conclusion is None            # 1 つでも不明なら不明
    cmax = _mk(slots, [_leaf("x", "請求額", ">=", {"max": ["実費", 8000]})])
    assert decide(cmax, {"請求額": D(9000), "実費": D(9000)}).conclusion == "対象"
    assert decide(cmax, {"請求額": D(8999), "実費": D(9000)}).conclusion == "対象外"
    assert decide(cmax, {"請求額": D(9000), "実費": None}).conclusion is None
    for bad in ({"min": ["実費"]}, {"max": []}, {"min": "実費"}):
        with pytest.raises(CardError):
            _mk(slots, [_leaf("x", "請求額", "<=", bad)])
    # 型の違う物を混ぜたカードの誤りは保留
    tc = _mk({**slots, **TIME_SLOT}, [_leaf("x", "請求額", "<=", {"min": ["実費", "退勤"]})])
    assert decide(tc, {"請求額": D(1), "実費": D(1), "退勤": Clock(60)}).conclusion is None


DATE_SLOTS = {"期限日": {"type": "date", "describe": "期限の日"}, "提出日": {"type": "date", "describe": "提出した日"}}


def _month_end_card():
    return _mk(DATE_SLOTS, [_leaf("翌月末まで", "提出日", "<=", {"month_end": {"add_months": ["期限日", 1]}})])


@pytest.mark.parametrize("deadline,submitted,expected", [
    ("2026-01-15", "2026-02-28", "対象"), ("2026-01-15", "2026-03-01", "対象外"),   # 翌月末は 2 月 28 日
    ("2028-01-15", "2028-02-29", "対象"), ("2028-01-15", "2028-03-01", "対象外"),   # うるう年は 29 日
    ("2026-12-10", "2027-01-31", "対象"), ("2026-12-10", "2027-02-01", "対象外"),   # 年をまたぐ
])
def test_month_end_of_next_month(deadline, submitted, expected):
    values = {"期限日": date.fromisoformat(deadline), "提出日": date.fromisoformat(submitted)}
    assert decide(_month_end_card(), values).conclusion == expected
    assert decide(_month_end_card(), {"期限日": None, "提出日": values["提出日"]}).conclusion is None


@pytest.mark.parametrize("start,k,expected", [
    ("2026-01-31", 1, "2026-02-28"), ("2028-01-31", 1, "2028-02-29"), ("2026-03-31", 1, "2026-04-30"),
    ("2026-01-30", 1, "2026-02-28"), ("2026-01-15", 1, "2026-02-15"), ("2026-11-30", 3, "2027-02-28"),
    ("2026-03-31", -1, "2026-02-28"), ("2026-01-31", 12, "2027-01-31"),
])
def test_add_months_rounds_to_month_end(start, k, expected):
    c = _mk(DATE_SLOTS, [_leaf("x", "提出日", "==", {"add_months": ["期限日", k]})])
    assert decide(c, {"期限日": date.fromisoformat(start), "提出日": date.fromisoformat(expected)}).conclusion == "対象"
    wrong = date.fromisoformat(expected) + timedelta(days=1)
    assert decide(c, {"期限日": date.fromisoformat(start), "提出日": wrong}).conclusion == "対象外"
    e = _mk(DATE_SLOTS, [_leaf("x", "提出日", "==", {"month_end": "期限日"})])
    assert decide(e, {"期限日": date(2026, 2, 3), "提出日": date(2026, 2, 28)}).conclusion == "対象"


def test_month_end_and_add_months_misuse_is_held_or_rejected():
    n = {"回数": {"type": "number", "describe": "回数"}, **DATE_SLOTS}
    d = decide(_mk(n, [_leaf("x", "提出日", "==", {"month_end": "回数"})]), {"回数": D(3), "提出日": date(2026, 1, 1)})
    assert d.conclusion is None and "計算できない" in d.hold_reason
    d = decide(_mk(n, [_leaf("x", "提出日", "==", {"add_months": ["期限日", 1.5]})]),
               {"期限日": date(2026, 1, 1), "提出日": date(2026, 1, 1)})
    assert d.conclusion is None and "計算できない" in d.hold_reason
    with pytest.raises(CardError, match="takes 2"):
        _mk(n, [_leaf("x", "提出日", "==", {"add_months": ["期限日"]})])


MIN_SLOTS = {"開始": {"type": "time", "describe": "開始"}, "終了": {"type": "time", "describe": "終了"}}


def _span_card():
    return _mk(MIN_SLOTS, [_leaf("4 時間以上", {"minutes_between": ["開始", "終了"]}, ">=", 240)])


@pytest.mark.parametrize("start,end,expected", [
    ("09:00", "13:00", "対象"), ("09:00", "12:59", "対象外"),
    ("22:00", "02:00", "対象"),    # 夜をまたぐ: 4 時間
    ("22:00", "01:59", "対象外"),
    ("23:30", "00:00", "対象外"),  # 30 分
    ("10:00", "10:00", "対象外"),  # 同時刻は 0 分（24 時間にしない）
])
def test_minutes_between_wraps_over_midnight(start, end, expected):
    v = {"開始": rule_cards.parse_time(start), "終了": rule_cards.parse_time(end)}
    assert decide(_span_card(), v).conclusion == expected
    assert decide(_span_card(), {**v, "終了": None}).conclusion is None


def test_minutes_between_needs_times():
    c = _mk({**MIN_SLOTS, "数": {"type": "number", "describe": "x"}},
            [_leaf("x", {"minutes_between": ["開始", "数"]}, ">=", 1)])
    d = decide(c, {"開始": Clock(0), "終了": None, "数": D(1)})
    assert d.conclusion is None and "計算できない" in d.hold_reason


# ----- 入れ子の条件（3 値） -----

B = {"a": {"type": "bool", "describe": "a"}, "b": {"type": "bool", "describe": "b"}}


def _group(kind, name="g"):
    return {"name": name, kind: [_leaf("a", "a", "==", "あり"), _leaf("b", "b", "==", "あり")]}


@pytest.mark.parametrize("kind,a,b,expected", [
    ("all", "あり", "あり", "対象"), ("all", "あり", "なし", "対象外"), ("all", "なし", None, "対象外"),   # 1 つでも不成立なら不成立
    ("all", "あり", None, None), ("all", None, None, None),                                          # 残りは不明
    ("any", "なし", "あり", "対象"), ("any", None, "あり", "対象"), ("any", "なし", "なし", "対象外"),  # 1 つでも成立なら成立
    ("any", "なし", None, None), ("any", None, None, None),
])
def test_nested_conditions_are_three_valued(kind, a, b, expected):
    c = _mk(B, [_group(kind)])
    assert decide(c, {"a": a, "b": b}).conclusion == expected


def test_nested_conditions_combine_with_siblings_and_nest_deeper():
    slots = {**B, "c": {"type": "bool", "describe": "c"}}
    deep = {"name": "deep", "all": [_group("any", "inner"), _leaf("c", "c", "==", "あり")]}
    c = _mk(slots, [deep])
    assert decide(c, {"a": None, "b": "あり", "c": "あり"}).conclusion == "対象"
    assert decide(c, {"a": None, "b": None, "c": "なし"}).conclusion == "対象外"
    assert decide(c, {"a": None, "b": None, "c": "あり"}).conclusion is None
    c2 = _mk(slots, [_group("any"), _leaf("c", "c", "==", "あり")], combine="all")
    assert decide(c2, {"a": "あり", "b": None, "c": "なし"}).conclusion == "対象外"


def test_nested_exception_and_unused_slot_detection():
    slots = {"金額": {"type": "number", "unit": "円", "describe": "金額"}, **B}
    c = _mk(slots, [_leaf("5万円超", "金額", ">", 50000)],
            exceptions=[{**_group("any", "例外"), "conclusion": "対象外", "if_unknown": "hold"}])
    assert decide(c, {"金額": D(60000), "a": "なし", "b": "あり"}).conclusion == "対象外"
    assert decide(c, {"金額": D(60000), "a": "なし", "b": "なし"}).conclusion == "対象"
    assert decide(c, {"金額": D(60000), "a": "なし", "b": None}).conclusion is None
    # 入れ子の中の枠も「使う枠」と数える（抽出の検査に通らなければ保留）
    q = "金額は60,000円です。"
    ex = {"金額": {"value": 60000, "quote": "60,000円"}, "a": {"value": "あり", "quote": "ある"},
          "b": {"value": None, "quote": None}}
    assert validate_extraction(c, q, ex)[1]


@pytest.mark.parametrize("bad", [
    {"name": "g", "all": []}, {"name": "g", "all": [_leaf("a", "a", "==", "あり")], "any": [_leaf("b", "b", "==", "あり")]},
    {"name": "g", "all": [_leaf("a", "a", "==", "あり")], "left": "a"}, {"name": "g", "all": "a"},
    {"name": "g", "all": [{"name": "x", "left": "zzz", "op": "==", "right": "あり"}]},
    {"name": "g", "any": [{"name": "x", "left": "a", "op": "=>", "right": "あり"}]},
    {"name": "g", "all": [_leaf("a", "a", "==", "あり", quote="")]},
])
def test_bad_nested_conditions_are_rejected(bad):
    with pytest.raises(CardError):
        _mk(B, [bad])


def test_nesting_depth_is_limited():
    node = _leaf("a", "a", "==", "あり")
    for i in range(8):
        node = {"name": f"n{i}", "all": [node]}
    with pytest.raises(CardError, match="too deep"):
        _mk(B, [node])


def test_quote_on_conditions_and_exceptions_is_optional_but_checked_for_shape():
    c = _mk(B, [_leaf("a", "a", "==", "あり", quote="根拠の文。")],
            exceptions=[_leaf("b", "b", "==", "あり", quote="ただし書き。", conclusion="対象外")])
    assert c.conditions[0]["quote"] == "根拠の文。" and decide(c, {"a": "あり", "b": "なし"}).conclusion == "対象"
    for bad in ({"quote": 5}, {"quote": ""}, {"quote": None}):
        with pytest.raises(CardError):
            _mk(B, [_leaf("a", "a", "==", "あり", **bad)])
    with pytest.raises(CardError):
        _mk(B, [_leaf("a", "a", "==", "あり")], exceptions=[_leaf("b", "b", "==", "あり", quote=7, conclusion="対象外")])


# ----- decide: 例外が不明でも結論が同じなら決める -----

EXC_SLOTS = {"金額": {"type": "number", "unit": "円", "describe": "金額"}, "緊急": {"type": "bool", "describe": "緊急か"}}


def _exc_card(exc_conclusion, *, if_unknown="hold", conds=None):
    return _mk(EXC_SLOTS, conds or [_leaf("5万円超", "金額", ">", 50000)], conclusions=("必要", "不要"),
               exceptions=[_leaf("緊急", "緊急", "==", "あり", conclusion=exc_conclusion, if_unknown=if_unknown)])


def test_unknown_exception_with_same_conclusion_does_not_hold():
    # 例外（緊急）の結論が「不要」。金額が 5 万円以下なら例外が無くても「不要」なので、緊急かどうか不明でも決まる
    c = _exc_card("不要")
    assert decide(c, {"金額": D(50000), "緊急": None}).conclusion == "不要"
    # 5 万円超なら例外の有無で結論が変わるので保留のまま
    d = decide(c, {"金額": D(50001), "緊急": None})
    assert d.conclusion is None and "例外" in d.hold_reason
    # 例外が無い時の結論が不明（金額も不明）なら、決まらない
    assert decide(c, {"金額": None, "緊急": None}).conclusion is None
    # 例外の結論が「必要」で、条件が成り立つ（必要）なら例外が不明でも決まる
    c2 = _exc_card("必要")
    assert decide(c2, {"金額": D(60000), "緊急": None}).conclusion == "必要"
    assert decide(c2, {"金額": D(50000), "緊急": None}).conclusion is None
    # 例外が成り立てば（不明でなければ）従来どおりその結論
    assert decide(c, {"金額": D(60000), "緊急": "あり"}).conclusion == "不要"
    assert decide(c, {"金額": D(60000), "緊急": "なし"}).conclusion == "必要"


def test_true_exception_decides_even_when_conditions_are_unknown():
    c = _exc_card("不要")
    assert decide(c, {"金額": None, "緊急": "あり"}).conclusion == "不要"
    # 例外が 2 つ。先の例外が不明（結論 不要）・後の例外が成立（結論 不要）→ 不要。後が「必要」なら保留
    two = _mk(EXC_SLOTS, [_leaf("5万円超", "金額", ">", 50000)], conclusions=("必要", "不要"), exceptions=[
        _leaf("緊急", "緊急", "==", "あり", conclusion="不要"),
        _leaf("小額", "金額", "<=", 1000, conclusion="不要")])
    assert decide(two, {"金額": D(500), "緊急": None}).conclusion == "不要"
    two2 = _mk(EXC_SLOTS, [_leaf("5万円超", "金額", ">", 50000)], conclusions=("必要", "不要"), exceptions=[
        _leaf("緊急", "緊急", "==", "あり", conclusion="不要"),
        _leaf("小額", "金額", "<=", 1000, conclusion="必要")])
    assert decide(two2, {"金額": D(500), "緊急": None}).conclusion is None


def test_ignore_exception_still_ignores_unknown():
    assert decide(_exc_card("不要", if_unknown="ignore"), {"金額": D(60000), "緊急": None}).conclusion == "必要"


def test_true_exception_skips_condition_evaluation():
    # 成立する例外があれば、条件の式が計算できなくても例外の結論で決まる（従来どおり）
    c = _mk(EXC_SLOTS, [_leaf("x", "金額", ">", {"date": "2026-01-01"})], conclusions=("必要", "不要"),
            exceptions=[_leaf("緊急", "緊急", "==", "あり", conclusion="不要")])
    assert decide(c, {"金額": D(1), "緊急": "あり"}).conclusion == "不要"


# ----- 同じ文書で結論の語の組が同じカード -----

def _write(dirpath, cid, source, concl, keywords):
    raw = dict(BY_ID["keihi-buchou-shounin"].raw, id=cid, doc={"source": source, "quote": "q"},
               match={"conclusions": list(concl), "keywords": list(keywords)},
               conclusion={"then": concl[0], "else": concl[1]})
    (dirpath / f"{cid}.json").write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")


def test_same_source_same_pair_is_rejected(tmp_path):
    _write(tmp_path, "a", "規程A.md", ("必要", "不要"), ["承認"])
    _write(tmp_path, "b", "規程A.md", ("不要", "必要"), ["部長"])   # 語の順が逆でも同じ組。語が重ならなければ許す
    assert len(load_cards(tmp_path, allow_draft=True)) == 2
    _write(tmp_path, "c", "規程A.md", ("必要", "不要"), ["承認の手続"])   # 「承認」を含むので重なる
    with pytest.raises(CardError, match="same conclusions pair"):
        load_cards(tmp_path, allow_draft=True)


def test_same_pair_is_allowed_across_sources_and_with_distinct_keywords(tmp_path):
    _write(tmp_path, "a", "規程A.md", ("必要", "不要"), ["承認"])
    _write(tmp_path, "b", "規程B.md", ("必要", "不要"), ["承認"])
    assert len(load_cards(tmp_path, allow_draft=True)) == 2
    _write(tmp_path, "c", "規程A.md", ("必要", "不要"), ["稟議"])
    assert len(load_cards(tmp_path, allow_draft=True)) == 3


def test_same_source_same_pair_without_keywords_is_rejected(tmp_path):
    _write(tmp_path, "a", "規程A.md", ("必要", "不要"), ["承認"])
    _write(tmp_path, "b", "規程A.md", ("必要", "不要"), [])
    with pytest.raises(CardError, match="same conclusions pair"):
        load_cards(tmp_path, allow_draft=True)


def test_current_twelve_cards_still_load_with_the_pair_check():
    # 福利厚生規程の住宅手当と人間ドックは同じ組だが keywords が重ならないので許される
    pairs = [c for c in CARDS if c.doc["source"].endswith("福利厚生規程.md") and set(c.conclusions) == {"対象になる", "対象にならない"}]
    assert {c.id for c in pairs} == {"fukuri-jutaku-teate", "fukuri-ningen-dock"}
    assert len(load_cards(CARDS_DIR, allow_draft=True)) == 12
