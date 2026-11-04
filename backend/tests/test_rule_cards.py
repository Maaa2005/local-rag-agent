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
from datetime import date
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
