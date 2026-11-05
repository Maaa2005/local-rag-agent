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


# ===== カードの選び方: 文字 2-gram の点・下限・差（誤選択 0 を優先し、迷う時は答えない） =====

_OPT_YN = "最後の行には「結論: 必要」または「結論: 不要」のどちらかだけを書いてください。"
DEV90_QUESTIONS = ROOT / "evaluation" / "v2" / "numeric_dev90" / "questions.jsonl"
DEV90_CARDS = Path.home() / ".cache" / "lr-gpu" / "dev90-2" / "cards"


def test_select_constants_are_the_counted_values():
    # 値を変えたら「正しく選ぶ・誤選択・答えない」を数え直す（docs/v2-rule-card-procedure.md）
    assert (rule_cards.SELECT_MIN_SCORE, rule_cards.SELECT_MIN_MARGIN, rule_cards.SELECT_KEYWORD_WEIGHT) == (14, 8, 3)


def test_select_weights_use_source_quote_slots_conditions_and_keywords():
    c = BY_ID["keihi-buchou-shounin"]
    w = rule_cards.card_select_weights(c)
    assert w["部門"] == rule_cards.SELECT_KEYWORD_WEIGHT          # keywords「部門長」
    assert w["経費"] == 1 and w["精算"] == 1                       # doc.source「経費精算規程.md」
    assert rule_cards.select_score(c, QUESTIONS[0]["question"]) >= rule_cards.SELECT_MIN_SCORE


def test_select_card_needs_a_floor_even_for_a_single_card():
    c = BY_ID["keihi-buchou-shounin"]
    assert select_card([c], QUESTIONS[0]["question"]) is c
    # 結論の語の組は合うが、規程の語が質問に無い（別の規程の問）→ 1 枚だけでも選ばない
    other = "旅行の積立金が3万円あります。友人に貸しても大丈夫ですか？" + _OPT_YN
    assert rule_cards.select_score(c, other) < rule_cards.SELECT_MIN_SCORE
    assert select_card([c], other) is None and select_card(CARDS, other) is None


def test_select_card_needs_a_margin_over_the_second():
    c = BY_ID["keihi-buchou-shounin"]
    q = QUESTIONS[0]["question"]
    near = parse_card(dict(c.raw, id="near", keywords=["部門長", "経費"]))
    assert select_card([c, near], q) is None                     # 同点
    far = parse_card(dict(c.raw, id="far", doc={**c.raw["doc"], "source": "x.md", "quote": "別の文。"},
                          slots={"額": {"type": "number", "unit": "円", "describe": "金"}},
                          conditions=[{"name": "条", "left": "額", "op": ">", "right": 1}], keywords=["別"]))
    assert select_card([far, c], q) is c and select_card([c, far], q) is c


@pytest.mark.parametrize("q", QUESTIONS, ids=lambda q: q["id"])
def test_no_wrong_card_is_picked_when_the_right_card_is_absent(q):
    # 規程のカードが無い（作れなかった）問: 同じ語の組の別の規程のカードを選ばない
    rest = [c for c in CARDS if c.id != RULE_TO_CARD[q["rule_id"]]]
    assert select_card(rest, q["question"]) is None


def test_dev90_questions_pick_none_of_the_other_rules_cards():
    rows = [json.loads(line) for line in DEV90_QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 90
    assert [r["id"] for r in rows if select_card(CARDS, r["question"]) is not None] == []


@pytest.mark.skipif(not DEV90_CARDS.is_dir(), reason="dev90 のカードは手元の置き場にだけある")
def test_dev90_selection_table_has_no_wrong_selection():
    cards = [rule_cards.load_card(p) for p in sorted(DEV90_CARDS.glob("*.json")) if p.name != "unable.json"]
    rows = [json.loads(line) for line in DEV90_QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    right = wrong = none = 0
    for r in rows:
        want = [c.id for c in cards if c.doc["source"] == r["rule_id"]
                and set(c.conclusions) == set(rule_cards.conclusion_options(r["question"]))]
        got = select_card(cards, r["question"])
        if got is None:
            none += 1
        elif [got.id] == want:
            right += 1
        else:
            wrong += 1
    assert wrong == 0 and right >= 88 and none <= 2   # 旧: 正 60・答えない 30


# ===== 値の照合: 時間の複合・年のない日付・derive の先・0 =====

def _dur_card(unit):
    return _mk({"運行時間": {"type": "number", "unit": unit, "describe": "運行の時間"}},
               [_leaf("6時間以内", "運行時間", "<=", 360 if unit == "分" else 6)])


@pytest.mark.parametrize("unit,quote,value", [
    ("分", "6時間10分", 370), ("分", "3時間半", 210), ("分", "8時間", 480), ("分", "1時間30分", 90), ("分", "45分", 45),
    ("時間", "6時間10分", 6.17), ("時間", "6時間10分", 6.16), ("時間", "3時間半", 3.5), ("時間", "8時間", 8),
    ("時間", "2.5時間", 2.5),
])
def test_duration_compound_matches_its_unit(unit, quote, value):
    q = f"社用車を{quote}使いました。"
    values, problems = validate_extraction(_dur_card(unit), q, {"運行時間": {"value": value, "quote": quote}})
    assert problems == [] and values["運行時間"] == Decimal(str(value))


@pytest.mark.parametrize("unit,quote,value", [
    ("分", "6時間10分", 10), ("分", "6時間10分", 6), ("分", "6時間10分", 371), ("分", "3時間半", 30), ("分", "8時間", 8),
    ("時間", "6時間10分", 6), ("時間", "6時間10分", 6.2), ("時間", "6時間10分", 10), ("時間", "3時間半", 3),
    ("時間", "3時間半", 4), ("時間", "8時間", 480),
])
def test_duration_compound_rejects_a_misread_part(unit, quote, value):
    q = f"社用車を{quote}使いました。"
    _, problems = validate_extraction(_dur_card(unit), q, {"運行時間": {"value": value, "quote": quote}})
    assert problems and "not the number" in problems[0]


def _span_dates(order=("開始", "終了")):
    slots = {n: {"type": "date", "describe": n + "した日"} for n in order}
    return _mk(slots, [_leaf("20日以内", {"sub": ["終了", "開始"]}, "<=", 20)])


def _dates_extraction(start, end):
    return {"開始": {"value": start[0], "quote": start[1]}, "終了": {"value": end[0], "quote": end[1]}}


def test_yearless_dates_across_the_new_year_use_the_nearest_year_and_ignore_the_models_year():
    q = "12月18日に始めて、1月5日に終えました。"
    ex = _dates_extraction(("2031-12-18", "12月18日に始めて"), ("1999-01-05", "1月5日に終えました"))
    for order in (("開始", "終了"), ("終了", "開始")):   # 枠の順が逆でも差は同じ（18 日）
        c = _span_dates(order)
        values, problems = validate_extraction(c, q, ex)
        assert problems == [] and abs((values["終了"] - values["開始"]).days) == 18
        assert (values["終了"] - values["開始"]).days == 18
        assert decide(c, values).conclusion == "対象"
    # 最初の年なし日付は固定の年（閏年）に置く
    values, _ = validate_extraction(_span_dates(), q, ex)
    assert values["開始"] == date(rule_cards.YEARLESS_BASE_YEAR, 12, 18) and values["終了"].year == values["開始"].year + 1


def test_yearless_date_follows_the_year_of_a_dated_date_in_the_same_question():
    q = "2025年12月20日に始めて、1月5日に終えました。"
    ex = _dates_extraction(("2025-12-20", "2025年12月20日に始めて"), ("2024-01-05", "1月5日に終えました"))
    values, problems = validate_extraction(_span_dates(), q, ex)
    assert problems == [] and values["終了"] == date(2026, 1, 5)
    # 年つきの日付は今までどおり引用の年と合う事
    bad = _dates_extraction(("2024-12-20", "2025年12月20日に始めて"), ("2026-01-05", "1月5日に終えました"))
    assert validate_extraction(_span_dates(), q, bad)[1]


def test_yearless_date_must_match_month_and_day_of_its_quote():
    q = "12月18日に始めて、1月5日に終えました。"
    for bad in (("2024-12-19", "12月18日に始めて"), ("2024-11-18", "12月18日に始めて")):
        _, problems = validate_extraction(_span_dates(), q, _dates_extraction(bad, ("2025-01-05", "1月5日に終えました")))
        assert problems and "not the date in its quote" in problems[0]
    q2 = "1/30に始めて、2/3に終えました。"   # 月/日の書き方
    ex2 = _dates_extraction(("2024-01-30", "1/30に始めて"), ("2024-02-03", "2/3に終えました"))
    values, problems = validate_extraction(_span_dates(), q2, ex2)
    assert problems == [] and (values["終了"] - values["開始"]).days == 4
    # 「2025年12月18日」の 12月18日 は年つきの日付の一部で、年のない日付ではない
    q3 = "2025年12月18日に始めて、2026年1月5日に終えました。"
    ex3 = _dates_extraction(("2024-12-18", "2025年12月18日に始めて"), ("2026-01-05", "2026年1月5日に終えました"))
    assert validate_extraction(_span_dates(), q3, ex3)[1]


def test_yearless_dates_170_days_apart_are_held_with_a_reason():
    c = _span_dates()
    # 1月1日 → 6月18日は 169 日（近い方が決まる）、6月19日は 170 日（決まらない）
    values, problems = validate_extraction(c, "1月1日に始めて、6月18日に終えました。",
                                           _dates_extraction(("2024-01-01", "1月1日に始めて"), ("2024-06-18", "6月18日に終えました")))
    assert problems == [] and (values["終了"] - values["開始"]).days == 169
    q = "1月1日に始めて、6月19日に終えました。"
    ex = _dates_extraction(("2024-01-01", "1月1日に始めて"), ("2024-06-19", "6月19日に終えました"))
    _, problems = validate_extraction(c, q, ex)
    assert problems and "終了" in problems[0] and f"{rule_cards.YEARLESS_AMBIGUOUS_DAYS} 日以上" in problems[0]
    text, d = run(rule_cards.answer_with_card(c, q, FakeExtractor(lambda m, s: ex)))
    assert d.conclusion is None and "年を決められない" in d.hold_reason and text.endswith("結論: 保留")


def test_feb_29_is_built_on_a_leap_year():
    c = _span_dates()
    values, problems = validate_extraction(c, "2月29日に始めて、3月5日に終えました。",
                                           _dates_extraction(("2024-02-29", "2月29日に始めて"), ("2024-03-05", "3月5日に終えました")))
    assert problems == [] and values["開始"] == date(2024, 2, 29) and (values["終了"] - values["開始"]).days == 5
    # 基準が閏年でない近くにあると 2 月 29 日の年は決められない → 保留
    _, problems = validate_extraction(c, "2025年3月1日に始めて、2月29日に終えました。",
                                      _dates_extraction(("2025-03-01", "2025年3月1日に始めて"), ("2024-02-29", "2月29日に終えました")))
    assert problems and "終了" in problems[0]


YUKYU_Q = "入社から6ヶ月が経過しました。この期間の全労働日は120日で、欠勤は30日でした。年次有給休暇は付与されますか？"


def _yukyu_raw(**change):
    raw = {"勤続月数": {"quote": "入社から6ヶ月が経過", "value": 6}, "全労働日数": {"quote": "全労働日は120日", "value": 120},
           "欠勤日数": {"quote": "欠勤は30日", "value": 30}, "出勤日数": {"quote": None, "value": None}}
    raw.update(change)
    return raw


def test_failed_derive_target_is_dropped_and_computed_from_its_sources():
    raw = _yukyu_raw(出勤日数={"quote": "欠勤は30日", "value": 80})    # 先の値が引用の数と違う
    values, problems = validate_extraction(YUKYU, YUKYU_Q, raw)
    assert problems == [] and values.get("出勤日数") is None
    d = decide(YUKYU, values)
    assert d.values["出勤日数"] == 90 and d.conclusion == "付与されない"   # 120 - 30 = 90 → 0.75
    # 引用が質問に無い時も同じ（保留にしない）
    values, problems = validate_extraction(YUKYU, YUKYU_Q, _yukyu_raw(出勤日数={"quote": "出勤は90日", "value": 90}))
    assert problems == [] and decide(YUKYU, values).values["出勤日数"] == 90


def test_failed_derive_target_stays_a_hold_when_a_source_is_missing_or_failed():
    bad_target = {"quote": "欠勤は30日", "value": 80}
    _, problems = validate_extraction(YUKYU, YUKYU_Q, _yukyu_raw(出勤日数=bad_target, 欠勤日数={"quote": None, "value": None}))
    assert any(p.startswith("出勤日数") for p in problems)           # 元の枠が無い → 従来どおり保留
    _, problems = validate_extraction(YUKYU, YUKYU_Q, _yukyu_raw(出勤日数=bad_target, 欠勤日数={"quote": "欠勤は30日", "value": 31}))
    assert any(p.startswith("出勤日数") for p in problems) and any(p.startswith("欠勤日数") for p in problems)
    # derive の先でない枠の失敗は今までどおり保留
    _, problems = validate_extraction(YUKYU, YUKYU_Q, _yukyu_raw(全労働日数={"quote": "全労働日は120日", "value": 12}))
    assert any(p.startswith("全労働日数") for p in problems)
    # 先の形の崩れ（value/quote が無い）は捨てない
    raw = _yukyu_raw()
    raw["出勤日数"] = {"value": 1}
    assert any("needs value and quote" in p for p in validate_extraction(YUKYU, YUKYU_Q, raw)[1])


def _zero_card():
    return _mk({"貸与中": {"type": "number", "unit": "円", "describe": "貸与中の機器の金額"}},
               [_leaf("0 円", "貸与中", "<", 1)])


@pytest.mark.parametrize("quote", ["他に借りている機器はありません", "機器は借りていない", "貸与中の機器はなし",
                                   "貸与中の機器は無い", "貸与中の機器は持っていない", "貸与中は0円"])
def test_zero_passes_only_with_a_negation_word_or_a_literal_zero(quote):
    values, problems = validate_extraction(_zero_card(), quote + "です。", {"貸与中": {"value": 0, "quote": quote}})
    assert problems == [] and values["貸与中"] == 0


@pytest.mark.parametrize("quote,value", [("他に3万円の機器を借りています", 0), ("機器を借りています", 0),
                                         ("機器は借りていません", 5), ("貸与中の機器はありません", 1)])
def test_zero_is_rejected_without_a_negation_word_and_negation_does_not_make_other_numbers(quote, value):
    _, problems = validate_extraction(_zero_card(), quote + "。", {"貸与中": {"value": value, "quote": quote}})
    assert problems and "not the number" in problems[0]


# ===== 抽出の形: quote を value より先に・2 回目は枠の順を逆にする =====

def test_extraction_schema_puts_quote_before_value_and_reverses_for_the_second_read():
    s = rule_cards.extraction_schema(YUKYU)
    assert list(s["properties"]) == list(YUKYU.slots) and s["required"] == list(YUKYU.slots)
    for item in s["properties"].values():
        assert list(item["properties"]) == ["quote", "value"] and item["required"] == ["quote", "value"]
    r = rule_cards.extraction_schema(YUKYU, reverse=True)
    assert list(r["properties"]) == list(YUKYU.slots)[::-1] and r["required"] == list(r["properties"])
    text = json.dumps(s, ensure_ascii=False)
    assert text.index('"quote"') < text.index('"value"')
    msg = rule_cards.extraction_messages(YUKYU, N07)[0]["content"]
    rev = rule_cards.extraction_messages(YUKYU, N07, reverse=True)[0]["content"]
    assert msg.index("先に") < msg.index("次に") and "そのまま" in msg
    names = list(YUKYU.slots)
    assert [msg.index(f"- {n}:") for n in names] == sorted(msg.index(f"- {n}:") for n in names)
    assert [rev.index(f"- {n}:") for n in names] == sorted((rev.index(f"- {n}:") for n in names), reverse=True)


# ===== 2 回読み（CONFIRM_PASSES）=====

class SeqExtractor:
    """呼び出しごとに決めた答えを返す抽出役（例外も投げられる）。last_stats は呼び出しごとに変わる。"""

    def __init__(self, *replies):
        self.replies, self.calls, self.last_stats = list(replies), [], {}

    async def extract(self, messages, schema):
        self.calls.append((messages, schema))
        self.last_stats = {}
        r = self.replies[min(len(self.calls), len(self.replies)) - 1]
        if isinstance(r, Exception):
            raise r
        self.last_stats = {"completion_tokens": 10 * len(self.calls), "finish_reason": "stop", "reasoning_chars": 0}
        return r


def _amount_card():
    slots = {"金額": {"type": "number", "unit": "円", "describe": "申請の金額"},
             "件数": {"type": "number", "unit": "件", "describe": "申請の件数"}}
    return _mk(slots, [_leaf("100円以上", "金額", ">=", 100)])


AQ = "金額は150円と50円です。件数は2件です。"


def _ex(amount, quote, n=2):
    return {"金額": {"quote": quote, "value": amount}, "件数": {"quote": "件数は2件", "value": n}}


def test_two_reads_agree_then_decide_and_record_both_conclusions():
    c = _amount_card()
    ex = SeqExtractor(_ex(150, "金額は150円"), _ex(150, "金額は150円"))
    text, d = run(rule_cards.answer_with_card(c, AQ, ex))
    assert len(ex.calls) == rule_cards.CONFIRM_PASSES == 2
    assert (d.conclusion, d.first_conclusion, d.second_conclusion, d.hold_reason) == ("対象", "対象", "対象", None)
    assert text.splitlines()[-1] == "結論: 対象"
    # 別の呼び出しで、枠の順を逆にして読む
    assert list(ex.calls[0][1]["properties"]) == ["金額", "件数"] and list(ex.calls[1][1]["properties"]) == ["件数", "金額"]
    assert ex.calls[0][0] != ex.calls[1][0]
    assert d.pass_stats == [{"completion_tokens": 10, "finish_reason": "stop", "reasoning_chars": 0},
                            {"completion_tokens": 20, "finish_reason": "stop", "reasoning_chars": 0}]
    assert ex.last_stats["completion_tokens"] == 20          # last_stats は直前の 1 回のまま
    assert d.raw == _ex(150, "金額は150円") and d.second_raw == d.raw


def test_two_reads_that_disagree_are_held_with_the_fixed_reason():
    c = _amount_card()
    ex = SeqExtractor(_ex(150, "金額は150円"), _ex(50, "50円"))
    text, d = run(rule_cards.answer_with_card(c, AQ, ex))
    assert d.conclusion is None and d.hold_reason == rule_cards.HOLD_MISMATCH == "2 回の読みが一致しない"
    assert (d.first_conclusion, d.second_conclusion) == ("対象", "対象外")
    assert text.splitlines()[-1] == "結論: 保留" and "2 回の読みが一致しない" in text


@pytest.mark.parametrize("second,why", [
    (_ex(None, None), "決まらない"),                                  # 2 回目が保留（値が読めない）
    (_ex(150, "金額は99円"), "検査に通らない"),                       # 2 回目が検査落ち
    (TimeoutError("slow"), "抽出に失敗"),                             # 2 回目の抽出の失敗
])
def test_second_read_that_holds_or_fails_makes_the_whole_answer_a_hold(second, why):
    ex = SeqExtractor(_ex(150, "金額は150円"), second)
    _, d = run(rule_cards.answer_with_card(_amount_card(), AQ, ex))
    assert d.conclusion is None and d.hold_reason == rule_cards.HOLD_MISMATCH
    assert d.first_conclusion == "対象" and d.second_conclusion is None and why in d.second_hold_reason


def test_first_read_that_holds_is_not_read_again():
    ex = SeqExtractor(_ex(None, None), _ex(150, "金額は150円"))
    _, d = run(rule_cards.answer_with_card(_amount_card(), AQ, ex))
    assert len(ex.calls) == 1 and d.conclusion is None and d.hold_reason != rule_cards.HOLD_MISMATCH
    assert (d.first_conclusion, d.second_conclusion) == (None, None) and len(d.pass_stats) == 1


def test_confirmation_can_be_turned_off_and_the_record_still_has_first_conclusion(monkeypatch):
    monkeypatch.setattr(rule_cards, "CONFIRM_PASSES", 1)
    ex = SeqExtractor(_ex(150, "金額は150円"), _ex(50, "50円"))
    _, d = run(rule_cards.answer_with_card(_amount_card(), AQ, ex))
    assert len(ex.calls) == 1 and d.conclusion == "対象" and (d.first_conclusion, d.second_conclusion) == ("対象", None)


def test_sealed_record_has_first_and_second_conclusions_without_breaking_the_old_fields():
    try:
        from evaluation.gpu_local import numeric_sealed as ns
    except ImportError:
        pytest.skip("evaluation を読み込めない")
    c = _amount_card()
    item = {"id": "q1", "question": AQ, "rule_id": "R", "should_hold": False, "expected_conclusion": "対象"}
    ex = SeqExtractor(_ex(150, "金額は150円"), _ex(150, "金額は150円"))
    _, d = run(rule_cards.answer_with_card(c, AQ, ex))
    rec = ns.make_record(item, 0, 1, c, d, d.pass_stats[0], 0.5)
    assert rec["outcome"] == "decided" and rec["conclusion"] == "対象"
    assert (rec["first_conclusion"], rec["second_conclusion"]) == ("対象", "対象")
    assert rec["completion_tokens"] == 10 and rec["second_completion_tokens"] == 20 and rec["finish_reason"] == "stop"
    ex = SeqExtractor(_ex(150, "金額は150円"), _ex(50, "50円"))
    _, d = run(rule_cards.answer_with_card(c, AQ, ex))
    rec = ns.make_record(item, 0, 1, c, d, d.pass_stats[0], 0.5)
    assert (rec["outcome"], rec["stop_reason"], rec["conclusion"]) == ("held", ns.CONFIRM_MISMATCH, None)
    assert (rec["first_conclusion"], rec["second_conclusion"]) == ("対象", "対象外")

# ---- 同じ量の別の言い方と、問題の枠を不明にする検査 (verify="equiv"・problem_as_unknown)


@pytest.mark.parametrize("value,quote,unit,ok", [
    (36, "勤続3年の正社員です", "か月", True),
    (13, "勤続は1年1か月です", "か月", True),
    (35, "勤続3年の正社員です", "か月", False),
    (11, "入社は2023年11月です", "か月", False),  # 年月は年の長さではない
    (4600, "合計は4千6百円です", "円", True),
    (4700, "合計は4千6百円です", "円", False),
    (85000, "5月に7万円、6月に1万5千円", "円", True),  # 単位の合う数の合計
    (5, "会計は5日分です", "回", False),  # 単位が違う数は通さない
])
def test_equiv_numbers(value, quote, unit, ok):
    quote = rule_cards._norm(quote)
    assert rule_cards._number_in_quote(Decimal(value), quote, unit, equiv=True) is ok


def test_equiv_is_off_by_default():
    assert rule_cards._number_in_quote(Decimal(36), "勤続3年", "か月") is False


def test_problem_as_unknown_decides_only_when_conclusion_does_not_depend():
    card = parse_card({
        "format": "rule-card@1", "id": "t", "status": "draft", "target": "t",
        "doc": {"source": "a.md", "quote": "q"}, "match": {"conclusions": ["可", "不可"], "keywords": ["t"]},
        "slots": {"日数": {"type": "number", "unit": "日", "describe": "d"},
                  "額": {"type": "number", "unit": "円", "describe": "a"}},
        "conditions": [{"name": "日数", "left": "日数", "op": ">=", "right": 5},
                       {"name": "額", "left": "額", "op": ">=", "right": 100}],
        "combine": "all", "conclusion": {"then": "可", "else": "不可"}})
    q = "日数は3日です。額は200円です。"
    raw = {"日数": {"quote": "日数は3日", "value": 3}, "額": {"quote": "額は200円", "value": 999}}
    _, problems = validate_extraction(card, q, raw)
    assert problems
    values, problems = validate_extraction(card, q, raw, problem_as_unknown=True)
    assert not problems and values["額"] is None
    assert decide(card, values).conclusion == "不可"  # 1 つ偽なら all は決まる
    raw["日数"]["value"] = 5
    raw["日数"]["quote"] = "額は200円"  # 日数の読みも落ちる → 両方不明で保留
    values, _ = validate_extraction(card, q, raw, problem_as_unknown=True)
    assert decide(card, values).conclusion is None


# ---- 話題の無い条件の既定と、あり/なしの全値代入 (decide_v2)


def _v2_card():
    return parse_card({
        "format": "rule-card@1", "id": "t", "status": "draft", "target": "t",
        "doc": {"source": "a.md", "quote": "q"}, "match": {"conclusions": ["可", "不可"], "keywords": ["t"]},
        "slots": {"時間": {"type": "number", "unit": "時間", "describe": "h"},
                  "短時間勤務か": {"type": "bool", "describe": "s"},
                  "再婚か": {"type": "bool", "describe": "r"}},
        "conditions": [{"name": "時間", "any": [
            {"name": "通常", "left": "時間", "op": ">=", "right": 4},
            {"name": "短時間", "all": [{"name": "短", "left": "短時間勤務か", "op": "==", "right": "あり"},
                                    {"name": "3h", "left": "時間", "op": ">=", "right": 3}]}]}],
        "combine": "all",
        "exceptions": [{"name": "再婚", "left": "再婚か", "op": "==", "right": "あり", "conclusion": "不可",
                        "unless_mentioned": False, "mention": ["再婚"]}],
        "conclusion": {"then": "可", "else": "不可"}})


def test_v2_exception_ignored_only_when_topic_is_absent():
    c = _v2_card()
    v = {"時間": Decimal(5), "短時間勤務か": None, "再婚か": None}
    assert rule_cards.decide_v2(c, v, "5時間です。").conclusion == "可"
    assert rule_cards.decide_v2(c, v, "5時間です。再婚かは分かりません。").conclusion is None


def test_v2_enumerates_bool_and_holds_when_split():
    c = _v2_card()
    assert rule_cards.decide_v2(c, {"時間": Decimal("2"), "短時間勤務か": None, "再婚か": None}, "2時間").conclusion == "不可"
    assert rule_cards.decide_v2(c, {"時間": Decimal("3.5"), "短時間勤務か": None, "再婚か": None}, "3.5時間").conclusion is None
    assert rule_cards.decide_v2(c, {"時間": None, "短時間勤務か": None, "再婚か": None}, "不明").conclusion is None


def test_v2_mention_needs_unless_mentioned():
    d = {"format": "rule-card@1", "id": "t", "status": "draft", "target": "t", "doc": {"source": "a.md", "quote": "q"},
         "match": {"conclusions": ["可", "不可"], "keywords": ["t"]}, "slots": {"x": {"type": "bool", "describe": "x"}},
         "conditions": [{"name": "x", "left": "x", "op": "==", "right": "あり", "mention": ["x"]}], "combine": "all",
         "conclusion": {"then": "可", "else": "不可"}}
    with pytest.raises(CardError):
        parse_card(d)


def test_v2_off_by_default():
    assert rule_cards.V2 is False


def test_in_md_range_handles_year_wrap_and_unknown():
    e = {"in_md_range": ["d", 12, 1, 1, 10]}
    for day, want in ((date(2026, 12, 1), "あり"), (date(2027, 1, 10), "あり"), (date(2026, 1, 11), "なし"), (date(2026, 11, 30), "なし")):
        assert rule_cards._eval(e, {"d": day}) == want
    assert rule_cards._eval({"in_md_range": ["d", 3, 1, 3, 31]}, {"d": date(2026, 3, 31)}) == "あり"
    assert rule_cards._eval({"in_md_range": ["d", 3, 1, 3, 31]}, {"d": date(2026, 4, 1)}) == "なし"
    assert rule_cards._eval(e, {"d": None}) is None


def test_same_units_only_when_card_says_so():
    q = "食堂の利用は5日分"
    assert not rule_cards._number_in_quote(Decimal(5), q, "回", equiv=True)  # カードに無ければ通さない
    assert rule_cards._number_in_quote(Decimal(5), q, "回", equiv=True, same_units=("日分",))
    assert not rule_cards._number_in_quote(Decimal(6), q, "回", equiv=True, same_units=("日分",))
    assert not rule_cards._number_in_quote(Decimal(5), q, "回", same_units=("日分",))  # 厳密の照合は今のまま


def test_select_card_v2_thresholds_are_only_for_v2(monkeypatch):
    calls = {"a": 23, "b": 19}
    cards = [SimpleNamespace(id=k, conclusions=("x", "y")) for k in calls]
    monkeypatch.setattr(rule_cards, "conclusion_options", lambda q: ("x", "y"))
    monkeypatch.setattr(rule_cards, "select_score", lambda c, q: calls[c.id])
    monkeypatch.setattr(rule_cards, "V2", False)
    assert rule_cards.select_card(cards, "q") is None
    monkeypatch.setattr(rule_cards, "V2", True)
    assert rule_cards.select_card(cards, "q").id == "a"
    calls["b"] = 20  # 差 3 は選ばない
    assert rule_cards.select_card(cards, "q") is None
    calls.update(a=7, b=0)  # 下限未満は選ばない
    assert rule_cards.select_card(cards, "q") is None


@pytest.mark.parametrize("value,quote,unit,ok", [
    (3, "片道3キロの道のり", "キロメートル", True),
    (3000, "片道3キロの道のり", "メートル", True),
    (900, "片道900メートルで", "m", True),
    (0.9, "片道900メートルで", "キロメートル", True),
    (900, "片道900円で", "メートル", False),
    (50000000, "対物は5千万円の保険", "円", True),
    (200000000, "対物は2億円の保険", "円", True),
    (5000000, "対物は5千万円の保険", "円", False),
])
def test_equiv_distance_and_large_numbers(value, quote, unit, ok):
    assert rule_cards._number_in_quote(Decimal(str(value)), rule_cards._norm(quote), unit, equiv=True) is ok


@pytest.mark.parametrize("a,b,want", [
    (date(2016, 4, 2), date(2026, 4, 1), 119), (date(2016, 4, 1), date(2026, 4, 1), 120),
    (date(2024, 1, 31), date(2024, 2, 28), 0), (date(2024, 1, 31), date(2024, 2, 29), 1),
    (date(2026, 5, 1), date(2026, 4, 1), -1),
])
def test_months_between(a, b, want):
    assert rule_cards._eval({"months_between": ["a", "b"]}, {"a": a, "b": b}) == Decimal(want)


def test_year_start():
    e = {"year_start": ["d", 4, 1]}
    assert rule_cards._eval(e, {"d": date(2026, 3, 31)}) == date(2025, 4, 1)
    assert rule_cards._eval(e, {"d": date(2026, 4, 1)}) == date(2026, 4, 1)


def _absent_card():
    return rule_cards.parse_card({
        "format": "rule-card@1", "id": "t", "status": "draft", "drafted_by": "x", "confirmed_by": None,
        "doc": {"source": "s", "section": "1", "quote": "q"}, "target": "t",
        "match": {"conclusions": ["出る", "出ない"], "keywords": ["代"]},
        "slots": {"時刻": {"type": "number", "unit": "時", "describe": "時刻"},
                  "妊娠中": {"type": "bool", "absent": "なし", "mention": ["妊娠"], "describe": "妊娠中か"}},
        "conditions": [{"name": "c", "any": [
            {"name": "a", "left": "時刻", "op": ">=", "right": 22},
            {"name": "b", "all": [{"name": "b1", "left": "妊娠中", "op": "==", "right": "あり"},
                                  {"name": "b2", "left": "時刻", "op": ">=", "right": 21}]}]}],
        "combine": "all", "conclusion": {"then": "出る", "else": "出ない"}})


def test_absent_slot_default_and_explicit_unknown():
    c = _absent_card()
    q = "21時半に退社しました。出ますか。"
    assert rule_cards.decide_v2(c, {"時刻": Decimal("21.5"), "妊娠中": None}, q).conclusion == "出ない"
    q2 = "妊娠中かもしれません。21時半に退社しました。出ますか。"
    assert rule_cards.decide_v2(c, {"時刻": Decimal("21.5"), "妊娠中": None}, q2).conclusion is None
    q3 = "同僚の事情は聞いていません。21時半に退社しました。出ますか。"  # 知らないと明言 → 既定を使わない
    assert rule_cards.decide_v2(c, {"時刻": Decimal("21.5"), "妊娠中": None}, q3).conclusion is None


def test_absent_needs_mention_on_bool():
    d = {"format": "rule-card@1", "id": "t", "status": "draft", "drafted_by": "x", "confirmed_by": None,
         "doc": {"source": "s", "section": "1", "quote": "q"}, "target": "t",
         "match": {"conclusions": ["出る", "出ない"], "keywords": ["代"]},
         "slots": {"b": {"type": "bool", "absent": "なし", "describe": "b"}},
         "conditions": [{"name": "c", "left": "b", "op": "==", "right": "あり"}],
         "combine": "all", "conclusion": {"then": "出る", "else": "出ない"}}
    with pytest.raises(CardError):
        rule_cards.parse_card(d)


def test_yearless_date_written_with_placeholder_year_is_accepted_in_equiv_only():
    c = rule_cards.parse_card({
        "format": "rule-card@1", "id": "t", "status": "draft", "drafted_by": "x", "confirmed_by": None,
        "doc": {"source": "s", "section": "1", "quote": "q"}, "target": "t",
        "match": {"conclusions": ["出る", "出ない"], "keywords": ["代"]},
        "slots": {"日": {"type": "date", "describe": "日"}},
        "conditions": [{"name": "c", "left": "日", "op": ">=", "right": {"date": "2000-01-01"}}],
        "combine": "all", "conclusion": {"then": "出る", "else": "出ない"}})
    q = "発表は7月1日です。\n最後の行は「結論: 出る」または「結論: 出ない」とする。"
    for raw_v in ("YYYY-07-01", "7月1日"):
        raw = {"日": {"quote": "発表は7月1日", "value": raw_v}}
        v, p = validate_extraction(c, q, raw, verify="equiv")
        assert not p and (v["日"].month, v["日"].day) == (7, 1)
        _, p2 = validate_extraction(c, q, raw)
        assert p2


@pytest.mark.parametrize("quote,ok", [("22時20分から22時40分の間だった", False), ("22時ごろ退社", False), ("22時20分に退社", True)])
def test_hedged_quote_gives_no_exact_value(quote, ok):
    c = rule_cards.parse_card({
        "format": "rule-card@1", "id": "t", "status": "draft", "drafted_by": "x", "confirmed_by": None,
        "doc": {"source": "s", "section": "1", "quote": "q"}, "target": "t",
        "match": {"conclusions": ["出る", "出ない"], "keywords": ["代"]},
        "slots": {"時刻": {"type": "time", "describe": "時刻"}},
        "conditions": [{"name": "c", "left": "時刻", "op": ">=", "right": {"time": "22:00"}}],
        "combine": "all", "conclusion": {"then": "出る", "else": "出ない"}})
    hh = "22:20" if "20分" in quote else "22:00"
    q = f"{quote}。\n最後の行は「結論: 出る」または「結論: 出ない」とする。"
    _, p = validate_extraction(c, q, {"時刻": {"quote": quote, "value": hh}}, verify="equiv")
    assert (not p) is ok
