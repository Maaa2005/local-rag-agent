"""規程カードの下書きの道具 (draft_cards.py) の prepare / collect を、架空の規程で確かめる。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evaluation.v2.sealed import draft_cards as dc

TEMPLATE = (
    "<!-- 先頭の注釈。渡さない -->\n"
    "<documents>\n<document>\n<source>{{SOURCE}}</source>\n<document_content>\n{{REGULATION}}\n"
    "</document_content>\n</document>\n</documents>\n\n"
    "<conclusion_pairs>\n{{OPTION_PAIRS}}\n</conclusion_pairs>\n\n本文。\n"
)
QUOTE_A = "架空の備品は購入日から14日以内に申請する。"
QUOTE_B = "架空の出張は3日を超える場合に承認を要する。"
REG_A = f"# 備品\n\n{QUOTE_A}\n"
REG_B = f"# 出張\n\n{QUOTE_B}\n"


def _card(cid: str, quote: str, source: str = "架空規程", **over) -> dict:
    c = {"format": "rule-card@1", "id": cid, "status": "draft", "drafted_by": "claude-fable-5-1",
         "confirmed_by": None, "doc": {"source": source, "section": "第1条", "quote": quote},
         "target": "申請の期限", "match": {"conclusions": ["受理", "不受理"], "keywords": ["備品"]},
         "slots": {"購入日": {"type": "date", "describe": "購入した日"}, "申請日": {"type": "date", "describe": "申請した日"}},
         "conditions": [{"name": "14日以内", "left": "申請日", "op": "<=", "right": {"add_days": ["購入日", 14]}}],
         "combine": "all", "conclusion": {"then": "受理", "else": "不受理"}}
    c.update(over)
    return c


def _answer(cards, unable="[]", raw_cards=None) -> str:
    body = raw_cards if raw_cards is not None else json.dumps(cards, ensure_ascii=False)
    return f"<quotes>x</quotes>\n<cards>{body}</cards>\n<unable>{unable}</unable>\n"


@pytest.fixture
def prepared(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "bihin.md").write_bytes(REG_A.encode("utf-8"))
    (corpus / "shucchou.md").write_bytes(REG_B.encode("utf-8"))
    options = tmp_path / "options.json"
    options.write_text(json.dumps({"format": "conclusion-options@1",
                                   "conclusion_options": [["不受理", "受理"], ["不要", "必要"]]}, ensure_ascii=False),
                       encoding="utf-8")
    prompt = tmp_path / "draft_prompt.md"
    prompt.write_text(TEMPLATE, encoding="utf-8")
    out = tmp_path / "drafts"
    n = dc.prepare(corpus, options, prompt, out)
    assert n == 2
    return tmp_path, out, prompt


def test_prepare_fills_and_fingerprints(prepared, capsys):
    tmp, out, prompt = prepared
    text = (out / "bihin.prompt.md").read_text(encoding="utf-8")
    assert "<!--" not in text and "{{" not in text
    assert "<source>bihin</source>" in text and QUOTE_A in text and QUOTE_B not in text
    assert "不受理 / 受理\n不要 / 必要" in text
    m = json.loads((out / "draft_manifest.json").read_text(encoding="utf-8"))
    assert m["prompt_sha256"] == hashlib.sha256(prompt.read_bytes()).hexdigest()
    assert m["rules_sha256"]["bihin"] == hashlib.sha256(REG_A.encode("utf-8")).hexdigest()
    assert set(m["rules_sha256"]) == {"bihin", "shucchou"}
    assert dc.regulation_of(text).strip() == REG_A.strip()


def test_prepare_cli_prints_count_only(tmp_path, prepared, capsys):
    tmp, _out, prompt = prepared
    rc = dc.main(["prepare", "--corpus", str(tmp / "corpus"), "--options", str(tmp / "options.json"),
                  "--prompt", str(prompt), "--out", str(tmp / "o2")])
    out = capsys.readouterr().out
    assert rc == 0 and "架空" not in out and "2" in out


def test_prepare_rejects_bad_options(tmp_path, prepared):
    tmp, _out, prompt = prepared
    bad = tmp / "bad.json"
    bad.write_text(json.dumps({"format": "x", "conclusion_options": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        dc.prepare(tmp / "corpus", bad, prompt, tmp / "o3")


def test_collect_normal(prepared):
    tmp, out, _ = prepared
    (out / "bihin.answer.md").write_text(_answer([_card("bihin-kigen", QUOTE_A)]), encoding="utf-8")
    (out / "shucchou.answer.md").write_text(_answer([_card("shucchou-shounin", QUOTE_B)]), encoding="utf-8")
    cards = tmp / "cards"
    assert dc.collect(out, cards) == (2, 0)
    assert sorted(p.name for p in cards.glob("*.json")) == ["bihin-kigen.json", "shucchou-shounin.json"]
    assert json.loads((out / "unable.json").read_text(encoding="utf-8")) == []


def test_collect_quote_not_in_regulation(prepared):
    tmp, out, _ = prepared
    (out / "bihin.answer.md").write_text(_answer([_card("bihin-kigen", "規程に無い文。")]), encoding="utf-8")
    (out / "shucchou.answer.md").write_text(_answer([_card("shucchou-shounin", QUOTE_B)]), encoding="utf-8")
    cards = tmp / "cards"
    assert dc.collect(out, cards) == (1, 1)
    assert not (cards / "bihin-kigen.json").exists()
    u = json.loads((out / "unable.json").read_text(encoding="utf-8"))
    assert u[0]["rule"] == "bihin-kigen" and "quote" in u[0]["reason"]


def test_collect_broken_json_and_bad_card(prepared):
    tmp, out, _ = prepared
    (out / "bihin.answer.md").write_text(_answer(None, raw_cards='[{"id": "x",'), encoding="utf-8")
    bad = _card("shucchou-shounin", QUOTE_B, combine="some")
    (out / "shucchou.answer.md").write_text(_answer([bad]), encoding="utf-8")
    cards = tmp / "cards"
    assert dc.collect(out, cards) == (0, 2)
    assert list(cards.glob("*.json")) == []
    u = {x["rule"]: x["reason"] for x in json.loads((out / "unable.json").read_text(encoding="utf-8"))}
    assert "壊れている" in u["bihin"] and "parse_card" in u["shucchou-shounin"]


def test_collect_missing_answer(prepared):
    tmp, out, _ = prepared
    (out / "bihin.answer.md").write_text(_answer([_card("bihin-kigen", QUOTE_A)]), encoding="utf-8")
    assert dc.collect(out, tmp / "cards") == (1, 1)
    u = json.loads((out / "unable.json").read_text(encoding="utf-8"))
    assert u == [{"rule": "shucchou", "reason": "答えのファイルが無い"}]


def test_collect_merges_unable_and_checks_status(prepared, capsys):
    tmp, out, _ = prepared
    unable = json.dumps([{"rule": "出張の時刻", "reason": "時刻の型が無い"}], ensure_ascii=False)
    (out / "bihin.answer.md").write_text(_answer([_card("bihin-kigen", QUOTE_A)], unable), encoding="utf-8")
    conf = _card("shucchou-shounin", QUOTE_B, status="confirmed", confirmed_by="x")
    (out / "shucchou.answer.md").write_text(_answer([conf], "[]"), encoding="utf-8")
    rc = dc.main(["collect", "--drafts", str(out), "--cards", str(tmp / "cards")])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "カード: 1・作れなかった: 2"
    u = json.loads((out / "unable.json").read_text(encoding="utf-8"))
    assert {"rule": "出張の時刻", "reason": "時刻の型が無い"} in u
    assert any(x["rule"] == "shucchou-shounin" and "status" in x["reason"] for x in u)


def _cond(name, quote=None, **over):
    c = {"name": name, "left": "申請日", "op": "<=", "right": {"add_days": ["購入日", 14]}}
    if quote is not None:
        c["quote"] = quote
    c.update(over)
    return c


def test_collect_checks_condition_and_exception_quotes(prepared):
    tmp, out, _ = prepared
    slots = {"購入日": {"type": "date", "describe": "購入した日"}, "申請日": {"type": "date", "describe": "申請した日"},
             "緊急": {"type": "bool", "describe": "緊急か"}}
    exc = {"name": "緊急", "left": "緊急", "op": "==", "right": "あり", "conclusion": "受理", "quote": QUOTE_A}
    # 条件の quote が規程に有る (改行や空白の違いは許す) ・例外の quote も有る → 通る
    ok = _card("bihin-kigen", QUOTE_A, slots=slots, conditions=[_cond("14日以内", QUOTE_A[:6] + " " + QUOTE_A[6:])],
               exceptions=[exc])
    (out / "bihin.answer.md").write_text(_answer([ok]), encoding="utf-8")
    (out / "shucchou.answer.md").write_text(_answer([_card("shucchou-shounin", QUOTE_B)]), encoding="utf-8")
    assert dc.collect(out, tmp / "cards1") == (2, 0)
    # quote が無い条件 (任意) も通る
    (out / "bihin.answer.md").write_text(_answer([_card("bihin-kigen", QUOTE_A)]), encoding="utf-8")
    assert dc.collect(out, tmp / "cards2") == (2, 0)


@pytest.mark.parametrize("where", ["condition", "exception", "nested", "nested-exception"])
def test_collect_rejects_condition_quote_not_in_regulation(prepared, where):
    tmp, out, _ = prepared
    slots = {"購入日": {"type": "date", "describe": "購入した日"}, "申請日": {"type": "date", "describe": "申請した日"},
             "緊急": {"type": "bool", "describe": "緊急か"}}
    bad = "規程に無い言い換えた文。"
    exc = {"name": "緊急", "left": "緊急", "op": "==", "right": "あり", "conclusion": "受理"}
    conds = [_cond("14日以内", QUOTE_A)]
    if where == "condition":
        conds = [_cond("14日以内", bad)]
    elif where == "exception":
        exc["quote"] = bad
    elif where == "nested":
        conds = [{"name": "g", "all": [_cond("14日以内", QUOTE_A), _cond("入れ子", bad)]}]
    else:
        child = {"name": "x", "left": "緊急", "op": "==", "right": "あり", "quote": bad}
        exc = {"name": "g", "any": [child], "conclusion": "受理"}
    card = _card("bihin-kigen", QUOTE_A, slots=slots, conditions=conds, exceptions=[exc])
    (out / "bihin.answer.md").write_text(_answer([card]), encoding="utf-8")
    (out / "shucchou.answer.md").write_text(_answer([_card("shucchou-shounin", QUOTE_B)]), encoding="utf-8")
    cards = tmp / "cards"
    assert dc.collect(out, cards) == (1, 1)
    assert not (cards / "bihin-kigen.json").exists()
    u = json.loads((out / "unable.json").read_text(encoding="utf-8"))
    assert u[0]["rule"] == "bihin-kigen" and "条件の quote" in u[0]["reason"]


def test_collect_accepts_time_card_and_rejects_bad_time(prepared):
    tmp, out, _ = prepared
    slots = {"退勤": {"type": "time", "describe": "退勤時刻"}}
    good = _card("bihin-kigen", QUOTE_A, slots=slots,
                 conditions=[{"name": "22時以降", "left": "退勤", "op": ">=", "right": {"time": "22:00"}}])
    bad = _card("shucchou-shounin", QUOTE_B, slots=slots,
                conditions=[{"name": "24時以降", "left": "退勤", "op": ">=", "right": {"time": "24:00"}}])
    (out / "bihin.answer.md").write_text(_answer([good]), encoding="utf-8")
    (out / "shucchou.answer.md").write_text(_answer([bad]), encoding="utf-8")
    assert dc.collect(out, tmp / "cards") == (1, 1)
    u = json.loads((out / "unable.json").read_text(encoding="utf-8"))
    assert "parse_card" in u[0]["reason"] and "bad time" in u[0]["reason"]
