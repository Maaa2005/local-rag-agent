"""封用の入口 (evaluation/gpu_local/numeric_sealed.py) と選択肢の取り出しのテスト。問・規程はすべて架空。

確かめること:
- 線の計算: 全体 90% (192 問なら 173)・規程ごと 80% (12 問なら 10)・誤った断定 0・保留すべき問を全部保留・言い換えの対
- 答えなかった問は保留に数えず、分母からも外さない
- freeze の指紋が違えば、サーバーを起動する前に止まる。--dev なしで draft は止まる
- 止まった理由 (カードに合わない・抽出の失敗・検査に通らない・不明で保留) を分ける
- extract_options は ID・本文・正解を出さない
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from evaluation.gpu_local import numeric_arms as na
from evaluation.gpu_local import numeric_sealed as ns
from evaluation.v2.sealed import extract_options as eo

# ---------------------------------------------------------------- 線の計算


def _rec(id_, rule, *, hold=False, outcome="decided", conclusion="必要", expected="必要", pair=None, run=1):
    if hold:
        expected = "保留"
        conclusion = None if outcome == "held" else conclusion
    elif outcome != "decided":
        conclusion = None
    reason = {"no_card": "no_card", "held": "unknown_hold"}.get(outcome)
    return {"run": run, "id": id_, "rule_id": rule, "should_hold": hold, "expected_conclusion": expected,
            "pair_id": pair, "outcome": outcome, "conclusion": conclusion, "stop_reason": reason}


def _full_set(undecided=(), broken_pairs=(), run=1):
    """16 規程 × (回答できる 12 + 保留 3)。各規程の先頭 4 問が 2 つの言い換えの対 (全部で 32 対)。"""
    rows = []
    for r in range(16):
        for j in range(12):
            id_ = f"R{r}-{j}"
            pair = f"P{r}-{j // 2}" if j < 4 else None
            outcome = "held" if id_ in undecided or id_ in broken_pairs else "decided"
            rows.append(_rec(id_, f"R{r}", outcome=outcome, pair=pair, run=run))
        for j in range(3):
            rows.append(_rec(f"R{r}-h{j}", f"R{r}", hold=True, outcome="held", run=run))
    return rows


def test_full_set_passes_all_lines():
    r = ns.evaluate_round(_full_set(), expected_n=240)
    assert (r["answerable"], r["should_hold"], r["pairs"]) == (192, 48, 32)
    assert r["ok"] and all(r["lines"].values())


def test_overall_threshold_exactly_90_percent():
    # 173/192 は届く (切り上げ)。172 は届かない
    ok = [f"R{r}-{j}" for r in range(16) for j in (11,)] + ["R0-10", "R1-10", "R2-10"]  # 16 + 3 = 19 問が未決定
    r = ns.evaluate_round(_full_set(undecided=ok), 240)
    assert (r["decided"], r["decided_required"]) == (173, 173)
    assert r["lines"]["decided_overall"] and r["lines"]["decided_per_rule"] and r["ok"]
    r = ns.evaluate_round(_full_set(undecided=ok + ["R3-10"]), 240)
    assert r["decided"] == 172 and not r["lines"]["decided_overall"] and not r["ok"]


def test_per_rule_threshold_80_percent():
    two = ["R0-11", "R0-10"]  # 規程 R0: 10/12 は届く
    r = ns.evaluate_round(_full_set(undecided=two), 240)
    assert r["per_rule"]["R0"]["decided"] == 10 and r["per_rule"]["R0"]["required"] == 10 and r["ok"]
    r = ns.evaluate_round(_full_set(undecided=two + ["R0-9"]), 240)  # 9/12 は届かない (全体は 189/192 で届く)
    assert r["per_rule"]["R0"]["ok"] is False
    assert r["lines"]["decided_overall"] and not r["lines"]["decided_per_rule"] and not r["ok"]


def test_one_wrong_assertion_fails():
    rows = _full_set()
    rows[0]["conclusion"] = "不要"  # 回答できる問で結論が正解と違う
    r = ns.evaluate_round(rows, 240)
    assert r["wrong_assertions"] == 1 and not r["lines"]["wrong_assertions_zero"] and not r["ok"]
    assert r["decided"] == 192  # 結論は出したので割合には数える


def test_missed_hold_counts_as_wrong_assertion_and_not_held():
    rows = _full_set()
    miss = next(x for x in rows if x["should_hold"])
    miss.update(outcome="decided", conclusion="必要", stop_reason=None)
    r = ns.evaluate_round(rows, 240)
    assert (r["held"], r["should_hold"], r["wrong_assertions"]) == (47, 48, 1)
    assert not r["lines"]["all_held"] and not r["lines"]["wrong_assertions_zero"]


def test_unanswered_hold_question_is_not_counted_as_held():
    rows = _full_set()
    next(x for x in rows if x["should_hold"]).update(outcome="no_card", stop_reason="no_card")
    r = ns.evaluate_round(rows, 240)
    # 答えなかった問は保留に数えない (47/48)。誤った断定でもない。分母 48 は変わらない
    assert (r["held"], r["should_hold"], r["wrong_assertions"]) == (47, 48, 0)
    assert not r["lines"]["all_held"] and not r["ok"]
    assert r["stop_reasons"]["no_card"] == 1


def test_unanswered_answerable_question_stays_in_denominator():
    rows = _full_set()
    rows[0].update(outcome="no_card", conclusion=None, stop_reason="no_card")
    r = ns.evaluate_round(rows, 240)
    assert (r["answerable"], r["decided"], r["wrong_assertions"]) == (192, 191, 0)
    assert r["ok"]  # 191 >= 173・規程 R0 は 11/12


def test_pair_threshold_90_percent():
    # 29/32 は届く (切り上げ)。壊す対は別の規程から (規程ごとの線に触れない)
    four = [f"R{r}-0" for r in range(3)]
    r = ns.evaluate_round(_full_set(broken_pairs=four[:3]), 240)
    assert (r["pairs_both_correct"], r["pairs_required"]) == (29, 29) and r["lines"]["pairs"] and r["ok"]
    r = ns.evaluate_round(_full_set(broken_pairs=four + ["R3-0"]), 240)
    assert r["pairs_both_correct"] == 28 and not r["lines"]["pairs"] and not r["ok"]


def test_pair_with_one_wrong_member_is_not_both_correct_and_single_member_pair_fails():
    rows = [_rec("a", "R0", pair="P1"), _rec("b", "R0", pair="P1", conclusion="不要"), _rec("c", "R0", pair="P2")]
    r = ns.evaluate_round(rows)
    assert (r["pairs"], r["pairs_both_correct"]) == (2, 0)


def test_all_runs_must_pass_and_incomplete_run_fails():
    rows = _full_set(run=1) + _full_set(run=2)
    assert ns.evaluate_all(rows, 2, 240)["ok"]
    rows2 = _full_set(run=1) + _full_set(run=2)[:100]
    out = ns.evaluate_all(rows2, 2, 240)
    assert not out["ok"] and out["rounds"]["1"]["ok"] and not out["rounds"]["2"]["complete"]
    assert not ns.evaluate_all(_full_set(run=1), 2, 240)["ok"]  # 2 回目が無い
    bad = _full_set(run=2)
    bad[0]["conclusion"] = "不要"
    assert not ns.evaluate_all(_full_set(run=1) + bad, 2, 240)["ok"]


def test_ceil_ratio_matches_issue_numbers():
    assert ns.ceil_ratio(192, ns.DECIDED_OVERALL) == 173
    assert ns.ceil_ratio(12, ns.DECIDED_PER_RULE) == 10
    assert ns.ceil_ratio(32, ns.PAIR_RATIO) == 29


# ---------------------------------------------------------------- カード・freeze

CARD = {"format": "rule-card@1", "id": "c-amount", "status": "confirmed", "confirmed_by": "本人",
        "doc": {"source": "架空規程", "section": "第 1 条", "quote": "1 万円以上は必要"},
        "target": "金額", "match": {"conclusions": ["必要", "不要"], "keywords": ["金額"]},
        "slots": {"金額": {"type": "number", "unit": "円", "describe": "申請の金額"}},
        "conditions": [{"name": "金額が 1 万円以上", "left": "金額", "op": ">=", "right": 10000}],
        "conclusion": {"then": "必要", "else": "不要"}}
OPT = "最後の行に「結論: 必要」か「結論: 不要」のどちらかを書いてください。"


def _make_dir(tmp_path: Path, status="confirmed", unable=None) -> tuple[Path, Path]:
    d = tmp_path / "cards"
    d.mkdir()
    card = dict(CARD, status=status)
    (d / "c-amount.json").write_text(json.dumps(card, ensure_ascii=False), encoding="utf-8")
    if unable is not None:
        (d / "unable.json").write_text(json.dumps(unable, ensure_ascii=False), encoding="utf-8")
    prompt = tmp_path / "prompt.md"
    prompt.write_text("下書きのプロンプト", encoding="utf-8")
    return d, prompt


def test_freeze_records_all_files_unable_prompt_and_rule_cards_py(tmp_path):
    d, prompt = _make_dir(tmp_path, unable=[{"rule": "架空規程 B", "reason": "原文が曖昧"}])
    f = ns.build_freeze(d, prompt, allow_draft=False)
    assert set(f["files"]) == {"c-amount.json", "unable.json"}
    assert f["unable"]["present"] and f["unable"]["rules"] == ["架空規程 B"] and len(f["unable"]["sha256"]) == 64
    assert len(f["prompt"]["sha256"]) == 64 and len(f["rule_cards_py"]["sha256"]) == 64
    assert ns.verify_freeze(f) == []


def test_freeze_stops_on_draft_without_allow_draft(tmp_path):
    d, prompt = _make_dir(tmp_path, status="draft")
    with pytest.raises(SystemExit, match="draft"):
        ns.build_freeze(d, prompt, allow_draft=False)
    assert ns.build_freeze(d, prompt, allow_draft=True)["allow_draft"] is True


def test_freeze_rejects_malformed_unable(tmp_path):
    d, prompt = _make_dir(tmp_path, unable=[{"rule": "x"}])
    with pytest.raises(SystemExit, match="unable.json"):
        ns.build_freeze(d, prompt, allow_draft=False)


def test_verify_freeze_detects_each_difference(tmp_path):
    d, prompt = _make_dir(tmp_path, unable=[])
    f = ns.build_freeze(d, prompt, allow_draft=False)
    prompt.write_text("書き換えた", encoding="utf-8")
    assert any("プロンプト" in p for p in ns.verify_freeze(f))
    prompt.write_text("下書きのプロンプト", encoding="utf-8")
    assert ns.verify_freeze(f) == []
    (d / "extra.json").write_text("{}", encoding="utf-8")
    assert any("extra.json" in p for p in ns.verify_freeze(f))
    (d / "extra.json").unlink()
    (d / "unable.json").write_text('[{"rule": "新", "reason": "後から"}]', encoding="utf-8")
    assert any("unable.json" in p for p in ns.verify_freeze(f))
    (d / "unable.json").write_text("[]", encoding="utf-8")
    bad = json.loads(json.dumps(f))
    bad["rule_cards_py"]["sha256"] = "0" * 64
    assert any("rule_cards.py" in p for p in ns.verify_freeze(bad))
    (d / "unable.json").unlink()
    assert any("unable.json" in p for p in ns.verify_freeze(f))


def _run_args(tmp_path, freeze_path, **kw):
    q = tmp_path / "q.jsonl"
    q.write_text(json.dumps({"id": "x", "question": "金額 5 円。" + OPT}, ensure_ascii=False) + "\n", encoding="utf-8")
    base = dict(set=str(q), freeze=str(freeze_path), server="no-such-server", gguf="no-such.gguf",
                out=str(tmp_path / "out"), runs=2, dev=False, sealed_dir=None, min_available_mib=0)
    base.update(kw)
    return ns.argparse.Namespace(**base)


@pytest.fixture
def no_server(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("サーバーを起動してはいけない")

    monkeypatch.setattr(na, "start_server", boom)


def test_run_stops_on_fingerprint_mismatch_before_server(tmp_path, no_server):
    d, prompt = _make_dir(tmp_path)
    fp = tmp_path / "freeze.json"
    fp.write_text(json.dumps(ns.build_freeze(d, prompt, False), ensure_ascii=False), encoding="utf-8")
    (d / "c-amount.json").write_text(json.dumps(dict(CARD, target="書き換え"), ensure_ascii=False), encoding="utf-8")
    with pytest.raises(SystemExit, match="sha256"):
        ns.cmd_run(_run_args(tmp_path, fp, dev=True))
    assert not (tmp_path / "out").exists()


def test_run_without_dev_stops_on_draft_and_on_sealed_mismatch(tmp_path, no_server, monkeypatch):
    d, prompt = _make_dir(tmp_path, status="draft")
    fp = tmp_path / "freeze.json"
    fp.write_text(json.dumps(ns.build_freeze(d, prompt, True), ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(ns, "verify_sealed", lambda *a: [])  # 封の照合は通った事にする
    with pytest.raises(SystemExit, match="allow-draft"):
        ns.cmd_run(_run_args(tmp_path, fp, sealed_dir="x"))
    # confirmed だけの freeze でも、封の照合が通らなければ止まる
    (tmp_path / "b").mkdir()
    d2, prompt2 = _make_dir(tmp_path / "b")
    fp2 = tmp_path / "freeze2.json"
    fp2.write_text(json.dumps(ns.build_freeze(d2, prompt2, False), ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(ns, "verify_sealed", lambda *a: ["規程の sha256 の組が違う"])
    with pytest.raises(SystemExit, match="規程の sha256"):
        ns.cmd_run(_run_args(tmp_path, fp2, sealed_dir="x"))


def test_run_without_dev_requires_sealed_dir(tmp_path, no_server):
    d, prompt = _make_dir(tmp_path)
    fp = tmp_path / "freeze.json"
    fp.write_text(json.dumps(ns.build_freeze(d, prompt, False), ensure_ascii=False), encoding="utf-8")
    with pytest.raises(SystemExit, match="sealed-dir"):
        ns.cmd_run(_run_args(tmp_path, fp))


def test_load_cards_checked_stops_on_draft_unless_allowed(tmp_path):
    d, _ = _make_dir(tmp_path, status="draft")
    with pytest.raises(SystemExit, match="draft"):
        ns.load_cards_checked(d, allow_draft=False)
    assert len(ns.load_cards_checked(d, allow_draft=True)) == 1


# ---------------------------------------------------------------- 1 回分を流す


class FakeExtractor:
    def __init__(self, reply):
        self.reply, self.calls, self.last_stats = reply, 0, {}

    async def extract(self, messages, schema):
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _item(id_, question, **kw):
    return {"id": id_, "question": question, **kw}


def _card():
    from app.services import rule_cards

    return rule_cards.parse_card(CARD)


def _round(rows, reply):
    ex = FakeExtractor(reply)
    return asyncio.run(ns.sealed_round(rows, [_card()], ex, 1)), ex


def test_round_classifies_stop_reasons_and_skips_model_for_no_card(monkeypatch):
    # 小さな架空カードなので、選ぶ下限はこの試験の中だけ下げる
    from app.services import rule_cards
    monkeypatch.setattr(rule_cards, "SELECT_MIN_SCORE", 1)
    monkeypatch.setattr(rule_cards, "SELECT_MIN_MARGIN", 0)
    q = "申請の金額は 12,000円です。" + OPT
    ok = {"金額": {"value": 12000, "quote": "12,000円"}}
    recs, ex = _round([_item("a", q, rule_id="R1", should_hold=False, expected_conclusion="必要")], ok)
    assert (recs[0]["outcome"], recs[0]["conclusion"], recs[0]["stop_reason"]) == ("decided", "必要", None)
    recs, _ = _round([_item("a", q)], RuntimeError("timeout"))
    assert (recs[0]["outcome"], recs[0]["stop_reason"]) == ("held", ns.EXTRACT_FAILED)
    recs, _ = _round([_item("a", q)], {"金額": {"value": 12000, "quote": "99,000円"}})
    assert recs[0]["stop_reason"] == ns.VALIDATION_FAILED
    recs, _ = _round([_item("a", q)], {"金額": {"value": None, "quote": None}})
    assert (recs[0]["outcome"], recs[0]["stop_reason"]) == ("held", ns.UNKNOWN_HOLD)
    # 選択肢の語が無い・手がかりの語が無い問はモデルに回さない
    recs, ex = _round([_item("n1", "金額は 5 円です。"), _item("n2", "日数は 3 日です。" + OPT)], ok)
    assert [r["outcome"] for r in recs] == ["no_card", "no_card"] and ex.calls == 0
    assert [r["stop_reason"] for r in recs] == [ns.NO_CARD] * 2


def test_round_tolerates_missing_fields():
    recs, _ = _round([{"question": "金額は 12,000円です。" + OPT}], {"金額": {"value": 12000, "quote": "12,000円"}})
    assert recs[0]["id"] == "row0" and recs[0]["rule_id"] is None and recs[0]["should_hold"] is False


def test_summary_writes_file_and_prints_no_question_text(tmp_path, capsys):
    out = tmp_path / "o"
    out.mkdir()
    (out / "run.json").write_text(json.dumps({"runs": 1, "n": 240, "dev": True, "unable_rules": ["架空規程 B"]},
                                             ensure_ascii=False), encoding="utf-8")
    (out / "raw.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in _full_set()) + "\n",
                                   encoding="utf-8")
    assert ns.cmd_summary(ns.argparse.Namespace(out=str(out))) == 0
    doc = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert doc["ok"] and doc["rounds"]["1"]["answerable"] == 192 and doc["unable_rules"] == ["架空規程 B"]
    assert "全体: 合格" in capsys.readouterr().out


# ---------------------------------------------------------------- extract_options


def test_extract_options_outputs_only_word_sets(tmp_path, capsys):
    rows = [
        {"id": "SECRET-ID-1", "question": "秘密の本文 1", "expected_conclusion": "正解 A", "conclusion_options": ["必要", "不要"]},
        {"id": "SECRET-ID-2", "question": "秘密の本文 2", "expected_conclusion": "正解 B", "conclusion_options": ["不要", "必要"]},
        {"id": "SECRET-ID-3", "question": "秘密の本文 3", "conclusion_options": ["受理される", "受理されない"]},
        {"id": "SECRET-ID-4", "question": "選択肢の欄が無い問"},
    ]
    src, dst = tmp_path / "q.jsonl", tmp_path / "opts.json"
    src.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    assert eo.main([str(src), "--out", str(dst)]) == 0
    cap = capsys.readouterr()
    assert cap.out.strip() == "2"
    text = dst.read_text(encoding="utf-8")
    for leak in ("SECRET", "秘密", "正解", "選択肢の欄が無い"):
        assert leak not in text and leak not in cap.out
    doc = json.loads(text)
    pairs = doc["conclusion_options"]
    assert pairs == sorted(pairs) and all(p == sorted(p) for p in pairs)
    assert pairs == [["不要", "必要"], ["受理されない", "受理される"]]
    assert set(doc) == {"format", "conclusion_options"}
