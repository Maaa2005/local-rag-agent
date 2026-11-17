"""evaluation.nugget.score のテスト。偽の判定役で記録を作り、手計算と照合する。"""
from __future__ import annotations

import pytest

from evaluation.metrics import correctness_keyword_score
from evaluation.nugget import judge, score
from evaluation.tests.test_nugget_judge import FakeJudge
from evaluation.tests.test_nugget_schema import CORPUS, make_rows

ROWS = {r["qid"]: r for r in make_rows()}


def rec(qid, text, run="r1", send=None):
    row = ROWS[qid]
    res = judge.judge_answer(
        send or FakeJudge(),
        rule_text=CORPUS[row["rule_id"]],
        question=row["question"],
        answer=text,
        nuggets=row["nuggets"],
    )
    return {"qid": qid, "run": run, "answer": text, **res}


def undetermined(qid, run="r1"):
    return rec(qid, "要点です。", run, send=FakeJudge(always={1: "壊れ"}))


def test_vital_keywords_union_without_duplicates():
    assert score.vital_keywords(ROWS["Q1"]) == ["承認", "6か月", "1回"]  # okay の事実は入れない
    row = {"nuggets": [
        {"nid": "a", "importance": "vital", "keywords": ["x", "y"]},
        {"nid": "b", "importance": "vital", "keywords": ["y", "z"]},
    ]}
    assert score.vital_keywords(row) == ["x", "y", "z"]


def test_score_record_v_strict_and_m6():
    text = "承認通知が届いた社員は6か月です<<Q1-n1:full>><<Q1-n2:partial>>。"
    s = score.score_record(rec("Q1", text), ROWS["Q1"])
    assert s["determined"]
    assert s["V"] == pytest.approx((1 + 0.5) / 2)       # 0.75
    assert s["V_strict"] == pytest.approx(1 / 2)         # full だけ 1
    assert s["C"] == 0.0 and s["contradictions"] == []
    # M6: 承認・6か月は含まれるが 1回 は含まれない → 2/3（metrics と同じ関数）
    assert s["m6"] == pytest.approx(2 / 3)
    assert s["m6"] == correctness_keyword_score(text, ["承認", "6か月", "1回"])
    assert s["points"] == {"Q1-n1": 1.0, "Q1-n2": 0.5}


def test_score_record_none_is_zero_and_all_full_is_one():
    assert score.score_record(rec("Q1", "要点です。"), ROWS["Q1"])["V"] == 0.0
    full = "要点です<<Q1-n1:full>><<Q1-n2:full>>。"
    s = score.score_record(rec("Q1", full), ROWS["Q1"])
    assert s["V"] == 1.0 and s["V_strict"] == 1.0


def test_contradiction_flag_and_label_counts():
    text = "要点です<<Q1-n1:full>><<Q1-n2:full>>。\n矛盾した断定です。\n補足の文です。"
    s = score.score_record(rec("Q1", text), ROWS["Q1"])
    assert s["C"] == 1.0
    assert s["contradictions"] == ["矛盾した断定です。"]
    assert s["label_counts"]["vital"] == 1
    assert s["label_counts"]["contradiction"] == 1
    assert s["label_counts"]["okay"] == 1
    assert s["label_counts"]["boilerplate"] == 0
    # 相反は V と足し合わせない: V は 1.0 のまま
    assert s["V"] == 1.0


def test_m6_is_none_without_keywords():
    row = {"qid": "Z", "rule_id": "r", "nuggets": [{"nid": "Z-n1", "text": "t", "importance": "vital", "quote": "q", "keywords": []}]}
    r = judge.judge_answer(FakeJudge(), rule_text="", question="q", answer="要点です<<Z-n1:full>>。", nuggets=row["nuggets"])
    s = score.score_record({"qid": "Z", "answer": "要点です<<Z-n1:full>>。", **r}, row)
    assert s["V"] == 1.0 and s["m6"] is None


def test_undetermined_record_has_no_score():
    s = score.score_record(undetermined("Q1"), ROWS["Q1"])
    assert s["determined"] is False and s["V"] is None and s["C"] is None


# ─── K 回の平均・除外 ───────────────────────────────────────────
def test_k_runs_average_per_question_not_pooled():
    run1 = [rec("Q1", "要点です<<Q1-n1:full>><<Q1-n2:partial>>。", "k1"), rec("Q2", "要点です<<Q2-n1:full>>。", "k1")]
    run2 = [rec("Q1", "要点です<<Q1-n1:full>><<Q1-n2:full>>。", "k2"), rec("Q2", "要点です<<Q2-n1:none>>。", "k2")]
    st = score.aggregate_setting([run1, run2], ROWS)
    assert st["n_runs"] == 2
    assert st["items"]["Q1"]["s"] == pytest.approx((0.75 + 1.0) / 2)   # 0.875
    assert st["items"]["Q2"]["s"] == pytest.approx((1.0 + 0.0) / 2)    # 0.5
    assert st["items"]["Q1"]["s_strict"] == pytest.approx((0.5 + 1.0) / 2)
    assert st["items"]["Q1"]["k_used"] == 2 and st["items"]["Q1"]["k_total"] == 2
    # 落ちた事実: Q1-n2 は (0.5+1)/2 = 0.75 < 1、Q1-n1 は 1.0 なので入らない
    assert st["items"]["Q1"]["lost"] == {"Q1-n2": pytest.approx(0.75)}
    assert st["items"]["Q2"]["lost"] == {"Q2-n1": pytest.approx(0.5)}
    assert st["missing"] == ["Q3", "Q4"]


def test_undetermined_run_is_dropped_from_mean_and_all_undetermined_question_is_excluded():
    run1 = [rec("Q1", "要点です<<Q1-n1:full>><<Q1-n2:full>>。", "k1"), undetermined("Q2", "k1")]
    run2 = [undetermined("Q1", "k2"), undetermined("Q2", "k2")]
    st = score.aggregate_setting([run1, run2], ROWS)
    # Q1: 判定できた 1 回だけの平均（0 点の回として混ぜない）
    assert st["items"]["Q1"]["s"] == 1.0
    assert st["items"]["Q1"]["k_used"] == 1 and st["items"]["Q1"]["k_total"] == 2
    # Q2: 1 回も判定できず → 外す（0 点にしない）
    assert "Q2" not in st["items"] and st["excluded"] == ["Q2"]
    summ = score.summarize(st)
    assert summ["n"] == 1 and summ["n_excluded"] == 1
    assert summ["mean_V"] == 1.0


def test_unknown_qid_is_ignored_and_reported():
    run = [{**rec("Q2", "要点です。"), "qid": "ZZZ"}]
    st = score.aggregate_setting([run], ROWS)
    assert st["unknown"] == ["ZZZ"] and st["items"] == {}


def test_summarize_values_and_m6():
    run = [
        rec("Q1", "承認 6か月 1回 です<<Q1-n1:full>><<Q1-n2:full>>。"),          # V=1, m6=1
        rec("Q2", "要点です<<Q2-n1:none>>。\n矛盾した断定です。"),                # V=0, C=1, m6=0 (1回 なし)
        rec("Q3", "要点です<<Q3-n1:full>><<Q3-n2:partial>>。"),                    # V=.75, m6=0
        rec("Q4", "要点です<<Q4-n1:full>>。10,000円"),                              # V=1, m6=1
    ]
    st = score.aggregate_setting([run], ROWS)
    summ = score.summarize(st)
    assert summ["n"] == 4 and summ["n_runs"] == 1
    assert summ["mean_V"] == pytest.approx((1 + 0 + 0.75 + 1) / 4)
    assert summ["mean_V_strict"] == pytest.approx((1 + 0 + 0.5 + 1) / 4)
    assert summ["contradiction_rate"] == pytest.approx(1 / 4)
    assert summ["m6"] == pytest.approx((1 + 0 + 0 + 1) / 4)
    # SE: s = [1, 0, .75, 1] → d̄ ではなく s の SE（CLT）
    s = [1, 0, 0.75, 1]
    m = sum(s) / 4
    var = sum((x - m) ** 2 for x in s) / 3
    assert summ["se_clt"] == pytest.approx((var / 4) ** 0.5)
    assert summ["se_cluster"] is not None
    assert score.summarize(st, use_cluster=False)["se_cluster"] is None
    assert score.clusters_of(ROWS) == {"Q1": "貸与規程", "Q2": "貸与規程", "Q3": "出張規程", "Q4": "出張規程"}
