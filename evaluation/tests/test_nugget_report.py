"""report と cli のテスト（freeze → judge → score → compare → report を偽の判定役で通す）。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.nugget import cli, report, schema, score, stats
from evaluation.tests.test_nugget_judge import FakeJudge
from evaluation.tests.test_nugget_schema import CORPUS, make_rows, make_set, write_jsonl


# ─── 表の部品 ───────────────────────────────────────────────────
def test_cell_escapes_pipe_and_newline():
    assert report.cell("a|b\nc") == "a\\|b c"


def test_fmt_helpers():
    assert report.fmt(None) == "-" and report.fmt(0.5) == "0.500"
    assert report.fmt_signed(-0.25) == "-0.250" and report.fmt_signed(0.25) == "+0.250"
    assert report.fmt_ci((-0.1, 0.2)) == "[-0.100, +0.200]" and report.fmt_ci(None) == "-"


def test_compare_table_marks_verdicts():
    cmp = {"n": 10, "dbar": 0.2, "se_clt": 0.04, "se_cluster": 0.05, "ci": stats.ci95(0.2, 0.05), "main": "cluster"}
    aa = {"n": 10, "dbar": 0.0, "ci": (-0.05, 0.04)}
    t = report.compare_table(cmp, aa, "base", "new")
    assert "new - base" in t and "+0.200" in t and "0.050" in t
    assert "0.050" in t and stats.VERDICT_BELIEVE in t  # w = 0.05, |d̄|=0.2, 区間は 0 を含まない
    t2 = report.compare_table(cmp, None)
    assert "A/A: 未実施" in t2 and stats.VERDICT_NO_AA in t2


# ─── 一連の流れ ─────────────────────────────────────────────────
def _answers(tmp: Path, name: str, texts: dict) -> str:
    return str(write_jsonl(tmp / f"{name}.jsonl", [{"id": q, "text": t} for q, t in texts.items()]))


def _mk(q1, q2, q3, q4):
    return {"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4}


FULL = "要点です<<{n}:full>>。"


@pytest.fixture()
def world(tmp_path):
    nug = make_set(tmp_path)
    assert cli.main(["freeze", "--nuggets", str(nug)]) == 0
    a1 = _mk("要点です<<Q1-n1:full>><<Q1-n2:partial>>。", "要点です<<Q2-n1:full>>。",
             "要点です<<Q3-n1:full>><<Q3-n2:none>>。", "要点です<<Q4-n1:full>>。")
    a2 = _mk("要点です<<Q1-n1:full>><<Q1-n2:full>>。", a1["Q2"], a1["Q3"], a1["Q4"])
    aa1 = {**a1, "Q3": "要点です<<Q3-n1:full>><<Q3-n2:partial>>。"}
    aa2 = a2
    b1 = _mk("要点です<<Q1-n1:full>><<Q1-n2:none>>。", "要点です<<Q2-n1:none>>。\n矛盾した断定です。",
             "要点です<<Q3-n1:full>><<Q3-n2:full>>。", a1["Q4"])
    b2 = {**b1, "Q2": "要点です<<Q2-n1:none>>。"}
    answers = tmp_path / "answers"
    answers.mkdir()
    files = {k: _answers(answers, k, v) for k, v in
             dict(a1=a1, a2=a2, aa1=aa1, aa2=aa2, b1=b1, b2=b2).items()}
    out = tmp_path / "judged"
    send = FakeJudge()
    rc = cli.main(["judge", "--nuggets", str(nug), "--answers", *files.values(), "--out-dir", str(out),
                   "--model", "fake-model"], send=send)
    assert rc == 0
    judged = {k: str(out / f"{k}.judged.jsonl") for k in files}
    return {"nug": str(nug), "judged": judged, "out": out, "send": send, "tmp": tmp_path}


def test_judge_records_carry_version_model_and_sha(world):
    recs = [json.loads(l) for l in Path(world["judged"]["a1"]).read_text(encoding="utf-8").splitlines()]
    assert len(recs) == 4
    r = recs[0]
    assert r["judge_version"] == "NJ-3" and r["judge_model"] == "fake-model"
    assert r["nuggets_sha256"] == schema.sha256_lf(world["nug"])
    assert r["status"] == "ok" and r["run"] == "a1" and r["qid"] == "Q1"
    assert r["labels"] == ["vital"] and r["facts"] == {"Q1-n1": "full", "Q1-n2": "partial"}
    assert r["answer"].startswith("要点です")


def test_score_command(world, capsys):
    rc = cli.main(["score", "--nuggets", world["nug"], "--judged", world["judged"]["a1"], world["judged"]["a2"]])
    assert rc == 0
    out = capsys.readouterr().out
    assert "| Q1 | 貸与規程 | 0.875 | 0.750 | 0.00 |" in out  # s=(.75+1)/2, strict=(.5+1)/2
    assert "| Q3 | 出張規程 | 0.500 |" in out
    assert "平均V" in out and "M6" in out


def test_compare_command_numbers(world, capsys):
    j = world["judged"]
    rc = cli.main(["compare", "--nuggets", world["nug"], "--a", j["a1"], j["a2"], "--b", j["b1"], j["b2"],
                   "--aa", j["aa1"], j["aa2"], "--cluster", "rule", "--json"])
    assert rc == 0
    got = json.loads(capsys.readouterr().out)
    # s_A = [.875, 1, .5, 1], s_B = [.5, 0, 1, 1] → d = [-.375, -1, .5, 0]
    d = [-0.375, -1.0, 0.5, 0.0]
    assert got["n"] == 4 and got["dbar"] == pytest.approx(sum(d) / 4)
    assert got["se_clt"] == pytest.approx(stats.se_clt(d))
    assert got["se_cluster"] == pytest.approx(stats.se_clustered(d, ["貸与規程", "貸与規程", "出張規程", "出張規程"]))
    # A/A: s_AA = [.875, 1, .625, 1] → d = [0, 0, .125, 0]
    aa_d = [0.0, 0.0, 0.125, 0.0]
    aa = stats.compare({"q%d" % i: 0 for i in range(4)}, {"q%d" % i: x for i, x in enumerate(aa_d)},
                       {"q0": "x", "q1": "x", "q2": "y", "q3": "y"})
    assert got["aa_width"] == pytest.approx(stats.aa_width(aa))
    assert got["verdict"] in (stats.VERDICT_BELIEVE, stats.VERDICT_NOISE)
    assert got["judge_version"] == "NJ-3" and got["judge_model"] == "fake-model"


def test_compare_without_aa_withholds_verdict(world, capsys):
    j = world["judged"]
    cli.main(["compare", "--nuggets", world["nug"], "--a", j["a1"], "--b", j["b1"]])
    assert stats.VERDICT_NO_AA in capsys.readouterr().out


def test_report_command_worst_first_and_columns(world, capsys, tmp_path):
    j = world["judged"]
    out = tmp_path / "report.md"
    rc = cli.main(["report", "--nuggets", world["nug"], "--a", j["a1"], j["a2"], "--b", j["b1"], j["b2"],
                   "--aa", j["aa1"], j["aa2"], "--name-a", "base", "--name-b", "new", "--top", "3",
                   "--out", str(out)])
    assert rc == 0
    md = out.read_text(encoding="utf-8")
    assert "## 設定ごとの要約" in md and "## 比較" in md and "## 悪化した問" in md
    assert "| base |" in md and "| new |" in md
    assert "NJ-3" in md and "fake-model" in md
    worst = md.split("## 悪化した問")[1]
    rows = [l for l in worst.splitlines() if l.startswith("| Q")]
    assert [r.split("|")[1].strip() for r in rows] == ["Q2", "Q1", "Q4"]  # 差 -1, -0.375, 0 の順で上位 3 つ
    q2 = rows[0]
    assert "貸与規程" in q2 and "1.000" in q2 and "0.000" in q2 and "-1.000" in q2
    assert "Q2-n1 延長は1回まで" in q2          # B で落ちた vital の事実
    assert "矛盾した断定です。" in q2            # B の相反の文
    assert "要点です<<Q2-n1:none>>。" in q2      # B の答えの頭 80 字
    assert "Q1-n2 期間は6か月で延長は1回" in rows[1]


def test_report_without_b_prints_summary_only(world, capsys):
    cli.main(["report", "--nuggets", world["nug"], "--a", world["judged"]["a1"]])
    out = capsys.readouterr().out
    assert "設定ごとの要約" in out and "## 比較" not in out


def test_worst_table_head_is_80_chars():
    long = "あ" * 200
    st_a = {"items": {"Q1": {"s": 1.0, "rule_id": "r", "lost": {}, "contradictions": [], "head": ""}}}
    st_b = {"items": {"Q1": {"s": 0.0, "rule_id": "r", "lost": {"Q1-n1": 0.0}, "contradictions": [], "head": long[:80]}}}
    t = report.worst_table(st_a, st_b, {"Q1": {"nuggets": [{"nid": "Q1-n1", "text": "事実"}]}})
    assert "あ" * 80 in t and "あ" * 81 not in t


# ─── 止める場面 ─────────────────────────────────────────────────
def test_judge_stops_when_freeze_mismatch(world, capsys, tmp_path):
    rows = make_rows()
    rows[0]["nuggets"][0]["text"] = "freeze の後に直した"
    write_jsonl(Path(world["nug"]), rows)
    rc = cli.main(["judge", "--nuggets", world["nug"], "--answers", world["judged"]["a1"],
                   "--out-dir", str(tmp_path / "x")], send=FakeJudge())
    assert rc == 1
    assert "食い違う" in capsys.readouterr().err


def test_score_and_compare_stop_when_freeze_mismatch(world, capsys):
    rows = make_rows()
    rows[1]["nuggets"][0]["text"] = "直した"
    write_jsonl(Path(world["nug"]), rows)
    assert cli.main(["score", "--nuggets", world["nug"], "--judged", world["judged"]["a1"]]) == 1
    assert cli.main(["compare", "--nuggets", world["nug"], "--a", world["judged"]["a1"], "--b", world["judged"]["b1"]]) == 1
    assert "食い違う" in capsys.readouterr().err


def test_old_version_judged_files_are_rejected_after_new_freeze(world, capsys):
    rows = make_rows()
    rows[1]["nuggets"][0]["text"] = "直した"
    for r in rows:
        r["version"] = "ng-2"
    write_jsonl(Path(world["nug"]), rows)
    assert cli.main(["freeze", "--nuggets", world["nug"]]) == 0
    rc = cli.main(["score", "--nuggets", world["nug"], "--judged", world["judged"]["a1"]])
    assert rc == 1
    assert "sha256" in capsys.readouterr().err  # 前の版で判定した記録とは混ぜない


def test_mixed_judge_models_are_rejected(world, capsys, tmp_path):
    p = Path(world["judged"]["b1"])
    recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    for r in recs:
        r["judge_model"] = "other-model"
    other = write_jsonl(tmp_path / "other.judged.jsonl", recs)
    rc = cli.main(["compare", "--nuggets", world["nug"], "--a", world["judged"]["a1"], "--b", str(other)])
    assert rc == 1 and "混在" in capsys.readouterr().err


def test_judge_all_undetermined_returns_2_and_score_excludes(tmp_path, capsys):
    nug = make_set(tmp_path)
    cli.main(["freeze", "--nuggets", str(nug)])
    ans = _answers(tmp_path, "run1", {"Q1": "要点です。", "Q2": "要点です。", "ZZ": "nuggets に無い問"})
    out = tmp_path / "j"
    rc = cli.main(["judge", "--nuggets", str(nug), "--answers", ans, "--out-dir", str(out)],
                  send=FakeJudge(always={1: "壊れ"}))
    assert rc == 2
    text = capsys.readouterr().out
    assert "判定できず 2 件" in text and "nuggets に無い qid 1 件" in text
    cli.main(["score", "--nuggets", str(nug), "--judged", str(out / "run1.judged.jsonl")])
    s = capsys.readouterr().out
    assert "判定できず（点から外した問）: Q1, Q2" in s


# ─── freeze / draft の CLI ──────────────────────────────────────
def test_freeze_cli_refuses_invalid_nuggets(tmp_path, capsys):
    rows = make_rows()
    rows[0]["nuggets"][0]["quote"] = "規程に無い文"
    nug = make_set(tmp_path, rows)
    assert cli.main(["freeze", "--nuggets", str(nug)]) == 1
    assert "逐語で無い" in capsys.readouterr().err
    assert not (tmp_path / "freeze.json").exists()


def test_draft_cli_writes_reviewable_rows(tmp_path):
    (tmp_path / "corpus").mkdir()
    for rid, body in CORPUS.items():
        (tmp_path / "corpus" / f"{rid}.md").write_text(body, encoding="utf-8")
    qs = write_jsonl(tmp_path / "q.jsonl", [{"id": "Q3", "question": "日当は？", "rule_id": "出張規程", "set": "toy"}])
    reply = json.dumps({"nuggets": [
        {"text": "日当は3,000円", "importance": "vital", "quote": "国内出張の日当は3,000円とする。", "keywords": ["3,000円"]},
        {"text": "でっち上げ", "importance": "okay", "quote": "ありもしない一文"},
    ]}, ensure_ascii=False)
    out = tmp_path / "nuggets.jsonl"
    rc = cli.main(["draft", "--questions", str(qs), "--corpus", str(tmp_path / "corpus"), "--out", str(out),
                   "--model", "drafter"], send=FakeJudge(queue=[reply]))
    assert rc == 0
    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert row["qid"] == "Q3" and row["reviewed_by"] == "" and row["drafted_by"] == "drafter"
    assert [n["nid"] for n in row["nuggets"]] == ["Q3-n1", "Q3-n2"]
    assert row["nuggets"][1]["flags"] == ["quote_not_verbatim"]
    # 下書きのままでは検査に通らない（quote が逐語でない）→ 人が直してから freeze
    _, errs = schema.validate_file(out)
    assert any("Q3-n2" in e for e in errs)
