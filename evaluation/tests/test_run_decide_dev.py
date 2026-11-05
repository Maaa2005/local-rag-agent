"""evaluation/v2/run_decide_dev.py: 開発用の問を「コードが決める」方式に流す道具の試験。

確かめること:
- 答えを作る側は正解欄・根拠欄・境目の欄を見ない（欄を壊しても engine の答えは変わらない）
- 閲覧レベルより上の文書は engine に渡らない
- 判断の記録は 1 問 1 件で、id と correct が足される。--out がリポジトリの中なら拒否する
- engine / records / parse / render は .jsonl を開く道具を持たない
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
SPEC = importlib.util.spec_from_file_location("run_decide_dev", ROOT / "evaluation/v2/run_decide_dev.py")
rdd = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rdd)


@pytest.fixture(scope="module")
def rows():
    return rdd.load_rows()


@pytest.fixture(scope="module")
def results(rows):
    # 確認済みの表（verified.json）にある記録だけで読む（既定の経路）
    return rdd.run_rows(rows)


@pytest.fixture(scope="module")
def results_with_rejected(rows, tmp_path_factory):
    # 採点で NG になり records_ng/ へ移した記録も戻した全記録で、表なしで読む（比べるためだけ。評価には使えない）
    d = tmp_path_factory.mktemp("all_records")
    for src in (rdd.RECORDS_DIR, rdd.ROOT / "v2/decide/records_ng"):
        for p in src.glob("R-*.json"):
            (d / p.name).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    return rdd.run_rows(rows, engine=rdd.make_engine(d, allow_unverified=True))


def test_all_dev_rows_are_run_in_file_order(rows, results):
    assert len(rows) == 36 and [r["id"] for r in results] == [r["id"] for r in rows]


def test_dev_rows_decided_ones_are_all_correct_and_the_rest_are_holds(results):
    # 計画 7.1 で変更: 表が許さない側（成り立たない側など）は unknown＝保留になり、開発用の問の一部は答えが決まらなくなった。
    # 「全問 decided・全問正解」は成り立たない。点が下がってもそのまま報告する（6 節）。
    # 守る事: 答えた問は 1 問も外れていない（間違った結論を言い切らない）。答えなかった問は保留で、モデルだけの回答には回らない
    wrong = [(r["id"], r["got"], r["expected"]) for r in results if r["kind"] == "decided" and not r["correct"]]
    assert not wrong, wrong
    assert {r["kind"] for r in results} <= {"decided", "held", "no_record"}
    assert any(r["kind"] == "decided" for r in results)
    assert all(not r["correct"] for r in results if r["kind"] != "decided")  # 開発用の問は「答えられる」問なので保留は不正解


def test_no_record_only_where_the_record_was_rejected_by_grading(results, results_with_rejected):
    # 7.2 の採点で NG の記録は表に載らず records_ng/ に移る。その分の問は記録なし（モデルだけの回答）に回る。
    # 守る事: 記録なしに回るのは NG で外した記録の問だけで、それ以外の問の答えは全記録のときと同じ
    rejected = {p.stem for p in (rdd.ROOT / "v2/decide/records_ng").glob("R-*.json")}
    assert rejected
    for a, b in zip(results, results_with_rejected, strict=True):
        if a["kind"] == "no_record" and b["kind"] != "no_record":
            assert b["rule_id"] in rejected, (a["id"], b["rule_id"])
        elif b["rule_id"] not in rejected:
            assert (a["kind"], a["hold_code"], a["got"]) == (b["kind"], b["hold_code"], b["got"]), a["id"]
    assert not [r["id"] for r in results_with_rejected if r["kind"] == "no_record"]  # 全記録なら記録なしは出ない


def test_answer_does_not_depend_on_grading_columns(rows):
    keep = ("id", "question", "user_access_level")
    stripped = [{k: r[k] for k in keep} for r in rows]  # 列名を推測せず、必要な 3 列以外を全部落とす
    assert set(rows[0]) - set(keep)  # 落とす列が実際にある
    a = rdd.run_rows(rows, allow_unverified=True)
    b = rdd.run_rows(stripped, allow_unverified=True)
    assert [(x["kind"], x["hold_code"], x["got"]) for x in a] == [(x["kind"], x["hold_code"], x["got"]) for x in b]
    assert not any(x["correct"] for x in b)  # 正解欄が無ければ採点は外れる（採点は答えの後）


def test_grading_is_after_the_engine_answers(rows):
    order = []

    class Spy:
        def decide(self, question, chunks):
            order.append("decide")
            return real.decide(question, chunks)

    class Row(dict):
        def get(self, k, d=None):
            if k == "expected_conclusion":
                order.append("grade")
            return super().get(k, d)

    real = rdd.make_engine(allow_unverified=True)
    rdd.run_rows([Row(rows[0])], engine=Spy())
    assert order == ["decide", "grade"]


def test_level_one_user_is_never_given_higher_documents():
    for level, allowed in ((1, {"general"}), (2, {"general", "manager"}), (3, {"general", "manager", "executive"})):
        ids = [c.document_id for c in rdd.visible_chunks(level)]
        assert ids and {i.split("/")[2] for i in ids} <= allowed
        assert all(c.access_level <= level for c in rdd.visible_chunks(level))
        assert len(ids) == len(set(ids))


def test_executive_question_is_not_decided_for_a_level_one_user(rows):
    execs = [r for r in rows if int(r["user_access_level"]) == 3]
    if not execs:
        pytest.skip("レベル 3 の問が無い")
    lowered = [{**r, "user_access_level": 1} for r in execs]
    for r in rdd.run_rows(lowered, allow_unverified=True):
        assert r["kind"] != "decided" and r["decision_record"]["final_answer"] is None


def test_decision_record_is_one_per_row_with_id_and_correct(results):
    ids = [r["decision_record"]["id"] for r in results]
    assert len(ids) == len(set(ids)) == 36
    for r in results:
        d = r["decision_record"]
        assert d["correct"] is r["correct"] and d["final_answer"] == (
            None if r["kind"] != "decided" else d["final_answer"])
        assert bool(d["doc_version"]) is (r["kind"] != "no_record")  # 記録なしの問だけ版が無い


def test_main_writes_jsonl_outside_repo_and_refuses_inside(tmp_path, capsys):
    out = tmp_path / "dev.jsonl"
    assert rdd.main(["--out", str(out), "--allow-unverified"]) == 0
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 36
    for k in ("id", "correct", "hold_code", "rule_id", "doc_version", "evidence", "extracted", "computation",
              "conclusion", "final_answer", "kind"):
        assert k in lines[0]
    printed = capsys.readouterr().out
    ok = sum(x["correct"] is True for x in lines)  # 7.1 の変更で 36/36 ではなくなった。数は出力の行と一致する
    assert f"正解 {ok}/36" in printed and "段ごと" in printed
    assert printed.count(rdd.UNVERIFIED_WARNING) == 1 and rdd.UNVERIFIED_WARNING == "未確認の記録で実行（評価には使えない）"
    inside = ROOT / "evaluation" / "_never_written.jsonl"
    assert rdd.main(["--out", str(inside), "--allow-unverified"]) == 2
    assert not inside.exists()
    assert rdd.main(["--out", str(ROOT / "x.jsonl"), "--allow-unverified"]) == 2


def _unstamped_copy(tmp_path):
    d = tmp_path / "rec"
    d.mkdir()
    for p in rdd.RECORDS_DIR.glob("R-*.json"):
        (d / p.name).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    assert not (d / "verified.json").exists()
    return d


def test_main_without_table_stops_at_startup_unless_allowed(tmp_path, capsys):
    d, out = _unstamped_copy(tmp_path), tmp_path / "dev.jsonl"
    assert rdd.main(["--out", str(out), "--records", str(d)]) == 2
    printed = capsys.readouterr().out
    assert "記録を読めない" in printed and rdd.UNVERIFIED_WARNING not in printed
    assert not out.exists()  # 起動で止まるので何も書かない
    assert rdd.main(["--out", str(out), "--records", str(d), "--allow-unverified"]) == 0
    assert capsys.readouterr().out.count(rdd.UNVERIFIED_WARNING) == 1 and out.exists()


def test_default_engine_requires_the_verified_table(tmp_path):
    from decide.records import RecordError
    d = _unstamped_copy(tmp_path)
    with pytest.raises(RecordError):
        rdd.make_engine(d)
    assert rdd.make_engine(d, allow_unverified=True).records


def test_engine_side_modules_do_not_open_jsonl_or_question_files():
    for name in ("engine", "records", "parse", "render", "__init__"):
        src = (ROOT / "v2/decide" / f"{name}.py").read_text(encoding="utf-8")
        for banned in ("numeric_boundary", "numeric_dev24", "holdout", "expected_conclusion", "expected_sources"):
            assert banned not in src, (name, banned)
        assert 'glob("*.jsonl")' not in src


def test_records_directory_holds_only_json_records():
    files = sorted(p.name for p in rdd.RECORDS_DIR.iterdir())
    assert files and all(f.endswith(".json") for f in files)


def test_default_path_runs_against_a_stamped_table(tmp_path, capsys):
    from decide.records import record_sha256
    d = _unstamped_copy(tmp_path)
    recs = {}
    for p in d.glob("R-*.json"):
        data = json.loads(p.read_text(encoding="utf-8"))
        recs[data["rule_id"]] = record_sha256(data)
    (d / "verified.json").write_text(json.dumps({"grade_report_sha256": "b" * 64, "graded_by": "tester",
                                                 "records": recs}), encoding="utf-8")
    out = tmp_path / "dev.jsonl"
    assert rdd.main(["--out", str(out), "--records", str(d)]) == 0  # --allow-unverified なし
    printed = capsys.readouterr().out
    assert rdd.UNVERIFIED_WARNING not in printed and "全 36 問" in printed
    assert len(out.read_text(encoding="utf-8").splitlines()) == 36
