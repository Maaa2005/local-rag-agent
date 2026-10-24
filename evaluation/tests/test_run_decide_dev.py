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
    return rdd.run_rows(rows)


def test_all_dev_rows_are_run_in_file_order(rows, results):
    assert len(rows) == 36 and [r["id"] for r in results] == [r["id"] for r in rows]


def test_dev_rows_are_all_decided_and_correct(results):
    # 開発用の問は規程の文書と記録を作る時に見ている集合なので、全問正解は「確認済み」であって汎化の数ではない
    bad = [(r["id"], r["hold_code"], r["got"], r["expected"]) for r in results if not r["correct"]]
    assert not bad, bad
    assert {r["kind"] for r in results} == {"decided"}


def test_answer_does_not_depend_on_grading_columns(rows):
    keep = ("id", "question", "user_access_level")
    stripped = [{k: r[k] for k in keep} for r in rows]  # 列名を推測せず、必要な 3 列以外を全部落とす
    assert set(rows[0]) - set(keep)  # 落とす列が実際にある
    a = rdd.run_rows(rows)
    b = rdd.run_rows(stripped)
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

    real = rdd.make_engine()
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
    for r in rdd.run_rows(lowered):
        assert r["kind"] != "decided" and r["decision_record"]["final_answer"] is None


def test_decision_record_is_one_per_row_with_id_and_correct(results):
    ids = [r["decision_record"]["id"] for r in results]
    assert len(ids) == len(set(ids)) == 36
    for r in results:
        d = r["decision_record"]
        assert d["correct"] is r["correct"] and d["doc_version"] and d["final_answer"] == (
            None if r["kind"] != "decided" else d["final_answer"])


def test_main_writes_jsonl_outside_repo_and_refuses_inside(tmp_path, capsys):
    out = tmp_path / "dev.jsonl"
    assert rdd.main(["--out", str(out)]) == 0
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 36
    for k in ("id", "correct", "hold_code", "rule_id", "doc_version", "evidence", "extracted", "computation",
              "conclusion", "final_answer", "kind"):
        assert k in lines[0]
    printed = capsys.readouterr().out
    assert "正解 36/36" in printed and "段ごと" in printed
    inside = ROOT / "evaluation" / "_never_written.jsonl"
    assert rdd.main(["--out", str(inside)]) == 2
    assert not inside.exists()
    assert rdd.main(["--out", str(ROOT / "x.jsonl")]) == 2


def test_engine_side_modules_do_not_open_jsonl_or_question_files():
    for name in ("engine", "records", "parse", "render", "__init__"):
        src = (ROOT / "v2/decide" / f"{name}.py").read_text(encoding="utf-8")
        for banned in ("numeric_boundary", "numeric_dev24", "holdout", "expected_conclusion", "expected_sources"):
            assert banned not in src, (name, banned)
        assert 'glob("*.jsonl")' not in src


def test_records_directory_holds_only_json_records():
    files = sorted(p.name for p in rdd.RECORDS_DIR.iterdir())
    assert files and all(f.endswith(".json") for f in files)
