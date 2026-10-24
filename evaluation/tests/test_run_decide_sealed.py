"""evaluation/v2/run_decide_sealed.py: 封をした問を「コードが決める」方式だけに流す道具の試験（plan 7.5）。

全部、tmp の人工の文書・記録・問だけで確かめる（封をした本物には触れない）。確かめること:
- 起動の検査（記録が確認済みの表と合う・1 件以上）は、問のファイルを開く前。失敗は終了 2 で、何も書かない
- --out / --summary がリポジトリの中なら拒否（終了 2）
- 答えを作る側は正解欄を見ない。採点は engine が返した後
- 採点: 保留が正解なら held が正、それ以外は decided で最終行が一致した時だけ正。no_record は不正解（モデルは動かさない）
- 画面には数だけ。問の文・正解・問の id は出ない
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
SPEC = importlib.util.spec_from_file_location("run_decide_sealed", ROOT / "evaluation/v2/run_decide_sealed.py")
rds = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rds)

from decide.records import record_sha256  # noqa: E402

DOC_NAME = "備品テスト規程.md"
DOC_TEXT = """# 備品テスト規程

## 第1条（目的）

この規程は、業務に用いる備品の購入手続を定める。

## 第2条（承認）

1件あたり5万円を超える備品の購入は、総務部長の承認を要する。

上記以外の備品は、所属長の承認で購入できる。
"""
SUFFIX = "最後の行には「結論: 必要」または「結論: 不要」のどちらかだけを書いてください。"
RECORD = {
    "rule_id": "R-test-approval",
    "doc_path": f"/watched/docs/{DOC_NAME}",
    "doc_version": hashlib.sha256(DOC_TEXT.encode("utf-8")).hexdigest(),
    "description": "1件あたり5万円を超える備品の購入は総務部長の承認を要する。それ以外は所属長の承認",
    "supported": True,
    "relation": "if_else",
    "subject": {"groups": [["備品"], ["承認", "総務部長"]]},
    "facts": [{"role": "amount", "label": "1件あたりの金額", "kind": "money", "cues": ["備品", "購入", "金額"]}],
    "conditions": [{"fact": "amount", "op": ">", "threshold": {"type": "number", "value": 50000},
                    "evidence": {"quote": "1件あたり5万円を超える備品の購入は、総務部長の承認を要する", "line": 9}}],
    "outcomes": {
        "all_true": {"value": "必要", "basis": "explicit",
                     "evidence": {"quote": "1件あたり5万円を超える備品の購入は、総務部長の承認を要する", "line": 9}},
        "any_false": {"value": "不要", "basis": "explicit",
                      "evidence": {"quote": "上記以外の備品は、所属長の承認で購入できる", "line": 11}},
    },
    "options": ["必要", "不要"],
}

QUESTIONS = [
    {"id": "S01", "question": f"1件49,000円の備品を購入します。総務部長の承認は必要ですか？{SUFFIX}",
     "expected_conclusion": "不要", "set": "alpha", "kind": "num"},
    {"id": "S02", "question": f"1件60,000円の備品を購入します。総務部長の承認は必要ですか？{SUFFIX}",
     "expected_conclusion": "必要", "set": "alpha", "kind": "num"},
    {"id": "S03", "question": f"1件49,000円の備品を購入します。総務部長の承認は必要ですか？{SUFFIX}",
     "expected_conclusion": "必要", "set": "beta", "kind": "num"},  # 正解欄がわざと逆: 不正解
    {"id": "S04", "question": f"備品を購入します。総務部長の承認は必要ですか？{SUFFIX}",
     "expected_conclusion": "保留", "set": "beta", "kind": "hold", "revised_to_hold": True},  # 金額が無い: 保留
    {"id": "S05", "question": f"1件60,000円の備品を購入します。総務部長の承認は必要ですか？{SUFFIX}",
     "expected_conclusion": "保留", "set": "beta", "kind": "hold"},  # 決まるのに保留が正解: 不正解
    {"id": "S06", "question": "出張旅費の日当は、国内ならいくらですか？", "expected_conclusion": "必要",
     "set": "beta", "kind": "other"},  # 記録に当たらない: no_record
]


def _write_table(rec_dir: Path, recs: dict[str, dict]) -> None:
    (rec_dir / "verified.json").write_text(json.dumps({
        "grade_report_sha256": "a" * 64, "graded_by": "tester",
        "records": {rid: record_sha256(d) for rid, d in recs.items()}}, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def world(tmp_path):
    docs = tmp_path / "root" / "docs"
    docs.mkdir(parents=True)
    (docs / DOC_NAME).write_text(DOC_TEXT, encoding="utf-8")
    rec_dir = tmp_path / "records"
    rec_dir.mkdir()
    (rec_dir / "R-test-approval.json").write_text(json.dumps(RECORD, ensure_ascii=False), encoding="utf-8")
    _write_table(rec_dir, {RECORD["rule_id"]: RECORD})
    q = tmp_path / "q.jsonl"
    q.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in QUESTIONS) + "\n", encoding="utf-8")
    return {"root": tmp_path / "root", "records": rec_dir, "questions": q, "tmp": tmp_path,
            "out": tmp_path / "out" / "rows.jsonl", "summary": tmp_path / "out" / "summary.json"}


def argv(w, **over):
    a = {"--questions": str(w["questions"]), "--docs-root": str(w["root"]), "--records": str(w["records"]),
         "--out": str(w["out"]), "--summary": str(w["summary"]), "--label": "run-1"}
    a.update(over)
    return [x for kv in a.items() for x in kv]


def rows_of(w):
    return [json.loads(x) for x in w["out"].read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def no_question_read(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("起動の検査より前に問のファイルを読んだ")
    monkeypatch.setattr(rds, "load_questions", boom)


# ---------------------------------------------------------------- 起動の検査

def test_startup_stops_without_verified_table_before_reading_questions(world, no_question_read, capsys):
    (world["records"] / "verified.json").unlink()
    assert rds.main(argv(world)) == 2
    assert "確認済みの表" in capsys.readouterr().out
    assert not world["out"].exists() and not world["summary"].exists()


def test_startup_stops_when_a_record_is_not_in_the_table(world, no_question_read, capsys):
    other = {**RECORD, "rule_id": "R-test-extra"}
    (world["records"] / "R-test-extra.json").write_text(json.dumps(other, ensure_ascii=False), encoding="utf-8")
    assert rds.main(argv(world)) == 2
    assert "R-test-extra" in capsys.readouterr().out and not world["out"].exists()


def test_startup_stops_when_a_record_hash_differs(world, no_question_read, capsys):
    changed = json.loads(json.dumps(RECORD))
    changed["outcomes"]["all_true"]["value"], changed["outcomes"]["any_false"]["value"] = "不要", "必要"  # 結論を逆に
    (world["records"] / "R-test-approval.json").write_text(json.dumps(changed, ensure_ascii=False), encoding="utf-8")
    assert rds.main(argv(world)) == 2
    assert "R-test-approval" in capsys.readouterr().out and not world["out"].exists()


def test_startup_stops_with_zero_records(world, no_question_read, capsys):
    (world["records"] / "R-test-approval.json").unlink()
    _write_table(world["records"], {})
    assert rds.main(argv(world)) == 2
    assert "0 件" in capsys.readouterr().out and not world["out"].exists()


def test_startup_stops_when_records_are_not_connected_to_documents(world, no_question_read, capsys):
    stale = {**RECORD, "doc_version": "0" * 64}  # 表には載せ直す（表との照合は通る）が、文書の版と合わない
    (world["records"] / "R-test-approval.json").write_text(json.dumps(stale, ensure_ascii=False), encoding="utf-8")
    _write_table(world["records"], {RECORD["rule_id"]: stale})
    assert rds.main(argv(world)) == 2
    printed = capsys.readouterr().out
    assert "R-test-approval" in printed and "1 / 1" in printed and not world["out"].exists()


def test_startup_stops_when_record_document_is_not_under_docs_root(world, no_question_read, capsys):
    moved = {**RECORD, "doc_path": "/watched/general/" + DOC_NAME}  # 別の置き場の記録
    (world["records"] / "R-test-approval.json").write_text(json.dumps(moved, ensure_ascii=False), encoding="utf-8")
    _write_table(world["records"], {RECORD["rule_id"]: moved})
    assert rds.main(argv(world)) == 2
    assert "R-test-approval" in capsys.readouterr().out and not world["out"].exists()


def test_startup_stops_with_empty_records_dir(world, no_question_read):
    empty = world["tmp"] / "empty"
    empty.mkdir()
    assert rds.main(argv(world, **{"--records": str(empty)})) == 2


def test_startup_stops_without_documents(world, no_question_read):
    (world["root"] / "docs" / DOC_NAME).unlink()
    assert rds.main(argv(world)) == 2


@pytest.mark.parametrize("flag", ["--out", "--summary"])
def test_out_and_summary_inside_repo_are_refused(world, flag, no_question_read):
    inside = ROOT / "evaluation" / "_never_written_sealed.json"
    assert rds.main(argv(world, **{flag: str(inside)})) == 2
    assert not inside.exists()
    assert rds.main(argv(world, **{flag: str(ROOT / "x.json")})) == 2
    assert not world["out"].exists() and not world["summary"].exists()


# ---------------------------------------------------------------- 流す・採点

def test_rows_are_graded_and_routed(world, capsys):
    assert rds.main(argv(world)) == 0
    rows = rows_of(world)
    assert [r["id"] for r in rows] == [q["id"] for q in QUESTIONS]
    got = {r["id"]: (r["kind"], r["route_final"], r["correct"]) for r in rows}
    assert got == {
        "S01": ("decided", "code", True), "S02": ("decided", "code", True),
        "S03": ("decided", "code", False),
        "S04": ("held", "held", True), "S05": ("decided", "code", False),
        "S06": ("no_record", "model_only_not_run", False),
    }
    by = {r["id"]: r for r in rows}
    assert by["S01"]["final_answer"].strip().split("\n")[-1] == "結論: 不要"
    assert by["S02"]["final_answer"].strip().split("\n")[-1] == "結論: 必要"
    for i in ("S04", "S06"):
        assert by[i]["final_answer"] is None
    for r in rows:  # 判断の記録に id・kind・route_final・final_answer・correct が足されている
        for k in ("rule_id", "doc_version", "evidence", "extracted", "computation", "conclusion", "hold_code",
                  "id", "kind", "route_final", "final_answer", "correct"):
            assert k in r
    assert by["S04"]["hold_code"]


def test_summary_numbers(world):
    assert rds.main(argv(world)) == 0
    s = json.loads(world["summary"].read_text(encoding="utf-8"))
    assert s["label"] == "run-1" and s["total"] == 6 and s["correct"] == 3
    assert s["by_kind"] == {"decided": 4, "held": 1, "no_record": 1}
    assert sum(s["by_hold_code_or_stage"].values()) == 6 and s["by_hold_code_or_stage"]["subject"] == 1
    assert s["by_set"] == {"alpha": {"correct": 2, "total": 2}, "beta": {"correct": 1, "total": 4}}
    assert s["by_question_kind"] == {"num": {"correct": 2, "total": 3}, "hold": {"correct": 1, "total": 2},
                                     "other": {"correct": 0, "total": 1}}
    assert s["revised_to_hold"] == {"total": 1, "correct": 1}
    assert s["n_records"] == 1
    assert s["verified_sha256"] == hashlib.sha256((world["records"] / "verified.json").read_bytes()).hexdigest()
    assert s["code_identity"] == rds.code_identity()
    assert "git" not in " ".join(s)  # git の版は持たない


def test_set_and_kind_groups_are_omitted_when_questions_have_no_such_fields(world):
    world["questions"].write_text(json.dumps({k: v for k, v in QUESTIONS[0].items() if k not in ("set", "kind")},
                                             ensure_ascii=False) + "\n", encoding="utf-8")
    assert rds.main(argv(world)) == 0
    s = json.loads(world["summary"].read_text(encoding="utf-8"))
    assert s["by_set"] == {} and s["by_question_kind"] == {} and s["revised_to_hold"] == {"total": 0, "correct": 0}
    assert s["total"] == 1 and s["correct"] == 1 and s["missing_expected"] == 0


def test_missing_expected_field_is_counted_not_silent(world):
    world["questions"].write_text("\n".join(json.dumps({k: v for k, v in q.items() if k != "expected_conclusion"},
                                                       ensure_ascii=False) for q in QUESTIONS) + "\n", encoding="utf-8")
    assert rds.main(argv(world)) == 0
    s = json.loads(world["summary"].read_text(encoding="utf-8"))
    assert s["missing_expected"] == 6 and s["correct"] == 0
    alt = [{**{k: v for k, v in q.items() if k != "expected_conclusion"}, "expected": q["expected_conclusion"]}
           for q in QUESTIONS]  # 別名 expected でも読める
    world["questions"].write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in alt) + "\n", encoding="utf-8")
    assert rds.main(argv(world)) == 0
    s = json.loads(world["summary"].read_text(encoding="utf-8"))
    assert s["missing_expected"] == 0 and s["correct"] == 3


def test_stdout_has_only_numbers_not_questions_or_ids(world, capsys):
    assert rds.main(argv(world)) == 0
    printed = capsys.readouterr().out
    assert json.loads(printed)["total"] == 6
    for q in QUESTIONS:
        assert q["question"] not in printed and q["id"] not in printed
    for word in ("総務部長", "49,000", "結論: "):
        assert word not in printed


def test_several_question_files_are_run_in_order(world):
    second = world["tmp"] / "q2.jsonl"
    second.write_text(json.dumps({**QUESTIONS[1], "id": "T01"}, ensure_ascii=False) + "\n", encoding="utf-8")
    argv_ = argv(world) + ["--questions", str(second)]
    assert rds.main(argv_) == 0
    ids = [r["id"] for r in rows_of(world)]
    assert ids == [q["id"] for q in QUESTIONS] + ["T01"]


def test_grading_is_after_the_engine_answers(world):
    order = []

    class Spy:
        def __init__(self, real):
            self.real = real

        def decide(self, question, chunks):
            order.append("decide")
            return self.real.decide(question, chunks)

    class Row(dict):
        def get(self, k, d=None):
            if k in rds.EXPECTED_FIELDS:
                order.append("grade")
            return super().get(k, d)

        def __getitem__(self, k):
            if k in rds.EXPECTED_FIELDS:
                order.append("grade")
            return super().__getitem__(k)

        def __contains__(self, k):
            if k in rds.EXPECTED_FIELDS:
                order.append("grade")
            return super().__contains__(k)

    engine, _ = rds.build_engine(world["records"], world["root"])
    rds.run_row(Spy(engine), Row(QUESTIONS[0]), rds.doc_chunks(world["root"]))
    assert order and order[0] == "decide" and order.count("decide") == 1 and set(order[1:]) == {"grade"}


def test_answer_does_not_depend_on_grading_columns(world):
    engine, _ = rds.build_engine(world["records"], world["root"])
    chunks = rds.doc_chunks(world["root"])
    for q in QUESTIONS:
        a, _t = rds.run_row(engine, q, chunks)
        b, tb = rds.run_row(engine, {"id": q["id"], "question": q["question"]}, chunks)
        assert (a["kind"], a["hold_code"], a["final_answer"]) == (b["kind"], b["hold_code"], b["final_answer"])
        assert b["correct"] is False  # 正解欄が無ければ、どの結果も正解にならない（採点は答えの後）


def test_rows_are_written_once_per_question_when_it_completes(world):
    engine, _ = rds.build_engine(world["records"], world["root"])
    seen = []
    real = engine.decide

    def spy(question, chunks):
        seen.append(len(world["out"].read_text(encoding="utf-8").splitlines()) if world["out"].exists() else 0)
        return real(question, chunks)
    engine.decide = spy
    import unittest.mock as m
    with m.patch.object(rds, "build_engine", return_value=(engine, 1)):
        assert rds.main(argv(world)) == 0
    assert seen == list(range(6))  # 問を流す前に、前の問までの行が既にファイルにある
    assert len(rows_of(world)) == 6


def test_chunks_are_one_per_document_at_level_one(world):
    (world["root"] / "docs" / "b.md").write_text("別の文書", encoding="utf-8")
    (world["root"] / "docs" / "not-md.txt").write_text("x", encoding="utf-8")
    chunks = rds.doc_chunks(world["root"])
    assert sorted(c.document_id for c in chunks) == sorted(["/watched/docs/b.md", f"/watched/docs/{DOC_NAME}"])
    assert all(c.access_level == 1 and c.content for c in chunks)


def test_bad_question_line_is_reported_without_echoing_it(world, capsys):
    world["questions"].write_text('{"id": "S99", "question": "秘密の問 {\n', encoding="utf-8")
    assert rds.main(argv(world)) == 2
    printed = capsys.readouterr().out
    assert "秘密の問" not in printed and "S99" not in printed and not world["out"].exists()


# ---------------------------------------------------------------- コードの同一性

def test_code_identity_follows_decide_sources_and_the_runner(tmp_path):
    dec = tmp_path / "decide"
    (dec / "sub").mkdir(parents=True)
    (dec / "a.py").write_text("A = 1\n", encoding="utf-8")
    (dec / "sub" / "b.py").write_text("B = 1\n", encoding="utf-8")
    (dec / "note.md").write_text("無関係", encoding="utf-8")
    runner = tmp_path / "runner.py"
    runner.write_text("R = 1\n", encoding="utf-8")
    base = rds.code_identity(dec, runner)
    assert base == rds.code_identity(dec, runner)
    (dec / "note.md").write_text("変えても同じ", encoding="utf-8")
    assert rds.code_identity(dec, runner) == base
    (dec / "sub" / "b.py").write_text("B = 2\n", encoding="utf-8")
    assert rds.code_identity(dec, runner) != base
    (dec / "sub" / "b.py").write_text("B = 1\n", encoding="utf-8")
    runner.write_text("R = 2\n", encoding="utf-8")
    assert rds.code_identity(dec, runner) != base
    runner.write_text("R = 1\n", encoding="utf-8")
    assert rds.code_identity(dec, runner) == base
    (dec / "sub" / "b.py").rename(dec / "sub" / "c.py")  # 中身が同じでも、パスが変われば変わる
    assert rds.code_identity(dec, runner) != base


def test_code_identity_of_real_tree_covers_decide_and_runner():
    src = ROOT / "evaluation/v2/run_decide_sealed.py"
    assert rds.code_identity() == rds.code_identity(ROOT / "v2/decide", src)
    assert len(rds.code_identity()) == 64


def test_runner_does_not_use_git_or_open_the_sealed_dir():
    src = (ROOT / "evaluation/v2/run_decide_sealed.py").read_text(encoding="utf-8")
    for banned in ("subprocess", "os.system", "rag-sealed"):
        assert banned not in src
