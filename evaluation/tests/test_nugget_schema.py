"""evaluation.nugget.schema のテスト（検査・freeze・答えの読み込み）。"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from evaluation.nugget import schema

# ─── 作り物の規程と事実（他の nugget テストからも使う） ───────────
CORPUS = {
    "貸与規程": (
        "第1条 在宅勤務の承認通知が届いた社員は、機器を貸与できる。\n"
        "第2条 貸与期間は6か月とし、延長は1回までとする。\n"
        "第3条 貸与機器は\n私物と同様に扱ってはならない。\n"
    ),
    "出張規程": (
        "第1条 国内出張の日当は3,000円とする。\n"
        "第2条 宿泊費の上限は10,000円とする。\n"
    ),
}


def _n(nid, text, imp, quote, kws=()):
    return {"nid": nid, "text": text, "importance": imp, "quote": quote, "keywords": list(kws)}


def make_rows() -> list[dict]:
    base = {"set": "toy", "excluded": [], "drafted_by": "m", "reviewed_by": "人", "version": "ng-1"}
    return [
        {**base, "qid": "Q1", "rule_id": "貸与規程", "question": "貸与の条件と期間は？", "nuggets": [
            _n("Q1-n1", "承認通知が届いた社員が対象", "vital", "在宅勤務の承認通知が届いた社員は、機器を貸与できる。", ["承認"]),
            _n("Q1-n2", "期間は6か月で延長は1回", "vital", "貸与期間は6か月とし、延長は1回までとする。", ["6か月", "1回"]),
            _n("Q1-n3", "私物扱い不可", "okay", "貸与機器は私物と同様に扱ってはならない。"),
        ]},
        {**base, "qid": "Q2", "rule_id": "貸与規程", "question": "延長は何回？", "nuggets": [
            _n("Q2-n1", "延長は1回まで", "vital", "延長は1回までとする。", ["1回"]),
        ]},
        {**base, "qid": "Q3", "rule_id": "出張規程", "question": "日当と宿泊費は？", "nuggets": [
            _n("Q3-n1", "日当は3,000円", "vital", "国内出張の日当は3,000円とする。", ["3,000円"]),
            _n("Q3-n2", "宿泊費上限は10,000円", "vital", "宿泊費の上限は10,000円とする。", ["10,000円"]),
        ]},
        {**base, "qid": "Q4", "rule_id": "出張規程", "question": "宿泊費の上限は？", "nuggets": [
            _n("Q4-n1", "宿泊費上限は10,000円", "vital", "宿泊費の上限は10,000円とする。", ["10,000円"]),
        ]},
    ]


def write_jsonl(path: Path, rows) -> Path:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def make_set(tmp_path: Path, rows=None) -> Path:
    """tmp_path/corpus/<rule>.md と tmp_path/nuggets.jsonl を作り nuggets.jsonl の Path を返す。"""
    (tmp_path / "corpus").mkdir(exist_ok=True)
    for rid, body in CORPUS.items():
        (tmp_path / "corpus" / f"{rid}.md").write_text(body, encoding="utf-8")
    return write_jsonl(tmp_path / "nuggets.jsonl", rows if rows is not None else make_rows())


# ─── 検査 ─────────────────────────────────────────────────────
def test_valid_rows_pass_even_with_newline_difference_in_quote():
    # Q1-n3 の quote は規程では途中で改行されている
    assert schema.validate_nuggets(make_rows(), CORPUS) == []


def test_quote_matching_ignores_whitespace_only():
    assert schema.quote_in_text("貸与機器は 私物と\n同様に", CORPUS["貸与規程"])
    assert schema.quote_in_text("貸与機器は　私物", CORPUS["貸与規程"])  # 全角空白
    assert not schema.quote_in_text("貸与機器は私物を同様に", CORPUS["貸与規程"])
    assert not schema.quote_in_text("   ", CORPUS["貸与規程"])


def test_duplicate_qid_and_nid():
    rows = make_rows()
    rows[1]["qid"] = "Q1"
    rows[3]["nuggets"][0]["nid"] = "Q3-n1"
    errs = schema.validate_nuggets(rows, CORPUS)
    assert any("qid が重複" in e for e in errs)
    assert any("nid が重複" in e for e in errs)


def test_importance_must_be_vital_or_okay():
    rows = make_rows()
    rows[0]["nuggets"][2]["importance"] = "minor"
    errs = schema.validate_nuggets(rows, CORPUS)
    assert any("importance" in e for e in errs)


def test_at_least_one_vital():
    rows = make_rows()
    rows[1]["nuggets"][0]["importance"] = "okay"
    errs = schema.validate_nuggets(rows, CORPUS)
    assert any("vital が 1 つも無い" in e and "Q2" in e for e in errs)


def test_quote_not_verbatim_is_reported():
    rows = make_rows()
    rows[2]["nuggets"][0]["quote"] = "国内出張の日当は4,000円とする。"
    errs = schema.validate_nuggets(rows, CORPUS)
    assert any("Q3-n1" in e and "逐語" in e for e in errs)


def test_unknown_rule_and_empty_quote():
    rows = make_rows()
    rows[3]["rule_id"] = "存在しない規程"
    rows[0]["nuggets"][0]["quote"] = ""
    errs = schema.validate_nuggets(rows, CORPUS)
    assert any("corpus に無い" in e for e in errs)
    assert any("Q1-n1" in e and "quote が空" in e for e in errs)


def test_validate_file_reads_corpus_dir(tmp_path):
    p = make_set(tmp_path)
    rows, errs = schema.validate_file(p)
    assert len(rows) == 4 and errs == []
    # 引数で dict を渡す形
    rows, errs = schema.validate_file(p, {"貸与規程": "", "出張規程": ""})
    assert errs  # 逐語が見つからない


def test_load_nuggets_reports_bad_json_line(tmp_path):
    p = tmp_path / "n.jsonl"
    p.write_text('{"qid": "a"}\n{oops\n', encoding="utf-8")
    with pytest.raises(schema.SchemaError, match=":2:"):
        schema.load_nuggets(p)


# ─── freeze ───────────────────────────────────────────────────
def test_sha256_ignores_line_endings(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_bytes(b'{"x":1}\n{"y":2}\n')
    b.write_bytes(b'{"x":1}\r\n{"y":2}\r\n')
    assert schema.sha256_lf(a) == schema.sha256_lf(b)
    b.write_bytes(b'{"x":1}\r\n{"y":3}\r\n')
    assert schema.sha256_lf(a) != schema.sha256_lf(b)


def test_freeze_write_and_check(tmp_path):
    p = make_set(tmp_path)
    now = dt.datetime(2026, 11, 18, 9, 30, 0, tzinfo=dt.timezone.utc)
    info = schema.write_freeze(p, now=now)
    assert info["version"] == "ng-1"
    assert info["frozen_at"] == "2026-11-18T09:30:00Z"
    assert info["sha256"] == schema.sha256_lf(p)
    saved = json.loads((tmp_path / "freeze.json").read_text(encoding="utf-8"))
    assert saved["sha256"] == info["sha256"] and saved["n_questions"] == 4
    assert schema.check_freeze(p)["version"] == "ng-1"


def test_freeze_mismatch_stops(tmp_path):
    p = make_set(tmp_path)
    schema.write_freeze(p)
    rows = make_rows()
    rows[0]["nuggets"][0]["text"] = "後から直した"
    write_jsonl(p, rows)
    with pytest.raises(schema.FreezeError, match="食い違う"):
        schema.check_freeze(p)


def test_freeze_missing_stops(tmp_path):
    p = make_set(tmp_path)
    with pytest.raises(schema.FreezeError, match="freeze が無い"):
        schema.check_freeze(p)


def test_freeze_same_version_cannot_be_overwritten_but_new_version_can(tmp_path):
    p = make_set(tmp_path)
    schema.write_freeze(p)
    rows = make_rows()
    rows[0]["nuggets"][0]["text"] = "直した"
    write_jsonl(p, rows)
    with pytest.raises(schema.FreezeError, match="版を上げる"):
        schema.write_freeze(p)
    for r in rows:
        r["version"] = "ng-2"
    write_jsonl(p, rows)
    assert schema.write_freeze(p)["version"] == "ng-2"
    assert schema.check_freeze(p)["version"] == "ng-2"


def test_freeze_survives_crlf_checkout(tmp_path):
    p = make_set(tmp_path)
    schema.write_freeze(p)
    lf = p.read_bytes().replace(b"\r\n", b"\n")
    p.write_bytes(lf.replace(b"\n", b"\r\n"))
    assert b"\r\n" in p.read_bytes()
    schema.check_freeze(p)
    p.write_bytes(lf)  # LF のチェックアウトでも同じ
    schema.check_freeze(p)


# ─── 答えの読み込み ───────────────────────────────────────────
def test_load_answers_accepts_id_text_and_qid_answer(tmp_path):
    f1 = write_jsonl(tmp_path / "runs_a.jsonl", [{"id": "Q1", "text": "答え1", "extra": 1}, {"id": "Q2", "text": "答え2"}])
    f2 = write_jsonl(tmp_path / "b.jsonl", [{"qid": "Q1", "answer": "答えB", "run": "seed7"}])
    a = schema.load_answers(f1)
    assert [(x["qid"], x["text"], x["run"]) for x in a] == [("Q1", "答え1", "runs_a"), ("Q2", "答え2", "runs_a")]
    b = schema.load_answers(f2)
    assert b == [{"qid": "Q1", "text": "答えB", "run": "seed7"}]


def test_load_answers_rejects_duplicate_qid_and_missing_fields(tmp_path):
    f = write_jsonl(tmp_path / "d.jsonl", [{"id": "Q1", "text": "a"}, {"id": "Q1", "text": "b"}])
    with pytest.raises(schema.SchemaError, match="2 回"):
        schema.load_answers(f)
    g = write_jsonl(tmp_path / "e.jsonl", [{"id": "Q1"}])
    with pytest.raises(schema.SchemaError):
        schema.load_answers(g)
