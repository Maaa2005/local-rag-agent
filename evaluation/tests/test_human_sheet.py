"""人手採点の表 (evaluation/human_sheet.py) のテスト。偽の raw.jsonl だけを使う。"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict

import pytest

from evaluation import human_sheet as hs
from evaluation.types import QuestionResult

LABELS = ("secret_label_alpha", "secret_label_beta")
MODEL_WORDS = ("secret-model-xyz",)


def _row(qid, answer, label, **kw):
    r = QuestionResult(
        id=qid, provider="vllm", question=f"{qid[-1]}番目の話題について教えてください", category="general",
        user_access_level=1, answer=answer, first_token_latency=7.7731, total_latency=123.4567,
        retrieved_chunks=[{"source_file": "規程.md", "content": f"規程の本文その{qid[-1]}。全文が出る。"}],
        expected_answer_keywords=["本文"],
    )
    return {**asdict(r), "run_label": label, "model": MODEL_WORDS[0], **kw}


def _write(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _raw(tmp_path, n=12, name="raw.jsonl"):
    rows = []
    for label in LABELS:
        for i in range(n):
            rows.append(_row(f"ZQ{i:03d}", f"第{i}の回答です。次の文です。", label))
    p = tmp_path / name
    _write(p, rows)
    return p


def _make(tmp_path, raw, seed=1, tag="a"):
    out_dir, key = tmp_path / f"sheet_{tag}", tmp_path / f"key_{tag}.json"
    code = hs.main(["make", "--raw", str(raw), "--seed", str(seed), "--out-dir", str(out_dir), "--key-out", str(key)])
    return code, out_dir, key


def _read_sheet(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.reader(f))


def _write_sheet(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerows(rows)


def _fill(rows, verdict="1", reason=""):
    return [rows[0]] + [r[:3] + [verdict, reason] for r in rows[1:]]


def _score(tmp_path, sheet, key, out="score.json"):
    return hs.main(["score", "--sheet", str(sheet), "--key", str(key), "--out", str(tmp_path / out)])


# ─── make ──────────────────────────────────────────────────────
def test_same_seed_same_order_different_seed_different_order(tmp_path):
    raw = _raw(tmp_path)
    _, _, k1 = _make(tmp_path, raw, 5, "a")
    _, _, k2 = _make(tmp_path, raw, 5, "b")
    _, _, k3 = _make(tmp_path, raw, 6, "c")
    order = lambda k: [(a["run_label"], a["question_id"]) for a in json.loads(k.read_text())["answers"].values()]
    assert order(k1) == order(k2)
    assert order(k1) != order(k3)
    assert len(order(k1)) == 24


def test_sheet_format_and_ids(tmp_path):
    code, out_dir, key = _make(tmp_path, _raw(tmp_path, 3))
    assert code == 0
    sheet = out_dir / "sheet.csv"
    assert sheet.read_bytes().startswith(b"\xef\xbb\xbf")
    rows = _read_sheet(sheet)
    assert rows[0] == ["回答ID", "文番号", "文", "判定", "理由"]
    assert rows[1][:3] == ["H001", "1", rows[1][2]] and rows[2][:2] == ["H001", "2"]
    assert all(r[3] == "" and r[4] == "" for r in rows[1:])
    assert len(rows) == 1 + 6 * 2
    assert {r[0] for r in rows[1:]} == {f"H00{i}" for i in range(1, 7)}


def test_sheet_and_read_do_not_leak_identity(tmp_path):
    _, out_dir, _ = _make(tmp_path, _raw(tmp_path))
    text = (out_dir / "sheet.csv").read_text(encoding="utf-8-sig") + (out_dir / "read.md").read_text(encoding="utf-8")
    for bad in (*LABELS, *MODEL_WORDS, "ZQ0", "7.7731", "123.4567", "raw.jsonl", "7.77", "123.45"):
        assert bad not in text


def test_read_md_has_rules_question_full_chunks_with_source_and_numbered_sentences(tmp_path):
    _, out_dir, _ = _make(tmp_path, _raw(tmp_path, 1))
    md = (out_dir / "read.md").read_text(encoding="utf-8")
    assert "保留" in md and "理由" in md
    assert "## H001" in md and "0番目の話題について教えてください" in md
    assert "規程.md" in md and "規程の本文その0。全文が出る。" in md
    assert "1. 第0の回答です。" in md and "2. 次の文です。" in md
    assert "本文にその答えがあれば 0、なければ 1" in md and "質問に答える内容" not in md


def test_sentence_cell_is_bracketed_so_spreadsheets_keep_text(tmp_path):
    sheet, key = _setup(tmp_path, [_row("ZQ000", "=1+1です。", LABELS[0])])
    rows = _read_sheet(sheet)
    assert [r[2] for r in rows[1:]] == ["「=1+1です。」"]
    _write_sheet(sheet, _fill(rows))
    assert _score(tmp_path, sheet, key) == 0


def test_unbracketed_sentence_counts_as_rewritten(tmp_path):
    sheet, key = _setup(tmp_path, [_row("ZQ000", "あ。", LABELS[0])])
    filled = _fill(_read_sheet(sheet))
    filled[1][2] = "あ。"
    _write_sheet(sheet, filled)
    assert _score(tmp_path, sheet, key) == 2


def test_auto_zero_not_in_sheet_but_in_key(tmp_path):
    rows = [
        _row("ZQ000", "普通の回答です。", LABELS[0]),
        _row("ZQ001", "", LABELS[0]),
        _row("ZQ002", "回答はある。", LABELS[0], error="run_failed: boom"),
    ]
    raw = tmp_path / "raw.jsonl"
    _write(raw, rows)
    code, out_dir, key = _make(tmp_path, raw)
    assert code == 0
    assert {r[0] for r in _read_sheet(out_dir / "sheet.csv")[1:]} == {"H001"}
    k = json.loads(key.read_text(encoding="utf-8"))
    assert [(z["run_label"], z["question_id"]) for z in k["auto_zero"]] == [(LABELS[0], "ZQ001"), (LABELS[0], "ZQ002")]
    assert k["auto_zero"][0]["reason"] == "empty_answer" and k["auto_zero"][1]["reason"].startswith("run_failed")
    assert len(k["answers"]) == 1


def test_key_contents(tmp_path):
    raw = _raw(tmp_path, 2)
    _, out_dir, key = _make(tmp_path, raw, seed=9)
    k = json.loads(key.read_text(encoding="utf-8"))
    assert k["seed"] == 9
    assert k["sheet_sha256"] == hashlib.sha256((out_dir / "sheet.csv").read_bytes()).hexdigest()
    assert k["read_sha256"] == hashlib.sha256((out_dir / "read.md").read_bytes()).hexdigest()
    assert k["raws"] == [{"path": str(raw), "sha256": hashlib.sha256(raw.read_bytes()).hexdigest()}]
    a = k["answers"]["H001"]
    assert set(a) == {"raw", "run_label", "question_id", "sentences"} and a["raw"] == str(raw)


def test_multiple_raws(tmp_path):
    r1 = tmp_path / "r1.jsonl"
    r2 = tmp_path / "r2.jsonl"
    _write(r1, [_row("ZQ000", "あ。い。", LABELS[0])])
    _write(r2, [_row("ZQ000", "う。", LABELS[1])])
    code, _, key = _make(tmp_path, [r1, r2][0])  # 単体
    assert code == 0
    out_dir, key2 = tmp_path / "multi", tmp_path / "multi_key.json"
    assert hs.main(["make", "--raw", str(r1), "--raw", str(r2), "--seed", "1", "--out-dir", str(out_dir), "--key-out", str(key2)]) == 0
    assert len(json.loads(key2.read_text())["answers"]) == 2


def test_key_out_inside_out_dir_refused(tmp_path):
    raw = _raw(tmp_path, 2)
    out_dir = tmp_path / "sheet"
    code = hs.main(["make", "--raw", str(raw), "--seed", "1", "--out-dir", str(out_dir), "--key-out", str(out_dir / "key.json")])
    assert code == 2 and not (out_dir / "key.json").exists() and not (out_dir / "sheet.csv").exists()


def test_make_refuses_existing_targets(tmp_path):
    raw = _raw(tmp_path, 2)
    out_dir = tmp_path / "sheet"
    out_dir.mkdir()
    (out_dir / "x.txt").write_text("x")
    assert hs.main(["make", "--raw", str(raw), "--seed", "1", "--out-dir", str(out_dir), "--key-out", str(tmp_path / "k.json")]) == 2
    empty = tmp_path / "empty"
    empty.mkdir()
    key = tmp_path / "k2.json"
    key.write_text("old")
    assert hs.main(["make", "--raw", str(raw), "--seed", "1", "--out-dir", str(empty), "--key-out", str(key)]) == 2
    assert key.read_text() == "old"


def test_make_refuses_over_max_answers(tmp_path):
    raw = _raw(tmp_path, 3)
    out_dir = tmp_path / "s"
    assert hs.main(["make", "--raw", str(raw), "--seed", "1", "--out-dir", str(out_dir), "--key-out", str(tmp_path / "k.json"), "--max-answers", "5"]) == 2


# ─── score ─────────────────────────────────────────────────────
def _setup(tmp_path, rows):
    raw = tmp_path / "raw.jsonl"
    _write(raw, rows)
    _, out_dir, key = _make(tmp_path, raw)
    return out_dir / "sheet.csv", key


def test_score_matches_hand_calculation_with_auto_zero(tmp_path):
    # A ラベル: Q0 = 2 文中 1 (0.5) / Q1 = 全 1 (1.0) / Q2 = auto_zero (0) → 平均 0.5, 3 問
    # B ラベル: Q0 = 全 1 → 1.0, 1 問
    rows = [
        _row("ZQ000", "あ。い。", LABELS[0]),
        _row("ZQ001", "う。え。", LABELS[0]),
        _row("ZQ002", "", LABELS[0]),
        _row("ZQ000", "お。か。", LABELS[1]),
    ]
    sheet, key = _setup(tmp_path, rows)
    k = json.loads(key.read_text())
    filled = _read_sheet(sheet)
    out = [filled[0]]
    for r in filled[1:]:
        a = k["answers"][r[0]]
        if a["run_label"] == LABELS[0] and a["question_id"] == "ZQ000" and r[1] == "2":
            out.append(r[:3] + ["0", "裏づけなし"])
        else:
            out.append(r[:3] + ["1", ""])
    _write_sheet(sheet, out)
    code = _score(tmp_path, sheet, key)
    res = json.loads((tmp_path / "score.json").read_text(encoding="utf-8"))
    a, b = res["runs"][LABELS[0]], res["runs"][LABELS[1]]
    assert (a["mean"], a["n_questions"], a["n_auto_zero"], a["result"]) == (0.5, 3, 1, "fail")
    assert (b["mean"], b["n_questions"], b["n_auto_zero"], b["result"]) == (1.0, 1, 0, "pass")
    assert code == 1 and res["result"] == "fail"


def test_score_all_pass_exit_0(tmp_path):
    sheet, key = _setup(tmp_path, [_row("ZQ000", "あ。い。", LABELS[0])])
    _write_sheet(sheet, _fill(_read_sheet(sheet)))
    assert _score(tmp_path, sheet, key) == 0


def test_score_exactly_point_nine_is_pass(tmp_path):
    rows = [_row(f"ZQ{i:03d}", "あ。", LABELS[0]) for i in range(10)]
    sheet, key = _setup(tmp_path, rows)
    filled = _fill(_read_sheet(sheet))
    filled[1][3], filled[1][4] = "0", "理由"
    _write_sheet(sheet, filled)
    code = _score(tmp_path, sheet, key)
    run = json.loads((tmp_path / "score.json").read_text())["runs"][LABELS[0]]
    assert run["mean"] == 0.9 and run["result"] == "pass" and code == 0


def test_score_just_below_point_nine_fails(tmp_path):
    rows = [_row(f"ZQ{i:03d}", "あ。", LABELS[0]) for i in range(9)] + [_row("ZQ998", "", LABELS[0]), _row("ZQ999", "", LABELS[0])]
    sheet, key = _setup(tmp_path, rows)
    _write_sheet(sheet, _fill(_read_sheet(sheet)))
    assert _score(tmp_path, sheet, key) == 1  # 9/11


def _incomplete(tmp_path, rows):
    code = _score(tmp_path, tmp_path / "sheet_a" / "sheet.csv", tmp_path / "key_a.json")
    res = json.loads((tmp_path / "score.json").read_text(encoding="utf-8"))
    return code, res


def test_pending_makes_incomplete(tmp_path):
    sheet, key = _setup(tmp_path, [_row("ZQ000", "あ。い。", LABELS[0])])
    filled = _fill(_read_sheet(sheet))
    filled[2][3] = "保留"
    _write_sheet(sheet, filled)
    code, res = _incomplete(tmp_path, None)
    assert code == 3 and res["result"] == "採点未完了" and res["runs"] is None
    assert [(x["回答ID"], x["文番号"], x["問題"]) for x in res["incomplete"]] == [("H001", 2, "保留")]


def test_zero_without_reason_makes_incomplete(tmp_path):
    sheet, key = _setup(tmp_path, [_row("ZQ000", "あ。", LABELS[0])])
    filled = _fill(_read_sheet(sheet), verdict="0", reason="  ")
    _write_sheet(sheet, filled)
    code, res = _incomplete(tmp_path, None)
    assert code == 3 and res["incomplete"][0]["問題"] == "0 なのに理由が空"


@pytest.mark.parametrize("cell", ["", "2", "0.5", "はい"])
def test_blank_or_invalid_verdict_makes_incomplete(tmp_path, cell):
    sheet, key = _setup(tmp_path, [_row("ZQ000", "あ。", LABELS[0])])
    _write_sheet(sheet, _fill(_read_sheet(sheet), verdict=cell))
    code, res = _incomplete(tmp_path, None)
    assert code == 3 and len(res["incomplete"]) == 1


def test_fullwidth_digits_are_accepted(tmp_path):
    sheet, key = _setup(tmp_path, [_row("ZQ000", "あ。", LABELS[0])])
    _write_sheet(sheet, _fill(_read_sheet(sheet), verdict="１"))
    assert _score(tmp_path, sheet, key) == 0


def test_rewritten_sentence_exits_2(tmp_path):
    sheet, key = _setup(tmp_path, [_row("ZQ000", "あ。い。", LABELS[0])])
    filled = _fill(_read_sheet(sheet))
    filled[1][2] = "書き換えた文。"
    _write_sheet(sheet, filled)
    assert _score(tmp_path, sheet, key) == 2
    assert not (tmp_path / "score.json").exists()


def test_rewritten_id_or_number_exits_2(tmp_path):
    sheet, key = _setup(tmp_path, [_row("ZQ000", "あ。い。", LABELS[0])])
    for col, val in ((0, "H009"), (1, "7")):
        filled = _fill(_read_sheet(sheet))
        filled[1][col] = val
        _write_sheet(sheet, filled)
        assert _score(tmp_path, sheet, key, out=f"s{col}.json") == 2


def test_missing_row_exits_2(tmp_path):
    sheet, key = _setup(tmp_path, [_row("ZQ000", "あ。い。", LABELS[0])])
    filled = _fill(_read_sheet(sheet))
    del filled[2]
    _write_sheet(sheet, filled)
    assert _score(tmp_path, sheet, key) == 2


def test_added_row_exits_2(tmp_path):
    sheet, key = _setup(tmp_path, [_row("ZQ000", "あ。い。", LABELS[0])])
    filled = _fill(_read_sheet(sheet))
    filled.append(["H001", "3", "足した文。", "1", ""])
    _write_sheet(sheet, filled)
    assert _score(tmp_path, sheet, key) == 2


def test_reordered_rows_are_allowed(tmp_path):
    sheet, key = _setup(tmp_path, [_row("ZQ000", "あ。い。", LABELS[0])])
    filled = _fill(_read_sheet(sheet))
    _write_sheet(sheet, [filled[0], filled[2], filled[1]])
    assert _score(tmp_path, sheet, key) == 0


def test_score_refuses_existing_out(tmp_path):
    sheet, key = _setup(tmp_path, [_row("ZQ000", "あ。", LABELS[0])])
    _write_sheet(sheet, _fill(_read_sheet(sheet)))
    (tmp_path / "score.json").write_text("old")
    assert _score(tmp_path, sheet, key) == 2
    assert (tmp_path / "score.json").read_text() == "old"
