"""decide.stamp: 採点の「可」の行だけを確認済みの表に載せる道具（計画 7.2）。一時ファイルだけを使う。"""
from __future__ import annotations

import hashlib
import json

import pytest

from decide.records import VERIFIED_FILE, load_records, load_verified_table, record_sha256
from decide.stamp import main

from .test_records import Source, money_record


def write_records(d, *ids):
    for rid in ids:
        (d / f"{rid}.json").write_text(json.dumps(money_record(rule_id=rid), ensure_ascii=False), encoding="utf-8")


def grade_file(path, lines, end=True, junk=True):
    out = ["採点を始める"] if junk else []
    out += [ln if isinstance(ln, str) else json.dumps(ln, ensure_ascii=False) for ln in lines]
    if end:
        out.append("===END===")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path


def ok(rid):
    return {"rule_id": rid, "verdict": "OK", "problems": []}


def run(d, g, capsys, *extra):
    code = main(["--records", str(d), "--grade", str(g), *extra])
    return code, capsys.readouterr()


def test_stamps_only_ok_lines_with_no_problems(tmp_path, capsys):
    write_records(tmp_path, "R-a", "R-b", "R-c", "R-d")
    g = grade_file(tmp_path / "g.txt", [
        ok("R-a"),
        {"rule_id": "R-b", "verdict": "NG", "problems": ["結論が逆"]},
        {"rule_id": "R-c", "verdict": "OK", "problems": ["引用が弱い"]},   # 「可」でも問題が残れば載せない
        {"missing_specs": [], "extra_records": []},                          # 記録の行でない JSON は無視
    ])
    code, out = run(tmp_path, g, capsys, "--graded-by", "codex-x")
    assert code == 0
    table = load_verified_table(tmp_path / VERIFIED_FILE)
    assert set(table) == {"R-a"}                       # R-b 不可・R-c 問題あり・R-d 採点の行なし
    assert table["R-a"] == record_sha256(json.loads((tmp_path / "R-a.json").read_text(encoding="utf-8")))
    raw = json.loads((tmp_path / VERIFIED_FILE).read_text(encoding="utf-8"))
    assert raw["graded_by"] == "codex-x"
    assert raw["grade_report_sha256"] == hashlib.sha256(g.read_bytes()).hexdigest()
    assert "載せた: R-a" in out.out and "R-b" in out.out and "R-c" in out.out and "R-d: 採点の行が無い" in out.out
    assert out.err == ""                                # 出力は標準出力だけ


def test_without_end_marker_exit_2_and_nothing_written(tmp_path, capsys):
    write_records(tmp_path, "R-a")
    g = grade_file(tmp_path / "g.txt", [ok("R-a")], end=False)
    code, _ = run(tmp_path, g, capsys)
    assert code == 2 and not (tmp_path / VERIFIED_FILE).exists()


def test_non_json_lines_ignored_and_lines_after_end_ignored(tmp_path, capsys):
    write_records(tmp_path, "R-a", "R-b")
    g = tmp_path / "g.txt"
    g.write_text("見出し\n{壊れた\n" + json.dumps(ok("R-a")) + "\n===END===\n" + json.dumps(ok("R-b")) + "\n", encoding="utf-8")
    assert run(tmp_path, g, capsys)[0] == 0
    assert set(load_verified_table(tmp_path / VERIFIED_FILE)) == {"R-a"}


def test_grade_line_for_missing_record_file_is_not_stamped(tmp_path, capsys):
    write_records(tmp_path, "R-a")
    g = grade_file(tmp_path / "g.txt", [ok("R-a"), ok("R-ghost")])
    code, out = run(tmp_path, g, capsys)
    assert code == 0 and set(load_verified_table(tmp_path / VERIFIED_FILE)) == {"R-a"}
    assert "R-ghost: 記録のファイルが無い" in out.out


def test_duplicate_grade_lines_one_bad_means_not_stamped(tmp_path, capsys):
    write_records(tmp_path, "R-a", "R-b")
    for order in ([ok("R-a"), {"rule_id": "R-a", "verdict": "NG", "problems": []}],
                  [{"rule_id": "R-a", "verdict": "NG", "problems": []}, ok("R-a")]):
        g = grade_file(tmp_path / "g.txt", [*order, ok("R-b")])
        assert run(tmp_path, g, capsys)[0] == 0
        assert set(load_verified_table(tmp_path / VERIFIED_FILE)) == {"R-b"}


def test_nothing_to_stamp_writes_nothing(tmp_path, capsys):
    write_records(tmp_path, "R-a")
    g = grade_file(tmp_path / "g.txt", [{"rule_id": "R-a", "verdict": "NG", "problems": ["x"]}])
    code, _ = run(tmp_path, g, capsys)
    assert code == 1 and not (tmp_path / VERIFIED_FILE).exists()


def test_missing_inputs_exit_2(tmp_path, capsys):
    assert run(tmp_path / "nope", tmp_path / "g.txt", capsys)[0] == 2


def test_stamped_dir_loads_with_require_verified(tmp_path, capsys):
    write_records(tmp_path, "R-a", "R-b")
    g = grade_file(tmp_path / "g.txt", [ok("R-a"), ok("R-b")])
    assert run(tmp_path, g, capsys)[0] == 0
    recs = load_records(tmp_path, Source(), require_verified=True)
    assert sorted(r.rule_id for r in recs) == ["R-a", "R-b"] and not any(r.stale for r in recs)
