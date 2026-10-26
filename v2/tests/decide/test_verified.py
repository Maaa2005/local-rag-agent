"""確認済みの表（verified.json）で、確認後に変えられた記録・表に無い記録が使う前に止まる事（計画 7.2 / F7）。

保証するのは「確認済みの物から変わったら止まる」「relation の表に合わない側は止まる」まで。
確認役が見逃した誤り（最初から結論が逆に書かれ、採点も通った記録）は止めない。
"""
from __future__ import annotations

import copy
import json

import pytest

from decide.engine import DecisionEngine
from decide.records import VERIFIED_FILE, RecordError, build_record, load_records, sha256_text
from decide.stamp import main as stamp_main

from .test_records import DOC, Source, money_record


def put(d, data, name=None):
    (d / (name or f"{data['rule_id']}.json")).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def read(d, rid):
    return json.loads((d / f"{rid}.json").read_text(encoding="utf-8"))


@pytest.fixture
def stamped(tmp_path, capsys):
    """if_else の合成の記録2つを置き、採点の「可」の行から道具で表を作った記録のディレクトリ。"""
    put(tmp_path, money_record(rule_id="R-money"))
    put(tmp_path, money_record(rule_id="R-money2"))
    g = tmp_path / "grade.txt"
    g.write_text("\n".join(json.dumps({"rule_id": r, "verdict": "OK", "problems": []}) for r in ("R-money", "R-money2"))
                 + "\n===END===\n", encoding="utf-8")
    assert stamp_main(["--records", str(tmp_path), "--grade", str(g)]) == 0
    capsys.readouterr()
    return tmp_path


def load(d):
    return load_records(d, Source(), require_verified=True)


def test_unchanged_stamped_dir_loads(stamped):
    recs = load(stamped)
    assert sorted(r.rule_id for r in recs) == ["R-money", "R-money2"] and not any(r.stale for r in recs)
    # 設定なし（従来どおり）でも読め、表は記録として数えない
    assert len(load_records(stamped, Source())) == 2


def test_a_reversed_outcomes_stop_before_decide(stamped):
    d = read(stamped, "R-money")
    o = d["outcomes"]
    o["all_true"]["value"], o["any_false"]["value"] = o["any_false"]["value"], o["all_true"]["value"]  # 結論を両側で入れ替え
    put(stamped, d)
    # 変更後の記録は、構造も引用も全て通る（機械の検査では見つからない）事を先に確かめる
    assert build_record(copy.deepcopy(d), Source()).all_true.value == "不要"
    with pytest.raises(RecordError, match="R-money.*確認済みの表と違う"):
        load(stamped)


def test_b_changed_threshold_stops(stamped):
    d = read(stamped, "R-money")
    d["conditions"][0]["threshold"]["value"] = 100001
    put(stamped, d)
    with pytest.raises(RecordError, match="確認済みの表と違う"):
        load(stamped)


def test_c_changed_doc_version_stops(stamped):
    d = read(stamped, "R-money")
    d["doc_version"] = sha256_text(DOC + "\n追記")
    put(stamped, d)
    # 版が違えば build_record は黙って stale にするだけ。表が先に止める
    assert build_record(copy.deepcopy(d), Source()).stale
    with pytest.raises(RecordError, match="確認済みの表と違う"):
        load(stamped)


def test_d_new_unstamped_record_stops(stamped):
    put(stamped, money_record(rule_id="R-new"))
    with pytest.raises(RecordError, match="R-new.*表に無い"):
        load(stamped)


def test_removed_record_stops(stamped):
    (stamped / "R-money2.json").unlink()
    with pytest.raises(RecordError, match="R-money2"):
        load(stamped)


def test_missing_table_stops_only_when_required(tmp_path):
    put(tmp_path, money_record())
    assert len(load_records(tmp_path, Source())) == 1
    with pytest.raises(RecordError, match="確認済みの表"):
        load_records(tmp_path, Source(), require_verified=True)


def test_malformed_table_stops(stamped):
    t = json.loads((stamped / VERIFIED_FILE).read_text(encoding="utf-8"))
    for bad in ({**t, "grade_report_sha256": "xyz"}, {**t, "graded_by": ""}, {**t, "records": {"R-money": "abc"}}, [1]):
        (stamped / VERIFIED_FILE).write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(RecordError):
            load(stamped)


def test_separate_verified_path_is_used(stamped, tmp_path_factory):
    other = tmp_path_factory.mktemp("table")
    (other / "v.json").write_text((stamped / VERIFIED_FILE).read_text(encoding="utf-8"), encoding="utf-8")
    (stamped / VERIFIED_FILE).unlink()
    assert len(load_records(stamped, Source(), verified=other / "v.json", require_verified=True)) == 2


def test_e_disallowed_side_stops_at_build_without_any_table():
    d = money_record(relation="if_then")  # if_then は any_false に結論を持てない
    with pytest.raises(RecordError, match="relation"):
        build_record(d, Source())


def test_changed_record_never_reaches_decide(stamped):
    """変えられた記録は、DecisionEngine を作る前（読み込み）で止まる。"""
    d = read(stamped, "R-money")
    d["outcomes"]["all_true"]["value"], d["outcomes"]["any_false"]["value"] = "不要", "必要"
    put(stamped, d)
    reached = []
    try:
        DecisionEngine(load(stamped), None)
        reached.append(True)
    except RecordError:
        pass
    assert not reached
