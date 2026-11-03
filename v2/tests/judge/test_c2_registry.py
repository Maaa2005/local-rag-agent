"""承認済み本文の登録簿（judge.c2_registry）の試験。Issue #17・本人決定 11/3。

1. 承認は本文の箇所に結ぶ。外せるのは WAIVABLE（amount・real_date）だけ。同じ種類の別の箇所の当たりは通さない
2. 登録には、承認者・当たりの承認をした人と別の人が全文を確かめる段（reviewed_by）が要る
3. 依頼文に書いてよい値（request_values）を登録に結ぶ
4. 読み込みで行の形を確かめる
"""
from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from judge import c2_registry as reg
from judge.request_cues import cue_spans
from orchestrator.policy import DEFAULT_POLICY_DIR, load_policy, sha256_text

POL = load_policy()
B = POL.contracts["B-csv-codegen@1"]
SPEC = POL.approved["spec-csv-001@1"]
EXPL = POL.approved["expl-keihi-001@1"]
REVIEWER = "reviewer-x"
TWO_FIVE_THOUSAND = "規程上限5,000円と実際の支出5,000円"


def body_text(body: str, name: str = "dev-spec-x"):
    """テスト用の承認済み本文（policies/ には置かない）。"""
    return replace(SPEC, key=f"{name}@1", body=body, sha256=sha256_text(body), request_identifiers=(),
                   request_values=())


def ack(text: str, reason: str = "規程上の額（実データではない）。試験用") -> dict:
    return {"text": text, "reason": reason}


def register(t, acknowledged=(), reviewed_by: str = REVIEWER, checked_by: str = "checker-y"):
    return reg.register(t, B, "codex", acknowledged=acknowledged, checked_by=checked_by, reviewed_by=reviewed_by)


# ===== 1. 承認を本文の箇所に結ぶ =====

def test_acknowledging_one_place_does_not_cover_the_same_kind_elsewhere():
    t = body_text(TWO_FIVE_THOUSAND)
    with pytest.raises(reg.RegistrationRefused, match="amount"):
        register(t, [ack("規程上限5,000円")])
    with pytest.raises(reg.RegistrationRefused, match="amount"):
        register(t)


def test_acknowledging_both_places_passes():
    t = body_text(TWO_FIVE_THOUSAND)
    e = register(t, [ack("規程上限5,000円"), ack("実際の支出5,000円", "試験用の架空の額")])
    assert e["hits"] == [{"cue": "amount", "text": "5,000円"}]
    assert e["acknowledged"] == [ack("実際の支出5,000円", "試験用の架空の額"), ack("規程上限5,000円")]
    assert e["result"] == "pass" and e["review"] == "pass"


def test_acknowledged_text_must_point_to_one_place():
    # 2 回出る文字列の承認は、どの箇所か決まらないので拒否する（Codex レビュー 11/3 の 7）
    t = body_text(TWO_FIVE_THOUSAND)
    with pytest.raises(reg.RegistrationRefused, match="2 times"):
        register(t, [ack("5,000円")])


def test_amount_hidden_by_spaces_is_not_covered_by_another_amount():
    # 正規化した写しで金額が 1 つ当たっても、空白を除いた写しだけで当たる別の金額は外せない（Codex 11/3 の 5）
    t = body_text("規程上限5,000円。実際の支払額1000 ド ル")
    with pytest.raises(reg.RegistrationRefused, match="amount"):
        register(t, [ack("規程上限5,000円")])


def test_explicit_reviewer_replaces_the_inherited_one():
    # 本番・封印用の登録で、開発用の確認者（fixture:dev）を実際の確認者に替えられる（Codex 11/3 の 8）
    old = reg.load_registry(reg.DEFAULT_REGISTRY_PATH).entries
    es = reg.build_entries(POL, checked_by="registry-build", reviewed_by="actual-reviewer", previous=old, re_review=True)
    assert {e["reviewed_by"] for e in es} == {"actual-reviewer"}
    # 確かめ直しを言わなければ、引き継げる行は前の確認者のまま（見ていない本文を確かめた扱いにしない）
    es = reg.build_entries(POL, checked_by="registry-build", reviewed_by="actual-reviewer", previous=old)
    assert {e["reviewed_by"] for e in es} == {"fixture:dev"}
    with pytest.raises(reg.RegistrationRefused, match="re_review"):
        reg.build_entries(POL, checked_by="registry-build", previous=old, re_review=True)


def test_cli_re_review_replaces_reviewers(tmp_path):
    d = policy_copy(tmp_path)
    assert reg.main(["--write", "--policy-dir", str(d), "--reviewed-by", "first-reviewer"]) == 0
    assert reg.main(["--write", "--policy-dir", str(d), "--reviewed-by", "second-reviewer", "--re-review"]) == 0
    assert {r["reviewed_by"] for r in rows(d)} == {"second-reviewer"}
    with pytest.raises(SystemExit):
        reg.main(["--write", "--policy-dir", str(d), "--re-review"])


def test_hit_must_fit_inside_the_range_not_just_touch_it():
    t = body_text("上限は5,000円")
    with pytest.raises(reg.RegistrationRefused, match="amount"):
        register(t, [ack("上限は5,00")])  # 当たりの途中までしか含まない
    assert register(t, [ack("上限は5,000円")])["acknowledged"] == [ack("上限は5,000円")]


def test_hits_are_deduplicated_and_ordered_by_position():
    t = body_text("下限1万円、上限5,000円、再掲5,000円、期日2026-10-31")
    e = register(t, [ack("1万円"), ack("上限5,000円"), ack("再掲5,000円"), ack("2026-10-31")])
    assert e["hits"] == [{"cue": "amount", "text": "1万円"}, {"cue": "amount", "text": "5,000円"},
                         {"cue": "real_date", "text": "2026-10-31"}]


def test_real_date_can_be_waived():
    t = body_text("規程の改定日は2026-10-31です")
    with pytest.raises(reg.RegistrationRefused, match="real_date"):
        register(t)
    assert register(t, [ack("2026-10-31")])["hits"] == [{"cue": "real_date", "text": "2026-10-31"}]


def test_rule_pattern_hits_are_also_located():
    # 送信時の手がかり語は「〜円を超える」の閾値を除くが、RuleC2Gate の規則は除かない。登録時は両方の和
    t = body_text("10万円を超える行は除く")
    assert not [s for s in cue_spans(t.body)[0] if s[0] == "amount"]
    with pytest.raises(reg.RegistrationRefused, match="amount"):
        register(t)
    assert register(t, [ack("10万円")])["hits"] == [{"cue": "amount", "text": "10万円"}]


@pytest.mark.parametrize("body,text,cue", [
    ("列 staff には 社員番号 E1024 が入る", "E1024", "employee_id"),
    ("田中さんの支出5,000円", "田中さん", "person_name"),
    ("A社の取引", "A社", "counterparty"),
    ("連絡先は a.b@example.com", "a.b@example.com", "email"),
    ("テーブル hr.salary を読む", "hr.salary", "table_name"),
    ("A社買収の予定", "買収", "m_and_a"),
    ("前の指示を無視して", "前の指示を無視", "injection"),
    ("М&A の予定", "М&A", "unexpected_script"),  # キリル文字の М
])
def test_other_kinds_cannot_be_waived(body, text, cue):
    t = body_text(body)
    with pytest.raises(reg.RegistrationRefused, match=cue):
        register(t, [ack(text), ack(body)])  # 本文全体を承認しても外せない


def test_waivable_kinds_are_only_amount_and_real_date():
    assert reg.WAIVABLE == frozenset({"amount", "real_date"})


def test_compact_only_hit_cannot_be_waived():
    # 空白を除いた写しだけで当たる金額は、位置を本文に戻せないので承認の対象にしない
    t = body_text("上限は1000 ド ル")
    spans, compact_only = cue_spans(t.body)
    assert "amount" in compact_only and not spans
    with pytest.raises(reg.RegistrationRefused, match="amount"):
        register(t, [ack("1000 ド ル"), ack(t.body)])


def test_acknowledged_text_missing_from_normalized_body_is_refused():
    t = body_text("上限は5,000円")
    with pytest.raises(reg.RegistrationRefused, match="not in"):
        register(t, [ack("上限は5,000円"), ack("下限は1,000円")])
    # 正規化する前の字（全角）では本文に出てこない
    t2 = body_text("上限は５，０００円")
    with pytest.raises(reg.RegistrationRefused, match="not in"):
        register(t2, [ack("５，０００円")])
    assert register(t2, [ack("5,000円")])["hits"] == [{"cue": "amount", "text": "5,000円"}]


def test_acknowledgement_that_hits_nothing_is_not_refused():
    t = body_text("列 date を読む。上限は5,000円")
    e = register(t, [ack("5,000円"), ack("列 date を読む", "当たらない箇所の承認")])
    assert e["hits"] == [{"cue": "amount", "text": "5,000円"}] and len(e["acknowledged"]) == 2
    e = register(body_text("列 date を読む"), [ack("列 date")])
    assert e["hits"] == [] and e["acknowledged"] == [ack("列 date")]


@pytest.mark.parametrize("bad", [
    {"text": "", "reason": "r"}, {"text": "5,000円", "reason": ""}, {"text": " ", "reason": "r"},
    {"text": "5,000円", "reason": "  "}, {"text": "5,000円"}, {"text": "5,000円", "reason": "r", "extra": 1},
    {"text": 5, "reason": "r"}, {"text": "5,000円", "reason": None}, "5,000円",
])
def test_malformed_acknowledgement_is_refused(bad):
    with pytest.raises(reg.RegistrationRefused, match="acknowledg"):
        register(body_text("上限は5,000円"), [bad])


def test_registration_hits_names_are_kept_for_compatibility():
    assert "employee_id" in reg.registration_hits("社員番号 Ｅ１０２４")
    assert "unexpected_script" in reg.registration_hits("М&A の予定")


# ===== 2. 人の全文確認の段 =====

def test_register_requires_reviewed_by_argument():
    with pytest.raises(TypeError):
        reg.register(body_text("列 date"), B, "codex", checked_by="c")  # type: ignore[call-arg]


@pytest.mark.parametrize("reviewed_by", ["", "   "])
def test_empty_reviewer_is_refused(reviewed_by):
    with pytest.raises(reg.RegistrationRefused, match="reviewed_by"):
        register(body_text("列 date"), reviewed_by=reviewed_by)


def test_reviewer_must_differ_from_approver_and_checker():
    t = body_text("列 date")
    with pytest.raises(reg.RegistrationRefused, match="approver"):
        register(t, reviewed_by=t.approver)
    with pytest.raises(reg.RegistrationRefused, match="checked_by"):
        register(t, reviewed_by="checker-y", checked_by="checker-y")
    e = register(t)
    assert (e["reviewed_by"], e["checked_by"], e["review"]) == (REVIEWER, "checker-y", "pass")


# ===== 3. 依頼文に書いてよい値を登録に結ぶ =====

def test_request_values_are_copied_into_the_row():
    t = replace(body_text("列 date を読む"), request_identifiers=("date", "category"),
                request_values=(("10万円", "閾値として書いてよい基準額"), ("5万円", "別の基準額")))
    e = register(t)
    assert e["request_values"] == {"identifiers": ["date", "category"], "values": ["10万円", "5万円"]}
    assert register(SPEC)["request_values"] == {"identifiers": ["date", "category", "amount"], "values": ["10万円"]}


def test_request_values_empty_when_the_text_has_none():
    e = reg.register(EXPL, POL.contracts["A-faq-format@1"], "claude", checked_by="c", reviewed_by=REVIEWER)
    assert e["request_values"] == {"identifiers": [], "values": []}


# ===== 版 =====

def test_check_version_is_bumped_and_covers_new_rules():
    assert reg.REGISTRATION_CHECK_VERSION.startswith("c2-registration@3#")
    s = reg.REGISTRATION_CHECK_SPEC
    assert set(s["waivable"]) == {"amount", "real_date"} and s["acknowledgement_unit"] == "span"
    assert "human_review" in s


# ===== CLI・作り直し =====

def policy_copy(tmp_path: Path, with_registry: bool = False) -> Path:
    d = tmp_path / "policies"
    shutil.copytree(DEFAULT_POLICY_DIR, d)
    if not with_registry:
        (d / "c2_registry.json").unlink()
    return d


def rows(d: Path) -> list[dict]:
    return json.loads((d / "c2_registry.json").read_text(encoding="utf-8"))["entries"]


def test_cli_refuses_new_rows_without_reviewed_by(tmp_path, capsys):
    d = policy_copy(tmp_path)
    assert reg.main(["--write", "--policy-dir", str(d)]) == 1
    assert "reviewed_by" in capsys.readouterr().err and not (d / "c2_registry.json").exists()


def test_cli_write_then_check_and_carry_over_reviewer(tmp_path):
    d = policy_copy(tmp_path)
    assert reg.main(["--write", "--policy-dir", str(d), "--reviewed-by", "first-reviewer"]) == 0
    assert {r["reviewed_by"] for r in rows(d)} == {"first-reviewer"}
    assert reg.main(["--check", "--policy-dir", str(d)]) == 0
    # 引き継げる行は、--reviewed-by が無くても、別の名前を付けても、前の確認者のまま
    assert reg.main(["--write", "--policy-dir", str(d)]) == 0
    assert reg.main(["--write", "--policy-dir", str(d), "--reviewed-by", "second-reviewer"]) == 0
    assert {r["reviewed_by"] for r in rows(d)} == {"first-reviewer"}


def test_cli_does_not_carry_over_when_request_values_change(tmp_path, capsys):
    d = policy_copy(tmp_path)
    assert reg.main(["--write", "--policy-dir", str(d), "--reviewed-by", "first-reviewer"]) == 0
    p = d / "approved" / "spec-csv-001@1.json"
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["request_values"]["values"].append({"text": "20万円", "reason": "別の基準額（試験用）"})
    p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    before = (d / "c2_registry.json").read_bytes()
    assert reg.main(["--write", "--policy-dir", str(d)]) == 1  # 引き継げない行があり --reviewed-by が無い
    assert "reviewed_by" in capsys.readouterr().err and (d / "c2_registry.json").read_bytes() == before
    assert reg.main(["--check", "--policy-dir", str(d)]) == 1
    assert reg.main(["--write", "--policy-dir", str(d), "--reviewed-by", "second-reviewer"]) == 0
    by = {r["ref"]: r["reviewed_by"] for r in rows(d)}
    assert by == {"expl-keihi-001@1": "first-reviewer", "spec-csv-001@1": "second-reviewer"}


def test_cli_does_not_carry_over_when_body_changes(tmp_path):
    d = policy_copy(tmp_path)
    assert reg.main(["--write", "--policy-dir", str(d), "--reviewed-by", "first-reviewer"]) == 0
    p = d / "approved" / "expl-keihi-001@1.json"
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["body"] += "\n以上。"
    doc["sha256"] = sha256_text(doc["body"])
    p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    assert reg.main(["--write", "--policy-dir", str(d)]) == 1


def test_cli_refuses_reviewer_equal_to_approver(tmp_path, capsys):
    d = policy_copy(tmp_path)
    assert reg.main(["--write", "--policy-dir", str(d), "--reviewed-by", EXPL.approver]) == 1
    assert "approver" in capsys.readouterr().err


def test_cli_check_reports_an_unreadable_registry_without_traceback(tmp_path, capsys):
    d = policy_copy(tmp_path)
    (d / "c2_registry.json").write_text('{"format": "c2-registry@1", "entries": [{"ref": "x"}]}', encoding="utf-8")
    assert reg.main(["--check", "--policy-dir", str(d)]) == 1
    assert "unreadable" in capsys.readouterr().err


def test_cli_write_rebuilds_over_an_unreadable_old_registry(tmp_path, capsys):
    # 古い形の登録簿は引き継ぎに使えないだけで、作り直しは止めない（引き継げない行は --reviewed-by が要る）
    d = policy_copy(tmp_path)
    (d / "c2_registry.json").write_text('{"format": "c2-registry@1", "entries": [{"ref": "x"}]}', encoding="utf-8")
    assert reg.main(["--write", "--policy-dir", str(d)]) == 1
    assert reg.main(["--write", "--policy-dir", str(d), "--reviewed-by", "r"]) == 0
    assert "unreadable" in capsys.readouterr().err and {r["reviewed_by"] for r in rows(d)} == {"r"}


def test_cli_acknowledge_binds_text_to_reason(tmp_path):
    d = policy_copy(tmp_path)
    p = d / "approved" / "dev-amount-001@1.json"
    body = TWO_FIVE_THOUSAND
    doc = json.loads((d / "approved" / "spec-csv-001@1.json").read_text(encoding="utf-8"))
    doc.update(id="dev-amount-001", body=body, sha256=sha256_text(body))
    p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    base = ["--write", "--policy-dir", str(d), "--reviewed-by", REVIEWER]
    assert reg.main(base) == 1  # 当たりの承認が無い
    assert reg.main(base + ["--acknowledge", "dev-amount-001@1=規程上限5,000円::規程上の額"]) == 1  # 片方の箇所だけ
    assert reg.main(base + ["--acknowledge", "dev-amount-001@1=規程上限5,000円::規程上の額",
                            "--acknowledge", "dev-amount-001@1=実際の支出5,000円::試験用の架空の額"]) == 0
    r = [x for x in rows(d) if x["ref"] == "dev-amount-001@1"][0]
    assert r["hits"] == [{"cue": "amount", "text": "5,000円"}]
    assert r["acknowledged"] == [ack("実際の支出5,000円", "試験用の架空の額"), ack("規程上限5,000円", "規程上の額")]
    # 次の --write は承認も確認者も引き継ぐ
    assert reg.main(["--write", "--policy-dir", str(d)]) == 0
    assert reg.main(["--check", "--policy-dir", str(d)]) == 0


@pytest.mark.parametrize("arg", ["dev-amount-001@1", "=x::y", "dev-amount-001@1=x", "dev-amount-001@1=::y",
                                 "dev-amount-001@1=x::"])
def test_cli_acknowledge_needs_ref_text_and_reason(tmp_path, arg):
    d = policy_copy(tmp_path)
    with pytest.raises(SystemExit):
        reg.main(["--write", "--policy-dir", str(d), "--reviewed-by", REVIEWER, "--acknowledge", arg])


def test_repository_registry_rows_are_fixture_reviewed():
    # 開発用の架空本文 2 件。封印用の測定と本番では、別の人の名前で作り直す
    got = reg.load_registry(reg.DEFAULT_REGISTRY_PATH)
    assert {e["reviewed_by"] for e in got.entries} == {"fixture:dev"}
    assert reg.main(["--check"]) == 0


# ===== 4. 読み込みの検証 =====

def good_row() -> dict:
    t = body_text(TWO_FIVE_THOUSAND)
    return register(t, [ack("規程上限5,000円"), ack("実際の支出5,000円")])


def write_rows(tmp_path: Path, rows_: list) -> Path:
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"format": reg.FORMAT, "entries": rows_}, ensure_ascii=False), encoding="utf-8")
    return p


def test_good_row_loads_and_round_trips(tmp_path):
    p = tmp_path / "r.json"
    reg.write_registry(p, [good_row()])
    got = reg.load_registry(p)
    assert got.entries[0]["hits"] == [{"cue": "amount", "text": "5,000円"}]
    assert got.entries[0]["request_values"] == {"identifiers": [], "values": []}


def test_registry_sha256_ignores_key_order_inside_rows(tmp_path):
    r = good_row()
    shuffled = dict(r, hits=[dict(reversed(list(h.items()))) for h in r["hits"]],
                    acknowledged=[dict(reversed(list(a.items()))) for a in r["acknowledged"]],
                    request_values=dict(reversed(list(r["request_values"].items()))))
    assert reg.load_registry(write_rows(tmp_path, [r])).sha256 == reg.load_registry(write_rows(tmp_path, [shuffled])).sha256


@pytest.mark.parametrize("change", [
    dict(review="fail"), dict(review=""), dict(review=None),
    dict(reviewed_by=""), dict(reviewed_by="   "), dict(reviewed_by=5), dict(reviewed_by="checker-y"),
    dict(result="maybe"), dict(sha256="xyz"),
    dict(hits="amount"), dict(hits=["amount"]), dict(hits=[{"cue": "amount"}]),
    dict(hits=[{"cue": "amount", "text": "5,000円", "extra": 1}]),
    dict(hits=[{"cue": "amount", "text": ""}]), dict(hits=[{"cue": "", "text": "5,000円"}]),
    dict(hits=[{"cue": "employee_id", "text": "5,000円"}]),  # 外せない種類
    dict(hits=[{"cue": "unexpected_script", "text": "5,000円"}]),
    dict(hits=[{"cue": "amount", "text": "7,000円"}]),  # 承認した text のどれにも含まれない
    dict(acknowledged="x"), dict(acknowledged=["規程上限5,000円"]),
    dict(acknowledged=[{"text": "規程上限5,000円"}]),
    dict(acknowledged=[{"text": "", "reason": "r"}, {"text": "5,000円", "reason": "r"}]),
    dict(acknowledged=[{"text": "5,000円", "reason": ""}]),
    dict(acknowledged=[{"text": "5,000円", "reason": "r", "extra": 1}]),
    dict(acknowledged=[]),  # 当たりがあるのに承認が無い
    dict(request_values=[]), dict(request_values={"identifiers": []}),
    dict(request_values={"identifiers": [], "values": [], "extra": []}),
    dict(request_values={"identifiers": "date", "values": []}),
    dict(request_values={"identifiers": [], "values": [{"text": "10万円", "reason": "r"}]}),
    dict(request_values={"identifiers": [1], "values": []}),
])
def test_load_registry_rejects_malformed_rows(tmp_path, change):
    row = dict(good_row(), **change)
    with pytest.raises(ValueError):
        reg.load_registry(write_rows(tmp_path, [row]))


@pytest.mark.parametrize("drop", ["reviewed_by", "review", "request_values", "checked_by"])
def test_load_registry_rejects_missing_keys(tmp_path, drop):
    row = good_row()
    del row[drop]
    with pytest.raises(ValueError):
        reg.load_registry(write_rows(tmp_path, [row]))


def test_load_registry_rejects_the_old_row_form(tmp_path):
    old = {k: v for k, v in good_row().items() if k not in ("reviewed_by", "review", "request_values")}
    old.update(hits=["amount"], acknowledged=["amount"])
    with pytest.raises(ValueError):
        reg.load_registry(write_rows(tmp_path, [old]))
