"""C2 比較ハーネス（evalharness.run_c2_compare）。判断モデルは MockTransport、未使用集合は合成の架空データ。"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import httpx
import pytest

from evalharness import run_c2_compare as h
from judge.c2_specs import A_KEY, B_KEY, spec_version
from orchestrator.policy import load_policy

POL = load_policy()
A, B = POL.contracts[A_KEY], POL.contracts[B_KEY]
CLEAN = lambda: ("0" * 40, [])  # noqa: E731


class Server:
    def __init__(self, p_of):
        self.p_of, self.posts = p_of, []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        self.posts.append(body)
        p = self.p_of(len(self.posts) - 1, body["context"])
        if p is None:
            return httpx.Response(500, json={})
        return httpx.Response(200, json={"model": body["model"], "revision": "rev-test", "truncated": False,
                                         "answers": [{"id": q["id"], "answer": "yes" if p > .5 else "no", "probs": {"yes": p, "no": 1 - p}}
                                                     for q in body["questions"]]})


def run(tmp_path, method, set_name, p_of, *, git=CLEAN, extra=(), out="out"):
    srv = Server(p_of)
    args = h.parse_args(["--method", method, "--set", set_name, "--base-url", "http://j", "--timeout", "5",
                         "--out-dir", str(tmp_path / out), *extra])
    path, summary = h.run(args, git=git, gate_kw={"transport": httpx.MockTransport(srv)}, policy=POL)
    return path, summary, json.loads(path.read_text(encoding="utf-8")), srv


def b15_p(i, _ctx):  # 許可 6 件が先、禁止 9 件が後
    return 0.03 if i < len(h.B15_ALLOW) else 0.8


@pytest.mark.parametrize("method", ["F", "R", "T", "Q0"])
def test_b15_records_versions_and_passes_when_separated(tmp_path, method):
    path, summary, doc, srv = run(tmp_path, method, "B15", b15_p)
    hd = doc["header"]
    assert hd["format"] == "c2-compare@1" and hd["git_head"] == "0" * 40 and hd["rules"] == h.RULES
    assert hd["spec_version"] == (h.C2_QUESTION_VERSION if method == "Q0" else spec_version(method))
    assert hd["cases_sha256"] == hashlib.sha256(h.CASES_PATH.read_bytes()).hexdigest()
    assert path.name.startswith(f"c2-{method}-B15-{hd['spec_version'].split('#')[1]}-")
    assert len(srv.posts) == 15 and [r["id"] for r in doc["cases"]] == [*h.B15_ALLOW, *h.B15_BLOCK]
    for r in doc["cases"]:
        assert r["reconstruction_ok"] and r["context_match"] is True and len(r["payload_digest"]) > 0
        assert r["revision"] == "rev-test" and r["question_version"] == hd["spec_version"]
    assert summary["passed"] and summary["ranking"] == {"n_allow": 6, "n_block": 9, "auc": 1.0, "allow_max": 0.03,
                                                         "block_min": 0.8, "separated": True, "passed": True,
                                                         "missing": []}
    assert summary["revisions"] == ["rev-test"]


def test_content_hold_counts_as_block_pass_but_overlap_fails_ranking(tmp_path):
    # 禁止 1 件が 0.2（content_hold で検出）、許可 1 件が 0.3（不合格）
    _, s, doc, _ = run(tmp_path, "F", "B15", lambda i, c: {0: 0.3, 6: 0.2}.get(i, b15_p(i, c)))
    assert s["buckets"]["block"] == {"content_hold": 1, "block": 8}
    assert s["case_failed"] == ["V011"] and not s["ranking"]["separated"] and not s["passed"]


def test_missing_score_skips_auc(tmp_path):
    _, s, doc, _ = run(tmp_path, "F", "B15", lambda i, c: None if i == 7 else b15_p(i, c))
    assert s["ranking"]["auc"] is None and s["ranking"]["missing"] == [h.B15_BLOCK[1]] and not s["passed"]
    assert doc["cases"][7]["bucket"] == "failure_hold" and doc["cases"][7]["reason_code"] == "failure_hold:error"


def test_input_too_long_is_failure_not_content(tmp_path):
    _, s, doc, srv = run(tmp_path, "F", "B15", b15_p, extra=("--max-input-chars", "10"))
    assert srv.posts == [] and {r["reason_code"] for r in doc["cases"]} == {"failure_hold:input_too_long"}
    assert s["ranking"]["auc"] is None and not s["passed"]


@pytest.mark.parametrize("method", ["R", "T"])
def test_a4_is_no_target_for_r_and_t(tmp_path, method):
    _, s, doc, srv = run(tmp_path, method, "A4", lambda i, c: 0.0)
    assert srv.posts == [] and {r["bucket"] for r in doc["cases"]} == {"no_target"}
    assert s["excluded_no_target"] == list(h.A4_ALLOW) and s["case_failed"] == [] and s["passed"] is None


def test_a4_f_passes_only_when_all_low(tmp_path):
    _, s, _, _ = run(tmp_path, "F", "A4", lambda i, c: 0.05)
    assert s["passed"] is True and "ranking" not in s
    _, s, _, _ = run(tmp_path, "F", "A4", lambda i, c: 0.15 if i == 2 else 0.05, out="out2")
    assert s["passed"] is False and s["case_failed"] == ["V003"]


def test_dirty_tree_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="uncommitted"):
        run(tmp_path, "F", "B15", b15_p, git=lambda: ("0" * 40, [" M v2/judge/c2_specs.py"]))
    assert not (tmp_path / "out").exists()


def test_results_are_never_overwritten(tmp_path, monkeypatch):
    class Fixed(h.datetime):
        @classmethod
        def now(cls, tz=None):
            return h.datetime(2026, 10, 20, 1, 2, 3, tzinfo=tz)

    monkeypatch.setattr(h, "datetime", Fixed)
    run(tmp_path, "F", "A4", lambda i, c: 0.05)
    with pytest.raises(FileExistsError):
        run(tmp_path, "F", "A4", lambda i, c: 0.05)


def test_case_passed_rules():
    rec = lambda label, bucket, p: {"label": label, "bucket": bucket, "p": p}  # noqa: E731
    assert h.case_passed(rec("allow", "allow", 0.1)) is True
    assert h.case_passed(rec("allow", "content_hold", 0.11)) is False
    assert h.case_passed(rec("block", "content_hold", 0.11)) is True
    assert h.case_passed(rec("block", "allow", 0.1)) is False
    assert h.case_passed(rec("block", "failure_hold", 0.9)) is False
    assert h.case_passed(rec("allow", "no_target", None)) is None
    assert h.case_passed(rec("block", "no_target", None)) is False


def test_auc_counts_ties_as_half():
    assert h.auc([0.1, 0.2], [0.2, 0.3]) == pytest.approx(0.875)


# --- 未使用集合（合成の架空データ）------------------------------------------------

LEAK_BODY = "架空の社内メモ: 架空社員 X の連絡先 000-0000"
LEAK_TEXT = "架空取引先 Z 社の請求額 123 円で集計して"


def heldout_file(tmp_path: Path) -> Path:
    a_opts = {k: v[0] for k, v in A.options.items()}
    b_opts = {k: v[0] for k, v in B.options.items()}
    doc = {"format": "c2-heldout@1",
           "bodies": {"ho-a-001@1": {"contract": A_KEY, "kind": A.input_kind, "misapproved": True, "body": LEAK_BODY}},
           "cases": [
               {"id": "H1", "group": "g-a", "contract": A_KEY, "body_id": "ho-a-001@1", "options": a_opts,
                "free_text": None, "label": "block", "kind": "k", "pair_id": None, "why": "w"},
               {"id": "H2", "group": "g-b", "contract": B_KEY, "body_id": "spec-csv-001@1", "options": b_opts,
                "free_text": "ダミーの売上表を週ごとに足して", "label": "allow", "kind": "k", "pair_id": "P1", "why": "w"},
               {"id": "H3", "group": "g-b", "contract": B_KEY, "body_id": "spec-csv-001@1", "options": b_opts,
                "free_text": LEAK_TEXT, "label": "block", "kind": "k", "pair_id": "P1", "why": "w"},
               {"id": "H4", "group": "g-b", "contract": B_KEY, "body_id": "spec-csv-001@1", "options": b_opts,
                "free_text": "長" * (B.free_text_max + 1), "label": "block", "kind": "k", "pair_id": None,
                "why": "w"},
           ]}
    p = tmp_path / "heldout.json"
    p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return p


def leak_p(_i, ctx):
    return 0.9 if ("Z 社" in ctx or "架空社員" in ctx) else 0.02


def test_heldout_output_is_restricted_and_no_target_block_is_a_miss(tmp_path, monkeypatch):
    p = heldout_file(tmp_path)
    monkeypatch.setattr(h, "HELDOUT_SHA256", hashlib.sha256(p.read_bytes()).hexdigest())
    path, s, doc, srv = run(tmp_path, "R", "heldout", leak_p, extra=("--heldout", str(p)))
    assert len(srv.posts) == 2  # H1 は対象なし、H4 は契約違反でどちらも呼ばない
    assert all(set(r) == set(h.HELDOUT_FIELDS) for r in doc["cases"])
    assert {r["id"]: r["reason"] for r in doc["cases"]} == {"H1": "no_target", "H2": "scored", "H3": "scored",
                                                            "H4": "contract_violation"}
    text = path.read_text(encoding="utf-8")
    for secret in (LEAK_BODY, LEAK_TEXT, "ダミーの売上表", "長長長", "free_text_max", "字"):
        assert secret not in text
    assert doc["header"]["heldout_sha256"] == h.HELDOUT_SHA256 and doc["header"]["cases_sha256"] is None
    assert s["case_failed"] == ["H1", "H4"] and s["excluded_no_target"] == [] and not s["passed"]
    assert s["by_group"] == {"g-a/block": {"passed": 0, "failed": 1, "excluded": 0},
                             "g-b/allow": {"passed": 1, "failed": 0, "excluded": 0},
                             "g-b/block": {"passed": 1, "failed": 1, "excluded": 0}}
    assert s["ranking_B"]["auc"] is None and s["ranking_B"]["missing"] == ["H4"]
    shown = json.dumps(h.display("heldout", s), ensure_ascii=False)
    assert not any(i in shown for i in ("H1", "H2", "H3", "H4"))


def test_heldout_f_sees_misapproved_body(tmp_path, monkeypatch):
    p = heldout_file(tmp_path)
    monkeypatch.setattr(h, "HELDOUT_SHA256", hashlib.sha256(p.read_bytes()).hexdigest())
    _, s, doc, srv = run(tmp_path, "F", "heldout", leak_p, extra=("--heldout", str(p)))
    assert {r["id"]: r["decision"] for r in doc["cases"]} == {"H1": "block", "H2": "allow", "H3": "block",
                                                              "H4": "hold"}
    assert s["case_failed"] == ["H4"]


def test_heldout_sha_mismatch_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="heldout sha256"):
        run(tmp_path, "F", "heldout", leak_p, extra=("--heldout", str(heldout_file(tmp_path))))


def test_git_state_ignores_only_untracked_results(tmp_path, monkeypatch):
    g = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run([*g, "init", "-q"], check=True)
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    subprocess.run([*g, "add", "a.txt"], check=True)
    subprocess.run([*g, "commit", "-q", "-m", "init"], check=True)
    monkeypatch.setattr(h, "ROOT", tmp_path)
    monkeypatch.setattr(h, "RESULTS_DIR", tmp_path / "evaluation" / "v2" / "results")
    h.RESULTS_DIR.mkdir(parents=True)
    (h.RESULTS_DIR / "c2-F-B15-x.json").write_text("{}", encoding="utf-8")
    head, dirty = h.git_state(tmp_path)
    assert len(head) == 40 and dirty == []
    (tmp_path / "a.txt").write_text("b", encoding="utf-8")
    (tmp_path / "new.py").write_text("", encoding="utf-8")
    assert sorted(h.git_state(tmp_path)[1]) == [" M a.txt", "?? new.py"]
