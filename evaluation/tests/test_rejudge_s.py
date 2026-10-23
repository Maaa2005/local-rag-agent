"""文ごと 0/1 の採点し直し (evaluation/rejudge_s.py) のテスト。採点役・vLLM は偽物。ネットワークは使わない。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from evaluation import rejudge_s
from evaluation.types import QuestionResult

SHA = "a" * 64
VLLM = {"url": "http://x/v1/models", "models": [{"id": "local-llm", "root": "org/judge-model", "max_model_len": 4096}]}


def _settings(temp=0.0, model="local-llm", max_tokens=2048):
    return SimpleNamespace(llm_temperature=temp, llm_model=model, llm_max_tokens=max_tokens,
                           vllm_base_url="http://x/v1")


def _precheck(**kw):
    d = {"result": "pass", "llm_model": "local-llm", "temperature": 0.0, "max_tokens": 2048,
         "prompt_sha256": SHA, "vllm": VLLM}
    d.update(kw)
    return d


def _row(id_, answer="一文目です。二文目です。", label="cand_a", **kw):
    r = QuestionResult(
        id=id_, provider="vllm", question="質問です", category="general", user_access_level=1,
        answer=answer, retrieved_chunks=[{"source_file": "a.md", "content": "本文"}],
        expected_answer_keywords=["本文"],
    )
    return {**asdict(r), "run_label": label, **kw}


def _write(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _judgement(answer, verdicts):
    from evaluation.metrics import split_sentences

    s = split_sentences(answer)
    return SimpleNamespace(sentences=s, verdicts=verdicts, reasons=["r"] * len(s),
                           faithfulness=sum(verdicts) / len(verdicts), relevance=1, raw={})


class FakeJudge:
    """verdict_for(answer) -> 0/1 の並び。呼ばれた回数を数える。"""

    def __init__(self, verdict_for=None):
        self.calls = []
        self.verdict_for = verdict_for or (lambda a: [1] * len(__import__("evaluation.metrics", fromlist=["x"]).split_sentences(a)))

    async def __call__(self, provider, question, chunks, answer):
        self.calls.append(answer)
        v = self.verdict_for(answer)
        if isinstance(v, Exception):
            raise v
        return _judgement(answer, v)


def _run(tmp_path, rows, judge, *, settings=None, precheck=None, out="out.json", vllm=VLLM):
    raw, pre = tmp_path / "raw.jsonl", tmp_path / "pre.json"
    _write(raw, rows)
    pre.write_text(json.dumps(precheck if precheck is not None else _precheck()), encoding="utf-8")
    code = rejudge_s.main(
        ["--raw", str(raw), "--precheck", str(pre), "--out", str(tmp_path / out)],
        judge_fn=judge, settings=settings or _settings(), prompt_sha=SHA, provider=object(),
        vllm_fetch=lambda url: vllm,
    )
    return code


def _load(tmp_path, out="out.json"):
    return json.loads((tmp_path / out).read_text(encoding="utf-8"))


# ─── 実行前の拒否 ──────────────────────────────────────────────
def test_refuses_nonzero_temperature(tmp_path):
    j = FakeJudge()
    assert _run(tmp_path, [_row("Q1")], j, settings=_settings(temp=0.1), precheck=_precheck(temperature=0.1)) == 2
    assert j.calls == [] and not (tmp_path / "out.json").exists()


@pytest.mark.parametrize("override", [
    {"result": "fail"}, {"result": "未判定"}, {"llm_model": "other-model"},
    {"temperature": 0.1}, {"prompt_sha256": "b" * 64},
    {"max_tokens": 1024}, {"max_tokens": None},
])
def test_refuses_bad_precheck(tmp_path, override):
    j = FakeJudge()
    assert _run(tmp_path, [_row("Q1")], j, precheck=_precheck(**override)) == 2
    assert j.calls == []


def test_refuses_when_vllm_roots_differ(tmp_path):
    other = {"url": "u", "models": [{"id": "local-llm", "root": "org/other", "max_model_len": 1}]}
    j = FakeJudge()
    assert _run(tmp_path, [_row("Q1")], j, vllm=other) == 2
    assert j.calls == []


@pytest.mark.parametrize("bad", [
    {"url": "u", "error": "URLError: down"}, {"url": "u", "models": []}, None,
    {"url": "u", "models": [{"id": "x", "root": None}]},
])
def test_refuses_when_current_vllm_unverifiable(tmp_path, bad):
    assert _run(tmp_path, [_row("Q1")], FakeJudge(), vllm=bad) == 2


@pytest.mark.parametrize("bad", [
    {"url": "u", "error": "boom"}, {"url": "u", "models": []},
])
def test_refuses_when_precheck_vllm_unverifiable(tmp_path, bad):
    assert _run(tmp_path, [_row("Q1")], FakeJudge(), precheck=_precheck(vllm=bad)) == 2


def test_refuses_when_precheck_has_no_vllm(tmp_path):
    pre = _precheck()
    del pre["vllm"]
    assert _run(tmp_path, [_row("Q1")], FakeJudge(), precheck=pre) == 2


def test_vllm_root_set_order_does_not_matter(tmp_path):
    two = {"url": "u", "models": [{"id": "a", "root": "r1"}, {"id": "b", "root": "r2"}]}
    rev = {"url": "u", "models": [{"id": "b", "root": "r2"}, {"id": "a", "root": "r1"}]}
    assert _run(tmp_path, [_row("Q1")], FakeJudge(), precheck=_precheck(vllm=two), vllm=rev) == 0


def test_refuses_existing_out(tmp_path):
    (tmp_path / "out.json").write_text("old", encoding="utf-8")
    j = FakeJudge()
    assert _run(tmp_path, [_row("Q1")], j) == 2
    assert (tmp_path / "out.json").read_text(encoding="utf-8") == "old"
    assert j.calls == []


# ─── 採点 ──────────────────────────────────────────────────────
def test_empty_and_failed_answers_are_auto_zero_without_calling_judge(tmp_path):
    j = FakeJudge()
    rows = [_row("Q1", answer=""), _row("Q2", answer="   "), _row("Q3", error="run_failed: boom")]
    code = _run(tmp_path, rows, j)
    run = _load(tmp_path)["runs"]["cand_a"]
    assert j.calls == []
    assert code == 1 and run["result"] == "fail"
    assert run["mean"] == 0.0 and run["n_questions"] == 3 and run["n_auto_zero"] == 3
    reasons = [q["auto_zero_reason"] for q in run["questions"]]
    assert reasons[0] == "empty_answer" and reasons[2].startswith("run_failed")


def test_any_exception_makes_run_undetermined(tmp_path):
    def vf(a):
        return RuntimeError("judge down") if "BAD" in a else [1] * 1

    rows = [_row("Q1", answer="良い。"), _row("Q2", answer="BAD。"), _row("Q3", answer="良い。")]
    code = _run(tmp_path, rows, FakeJudge(vf))
    run = _load(tmp_path)["runs"]["cand_a"]
    assert code == 3 and run["result"] == "未判定"
    assert run["n_failed"] == 1 and run["mean"] is None
    assert "judge down" in run["questions"][1]["error"]


def test_custom_exception_class_is_recorded(tmp_path):
    class ScoringFailure(Exception):
        pass

    code = _run(tmp_path, [_row("Q1", answer="良い。")], FakeJudge(lambda a: ScoringFailure("bad json")))
    assert code == 3
    assert "ScoringFailure" in _load(tmp_path)["runs"]["cand_a"]["questions"][0]["error"]


@pytest.mark.parametrize("make_bad", [
    lambda a: _judgement(a, [1]),                    # verdicts が足りない
    lambda a: _judgement(a, [1, 2]),                 # 0/1 でない
    lambda a: _judgement(a, [True, True]),           # bool
    lambda a: SimpleNamespace(sentences=["別の文"], verdicts=[1, 1], reasons=[], faithfulness=1, relevance=1, raw={}),
])
def test_malformed_judgement_is_undetermined_not_filled(tmp_path, make_bad):
    async def bad(provider, q, c, a):
        return make_bad(a)

    code = _run(tmp_path, [_row("Q1")], bad)
    assert code == 3 and _load(tmp_path)["runs"]["cand_a"]["result"] == "未判定"


def test_all_ones_pass(tmp_path):
    code = _run(tmp_path, [_row("Q1"), _row("Q2")], FakeJudge())
    doc = _load(tmp_path)
    run = doc["runs"]["cand_a"]
    assert code == 0 and doc["result"] == "pass" and run["mean"] == 1.0
    q = run["questions"][0]
    assert q["sentences"] == ["一文目です。", "二文目です。"] and q["verdicts"] == [1, 1]
    assert q["faithfulness"] == 1.0 and q["relevance"] == 1 and q["error"] is None


def _fraction_rows(n_total, n_bad):
    """n_bad 問が全文 0、残りが全文 1 になる行。点は (n_total-n_bad)/n_total。"""
    return [_row(f"Q{i}", answer="良い。" if i >= n_bad else "悪い。") for i in range(n_total)]


def _bad_is_zero(a):
    return [0] if "悪い" in a else [1]


def test_exactly_point_nine_passes(tmp_path):
    code = _run(tmp_path, _fraction_rows(10, 1), FakeJudge(_bad_is_zero))
    run = _load(tmp_path)["runs"]["cand_a"]
    assert run["mean"] == 0.9 and run["result"] == "pass" and code == 0


def test_point_eight_nine_fails(tmp_path):
    code = _run(tmp_path, _fraction_rows(100, 11), FakeJudge(_bad_is_zero))
    run = _load(tmp_path)["runs"]["cand_a"]
    assert run["mean"] == pytest.approx(0.89) and run["result"] == "fail" and code == 1


def test_question_score_is_mean_of_sentences(tmp_path):
    # 1 問: 4 文中 3 文が 1 → 0.75
    ans = "あ。い。う。え。"
    code = _run(tmp_path, [_row("Q1", answer=ans)], FakeJudge(lambda a: [1, 1, 1, 0]))
    run = _load(tmp_path)["runs"]["cand_a"]
    assert run["mean"] == 0.75 and run["result"] == "fail" and code == 1


def test_exit_code_priority_and_per_label(tmp_path):
    rows = [_row("Q1", label="good", answer="良い。"), _row("Q1", label="bad", answer="悪い。"), _row("Q2", label="unk", answer="BAD。")]

    def vf(a):
        if "BAD" in a:
            return RuntimeError("x")
        return _bad_is_zero(a)

    code = _run(tmp_path, rows, FakeJudge(vf))
    runs = _load(tmp_path)["runs"]
    assert (runs["good"]["result"], runs["bad"]["result"], runs["unk"]["result"]) == ("pass", "fail", "未判定")
    assert code == 3  # 未判定が fail より優先


def test_fail_without_undetermined_exits_1(tmp_path):
    rows = [_row("Q1", label="good", answer="良い。"), _row("Q1", label="bad", answer="悪い。")]
    assert _run(tmp_path, rows, FakeJudge(_bad_is_zero)) == 1


def test_output_records_provenance(tmp_path):
    _run(tmp_path, [_row("Q1")], FakeJudge())
    doc = _load(tmp_path)
    raw, pre = tmp_path / "raw.jsonl", tmp_path / "pre.json"
    assert doc["raw"] == {"path": str(raw), "sha256": hashlib.sha256(raw.read_bytes()).hexdigest()}
    assert doc["precheck"] == {"path": str(pre), "sha256": hashlib.sha256(pre.read_bytes()).hexdigest()}
    assert (doc["judge"], doc["llm_model"], doc["temperature"], doc["prompt_sha256"]) == ("vllm", "local-llm", 0.0, SHA)
    assert doc["max_tokens"] == 2048
    assert doc["vllm"] == VLLM
    assert doc["run_meta"]["judged_from"]["sha256"] == doc["raw"]["sha256"]
    assert doc["run_meta"]["status"] == "done"
