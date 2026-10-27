"""採点役の事前確認 (evaluation/judge_precheck.py) のテスト。採点役は偽物 (例ごとに決めた文字列を流す)。

確かめること:
- 5 例が Q018 と同じ問・期待キーワードで、チャンクが本物のコーパスの文そのまま
- 良い答えが一番高ければ pass、同点・逆転なら fail
- JSON が取れない・値が欠ける・0〜1 の数でない・呼び出しが落ちた採点が 1 つでもあれば未判定
- 本物の judges.judge を通しても同じ判定になり、CLI は JSON を残して終了コードで分ける
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from evaluation import judge_precheck as jp

ROOT = Path(__file__).resolve().parents[2]
CORPUS_FILE = ROOT / "evaluation" / "corpus" / "general" / "経費精算規程.md"


def _ok(faith, rel=1.0):
    return json.dumps({
        "context_relevance": 1.0, "answer_relevance": rel, "faithfulness": faith,
        "sentence_grounded": [1], "rationale": "テスト",
    }, ensure_ascii=False)


GOOD = {
    "faithful": _ok(1.0),
    "unsupported": _ok(0.0),
    "opposite": _ok(0.0),
    "partial": _ok(0.5),
    "refusal": _ok(1.0, rel=0.0),
}


class _Judge:
    """プロンプトの【RAG の回答】から例を見分け、決めた文字列を数文字ずつ流す。"""

    def __init__(self, by_kind):
        rows = jp.load_fixture(jp.FIXTURE)
        self.by_answer = {r["answer"]: by_kind[r["kind"]] for r in rows}
        self.last_usage = None
        self.calls = 0

    async def stream(self, messages):
        self.calls += 1
        prompt = messages[-1]["content"]
        for answer, text in self.by_answer.items():
            if f"【RAG の回答】\n{answer}\n" in prompt:
                if isinstance(text, Exception):
                    raise text
                for i in range(0, len(text), 7):
                    yield text[i:i + 7]
                return
        raise AssertionError("知らない回答")


def _scored(**override):
    """kind → {"raw", "error"}。override で例ごとに差し替える。"""
    out = {k: {"raw": json.loads(v), "error": None} for k, v in GOOD.items()}
    for kind, v in override.items():
        out[kind] = v
    return out


def test_fixture_matches_q018_and_corpus():
    rows = jp.load_fixture(jp.FIXTURE)
    q018 = next(
        json.loads(line) for line in (ROOT / "evaluation" / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line)["id"] == "Q018"
    )
    corpus = CORPUS_FILE.read_text(encoding="utf-8")
    assert [r["kind"] for r in rows] == list(jp.KINDS)
    assert len({r["id"] for r in rows}) == len({r["answer"] for r in rows}) == 5
    for r in rows:
        assert r["question"] == q018["question"]
        assert r["expected_keywords"] == q018["expected_answer_keywords"]
        assert r["chunks"] == rows[0]["chunks"]  # 5 例とも同じ根拠 (答えだけを変える)
        for c in r["chunks"]:
            assert c["source_file"] in q018["expected_sources"]
            assert c["content"] in corpus


def test_fixture_with_missing_kind_is_refused(tmp_path):
    rows = jp.load_fixture(jp.FIXTURE)[:4]
    path = tmp_path / "f.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="kind"):
        jp.load_fixture(path)


def test_good_judge_passes():
    v = jp.evaluate(_scored())
    assert v["result"] == "pass"
    assert v["broken"] == []
    assert [c["ok"] for c in v["checks"]] == [True] * 4


@pytest.mark.parametrize("kind,raw", [
    ("opposite", {"faithfulness": 1.0, "answer_relevance": 1.0}),   # 結論が逆でも満点 (同点)
    ("partial", {"faithfulness": 1.0, "answer_relevance": 1.0}),
    ("refusal", {"faithfulness": 1.0, "answer_relevance": 1.0}),    # 拒否なのに答えていると採点
])
def test_tie_or_reversal_fails(kind, raw):
    v = jp.evaluate(_scored(**{kind: {"raw": raw, "error": None}}))
    assert v["result"] == "fail"
    assert v["broken"] == []
    assert [c["ok"] for c in v["checks"]].count(False) == 1


@pytest.mark.parametrize("entry", [
    {"raw": {}, "error": None},                                            # JSON が取れない
    {"raw": {"faithfulness": 0.0}, "error": None},                         # answer_relevance が欠ける
    {"raw": {"faithfulness": "低い", "answer_relevance": 0.0}, "error": None},
    {"raw": {"faithfulness": 1.5, "answer_relevance": 0.0}, "error": None},  # 範囲外
    {"raw": {"faithfulness": True, "answer_relevance": 0.0}, "error": None},
    {"raw": {}, "error": "APIConnectionError: refused"},
])
def test_broken_scores_are_undecided(entry):
    # 壊れた側が 0.0 扱いなら pass に見える (judges は欠けた値を 0.0 にする)
    v = jp.evaluate(_scored(opposite=entry))
    assert v["result"] == "未判定"
    assert v["broken"] == ["opposite"]


def test_broken_good_answer_is_undecided_not_fail():
    v = jp.evaluate(_scored(faithful={"raw": {}, "error": None}))
    assert v["result"] == "未判定"
    assert all(c["ok"] is None for c in v["checks"])


def test_numeric_strings_count_like_judges():
    raw = {"faithfulness": "0.0", "answer_relevance": "1"}
    assert jp.evaluate(_scored(opposite={"raw": raw, "error": None}))["result"] == "pass"


def _score(by_kind):
    import asyncio

    judge = _Judge(by_kind)
    return asyncio.run(jp.score_all(judge, jp.load_fixture(jp.FIXTURE))), judge


def test_score_all_through_real_judges():
    scored, judge = _score(GOOD)
    assert judge.calls == 5
    assert jp.evaluate(scored)["result"] == "pass"
    assert scored["partial"]["raw"]["faithfulness"] == 0.5
    assert scored["faithful"]["id"] == "JP1"


def test_score_all_keeps_going_after_failures():
    by_kind = {**GOOD, "partial": "採点できませんでした", "unsupported": ConnectionError("refused")}
    scored, judge = _score(by_kind)
    assert judge.calls == 5
    assert (scored["partial"]["raw"], scored["partial"]["error"]) == ({}, None)  # JSON が無い応答
    assert scored["unsupported"]["error"] == "ConnectionError: refused"
    v = jp.evaluate(scored)
    assert v["result"] == "未判定"
    assert v["broken"] == ["unsupported", "partial"]


@pytest.fixture
def fake_judge(monkeypatch):
    def install(by_kind):
        judge = _Judge(by_kind)

        async def get_provider(name):
            assert name == "vllm"
            return judge

        monkeypatch.setattr(jp, "_get_provider", get_provider)
        monkeypatch.setattr(jp.run_meta, "fetch_vllm_models", lambda url, **kw: {"url": url, "error": "offline"})
        return judge

    return install


@pytest.mark.parametrize("by_kind,result,code", [
    (GOOD, "pass", 0),
    ({**GOOD, "opposite": _ok(1.0)}, "fail", 1),
    ({**GOOD, "refusal": "{壊れた"}, "未判定", 3),
])
def test_main_writes_json_and_exit_code(tmp_path, fake_judge, by_kind, result, code):
    fake_judge(by_kind)
    out = tmp_path / "step05" / "judge_precheck.json"
    assert jp.main(["--out", str(out)]) == code
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["result"] == result
    assert saved["judge"] == "vllm"
    assert saved["fixture"]["rows"] == 5
    assert saved["fixture"]["sha256"] == jp.run_meta.file_sha256(jp.FIXTURE)
    assert set(saved["scores"]) == set(jp.KINDS)
    assert saved["vllm"]["error"] == "offline"


def test_main_refuses_existing_out(tmp_path, fake_judge):
    judge = fake_judge(GOOD)
    out = tmp_path / "judge_precheck.json"
    out.write_text("前の結果\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        jp.main(["--out", str(out)])
    assert exc.value.code == 2
    assert out.read_text(encoding="utf-8") == "前の結果\n"
    assert judge.calls == 0


def test_module_form_help_runs_from_repo_root():
    # ノートブックと同じ呼び方 (repo ルートで -m)。evaluation/types.py が標準の types を隠さない形
    p = subprocess.run(
        [sys.executable, "-m", "evaluation.judge_precheck", "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    assert p.returncode == 0, p.stderr
    assert "--out" in p.stdout
