"""run_eval の配線のテスト。検索・生成・採点を偽物にして main を端から端まで流す。

確かめること:
- 回答のトークン数 (vLLM の usage) が、採点役が同じインスタンスで usage を上書きしても回答の値で残る
- raw.jsonl は 1 問ごとに書き足され、途中で落ちてもそこまでの行と run_meta.json (failed) が残る
- --dataset の繰り返し・--ids・--meta・--out-dir が run_meta.json に残る
"""
from __future__ import annotations

import asyncio
import importlib
import json
import sys
import types
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def run_eval():
    # 埋め込みモデルの重い依存はここでは使わない。入っていなければ空の偽物で import だけ通す
    try:
        importlib.import_module("sentence_transformers")
    except ImportError:
        stub = types.ModuleType("sentence_transformers")
        stub.SentenceTransformer = object
        stub.CrossEncoder = object
        sys.modules["sentence_transformers"] = stub
    return importlib.import_module("evaluation.run_eval")


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def _item(qid, question="質問", **kw):
    return {
        "id": qid,
        "question": question,
        "category": "general",
        "user_access_level": 1,
        "expected_sources": ["a.md"],
        "expected_answer_keywords": ["答え"],
        **kw,
    }


class _FakeProvider:
    def __init__(self):
        self.last_usage = None


def _install_fakes(monkeypatch, run_eval, fail_on=None):
    provider = _FakeProvider()

    async def get_provider(name):
        return provider

    async def collect(question, level, provider_name):
        if fail_on and question == fail_on:
            raise KeyboardInterrupt  # Colab の中断を真似る
        provider.last_usage = SimpleNamespace(prompt_tokens=len(question) * 100, completion_tokens=7)
        return run_eval.TimedAnswer(
            chunks=[{"source_file": "a.md", "content": "本文", "access_level": 1}],
            answer="答えです",
            retrieval_latency=0.1,
            first_token_latency=0.5,
            total_latency=1.0,
        )

    async def judge(judge_provider, question, chunks, answer, keywords):
        # 採点役も同じインスタンスなので usage を上書きする
        judge_provider.last_usage = SimpleNamespace(prompt_tokens=99999, completion_tokens=99999)
        return SimpleNamespace(faithfulness=1.0, answer_relevance=1.0, context_relevance=1.0)

    async def init_db():
        return None

    import app.database

    monkeypatch.setattr(run_eval, "get_provider", get_provider)
    monkeypatch.setattr(run_eval, "_collect_answer_and_chunks", collect)
    monkeypatch.setattr(run_eval.judges, "judge", judge)
    monkeypatch.setattr(app.database, "init_db", init_db)
    monkeypatch.setattr(
        run_eval.run_meta, "fetch_vllm_models", lambda url, **kw: {"url": url, "error": "offline"}
    )
    return provider


def _run(monkeypatch, run_eval, argv):
    monkeypatch.setattr(sys, "argv", ["run_eval.py", *argv])
    asyncio.run(run_eval.main())


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_tokens_survive_judge_and_meta_is_written(tmp_path, monkeypatch, run_eval):
    a = _jsonl(tmp_path / "a.jsonl", [_item("Q001", "あ"), _item("Q002", "いい")])
    b = _jsonl(tmp_path / "b.jsonl", [_item("N01", "ううう", expected_conclusion="該当")])
    _install_fakes(monkeypatch, run_eval)
    out = tmp_path / "run1"

    _run(monkeypatch, run_eval, [
        "--provider", "vllm", "--judge", "vllm", "--rerank", "off",
        "--dataset", str(a), "--dataset", str(b), "--ids", "N01,Q002",
        "--meta", "gpu=T4", "--meta", "gpu_util=0.80", "--out-dir", str(out),
    ])

    rows = _rows(out / "raw.jsonl")
    assert [r["id"] for r in rows] == ["Q002", "N01"]
    assert [r["prompt_tokens"] for r in rows] == [200, 300]  # 採点役の 99999 ではない
    assert [r["completion_tokens"] for r in rows] == [7, 7]
    assert all(r["faithfulness"] == 1.0 for r in rows)
    assert {r["run_label"] for r in rows} == {"vllm"}

    meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "done"
    assert meta["results"] == {"vllm": 2}
    assert meta["ids"] == ["N01", "Q002"]
    assert meta["n_questions"] == 2
    assert [d["path"] for d in meta["datasets"]] == [str(a), str(b)]
    assert meta["extras"] == {"gpu": "T4", "gpu_util": "0.80"}
    assert meta["rerank_modes"] == [False]
    assert meta["vllm"]["error"] == "offline"
    assert (out / "report.md").exists()


def test_interrupt_keeps_written_rows_and_marks_failed(tmp_path, monkeypatch, run_eval):
    a = _jsonl(tmp_path / "a.jsonl", [_item("Q001", "あ"), _item("Q002", "止まる"), _item("Q003", "う")])
    _install_fakes(monkeypatch, run_eval, fail_on="止まる")
    out = tmp_path / "run1"

    with pytest.raises(KeyboardInterrupt):
        _run(monkeypatch, run_eval, [
            "--provider", "vllm", "--no-judge", "--dataset", str(a), "--out-dir", str(out),
        ])

    assert [r["id"] for r in _rows(out / "raw.jsonl")] == ["Q001"]
    meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "failed"
    assert meta["results"] == {"vllm": 1}
    assert meta["error"].startswith("KeyboardInterrupt")
    assert not (out / "report.md").exists()


def test_non_empty_out_dir_is_refused(tmp_path, monkeypatch, run_eval):
    a = _jsonl(tmp_path / "a.jsonl", [_item("Q001")])
    _install_fakes(monkeypatch, run_eval)
    out = tmp_path / "run1"
    out.mkdir()
    (out / "raw.jsonl").write_text("前の結果\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        _run(monkeypatch, run_eval, [
            "--provider", "vllm", "--no-judge", "--dataset", str(a), "--out-dir", str(out),
        ])
    assert (out / "raw.jsonl").read_text(encoding="utf-8") == "前の結果\n"


@pytest.mark.parametrize(
    "argv",
    [
        ["--ids", "Q999"],
        ["--ids", " , "],
        ["--meta", "gpu"],
        ["--judge-from", "x.jsonl", "--ids", "Q001"],
        ["--judge-from", "x.jsonl", "--dataset", "a.jsonl"],
    ],
)
def test_bad_args_stop_before_running(tmp_path, monkeypatch, run_eval, argv):
    a = _jsonl(tmp_path / "a.jsonl", [_item("Q001")])
    _install_fakes(monkeypatch, run_eval)
    full = ["--provider", "vllm", "--no-judge", "--dataset", str(a), "--out-dir", str(tmp_path / "o")]
    if argv[0] == "--judge-from":
        full = []
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, run_eval, full + argv)
    assert exc.value.code == 2  # argparse の使い方エラー
    assert not (tmp_path / "o").exists()
