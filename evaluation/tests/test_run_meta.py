"""実行条件の記録 (evaluation/run_meta.py) のテスト。"""
from __future__ import annotations

import io
import json
import subprocess
from types import SimpleNamespace

import pytest

from evaluation import run_meta


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


# ─── 問の読み込み ──────────────────────────────────────────────
def test_load_questions_concatenates_files_in_order(tmp_path):
    a = _jsonl(tmp_path / "a.jsonl", [{"id": "Q001"}, {"id": "Q002"}])
    b = _jsonl(tmp_path / "b.jsonl", [{"id": "N01"}])
    rows, infos = run_meta.load_questions([a, b])
    assert [r["id"] for r in rows] == ["Q001", "Q002", "N01"]
    assert [i["rows"] for i in infos] == [2, 1]
    assert infos[0]["path"] == str(a)
    assert infos[0]["sha256"] == run_meta.file_sha256(a)


def test_load_questions_ids_keep_file_order_not_ids_order(tmp_path):
    a = _jsonl(tmp_path / "a.jsonl", [{"id": "Q001"}, {"id": "Q002"}, {"id": "Q003"}])
    b = _jsonl(tmp_path / "b.jsonl", [{"id": "N01"}])
    rows, infos = run_meta.load_questions([a, b], ids=["N01", "Q003"])
    assert [r["id"] for r in rows] == ["Q003", "N01"]
    # 記録に残す行数は絞る前のファイル全体
    assert [i["rows"] for i in infos] == [3, 1]


def test_load_questions_rejects_duplicate_id_across_files(tmp_path):
    a = _jsonl(tmp_path / "a.jsonl", [{"id": "Q001"}])
    b = _jsonl(tmp_path / "b.jsonl", [{"id": "Q001"}])
    with pytest.raises(ValueError, match="Q001"):
        run_meta.load_questions([a, b])


def test_load_questions_rejects_duplicate_id_in_one_file(tmp_path):
    a = _jsonl(tmp_path / "a.jsonl", [{"id": "Q001"}, {"id": "Q001"}])
    with pytest.raises(ValueError, match="Q001"):
        run_meta.load_questions([a])


def test_load_questions_rejects_unknown_ids(tmp_path):
    a = _jsonl(tmp_path / "a.jsonl", [{"id": "Q001"}])
    with pytest.raises(ValueError, match="Q999"):
        run_meta.load_questions([a], ids=["Q001", "Q999"])


def test_load_questions_skips_blank_lines(tmp_path):
    a = tmp_path / "a.jsonl"
    a.write_text('{"id": "Q001"}\n\n{"id": "Q002"}\n', encoding="utf-8")
    rows, infos = run_meta.load_questions([a])
    assert [r["id"] for r in rows] == ["Q001", "Q002"]
    assert infos[0]["rows"] == 2


def test_parse_ids():
    assert run_meta.parse_ids(None) is None
    assert run_meta.parse_ids("Q010, N02 ,") == ["Q010", "N02"]
    with pytest.raises(ValueError):
        run_meta.parse_ids(" , ")


def test_parse_meta_args():
    assert run_meta.parse_meta_args(None) == {}
    assert run_meta.parse_meta_args(["gpu=T4", "note=a=b", "empty="]) == {
        "gpu": "T4",
        "note": "a=b",
        "empty": "",
    }
    for bad in (["gpu"], ["=T4"]):
        with pytest.raises(ValueError):
            run_meta.parse_meta_args(bad)
    with pytest.raises(ValueError, match="gpu"):
        run_meta.parse_meta_args(["gpu=T4", "gpu=L4"])


# ─── 実行環境 ──────────────────────────────────────────────────
def test_git_info_reports_head_and_changes():
    def run(cmd, **kw):
        out = "abc123\n" if "rev-parse" in cmd else " M evaluation/run_eval.py\n?? x.py\n"
        return SimpleNamespace(stdout=out)

    info = run_meta.git_info("/repo", run=run)
    assert info == {
        "head": "abc123",
        "dirty": True,
        "changes": [" M evaluation/run_eval.py", "?? x.py"],
    }


def test_git_info_clean_tree():
    def run(cmd, **kw):
        return SimpleNamespace(stdout="abc123\n" if "rev-parse" in cmd else "")

    assert run_meta.git_info("/repo", run=run) == {"head": "abc123", "dirty": False, "changes": []}


def test_git_info_failure_is_recorded_not_raised():
    def run(cmd, **kw):
        raise subprocess.CalledProcessError(128, cmd)

    info = run_meta.git_info("/repo", run=run)
    assert info["head"] is None
    assert "error" in info


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_fetch_vllm_models_reads_id_root_and_max_model_len():
    seen = {}

    def opener(url, timeout):
        seen["url"] = url
        body = {
            "data": [
                {"id": "local-llm", "root": "org/model-7b", "max_model_len": 4096, "owned_by": "vllm"}
            ]
        }
        return _Resp(json.dumps(body).encode("utf-8"))

    out = run_meta.fetch_vllm_models("http://localhost:8001/v1/", opener=opener)
    assert seen["url"] == "http://localhost:8001/v1/models"
    assert out["models"] == [{"id": "local-llm", "root": "org/model-7b", "max_model_len": 4096}]


def test_fetch_vllm_models_failure_is_recorded_not_raised():
    def opener(url, timeout):
        raise OSError("connection refused")

    out = run_meta.fetch_vllm_models("http://localhost:8001/v1", opener=opener)
    assert "models" not in out
    assert "connection refused" in out["error"]


# ─── 記録の中身 ────────────────────────────────────────────────
def _meta(**kw):
    settings = SimpleNamespace(llm_model="local-llm", llm_max_tokens=1024, rerank_enabled=False)
    base = dict(
        argv=["run_eval.py", "--provider", "vllm"],
        datasets=[{"path": "a.jsonl", "sha256": "x", "rows": 1}],
        ids=["Q001"],
        n_questions=1,
        providers=["vllm"],
        rerank_modes=[False],
        judge=None,
        settings=settings,
        extras={"gpu": "T4"},
        git={"head": "abc", "dirty": False, "changes": []},
        vllm=None,
    )
    base.update(kw)
    return run_meta.build_run_meta(**base)


def test_build_run_meta_starts_running_and_copies_settings():
    meta = _meta()
    assert meta["status"] == "running"
    assert meta["settings"]["llm_model"] == "local-llm"
    assert meta["settings"]["rerank_enabled"] is False
    # settings に無い欄は None で残す (欄の有無で比べられるように)
    assert meta["settings"]["qdrant_collection"] is None
    assert {"retrieval_top_k", "chunk_size", "data_dir", "qdrant_url"} <= set(meta["settings"])
    assert meta["extras"] == {"gpu": "T4"}
    json.dumps(meta)  # そのまま JSON にできる


def test_write_run_meta_then_finish(tmp_path):
    meta = _meta()
    path = run_meta.write_run_meta(tmp_path, meta)
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "running"

    run_meta.write_run_meta(tmp_path, run_meta.finish_run_meta(meta, {"vllm+rerank_off": 6}))
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["status"] == "done"
    assert saved["results"] == {"vllm+rerank_off": 6}
    assert "finished_at" in saved
    assert "error" not in saved
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run_meta.json"]


def test_finish_run_meta_with_error_is_failed():
    meta = run_meta.finish_run_meta(_meta(), {"vllm": 2}, error="KeyboardInterrupt: ")
    assert meta["status"] == "failed"
    assert meta["results"] == {"vllm": 2}
    assert meta["error"] == "KeyboardInterrupt: "


def test_judged_meta_links_source_run_meta(tmp_path):
    src = tmp_path / "run1"
    src.mkdir()
    raw = src / "raw.jsonl"
    raw.write_text('{"id": "Q001"}\n', encoding="utf-8")
    run_meta.write_run_meta(src, {"status": "done", "extras": {"gpu": "T4"}})

    meta = run_meta.judged_meta(raw, "vllm", "judge-llm", ["run_eval.py"], {"note": "x"})
    assert meta["judged_from"] == {"path": str(raw), "sha256": run_meta.file_sha256(raw)}
    assert meta["judge"] == {"provider": "vllm", "model": "judge-llm"}
    assert meta["source_run_meta"]["extras"] == {"gpu": "T4"}
    assert meta["extras"] == {"note": "x"}


def test_judged_meta_without_source_run_meta(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text('{"id": "Q001"}\n', encoding="utf-8")
    meta = run_meta.judged_meta(raw, "vllm", "judge-llm", [], {})
    assert meta["source_run_meta"] is None
