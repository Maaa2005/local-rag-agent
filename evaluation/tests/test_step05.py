"""Step 0.5 の部品 (evaluation/colab/step05.py) のテスト。

確かめること:
- 予算から出す GPU_UTIL が T4 で計画どおり (8GB 0.50・12GB 0.76) で、超えた時は下げるだけ
- max-model-len は (入力の最大 + 1024) を 512 単位で切り上げ、測れなかった問があれば止まる
- GPU の記録は起動と実行の両方の区間で見て、記録の無い区間があれば「未確認」(None)
- 試運転の確かめは、取れなかった項目を True にしない
- ノートが組む run_eval の引数を run_eval 本体が受け付け、--judge-from に GPU の記録が写る
- 試運転ノート (step05_trial.ipynb) の全セルが Python として読め、step05 に無い名前を使わない
- Step 1 の部品: 全問の ID・採点役の失敗の数え方・Step 1 ノート (step1_measure.ipynb) の約束
"""
from __future__ import annotations

import ast
import io
import json
import re
from pathlib import Path

import pytest

from evaluation import run_meta
from evaluation.colab import step05
from evaluation.tests.test_run_eval_cli import _install_fakes, _item, _jsonl, _rows, _run, run_eval  # noqa: F401

T4_MIB = 15360


# ─── 予算と長さ ────────────────────────────────────────────────
def test_gpu_util_for_budget_matches_plan_on_t4():
    assert step05.gpu_util_for_budget(7680, T4_MIB) == 0.50
    assert step05.gpu_util_for_budget(11776, T4_MIB) == 0.76  # 0.7666… を切り捨て
    # 埋め込みを GPU に載せた分だけ下がる
    assert step05.gpu_util_for_budget(7680, T4_MIB, resident_mib=768) == 0.45


@pytest.mark.parametrize(
    "budget,total,resident",
    [(7680, 0, 0), (16000, T4_MIB, 0), (7680, T4_MIB, 7680)],
)
def test_gpu_util_for_budget_rejects_impossible(budget, total, resident):
    with pytest.raises(ValueError):
        step05.gpu_util_for_budget(budget, total, resident)


def test_adjusted_gpu_util_only_lowers():
    assert step05.adjusted_gpu_util(0.50, 7680, 7680, T4_MIB) is None  # ちょうどは収まっている
    assert step05.adjusted_gpu_util(0.50, 7000, 7680, T4_MIB) is None  # 余っても上げない
    # 1,536 MiB 超え = 0.10 分下げる
    assert step05.adjusted_gpu_util(0.50, 7680 + 1536, 7680, T4_MIB) == 0.40
    # 1 MiB でも超えたら必ず下がる
    assert step05.adjusted_gpu_util(0.76, 11777, 11776, T4_MIB) < 0.76
    with pytest.raises(ValueError):
        step05.adjusted_gpu_util(0.10, 7680 + 4000, 7680, T4_MIB)


@pytest.mark.parametrize(
    "prompt,expected",
    [(0, 1024), (1, 1536), (512, 1536), (513, 2048), (3072, 4096), (3073, 4608)],
)
def test_needed_max_model_len_rounds_up_to_512(prompt, expected):
    assert step05.needed_max_model_len(prompt) == expected


def test_prompt_token_stats_picks_longest(tmp_path):
    raw = _jsonl(tmp_path / "raw.jsonl", [
        {"id": "Q001", "prompt_tokens": 900, "error": None},
        {"id": "N05", "prompt_tokens": 2100, "error": None},
        {"id": "Q002", "prompt_tokens": 1500},
    ])
    assert step05.prompt_token_stats(raw) == {"max_prompt_tokens": 2100, "max_id": "N05", "n": 3}


def test_prompt_token_stats_stops_on_unmeasured(tmp_path):
    raw = _jsonl(tmp_path / "raw.jsonl", [
        {"id": "Q001", "prompt_tokens": 900, "error": None},
        {"id": "Q002", "prompt_tokens": None, "error": None},
        {"id": "Q003", "prompt_tokens": 100, "error": "TimeoutError"},
    ])
    with pytest.raises(ValueError, match=r"2 件.*Q002.*Q003"):
        step05.prompt_token_stats(raw)
    with pytest.raises(ValueError):
        step05.prompt_token_stats(_jsonl(tmp_path / "empty.jsonl", []))


def test_trial_ids_adds_longest_once():
    assert step05.trial_ids("N07") == [*step05.TRIAL_IDS, "N07"]
    assert step05.trial_ids("Q017") == list(step05.TRIAL_IDS)
    assert step05.trial_ids(None) == list(step05.TRIAL_IDS)


def test_trial_ids_exist_in_datasets():
    root = step05.Path(step05.__file__).resolve().parents[2]
    rows, _ = run_meta.load_questions([root / d for d in step05.DATASETS], None)
    ids = {r["id"] for r in rows}
    assert set(step05.TRIAL_IDS) <= ids
    assert len(ids) == 62  # 50 問 + 数値の境目 12 問


# ─── GPU メモリの記録 ──────────────────────────────────────────
def test_parse_smi_mib():
    assert step05.parse_smi_mib("1234\n") == 1234
    assert step05.parse_smi_mib("\n 5678 MiB\n9999\n") == 5678
    with pytest.raises(ValueError):
        step05.parse_smi_mib("No devices were found\n")


SAMPLES = [(100.0, 300), (101.0, 8000), (102.0, 6500), (110.0, 7000), (111.0, 7200), (120.0, 9000)]


def test_gpu_mem_record_uses_both_windows():
    rec = step05.gpu_mem_record(SAMPLES, {"launch": (100.0, 102.0), "run": (110.0, 111.5)}, 7680, T4_MIB, 300)
    assert rec["windows"]["launch"]["peak_mib"] == 8000  # 起動中の山
    assert rec["windows"]["run"]["peak_mib"] == 7200
    assert rec["peak_mib"] == 8000
    assert rec["within_budget"] is False
    assert rec["baseline_mib"] == 300
    # 区間の外 (120 秒の 9000) は入れない。時刻は起動の始まりからの秒
    assert rec["samples"] == [[0.0, 300], [1.0, 8000], [2.0, 6500], [10.0, 7000], [11.0, 7200]]


def test_gpu_mem_record_within_budget():
    rec = step05.gpu_mem_record(SAMPLES, {"launch": (102.0, 102.0), "run": (110.0, 111.0)}, 7680, T4_MIB)
    assert rec["peak_mib"] == 7200
    assert rec["within_budget"] is True


def test_gpu_mem_record_without_samples_is_unknown():
    rec = step05.gpu_mem_record(SAMPLES, {"launch": (100.0, 102.0), "run": (130.0, 140.0)}, 9000, T4_MIB)
    assert rec["windows"]["run"] == {"start": 130.0, "end": 140.0, "peak_mib": None, "n_samples": 0}
    assert rec["peak_mib"] == 8000
    assert rec["within_budget"] is None  # 実行中を測れていないので「収まった」にしない
    assert step05.gpu_mem_record([], {}, 7680, T4_MIB)["within_budget"] is None


def test_wait_gpu_released():
    clock = iter(range(100)).__next__
    reads = iter([5000, 2000, 340]).__next__
    assert step05.wait_gpu_released(reads, 300, sleep=lambda s: None, clock=clock) == 340
    with pytest.raises(RuntimeError, match="空かない"):
        step05.wait_gpu_released(lambda: 5000, 300, limit=3, sleep=lambda s: None, clock=iter(range(100)).__next__)


# ─── vLLM の記録 ───────────────────────────────────────────────
LOG = """INFO 10-20 12:00:00 engine.py:1] Avg prompt throughput: 10 tokens/s
WARNING 10-20 12:00:01 scheduler.py:1] Sequence group 3 is preempted by PreemptionMode.RECOMPUTE
ERROR 10-20 12:00:02 worker.py:1] torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB
"""


def test_scan_vllm_log():
    found = step05.scan_vllm_log(LOG)
    assert len(found["oom"]) == 1 and "CUDA out of memory" in found["oom"][0]
    assert len(found["preempt"]) == 1 and "preempted" in found["preempt"][0]
    assert step05.scan_vllm_log("INFO ok\n") == {"oom": [], "preempt": []}
    assert len(step05.scan_vllm_log("preempt\n" * 50)["preempt"]) == 20


METRICS = """# HELP vllm:num_preemptions_total Cumulative number of preemption from the engine.
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total{engine="0",model_name="local-llm"} 2.0
vllm:num_preemptions_total{engine="1",model_name="local-llm"} 1.0
vllm:num_requests_running{model_name="local-llm"} 0.0
"""


def test_parse_prometheus_counter_sums_labels():
    assert step05.parse_prometheus_counter(METRICS) == 3.0
    assert step05.parse_prometheus_counter("vllm:num_preemptions_total 4\n") == 4.0
    assert step05.parse_prometheus_counter(METRICS, "vllm:num_requests_running") == 0.0
    # 名前が無い版では None (0 にしない)
    assert step05.parse_prometheus_counter("# HELP x\nother_total 1\n") is None
    # 前方一致の別名を拾わない
    assert step05.parse_prometheus_counter("vllm:num_preemptions_total_created 9\n") is None


def test_metric_lines_and_delta():
    assert step05.metric_lines(METRICS) == METRICS.splitlines()[:4]
    assert step05.counter_delta(2.0, 5.0) == 3.0
    assert step05.counter_delta(None, 5.0) is None
    assert step05.counter_delta(2.0, None) is None


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def test_hf_revision():
    seen = {}

    def opener(url, timeout):
        seen["url"], seen["timeout"] = url, timeout
        return _Resp(json.dumps({"id": "org/m", "sha": "abc123"}).encode())

    assert step05.hf_revision("org/m", opener=opener) == {"model": "org/m", "sha": "abc123"}
    assert seen == {"url": "https://huggingface.co/api/models/org/m", "timeout": 10.0}

    def broken(url, timeout):
        raise OSError("offline")

    got = step05.hf_revision("org/m", opener=broken)
    assert got["sha"] is None and got["error"] == "OSError: offline"
    assert step05.hf_revision("org/m", opener=lambda u, timeout: _Resp(b"{}"))["error"] == "sha が無い"


def test_vllm_argv_pins_revision():
    argv = step05.vllm_argv("/v/bin/vllm", "org/m", 8001, 4096, 0.5, revision="abc", extra=["--enforce-eager"])
    assert argv == [
        "/v/bin/vllm", "serve", "org/m", "--served-model-name", "local-llm", "--port", "8001",
        "--dtype", "auto", "--max-model-len", "4096", "--gpu-memory-utilization", "0.50",
        "--revision", "abc", "--tokenizer-revision", "abc", "--enforce-eager",
    ]
    assert "--revision" not in step05.vllm_argv("vllm", "org/m", 8001, 4096, 0.76)


# ─── 試運転の確かめ ────────────────────────────────────────────
def _trial_raw(tmp_path, rows):
    return _jsonl(tmp_path / "raw.jsonl", rows)


NO_LOG = {"oom": [], "preempt": []}
IN_BUDGET = {"within_budget": True}


def test_trial_checks_all_ok(tmp_path):
    raw = _trial_raw(tmp_path, [{"id": "Q010", "error": None}, {"id": "N07", "error": None}])
    got = step05.trial_checks(raw, ["Q010", "N07"], "N07", IN_BUDGET, 0.0, NO_LOG)
    assert got["ok"] is True
    assert got["missing"] == []
    assert got["note"] == step05.TRIAL_NOTE


def test_trial_checks_missing_metric_is_never_ok(tmp_path):
    raw = _trial_raw(tmp_path, [{"id": "Q010", "error": None}])
    got = step05.trial_checks(raw, ["Q010"], "Q010", IN_BUDGET, None, NO_LOG)
    assert got["no_preemption"] is None
    assert got["ok"] is None
    # GPU の記録が未確認でも同じ
    got = step05.trial_checks(raw, ["Q010"], "Q010", {"within_budget": None}, 0.0, NO_LOG)
    assert got["ok"] is None


def test_trial_checks_failures(tmp_path):
    raw = _trial_raw(tmp_path, [{"id": "Q010", "error": None}, {"id": "N07", "error": "TimeoutError"}])
    got = step05.trial_checks(raw, ["Q010", "Q039", "N07"], "N07", IN_BUDGET, 0.0, NO_LOG)
    assert got["all_answered"] is False
    assert got["longest_ok"] is False
    assert got["missing"] == ["Q039", "N07"]
    assert got["ok"] is False
    # 未確認が混ざっていても、失敗が 1 つあれば False
    got = step05.trial_checks(raw, ["Q010"], None, {"within_budget": None}, None,
                              {"oom": ["CUDA out of memory"], "preempt": []})
    assert got["no_oom"] is False and got["ok"] is False
    # 指標が 0 でもログに追い出しがあれば False
    got = step05.trial_checks(raw, ["Q010"], None, IN_BUDGET, 0.0, {"oom": [], "preempt": ["preempted"]})
    assert got["no_preemption"] is False


def test_trial_checks_without_raw(tmp_path):
    got = step05.trial_checks(tmp_path / "none.jsonl", ["Q010"], "Q010", IN_BUDGET, 0.0, NO_LOG)
    assert got["all_answered"] is False and got["ok"] is False


# ─── run_eval との配線 ─────────────────────────────────────────
def test_eval_env_keeps_index_and_forces_cpu():
    env = step05.eval_env({"DATA_DIR": "/w/data", "QDRANT_COLLECTION": "c"}, "http://127.0.0.1:8001/v1",
                          LLM_MAX_TOKENS="1")
    assert env == {
        "DATA_DIR": "/w/data", "QDRANT_COLLECTION": "c", "EMBED_DEVICE": "cpu", "RERANK_DEVICE": "cpu",
        "VLLM_BASE_URL": "http://127.0.0.1:8001/v1", "LLM_MAX_TOKENS": "1",
    }


def test_cmds_use_module_form():
    cmd = step05.eval_cmd("py", "o", ids=["Q010", "N02"], meta={"gpu": "T4"})
    assert cmd[:3] == ["py", "-m", "evaluation.run_eval"]
    assert cmd[cmd.index("--ids") + 1] == "Q010,N02"
    assert [cmd[i + 1] for i, a in enumerate(cmd) if a == "--dataset"] == list(step05.DATASETS)
    assert "--ids" not in step05.eval_cmd("py", "o")
    assert step05.judge_cmd("py", "r/raw.jsonl", "j", meta={"judge_precheck": "pass"})[:3] == cmd[:3]


def test_attach_to_run_meta_refuses_overwrite(tmp_path):
    with pytest.raises(FileNotFoundError):
        step05.attach_to_run_meta(tmp_path, "gpu_mem", {})
    run_meta.write_run_meta(tmp_path, {"status": "done"})
    step05.attach_to_run_meta(tmp_path, "gpu_mem", {"peak_mib": 7000})
    with pytest.raises(ValueError, match="既にある"):
        step05.attach_to_run_meta(tmp_path, "gpu_mem", {"peak_mib": 1})
    meta = json.loads((tmp_path / run_meta.META_NAME).read_text(encoding="utf-8"))
    assert meta == {"status": "done", "gpu_mem": {"peak_mib": 7000}}


def test_notebook_argv_runs_through_run_eval(tmp_path, monkeypatch, run_eval):  # noqa: F811
    """ノートの eval_cmd → attach → judge_cmd の順で、run_eval が引数を受け付け、記録が採点側に写る。"""
    a = _jsonl(tmp_path / "a.jsonl", [_item("Q010", "あ"), _item("Q017", "いいい"), _item("Q099", "う")])
    b = _jsonl(tmp_path / "b.jsonl", [_item("N02", "ええ", expected_conclusion="不要")])
    _install_fakes(monkeypatch, run_eval)
    out, judged = tmp_path / "trial-8GB", tmp_path / "trial-8GB-judged"

    cmd = step05.eval_cmd("py", out, ids=["Q010", "Q017", "N02"], meta={"budget": "8GB", "gpu_util": "0.50"},
                          datasets=(a, b))
    _run(monkeypatch, run_eval, cmd[3:])
    rows = _rows(out / "raw.jsonl")
    assert [r["id"] for r in rows] == ["Q010", "Q017", "N02"]
    assert all(r["faithfulness"] == 0.0 for r in rows)  # --no-judge は採点しない (既定値のまま)
    assert step05.prompt_token_stats(out / "raw.jsonl") == {"max_prompt_tokens": 300, "max_id": "Q017", "n": 3}

    record = step05.gpu_mem_record([(1.0, 7000)], {"launch": (0.0, 2.0), "run": (0.5, 1.5)}, 7680, T4_MIB)
    step05.attach_to_run_meta(out, "gpu_mem", record)
    checks = step05.trial_checks(out / "raw.jsonl", ["Q010", "Q017", "N02"], "Q017", record, 0.0, NO_LOG)
    step05.attach_to_run_meta(out, "trial_checks", checks)
    assert checks["ok"] is True

    _run(monkeypatch, run_eval, step05.judge_cmd("py", out / "raw.jsonl", judged, meta={"judge_precheck": "pass"})[3:])
    meta = json.loads((judged / run_meta.META_NAME).read_text(encoding="utf-8"))
    assert meta["status"] == "done"
    assert meta["extras"] == {"judge_precheck": "pass"}
    assert meta["source_run_meta"]["extras"] == {"budget": "8GB", "gpu_util": "0.50"}
    assert meta["source_run_meta"]["gpu_mem"]["peak_mib"] == 7000
    assert meta["source_run_meta"]["trial_checks"]["ok"] is True
    assert all(r["faithfulness"] == 1.0 for r in _rows(judged / "raw.jsonl"))


# ─── 試運転ノート ──────────────────────────────────────────────
NOTEBOOK = Path(__file__).resolve().parents[1] / "colab" / "step05_trial.ipynb"


def _notebook_code(path: Path = NOTEBOOK) -> list[str]:
    """コードセルを、! のシェル行 (\\ の継続行を含む) を pass に替えた Python として返す。"""
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = []
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        out, cont = [], False
        for line in "".join(c["source"]).splitlines():
            body = line.lstrip()
            if cont or body.startswith(("!", "%")):
                out.append("" if cont else line[: len(line) - len(body)] + "pass")
                cont = line.rstrip().endswith("\\")
                continue
            out.append(line)
        cells.append("\n".join(out))
    return cells


def test_trial_notebook_cells_parse_and_use_existing_names():
    cells = _notebook_code()
    assert len(cells) >= 8
    for i, src in enumerate(cells):
        ast.parse(src, filename=f"step05_trial.ipynb セル {i}")
    used = set(re.findall(r"\bs5\.(\w+)", "\n".join(cells)))
    assert {"vllm_argv", "eval_cmd", "judge_cmd", "gpu_mem_record", "trial_checks", "attach_to_run_meta"} <= used
    assert sorted(n for n in used if not hasattr(step05, n)) == []


def test_trial_notebook_states_what_it_skips():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text = "".join("".join(c["source"]) for c in nb["cells"])
    assert "合否には使わない" in text
    assert "llama-server" in text                      # 計画 6 はやらないと書いてある
    assert "flow-check-only" in text                   # 採点役は代役
    assert "evaluation.judge_precheck" in text
    assert nb["metadata"]["colab"]["gpuType"] == "T4"


# ─── Step 1 ────────────────────────────────────────────────────
STEP1_NOTEBOOK = NOTEBOOK.with_name("step1_measure.ipynb")


def test_dataset_ids_covers_both_datasets_without_duplicates():
    repo = Path(__file__).resolve().parents[2]
    ids = step05.dataset_ids(repo)
    assert len(ids) == 62 and len(set(ids)) == 62
    assert {"Q001", "N01", "N12"} <= set(ids)


def test_dataset_ids_stops_on_duplicates(tmp_path):
    for name in ("a.jsonl", "b.jsonl"):
        (tmp_path / name).write_text(json.dumps({"id": "X1"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="X1"):
        step05.dataset_ids(tmp_path, datasets=("a.jsonl", "b.jsonl"))


def test_judge_failures_counts_only_judge_errors(tmp_path):
    raw = tmp_path / "raw.jsonl"
    rows = [{"id": "Q1", "error": None}, {"id": "Q2", "error": "judge_failed: context length"},
            {"id": "Q3", "error": "timeout"}, {"id": "Q4"}]
    raw.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert step05.judge_failures(raw) == {"n_rows": 4, "judge_failed": ["Q2"]}
    assert step05.judge_failures(tmp_path / "none.jsonl") == {"n_rows": 0, "judge_failed": []}


def test_trial_checks_note_can_be_replaced(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps({"id": "Q1", "error": None, "prompt_tokens": 10}) + "\n", encoding="utf-8")
    mem = {"within_budget": True}
    scan = {"oom": False, "preempt": False}
    assert step05.trial_checks(raw, ["Q1"], None, mem, 0.0, scan)["note"] == step05.TRIAL_NOTE
    assert step05.trial_checks(raw, ["Q1"], None, mem, 0.0, scan, note="基準線")["note"] == "基準線"


def test_step1_notebook_cells_parse_and_use_existing_names():
    cells = _notebook_code(STEP1_NOTEBOOK)
    assert len(cells) >= 9
    for i, src in enumerate(cells):
        ast.parse(src, filename=f"step1_measure.ipynb セル {i}")
    code = "\n".join(cells)
    used = set(re.findall(r"\bs5\.(\w+)", code))
    assert {"dataset_ids", "judge_failures", "trial_checks", "gpu_util_for_budget", "judge_cmd"} <= used
    assert sorted(n for n in used if not hasattr(step05, n)) == []
    # V は救わない: 下げて起動し直す部品も、8 問の抜き出しも使わない
    assert "adjusted_gpu_util" not in used and "trial_ids" not in used
    assert "s5.TRIAL_NOTE" not in code


def test_step1_notebook_gates_scoring_and_runs_g0_last():
    nb = json.loads(STEP1_NOTEBOOK.read_text(encoding="utf-8"))
    assert nb["metadata"]["colab"]["gpuType"] == "T4"
    cells = _notebook_code(STEP1_NOTEBOOK)
    code = "\n".join(cells)
    assert 'MODEL = "sbintuitions/sarashina2.2-3b-instruct-v0.1"' in code
    assert 'JUDGE_MODEL = "stelterlab/phi-4-AWQ"' in code
    assert '"real-judge"' in code and "flow-check-only" not in code
    assert 'JUDGE["precheck"] != "pass"' in code      # 採点役の確認に通らなければ採点しない
    judge_at = next(i for i, c in enumerate(cells) if "s5.judge_cmd" in c)
    g0_at = next(i for i, c in enumerate(cells) if "rag_connection_check.py" in c)
    summary_at = next(i for i, c in enumerate(cells) if "summary.json" in c)
    assert judge_at < g0_at < summary_at              # G0 は両サーバーを落とすので採点の後
    assert '"--rerank", "off"' in cells[g0_at]


def test_numbers_only_drops_text_and_summarises_times(tmp_path):
    raw = tmp_path / "raw.jsonl"
    rows = [{"id": "Q1", "category": "c", "question": "本文", "answer": "回答", "retrieved_chunks": [{"content": "x"}],
             "error": None, "total_latency": 2.0, "first_token_latency": 1.0, "retrieval_latency": 0.5, "faithfulness": 1.0},
            {"id": "Q2", "category": "c", "question": "本文", "answer": "回答", "error": None, "total_latency": 6.0,
             "first_token_latency": 3.0, "retrieval_latency": 0.7},
            {"id": "Q3", "category": "c", "question": "本文", "answer": "", "error": "timeout", "total_latency": 99.0}]
    raw.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    out = step05.numbers_only(raw)
    assert out["n_rows"] == 3 and out["n_error"] == 1
    assert out["times"]["total_latency"] == {"median": 4.0, "max": 6.0, "n": 2}  # 失敗した問は時間に入れない
    text = json.dumps(out, ensure_ascii=False)
    assert "本文" not in text and "回答" not in text and "content" not in text
    assert out["rows"][2] == {"id": "Q3", "category": "c", "error": "timeout", "total_latency": 99.0}
