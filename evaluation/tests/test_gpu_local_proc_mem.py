"""プロセスのメモリ記録 (evaluation/gpu_local/proc_mem.py) と検索だけの集計のテスト。

確かめること:
- /proc の status から VmRSS などを kB で読み、stat の majflt をコマンド名の括弧や空白に惑わされず読む
- コマンドラインの文字列で対象を選び、同じ名前の複数プロセスは足す
- 区間の最大 (MiB) と majflt の増え方を出す
- 検索だけの集計は Q の 50 問で M1・M2、検索の時間は全問で取り、失敗は無限大
"""
from __future__ import annotations

import json

from evaluation.gpu_local import proc_mem, summary

STATUS = "Name:\tllama\nVmHWM:\t 2048 kB\nVmRSS:\t 1024 kB\nRssAnon:\t 512 kB\nRssFile:\t 256 kB\nThreads:\t4\n"
# 3 番目 (state) から数えて 10 番目が 12 番目の欄 majflt (man 5 proc)
STAT = "123 (llama (server) x) S 1 2 3 4 5 6 7 8 777 11 12 13 14 15 16 17 18"


def _fake_proc(tmp_path, pid, cmd, status=STATUS, stat=STAT):
    d = tmp_path / str(pid)
    d.mkdir()
    (d / "cmdline").write_bytes(cmd.replace(" ", "\0").encode())
    (d / "status").write_text(status)
    (d / "stat").write_text(stat)


def test_parse_status_and_majflt():
    assert proc_mem.parse_status(STATUS) == {"VmHWM": 2048, "VmRSS": 1024, "RssAnon": 512, "RssFile": 256}
    assert proc_mem.parse_majflt(STAT) == 777


def test_sample_sums_matching_processes(tmp_path):
    _fake_proc(tmp_path, 10, "/x/ld.so /x/llama-server -m a")
    _fake_proc(tmp_path, 11, "/x/llama-server --port 1")
    _fake_proc(tmp_path, 12, "python -m evaluation.run_eval")
    (tmp_path / "self").mkdir()
    out = proc_mem.sample({"llama": "llama-server", "eval": "evaluation.run_eval", "none": "qdrant"}, tmp_path)
    assert out["llama"]["VmRSS"] == 2048 and out["llama"]["majflt"] == 1554
    assert out["eval"]["RssAnon"] == 512
    assert "none" not in out


def test_window_stats_max_and_majflt_delta():
    s = [(1.0, {"llama": {"VmRSS": 1024, "VmHWM": 1024, "RssAnon": 512, "RssFile": 512, "majflt": 10}}),
         (2.0, {"llama": {"VmRSS": 4096, "VmHWM": 4096, "RssAnon": 1024, "RssFile": 2048, "majflt": 25}}),
         (9.0, {"llama": {"VmRSS": 9999, "VmHWM": 9999, "RssAnon": 9999, "RssFile": 9999, "majflt": 99}})]
    st = proc_mem.window_stats(s, 0.5, 2.5)["llama"]
    assert st["max_mib"]["VmRSS"] == 4.0 and st["max_mib"]["RssAnon_plus_File"] == 3.0
    assert st["majflt_delta"] == 15 and st["n_samples"] == 2


def test_retrieval_numbers(tmp_path):
    rows = [{"id": "Q001", "provider": "vllm", "question": "q", "category": "normal", "user_access_level": 1,
             "recall_at_5": 1.0, "mrr": 1.0, "retrieval_latency": 0.2},
            {"id": "Q002", "provider": "vllm", "question": "q", "category": "normal", "user_access_level": 1,
             "recall_at_5": 0.0, "mrr": 0.0, "retrieval_latency": 0.0, "error": "run_failed: X: y"},
            {"id": "N01", "provider": "vllm", "question": "q", "category": "numeric", "user_access_level": 1,
             "expected_conclusion": "必要", "retrieval_latency": 0.4}]
    p = tmp_path / "raw.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    r = summary.retrieval_numbers(summary.load_results(p))
    assert r["n_q"] == 2 and r["n_all"] == 3 and r["answer_failed"] == 1
    assert r["M1_recall_at_5"] == 0.5
    assert r["retrieval_max"] == float("inf")
