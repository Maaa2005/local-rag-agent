"""Step 2 の測り方を手元の Linux GPU (WSL2 など・Docker を使わない) で流す。Issue #1 の 6GB 実機用。

evaluation/colab/step2_measure_llamacpp.ipynb @ 88cd974 (Gemma 4 E4B Q4_K_M・考え OFF・C ON) と同じ手順:
索引 → 入力トークン数の測定 (ctx 8192・出力 1) → 必要な文脈で別々に起動して 62 問 × runs 回 → G0 → summary.json。
違いは、予算が 1 つ (--budget-mib) で 8GB/12GB に分けない事と、P を換算せず測定値のまま見る事
(線は 2026-11-02 の Issue #1 のコメントで測る前に固定した)。

準備 (README.md):
    .work/qdrant/bin/qdrant, .work/llama-pre/**/llama-server, .work/gguf/*.gguf, .work/rag-env (Colab と同じ版)
流し方 (repo のルートで):
    .work/rag-env/bin/python -m evaluation.gpu_local.step2_local --work .work --out .work/out-<日時>
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from evaluation.colab import step05 as s5
from evaluation.gpu_local import mem_pressure
from evaluation.gpu_local import proc_mem
from evaluation.gpu_local import summary as sm

MODEL = "google/gemma-4-E4B-it"
GGUF_REPO = "unsloth/gemma-4-E4B-it-GGUF"
GGUF_REV = "bfc15c382204943c3a8fff0c750b94ae2364d7a3"
GGUF_FILE = "gemma-4-E4B-it-Q4_K_M.gguf"
QUANT = "Q4_K_M"
LLAMA_TAG = "b11836"
NUMERIC_CHECK = "true"
MEASURE_CTX = 8192
FAULT_TIMEOUT = 15
PORT = 8001
QDRANT_URL = "http://127.0.0.1:6333"
BASE_URL = f"http://127.0.0.1:{PORT}/v1"
# 換算の確かめ (合否と別): T4 の 88cd974 の回の出力速度 (8GB の回の中央値) と帯域
T4_TPS = 51.1
T4_BW_GBS = 320.0

PRECHECK_PY = r'''
import json, re, sys
from openai import OpenAI
c = OpenAI(base_url=sys.argv[1], api_key="NONE")
s = c.chat.completions.create(model="local-llm", messages=[{"role": "user", "content": "日本の首都はどこですか。一文で答えてください。"}],
                              max_tokens=64, temperature=0.1, stream=True, stream_options={"include_usage": True})
content, reasoning, usage = "", "", None
for ch in s:
    if getattr(ch, "usage", None) is not None:
        usage = ch.usage
    if ch.choices:
        d = ch.choices[0].delta
        content += d.content or ""
        reasoning += ((getattr(d, "model_extra", None) or {}).get("reasoning_content") or "")
print(json.dumps({"prompt_tokens": getattr(usage, "prompt_tokens", None), "content_chars": len(content),
                  "reasoning_chars": len(reasoning), "japanese": bool(re.search(r"[぀-ヿ]", content)),
                  "thinking_marks": bool(re.search(r"<\|?channel|<\|?think", content))}))
'''


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def launch(argv: list[str], log_path: Path, env: dict | None = None) -> subprocess.Popen:
    # 新しいプロセスグループで起動する (落とすときに killpg を使うため。pid = pgid)
    f = open(log_path, "w")
    return subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT, start_new_session=True, env=env)


def wait_up(url: str, proc: subprocess.Popen, limit: float) -> bool:
    t = time.time()
    while time.time() - t < limit:
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(url, timeout=3)
            log(f"ready: {url} {time.time() - t:.0f} 秒")
            return True
        except Exception:
            time.sleep(2)
    return False


def kill(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def stop_gently(proc: subprocess.Popen | None, wait_s: float = 15.0) -> None:
    """SIGTERM で止め (持ったメモリと powershell.exe を片付けさせる)、止まらなければ kill。"""
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=wait_s)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        kill(proc)


def read_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return s5.parse_smi_mib(out)


class GpuLog:
    """カード全体の使用量を 1 秒ごとに記録する (Colab 版と同じ間隔)。"""

    def __init__(self) -> None:
        self.samples: list[tuple[float, int]] = []
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while True:
            try:
                self.samples.append((time.time(), read_mib()))
            except Exception:
                pass
            time.sleep(1)


class Bench:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.repo = Path.cwd()
        self.out = Path(args.out).resolve()
        if self.out.exists() and any(self.out.iterdir()):
            raise SystemExit(f"{self.out} が空でない (前の結果と混ざるので止める)")
        self.out.mkdir(parents=True, exist_ok=True)
        work = Path(args.work).resolve()
        self.py = str(work / "rag-env" / "bin" / "python")
        # 配布版は glibc 2.38 以上が要る。古い OS では新しい glibc の ld.so で包んだ起動口を --server-bin で渡す
        self.server_bin = args.server_bin or str(sorted(work.glob("llama-pre/**/llama-server"))[0])
        libdirs = sorted({str(p.parent) for p in work.glob("llama-pre/**/*.so*")})
        # LLAMA_WORK は llama_server_noble.sh が配布版と glibc を探す場所。--work と食い違わないように渡す
        self.llama_env = dict(os.environ, LD_LIBRARY_PATH=":".join(libdirs + [os.environ.get("LD_LIBRARY_PATH", "")]),
                              LLAMA_WORK=str(work))
        self.gguf = str(work / "gguf" / GGUF_FILE)
        self.qdrant_bin = str(work / "qdrant" / "bin" / "qdrant")
        # WSL2 の ext4 では索引の作成が 5 秒を超え、qdrant-client の既定の時間切れ (5 秒) で落ちた (11/2)。
        # 保存先は --qdrant-storage でメモリ上 (/dev/shm) に置ける。中身は評価用の架空の文書だけ
        self.qdrant_storage = args.qdrant_storage or str(self.out / "qdrant_storage")
        self.windows: dict[str, tuple[float, float]] = {}
        self.gpu = GpuLog()
        # Issue #14: 部品ごとの常駐メモリ (llama-server・評価・索引)
        self.procs = proc_mem.ProcLog({"llama": "llama-server", "eval": "evaluation.run_eval",
                                       "index": "corpus_index.py"})
        self.server_extra = args.server_extra.split() if args.server_extra else []
        name, cap, total = [x.strip() for x in subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True).stdout.splitlines()[0].split(",")]
        self.gpu_name, self.cap, self.total_mib = name, cap, int(total)
        self.baseline = read_mib()
        v = subprocess.run([self.server_bin, "--version"], capture_output=True, text=True, env=self.llama_env)
        self.server_ver = (v.stdout + v.stderr).strip().splitlines()[-2:]
        self.meta_common = {"step": "2-local", "gpu": name, "compute_cap": cap, "model": MODEL,
                            "model_revision": GGUF_REV, "runtime": f"llama.cpp {LLAMA_TAG}",
                            "gguf": f"{GGUF_REPO}/{GGUF_FILE}", "quant": QUANT, "rerank": "off",
                            "numeric_check": "on", "note": "Issue #1 6GB 実機 (step2_local.py)"}
        log(f"{name} cap {cap} 全体 {self.total_mib} MiB / 起動前 {self.baseline} MiB / {self.server_ver}")

    # ─── 本体のメモリを絞る (Issue #14 E3) ───
    def start_pressure(self, label: str) -> tuple[subprocess.Popen, Path] | None:
        """--pressure-target-mib のときだけ、空きを目標に保つ別のプロセスを起動し、空きが幅に入るまで待つ。"""
        a = self.args
        if a.pressure_target_mib is None:
            return None
        plog = self.out / f"pressure_{label}.jsonl"
        proc = launch([sys.executable, "-m", "evaluation.gpu_local.mem_pressure", "--target-mib",
                       str(a.pressure_target_mib), "--probe", a.pressure_probe, "--band-mib", "128",
                       "--floor-mib", str(a.pressure_floor_mib), "--log", str(plog)],
                      self.out / f"pressure_{label}.log")
        t, steady = time.time(), 0
        while time.time() - t < a.pressure_settle_s and proc.poll() is None:
            rows = mem_pressure.load_log(plog)
            last = rows[-1] if rows else None
            ok = last is not None and abs(last["free_mib"] - a.pressure_target_mib) <= a.pressure_valid_band_mib
            steady = steady + 1 if ok else 0
            if steady >= 5:
                break
            time.sleep(1)
        rows = mem_pressure.load_log(plog)
        log(f"[{label}] 空き {rows[0]['free_mib'] if rows else '?'} → {rows[-1]['free_mib'] if rows else '?'} MiB"
            f" (目標 {a.pressure_target_mib}・{a.pressure_probe}) {'安定' if steady >= 5 else '安定しない'}")
        return proc, plog

    def pressure_record(self, started: tuple[subprocess.Popen, Path] | None, window: tuple[float, float]) -> dict | None:
        if started is None:
            return None
        proc, plog = started
        # 止める前に終わっていたら途中で落ちた（OOM で消された時も含む）
        exited_early = proc.poll() is not None
        stop_gently(proc)
        rows = mem_pressure.load_log(plog)
        rec = mem_pressure.window_summary(rows, *window, self.args.pressure_target_mib,
                                          self.args.pressure_valid_band_mib)
        rec["probe"] = self.args.pressure_probe
        rec["before_free_mib"] = rows[0]["free_mib"] if rows else None
        rec["exited_early"] = exited_early
        return rec

    # ─── 部品 ───
    def start_server(self, ctx: int, log_path: Path) -> tuple[subprocess.Popen, bool, dict, object]:
        proc = launch(s5.llama_server_argv(self.server_bin, self.gguf, PORT, ctx, extra=self.server_extra),
                      log_path, self.llama_env)
        up = wait_up(f"http://127.0.0.1:{PORT}/health", proc, 600)
        gpu = s5.llama_gpu_check(log_path.read_text(errors="replace"))
        warm = None
        if up and gpu["ok"]:
            body = json.dumps({"model": "local-llm", "messages": [{"role": "user", "content": "こんにちは"}],
                               "temperature": 0.1, "max_tokens": 16}).encode()
            t = time.time()
            try:
                req = urllib.request.Request(f"{BASE_URL}/chat/completions", body, {"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=300).read()
                warm = round(time.time() - t, 1)
            except Exception as e:
                warm = f"失敗: {repr(e)[:200]}"
        log(f"[{log_path.name}] 起動 {up} / GPU {gpu['offloaded']} ok={gpu['ok']} / 数えない問 {warm}")
        return proc, up, gpu, warm

    def run_eval(self, argv: list[str], env: dict, name: str) -> int:
        t0 = time.time()
        with open(self.out / f"{name.replace(':', '_')}.log", "w") as f:
            rc = subprocess.run(argv, cwd=self.repo, env=dict(os.environ, **env), stdout=f,
                                stderr=subprocess.STDOUT).returncode
        self.windows[name] = (t0, time.time())
        log(f"[{name}] rc={rc} {time.time() - t0:.0f} 秒")
        return rc

    def index(self) -> dict:
        t0 = time.time()
        p = subprocess.run([self.py, "evaluation/colab/corpus_index.py", "--repo", str(self.repo),
                            "--work", str(self.out / "index_work"), "--qdrant-url", QDRANT_URL,
                            "--collection", f"local_{int(time.time())}"],
                           cwd=self.repo, capture_output=True, text=True)
        self.windows["index"] = (t0, time.time())
        if p.returncode != 0:
            raise SystemExit(f"索引に失敗: {p.stderr[-2000:]}")
        return json.loads((self.out / "index_work" / "index.json").read_text())

    # ─── 流れ ───
    def measure(self, index: dict) -> dict:
        s5.wait_gpu_released(read_mib, self.baseline)
        srv, up, gpu, warm = self.start_server(MEASURE_CTX, self.out / "llama_measure.log")
        if not (up and gpu["ok"]):
            kill(srv)
            raise SystemExit("測定用の llama-server が起動しない・GPU に全層が載らない")
        pc = subprocess.run([self.py, "-c", PRECHECK_PY, BASE_URL], capture_output=True, text=True)
        try:
            pre = json.loads(pc.stdout.strip().splitlines()[-1])
        except Exception:
            pre = {"error": (pc.stdout + pc.stderr)[-500:]}
        pre["ok"] = bool((pre.get("prompt_tokens") or 0) > 0 and pre.get("content_chars", 0) > 0
                         and pre.get("reasoning_chars") == 0 and not pre.get("thinking_marks"))
        if not pre["ok"]:
            kill(srv)
            raise SystemExit(f"流し読みの事前確認が通らない: {pre}")
        rc = self.run_eval(s5.eval_cmd(self.py, self.out / "measure",
                                       meta={**self.meta_common, "purpose": "prompt_tokens", "ctx": MEASURE_CTX,
                                             "llm_max_tokens": 1}),
                           s5.eval_env(index["env"], BASE_URL, LLM_MAX_TOKENS="1", NUMERIC_CHECK_ENABLED=NUMERIC_CHECK),
                           "measure")
        kill(srv)
        stats = s5.prompt_token_stats(self.out / "measure" / "raw.jsonl")
        ctx = s5.needed_max_model_len(stats["max_prompt_tokens"])
        log(f"入力の最大 {stats['max_prompt_tokens']} ({stats['max_id']}) → 文脈 {ctx}")
        # 出力 1 トークンの回なので答えの指標は使えないが、検索の指標 (M1・M2・検索の時間) はそのまま読める
        retrieval = sm.retrieval_numbers(sm.load_results(self.out / "measure" / "raw.jsonl"))
        log(f"検索: {retrieval}")
        return {**stats, "ctx": MEASURE_CTX, "rc": rc, "needed_ctx": ctx, "gpu_check": gpu,
                "warmup_seconds": warm, "precheck": pre, "retrieval": retrieval,
                "proc_mem": self.procs.stats(*self.windows["measure"]),
                "index_proc_mem": self.procs.stats(*self.windows["index"]) if "index" in self.windows else None}

    def run_once(self, label: str, index: dict, ctx: int, all_ids: list[str], longest_id: str) -> dict:
        budget = self.args.budget_mib
        s5.wait_gpu_released(read_mib, self.baseline)
        llog = self.out / f"llama_{label}.log"
        t_launch = time.time()
        proc, up, gpu, warm = self.start_server(ctx, llog)
        self.windows[f"{label}:launch"] = (t_launch, time.time())
        rec: dict = {"label": label, "ctx": ctx, "budget_mib": budget, "gpu_check": gpu, "warmup_seconds": warm}
        if not (up and gpu["ok"]):
            kill(proc)
            rec.update(started=up, gpu_ok=gpu["ok"],
                       peak_mib=s5.peak_in_window(self.gpu.samples, t_launch, time.time())["peak_mib"],
                       lines={"all": False}, note="起動しない・GPU に全層が載らない")
            return rec
        log_start = len(llog.read_text(errors="replace"))
        out_dir = self.out / f"run_{label}"
        pressure = self.start_pressure(label)
        try:
            rc = self.run_eval(s5.eval_cmd(self.py, out_dir, meta={**self.meta_common, "budget": label,
                                                                   "budget_mib": budget, "ctx": ctx}),
                               s5.eval_env(index["env"], BASE_URL, NUMERIC_CHECK_ENABLED=NUMERIC_CHECK),
                               f"{label}:run")
        finally:
            run_window = self.windows.get(f"{label}:run", (time.time(), time.time()))
            rec["pressure"] = self.pressure_record(pressure, run_window)
        text = llog.read_text(errors="replace")
        kill(proc)
        whole, run_part = s5.scan_llama_log(text), s5.scan_llama_log(text[log_start:])
        log_scan = dict(whole, debug_active=run_part["debug_active"], slot_n_ctx=run_part["slot_n_ctx"])
        mem = s5.gpu_mem_record(self.gpu.samples, {"launch": self.windows[f"{label}:launch"],
                                                   "run": self.windows[f"{label}:run"]},
                                budget, self.total_mib, self.baseline)
        checks = s5.trial_checks(out_dir / "raw.jsonl", all_ids, longest_id, mem,
                                 s5.llama_preempt_delta(log_scan), log_scan, note="Issue #1 6GB 実機")
        if (out_dir / "run_meta.json").exists():
            for key, val in (("gpu_mem", mem), ("run_checks", checks), ("llama_log_scan", log_scan),
                             ("gpu_check", gpu)):
                s5.attach_to_run_meta(out_dir, key, val)
        nums = sm.run_numbers(sm.load_results(out_dir / "raw.jsonl"))
        rec.update(started=True, gpu_ok=True, rc=rc, out_dir=str(out_dir), peak_mib=mem["peak_mib"],
                   v_checks=checks, slot_n_ctx=log_scan["slot_n_ctx"], debug_active=log_scan["debug_active"],
                   oom_lines=len(log_scan["oom"]), overflow_lines=len(log_scan["preempt"]), numbers=nums,
                   lines=sm.line_checks(nums, mem["peak_mib"], budget, checks),
                   proc_mem=self.procs.stats(*self.windows[f"{label}:run"]),
                   lazy_lines=[ln[:200] for ln in text.splitlines() if "lazy" in ln.lower()][:5])
        log(f"[{label}] 最大 {mem['peak_mib']} MiB / 予算 {budget} → {rec['lines']} / N {nums['N']}")
        return rec

    def g0(self, ctx: int, qdrant: subprocess.Popen) -> dict:
        s5.wait_gpu_released(read_mib, self.baseline)
        srv, up, gpu, _ = self.start_server(ctx, self.out / "llama_g0.log")
        res: dict = {"rerank": "off", "fault_timeout": FAULT_TIMEOUT, "ctx": ctx, "gpu_check": gpu}
        if up and gpu["ok"]:
            work = self.out / "g0_work"
            p = subprocess.run([self.py, "evaluation/colab/rag_connection_check.py", "--repo", str(self.repo),
                                "--work", str(work), "--qdrant-url", QDRANT_URL, "--vllm-url", BASE_URL,
                                "--qdrant-pgid", str(qdrant.pid), "--vllm-pgid", str(srv.pid),
                                "--fault-timeout", str(FAULT_TIMEOUT), "--rerank", "off",
                                "--label", f"local-{self.gpu_name.replace(' ', '-')}-rerank-off"],
                               cwd=self.repo, capture_output=True, text=True,
                               env=dict(os.environ, NUMERIC_CHECK_ENABLED=NUMERIC_CHECK))
            (self.out / "g0.log").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
            result = work / "rag_check_result.json"
            res["rc"] = p.returncode
            res["ok"] = json.loads(result.read_text()).get("ok") if result.exists() else None
        else:
            res["skipped"] = "G0 用の llama-server が起動しない"
        kill(srv)
        return res

    def main(self) -> None:
        q_env = dict(os.environ, QDRANT__STORAGE__STORAGE_PATH=self.qdrant_storage,
                     QDRANT__SERVICE__HTTP_PORT="6333")
        summary: dict = {"issue": 1, "gpu": self.gpu_name, "compute_cap": self.cap, "total_mib": self.total_mib,
                         "baseline_mib": self.baseline, "budget_mib": self.args.budget_mib,
                         "runtime": f"llama.cpp {LLAMA_TAG}", "server_version": self.server_ver, "model": MODEL,
                         "gguf": {"repo": GGUF_REPO, "file": GGUF_FILE, "revision": GGUF_REV}, "quant": QUANT,
                         "embedding": f"{os.environ.get('EMBED_MODEL', 'intfloat/multilingual-e5-large')} (CPU)",
                         "rerank": "off", "numeric_check": NUMERIC_CHECK, "server_extra": self.server_extra,
                         "measure_only": self.args.measure_only, "runs": [],
                         "pressure": None if self.args.pressure_target_mib is None else {
                             "target_mib": self.args.pressure_target_mib, "probe": self.args.pressure_probe,
                             "valid_band_mib": self.args.pressure_valid_band_mib,
                             "floor_mib": self.args.pressure_floor_mib}}
        qdrant = launch([self.qdrant_bin], self.out / "qdrant.log", q_env)
        try:
            # 起動しない時も finally で落とす (残ると次の回が古い Qdrant に繋がる)
            if not wait_up(f"{QDRANT_URL}/readyz", qdrant, 60):
                raise SystemExit("Qdrant が起動しない")
            index = self.index()
            summary["index"] = {k: index.get(k) for k in ("documents", "chunks", "seconds")}
            measure = self.measure(index)
            summary["measure"] = measure
            ctx = measure["needed_ctx"]
            summary["server_args"] = s5.llama_server_argv("llama-server", GGUF_FILE, PORT, ctx,
                                                          extra=self.server_extra)[1:]
            if self.args.measure_only:
                # Issue #14 E1: 検索とメモリだけを見る回。回答の回と G0 は流さない
                self._write(summary)
                log(f"終わり (入力長の測定だけ): {self.out / 'summary.json'}")
                return
            all_ids = s5.dataset_ids(self.repo)
            for i in range(1, self.args.runs + 1):
                summary["runs"].append(self.run_once(f"{self.args.label}-{i}", index, ctx, all_ids,
                                                     measure["max_id"]))
                self._write(summary)
            if not self.args.skip_g0:
                summary["g0"] = self.g0(ctx, qdrant)
        finally:
            kill(qdrant)
        started = [r for r in summary["runs"] if r.get("started")]
        # 回ごとに確かめ、全部の回で予測の幅に入った時だけ「外れていない」
        conv = {r["label"]: sm.conversion_check(T4_TPS, T4_BW_GBS, self.args.bandwidth_gbs,
                                                r["numbers"]["decode_tps_median"]) for r in started}
        summary["conversion_check"] = {"runs": conv, "within_all": len(conv) == self.args.runs and all(
            c["within"] is True for c in conv.values())}
        summary["verdict_all_lines"] = bool(started) and len(started) == self.args.runs and all(
            r["lines"]["all"] for r in started) and summary.get("g0", {}).get("ok") is True
        self._write(summary)
        log(f"終わり: {self.out / 'summary.json'}")

    def _write(self, summary: dict) -> None:
        (self.out / "summary.json").write_text(json.dumps(sm.json_safe(summary), ensure_ascii=False, indent=2) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work", required=True, help="準備物の置き場 (.work)")
    p.add_argument("--out", required=True, help="結果の置き場 (空のディレクトリ)")
    p.add_argument("--runs", type=int, default=2, help="別々に起動して流す回数 (最終判定は 2 回)")
    p.add_argument("--budget-mib", type=int, default=5632, help="V の予算 (6GB: 6,144 − 512)")
    p.add_argument("--bandwidth-gbs", type=float, default=168.0, help="このカードのメモリ帯域 (換算の確かめ用)")
    p.add_argument("--label", default="6GB")
    p.add_argument("--server-bin", help="llama-server の起動口 (既定は .work/llama-pre の配布版)")
    p.add_argument("--qdrant-storage", help="Qdrant の保存先 (既定は --out の下)")
    p.add_argument("--measure-only", action="store_true",
                   help="索引と入力長の測定 (出力 1 トークン) だけで止める。検索の指標とメモリを見る (Issue #14 E1)")
    p.add_argument("--server-extra", default="", help="llama-server に足す引数 (空白区切り。例: '-lzm on')")
    p.add_argument("--skip-g0", action="store_true")
    # Issue #14 E3: 回答の回の間だけ、本体の空きを目標に保つ (evaluation/gpu_local/mem_pressure.py)
    p.add_argument("--pressure-target-mib", type=int, default=None,
                   help="回答の回の間に保つ空き (MiB)。E3 の S1 は 1024。無ければ絞らない (今までと同じ)")
    p.add_argument("--pressure-probe", choices=["windows", "linux"], default="windows",
                   help="どの空きを目標に合わせるか (WSL では Windows の空き)")
    p.add_argument("--pressure-valid-band-mib", type=int, default=256,
                   help="有効な回: 回答中の空きの中央値が 目標 ± この幅 (E3 で測る前に固定)")
    p.add_argument("--pressure-floor-mib", type=int, default=256, help="これを下回ったら持った分を全部放す")
    p.add_argument("--pressure-settle-s", type=float, default=180.0, help="空きが幅に入るまで待つ上限 (秒)")
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(Bench(parse_args()).main())
