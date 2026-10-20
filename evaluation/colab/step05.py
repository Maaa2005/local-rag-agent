"""Step 0.5 (試運転) の部品。docs/v2-light-model-plan.md §1 V・§4 Step 0.5。

ノート (step05_trial.ipynb) の kernel から import するので標準ライブラリだけで書く。
ここで出す確かめ (trial_checks) は試運転用で、合否には使わない。

流れ (ノート):
1. 索引 (corpus_index) → 2. 入力トークンだけ測る (LLM_MAX_TOKENS=1・62 問) → max-model-len を決める
→ 3. 8GB・12GB の予算ごとに GPU_UTIL を出して起動し、8 問 + 最長の問を GPU の記録の中で流す
→ 4. 採点役を別に起動し、採点役の確認 (judge_precheck) と --judge-from。
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path

from evaluation import run_meta

BUDGETS_MIB = {"8GB": 7680, "12GB": 11776}  # カード容量 − 512 MiB (§1 V)
OUTPUT_TOKENS = 1024  # 最大出力 (backend/app/config.py の llm_max_tokens)
LEN_STEP = 512
DATASETS = ("evaluation/dataset.jsonl", "evaluation/v2/numeric_boundary.jsonl")
# 50 問から権限 2・拒否 2・数値 2、数値の境目 (N) から 2
TRIAL_IDS = ("Q010", "Q039", "Q012", "Q045", "Q017", "Q018", "N02", "N03")
PREEMPT_METRIC = "vllm:num_preemptions_total"
TRIAL_NOTE = "試運転・合否には使わない"


# ─── 予算と長さ ────────────────────────────────────────────────
def gpu_util_for_budget(budget_mib: int, total_mib: int, resident_mib: int = 0) -> float:
    """GPU_UTIL = (予算 − GPU に載せた埋め込み・リランカーの MiB) ÷ カード全体。小数 2 桁で切り捨て。

    T4 (15,360 MiB) で埋め込み・リランカーが CPU なら 8GB は 0.50、12GB は 0.76。
    浮動小数の誤差で 1 桁上がらないよう整数で割る。
    """
    if total_mib <= 0:
        raise ValueError(f"カード全体の MiB が不正: {total_mib}")
    if budget_mib > total_mib:
        raise ValueError(f"予算 {budget_mib} MiB がカード全体 {total_mib} MiB を超える")
    usable = budget_mib - resident_mib
    if usable <= 0:
        raise ValueError(f"予算 {budget_mib} MiB から常駐 {resident_mib} MiB を引くと残らない")
    return (usable * 100 // total_mib) / 100


def adjusted_gpu_util(util: float, peak_mib: int, budget_mib: int, total_mib: int) -> float | None:
    """記録した最大値が予算を超えた分だけ GPU_UTIL を下げた値 (2 回目の起動用)。超えていなければ None。

    下げるだけで上げない (起動できなかった時に上げて救うことはしない)。目安で、2 回目も記録で確かめる。
    """
    if peak_mib <= budget_mib:
        return None
    pct = round(util * 100)
    new_pct = (pct * total_mib - (peak_mib - budget_mib) * 100) // total_mib
    if new_pct <= 0:
        raise ValueError(f"超過 {peak_mib - budget_mib} MiB を引くと GPU_UTIL が残らない")
    return new_pct / 100


def needed_max_model_len(max_prompt_tokens: int, max_tokens: int = OUTPUT_TOKENS, step: int = LEN_STEP) -> int:
    """必要な max-model-len = 入力の最大トークン数 + 出力、を step 単位で切り上げた値 (§1 V)。"""
    if max_prompt_tokens < 0 or max_tokens <= 0 or step <= 0:
        raise ValueError("トークン数が不正")
    return -(-(max_prompt_tokens + max_tokens) // step) * step


def _read_rows(raw_path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(raw_path).read_text(encoding="utf-8").splitlines() if line.strip()]


def prompt_token_stats(raw_path: Path) -> dict:
    """測定の raw.jsonl から入力の最大トークン数とその問を出す。

    1 問でも失敗・トークン数なしがあれば止める (その問の入力が最長かもしれず、最大値が小さく出るため)。
    """
    rows = _read_rows(raw_path)
    if not rows:
        raise ValueError(f"{raw_path} に行が無い")
    bad = [r.get("id") for r in rows if r.get("error") or r.get("prompt_tokens") is None]
    if bad:
        raise ValueError(f"入力トークン数が取れなかった問 {len(bad)} 件: {bad}")
    top = max(rows, key=lambda r: r["prompt_tokens"])
    return {"max_prompt_tokens": top["prompt_tokens"], "max_id": top["id"], "n": len(rows)}


def trial_ids(longest_id: str | None) -> list[str]:
    """試運転の問 (TRIAL_IDS) に、入力が最長の問を足す (既に入っていれば足さない)。"""
    ids = list(TRIAL_IDS)
    if longest_id and longest_id not in ids:
        ids.append(longest_id)
    return ids


# ─── GPU メモリの記録 ──────────────────────────────────────────
def parse_smi_mib(stdout: str) -> int:
    """`nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits` の 1 行目を MiB の整数にする。"""
    for line in stdout.splitlines():
        m = re.search(r"\d+", line)
        if m:
            return int(m.group())
    raise ValueError(f"nvidia-smi の出力に数が無い: {stdout!r}")


def peak_in_window(samples: list[tuple[float, int]], start: float, end: float) -> dict:
    """[start, end] の間の記録の最大値。記録が 1 つも無ければ peak_mib は None。"""
    inside = [mib for t, mib in samples if start <= t <= end]
    return {"start": start, "end": end, "peak_mib": max(inside) if inside else None, "n_samples": len(inside)}


def gpu_mem_record(
    samples: list[tuple[float, int]],
    windows: dict[str, tuple[float, float]],
    budget_mib: int,
    total_mib: int,
    baseline_mib: int | None = None,
) -> dict:
    """区間ごと (起動: launch→ready、実行: run 開始→終了) の最大値と、予算に収まったか。

    起動中のプロファイル・CUDA グラフの確保で実行中より高くなりうるので、両方の大きい方で見る。
    1 つでも記録の無い区間があれば within_budget は None (未確認)。
    """
    per = {name: peak_in_window(samples, s, e) for name, (s, e) in windows.items()}
    peaks = [w["peak_mib"] for w in per.values()]
    known = [p for p in peaks if p is not None]
    peak = max(known) if known else None
    if not per or None in peaks:
        within = None
    else:
        within = peak <= budget_mib
    t0 = min(s for s, _ in windows.values()) if windows else 0.0
    t1 = max(e for _, e in windows.values()) if windows else 0.0
    return {
        "peak_mib": peak,
        "budget_mib": budget_mib,
        "total_mib": total_mib,
        "baseline_mib": baseline_mib,
        "within_budget": within,
        "windows": per,
        # 起動の始まりからの秒と MiB。後でグラフにできるように残す
        "samples": [[round(t - t0, 1), mib] for t, mib in samples if t0 <= t <= t1],
    }


def wait_gpu_released(read_mib, baseline_mib: int, tol_mib: int = 64, limit: float = 120,
                      sleep=time.sleep, clock=time.monotonic) -> int:
    """前のサーバーを止めた後、GPU の使用量が起動前 + tol に戻るまで待つ。戻らなければ止める。"""
    deadline = clock() + limit
    while True:
        mib = read_mib()
        if mib <= baseline_mib + tol_mib:
            return mib
        if clock() >= deadline:
            raise RuntimeError(f"GPU が空かない: {mib} MiB (起動前 {baseline_mib} MiB)。前のサーバーが残っている")
        sleep(1)


# ─── vLLM の記録 ───────────────────────────────────────────────
_OOM = re.compile(r"CUDA out of memory|OutOfMemoryError")
_PREEMPT = re.compile(r"preempt", re.IGNORECASE)


def scan_vllm_log(text: str, keep: int = 20) -> dict:
    """vLLM のログ (実行の間の部分) から OOM と KV の追い出しの行を拾う。各 keep 行まで。"""
    lines = text.splitlines()
    return {
        "oom": [ln for ln in lines if _OOM.search(ln)][:keep],
        "preempt": [ln for ln in lines if _PREEMPT.search(ln)][:keep],
    }


def metric_lines(text: str, word: str = "preempt") -> list[str]:
    """/metrics の中で word を含む行 (版で名前が違うので、ノートで実物を表示する用)。"""
    return [ln for ln in text.splitlines() if word in ln]


def parse_prometheus_counter(text: str, name: str = PREEMPT_METRIC) -> float | None:
    """/metrics の name の値をラベル違いの行も合わせて足す。1 行も無ければ None。"""
    total, found = 0.0, False
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = re.match(rf"{re.escape(name)}(\{{[^}}]*\}})?\s+(\S+)", line)
        if m:
            total += float(m.group(2))
            found = True
    return total if found else None


def counter_delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return after - before


def hf_revision(model: str, opener=urllib.request.urlopen, timeout: float = 10.0) -> dict:
    """Hugging Face の今の revision (コミット) を取る。失敗は止めずに記録する。"""
    url = f"https://huggingface.co/api/models/{model}"
    try:
        with opener(url, timeout=timeout) as resp:
            sha = json.loads(resp.read().decode("utf-8")).get("sha")
    except Exception as exc:  # noqa: BLE001 — 記録用。取れなくても試運転は続ける
        return {"model": model, "sha": None, "error": f"{type(exc).__name__}: {exc}"}
    if not sha:
        return {"model": model, "sha": None, "error": "sha が無い"}
    return {"model": model, "sha": sha}


def vllm_argv(vllm_bin: str, model: str, port: int, max_model_len: int, gpu_util: float,
              revision: str | None = None, served_name: str = "local-llm", extra=()) -> list[str]:
    """vllm serve の引数。revision があれば重みとトークナイザを同じコミットに固定する。"""
    argv = [
        vllm_bin, "serve", model,
        "--served-model-name", served_name,
        "--port", str(port),
        "--dtype", "auto",
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", f"{gpu_util:.2f}",
    ]
    if revision:
        argv += ["--revision", revision, "--tokenizer-revision", revision]
    return argv + list(extra)


# ─── run_eval を別プロセスで流す ────────────────────────────────
def eval_env(index_env: dict[str, str], vllm_base_url: str, **extra: str) -> dict[str, str]:
    """run_eval に渡す環境変数。索引と同じ場所 (index.json の env) を見て、埋め込み・リランカーは CPU。"""
    env = dict(index_env)
    env.update({"EMBED_DEVICE": "cpu", "RERANK_DEVICE": "cpu", "VLLM_BASE_URL": vllm_base_url})
    env.update(extra)
    return env


def _meta_args(meta: dict[str, object] | None) -> list[str]:
    out: list[str] = []
    for k, v in (meta or {}).items():
        out += ["--meta", f"{k}={v}"]
    return out


def eval_cmd(python: str, out_dir: Path, ids: list[str] | None = None, meta: dict | None = None,
             datasets=DATASETS) -> list[str]:
    """回答だけ作る (--no-judge・リランク off)。repo のルートを cwd にして -m で流す。

    `python evaluation/run_eval.py` の形は evaluation/types.py が標準の types を隠して起動しない。
    """
    argv = [python, "-m", "evaluation.run_eval", "--provider", "vllm", "--no-judge", "--rerank", "off"]
    for d in datasets:
        argv += ["--dataset", str(d)]
    if ids:
        argv += ["--ids", ",".join(ids)]
    return argv + ["--out-dir", str(out_dir)] + _meta_args(meta)


def judge_cmd(python: str, raw_path: Path, out_dir: Path, meta: dict | None = None) -> list[str]:
    """保存した回答を、今 vLLM に載っている採点役で採点する。"""
    return [
        python, "-m", "evaluation.run_eval", "--judge-from", str(raw_path), "--judge", "vllm",
        "--out-dir", str(out_dir),
    ] + _meta_args(meta)


def attach_to_run_meta(out_dir: Path, key: str, value) -> Path:
    """run_eval が書いた run_meta.json に欄を足す。既にある欄は上書きしない。

    --judge-from は元の run_meta.json をその時点で写すので、採点の前に足すこと。
    """
    path = Path(out_dir) / run_meta.META_NAME
    meta = json.loads(path.read_text(encoding="utf-8"))
    if key in meta:
        raise ValueError(f"{path} に {key} が既にある")
    meta[key] = value
    return run_meta.write_run_meta(out_dir, meta)


# ─── 試運転の確かめ ────────────────────────────────────────────
def _tri(values: list[bool | None]) -> bool | None:
    """1 つでも False なら False、False が無く 1 つでも None なら None (未確認)、全部 True なら True。"""
    if any(v is False for v in values):
        return False
    if any(v is None for v in values):
        return None
    return True


def trial_checks(raw_path: Path, expected_ids: list[str], longest_id: str | None, gpu_mem: dict,
                 preempt_delta: float | None, log_scan: dict) -> dict:
    """§1 V の合格の条件を試運転の範囲で確かめる。取れなかった項目は None (未確認) で、True にしない。"""
    rows = {r["id"]: r for r in _read_rows(raw_path)} if Path(raw_path).exists() else {}
    answered = [i for i in expected_ids if i in rows and not rows[i].get("error")]
    if log_scan["preempt"]:
        no_preempt = False
    elif preempt_delta is None:
        no_preempt = None
    else:
        no_preempt = preempt_delta == 0
    checks = {
        "all_answered": len(answered) == len(expected_ids),
        "longest_ok": None if not longest_id else (longest_id in rows and not rows[longest_id].get("error")),
        "no_oom": not log_scan["oom"],
        "no_preemption": no_preempt,
        "within_budget": gpu_mem.get("within_budget"),
    }
    return {
        **checks,
        "missing": [i for i in expected_ids if i not in answered],
        "ok": _tri(list(checks.values())),
        "note": TRIAL_NOTE,
    }
