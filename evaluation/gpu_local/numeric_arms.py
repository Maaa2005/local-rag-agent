"""数の比べの安い腕と規程カード方式を、N 12 問 × 2 回と開発用 24 問で比べる (Issue #16 の手順 1・2)。

step2_local と同じ回し方 (別々に起動した llama-server に run_eval を流す・リランク off) を、腕ごとに繰り返す。
違いは、Windows でそのまま動く事 (WSL・Docker を使わない) と、本体のメモリの空きが足りない時は流さない事。
封の問・確かめ用 48 問は使わない。

direct はカードの抽出だけを、検索 (Qdrant と埋め込み) なしで流す。カードは質問の文だけで決めるので、結論・保留・値の一致は
run と同じ物が測れる。時間は検索を含まない。

流し方 (repo のルートで・PYTHONUTF8=1):
    python -m evaluation.gpu_local.numeric_arms direct --server <llama-server> --gguf <gguf> --out <dir>
    python -m evaluation.gpu_local.numeric_arms prepare --server <llama-server> --gguf <gguf> --qdrant <qdrant> --out <dir>
    python -m evaluation.gpu_local.numeric_arms run --arm c0 --server ... --gguf ... --qdrant ... --out <dir>
    python -m evaluation.gpu_local.numeric_arms summary --out <dir>
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path

from evaluation import metrics
from evaluation.colab import step05 as s5
from evaluation.gpu_local import summary as sm
from evaluation.gpu_local.step2_local import BASE_URL, MEASURE_CTX, PORT, QDRANT_URL, GpuLog, log, read_mib, wait_up
from evaluation.reporter import answer_failed, conclusion_tally
from evaluation.types import QuestionResult

N12 = "evaluation/v2/numeric_boundary.jsonl"
DEV24 = "evaluation/v2/numeric_dev24.jsonl"
CARDS_DIR = "evaluation/v2/rule_cards"
HOLD = "保留"
# 控えの既定 (8GB) は本体のメモリが足りない PC で落ちる。控えが無くても入力量の記録は変わらない
SERVER_EXTRA = ["--cache-ram", "1024"]
PROMPT_FLAGS = {"NUMERIC_NORMALIZE_ENABLED": "true", "NUMERIC_CONTRAST_ENABLED": "true",
                "NUMERIC_CODEFORM_ENABLED": "true"}
ARMS = {
    "c0": {},
    "norm": {"NUMERIC_NORMALIZE_ENABLED": "true"},
    "contrast": {"NUMERIC_CONTRAST_ENABLED": "true"},
    "codeform": {"NUMERIC_CODEFORM_ENABLED": "true"},
    "cards": {"RULE_CARDS_ENABLED": "true", "RULE_CARDS_DIR": CARDS_DIR, "RULE_CARDS_ALLOW_DRAFT": "true",
              "RULE_CARDS_EXTRACT_MODEL": "local-llm"},
}
ROUNDS = (("n1", N12, "main"), ("n2", N12, "main"), ("dev", DEV24, "dev"))
DIRECT_ARM = "cards_direct"
DIRECT_CTX = 4096  # 抽出の入力は質問とカードの説明だけ。あふれは記録の overflow_lines で確かめる


class MemLog:
    """本体のメモリの空きと、部品ごとの常駐の最大を 1 秒ごとに記録する (Windows でも動く psutil を使う)。"""

    def __init__(self) -> None:
        import psutil

        self.psutil = psutil
        self.min_available_mib: int | None = None
        self.peak_rss_mib: dict[str, int] = {}
        threading.Thread(target=self._loop, daemon=True).start()

    def available_mib(self) -> int:
        return self.psutil.virtual_memory().available // (1024 * 1024)

    def _loop(self) -> None:
        while True:
            try:
                self._sample()
            except Exception:
                pass
            time.sleep(1)

    def _sample(self) -> None:
        avail = self.available_mib()
        self.min_available_mib = avail if self.min_available_mib is None else min(self.min_available_mib, avail)
        totals: dict[str, int] = {}
        for p in self.psutil.process_iter(["name", "cmdline", "memory_info"]):
            cmd = " ".join(p.info["cmdline"] or [])
            for key, pat in (("llama", "llama-server"), ("eval", "evaluation.run_eval"), ("qdrant", "qdrant.exe")):
                if pat in (p.info["name"] or "") or pat in cmd:
                    totals[key] = totals.get(key, 0) + p.info["memory_info"].rss // (1024 * 1024)
                    break
        for key, val in totals.items():
            self.peak_rss_mib[key] = max(self.peak_rss_mib.get(key, 0), val)


def launch(argv: list[str], log_path: Path, env: dict | None = None) -> subprocess.Popen:
    f = open(log_path, "w", encoding="utf-8")
    return subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT, env=env)


def stop(proc: subprocess.Popen | None) -> None:
    if proc is not None:
        proc.kill()
        proc.wait()


def start_qdrant(out: Path, name: str) -> subprocess.Popen:
    q_env = dict(os.environ, QDRANT__STORAGE__STORAGE_PATH=str(out / "qdrant_storage"),
                 QDRANT__SERVICE__HOST="127.0.0.1", QDRANT__SERVICE__HTTP_PORT="6333",
                 QDRANT__TELEMETRY_DISABLED="true")
    return launch([ARGS.qdrant], out / f"qdrant_{name}.log", q_env)


def start_server(ctx: int, log_path: Path) -> tuple[subprocess.Popen, dict]:
    argv = s5.llama_server_argv(ARGS.server, ARGS.gguf, PORT, ctx, extra=SERVER_EXTRA)
    proc = launch(argv, log_path, dict(os.environ))
    up = wait_up(f"http://127.0.0.1:{PORT}/health", proc, 600)
    gpu = s5.llama_gpu_check(log_path.read_text(encoding="utf-8", errors="replace"))
    if not (up and gpu["ok"]):
        stop(proc)
        raise SystemExit(f"llama-server が起動しない・全層が GPU に載らない: up={up} {gpu}")
    # 数えない問 (step2_local と同じ。最初の問だけ読み込みの分が乗らないように)
    body = json.dumps({"model": "local-llm", "messages": [{"role": "user", "content": "こんにちは"}],
                       "temperature": 0.1, "max_tokens": 16}).encode()
    req = urllib.request.Request(f"{BASE_URL}/chat/completions", body, {"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=300).read()
    return proc, gpu


def eval_once(out_dir: Path, dataset: str, index: dict, extra_env: dict, meta: dict, log_path: Path) -> int:
    argv = s5.eval_cmd(sys.executable, out_dir, meta=meta, datasets=(dataset,))
    env = dict(os.environ, **s5.eval_env(index["env"], BASE_URL, NUMERIC_CHECK_ENABLED="true", **extra_env))
    with open(log_path, "w", encoding="utf-8") as f:
        return subprocess.run(argv, env=env, stdout=f, stderr=subprocess.STDOUT).returncode


def prepare(out: Path) -> None:
    """索引 2 つ (N は本番の文書・開発用は別の文書) と、必要な文脈の測定。全部の腕で同じ文脈を使う。"""
    out.mkdir(parents=True, exist_ok=True)
    qd = start_qdrant(out, "prepare")
    state: dict = {"indexes": {}}
    try:
        if not wait_up(f"{QDRANT_URL}/readyz", qd, 60):
            raise SystemExit("Qdrant が起動しない")
        for name, corpus in (("main", "evaluation/corpus"), ("dev", "evaluation/v2/corpus_nx_dev")):
            work = out / f"index_{name}"
            p = subprocess.run([sys.executable, "evaluation/colab/corpus_index.py", "--repo", ".", "--work", str(work),
                                "--qdrant-url", QDRANT_URL, "--collection", f"arms_{name}", "--corpus", corpus],
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            if p.returncode != 0:
                raise SystemExit(f"索引に失敗 ({name}): {p.stderr[-1500:]}")
            state["indexes"][name] = json.loads((work / "index.json").read_text(encoding="utf-8"))
            log(f"索引 {name}: {state['indexes'][name]['documents']} 文書 / {state['indexes'][name]['chunks']} 片")
        # 入力が一番長くなる形 (カード以外の腕を全部 ON) で 1 トークンだけ出して入力の長さを測る
        srv, gpu = start_server(MEASURE_CTX, out / "llama_measure.log")
        try:
            tops = []
            for name, dataset, idx in (("n", N12, "main"), ("dev", DEV24, "dev")):
                rc = eval_once(out / f"measure_{name}", dataset, state["indexes"][idx],
                               {"LLM_MAX_TOKENS": "1", **PROMPT_FLAGS}, {"purpose": "prompt_tokens"},
                               out / f"measure_{name}.log")
                if rc != 0:
                    raise SystemExit(f"入力の長さの測定に失敗 ({name}): rc={rc}")
                tops.append(s5.prompt_token_stats(out / f"measure_{name}" / "raw.jsonl"))
        finally:
            stop(srv)
    finally:
        stop(qd)
    top = max(tops, key=lambda t: t["max_prompt_tokens"])
    state.update(measure=tops, ctx=s5.needed_max_model_len(top["max_prompt_tokens"]), gpu_check=gpu)
    (out / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"入力の最大 {top['max_prompt_tokens']} ({top['max_id']}) → 文脈 {state['ctx']}")


def run_arm(out: Path, arm: str) -> int:
    """1 つの腕を流す。空きが足りなければ流さず理由を記録して 3 で終わる (測っていない値を残さない)。"""
    state = json.loads((out / "state.json").read_text(encoding="utf-8"))
    mem, gpu_log = MemLog(), GpuLog()
    baseline = read_mib()
    arm_dir = out / f"arm_{arm}"
    if arm_dir.exists() and any(arm_dir.iterdir()):
        raise SystemExit(f"{arm_dir} が空でない (前の結果と混ざるので止める)")
    arm_dir.mkdir(exist_ok=True)
    qd = start_qdrant(out, arm)
    rec: dict = {"arm": arm, "env": ARMS[arm], "ctx": state["ctx"], "baseline_mib": baseline, "rounds": {}}
    try:
        if not wait_up(f"{QDRANT_URL}/readyz", qd, 60):
            raise SystemExit("Qdrant が起動しない")
        for name, dataset, idx in ROUNDS:
            avail = mem.available_mib()
            if avail < ARGS.min_available_mib:
                rec["stopped"] = f"{name} の前の空き {avail} MiB < {ARGS.min_available_mib} MiB"
                log(rec["stopped"])
                return 3
            s5.wait_gpu_released(read_mib, baseline, tol_mib=ARGS.gpu_tol_mib)
            t0 = time.time()
            srv, _ = start_server(state["ctx"], arm_dir / f"llama_{name}.log")
            try:
                t1 = time.time()
                rc = eval_once(arm_dir / name, dataset, state["indexes"][idx], ARMS[arm],
                               {"arm": arm, "round": name, "gpu": ARGS.gpu_name, "ctx": state["ctx"]},
                               arm_dir / f"eval_{name}.log")
            finally:
                stop(srv)
            text = (arm_dir / f"llama_{name}.log").read_text(encoding="utf-8", errors="replace")
            scan = s5.scan_llama_log(text)
            rec["rounds"][name] = {"rc": rc, "available_before_mib": avail, "launch_seconds": round(t1 - t0, 1),
                                   "eval_seconds": round(time.time() - t1, 1), "oom_lines": len(scan["oom"]),
                                   "overflow_lines": len(scan["preempt"]),
                                   "peak_gpu_mib": s5.peak_in_window(gpu_log.samples, t0, time.time())["peak_mib"]}
            log(f"[{arm}/{name}] {rec['rounds'][name]}")
        return 0
    finally:
        stop(qd)
        rec.update(min_available_mib=mem.min_available_mib, peak_rss_mib=mem.peak_rss_mib)
        (arm_dir / "arm.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")


async def _direct_round(dataset: str, cards: list) -> list[QuestionResult]:
    from app.services import rule_cards

    extractor = rule_cards.OpenAICompatExtractor(BASE_URL, "local-llm")
    rows = [json.loads(line) for line in Path(dataset).read_text(encoding="utf-8").splitlines() if line.strip()]
    results = []
    for item in rows:
        r = QuestionResult(id=item["id"], provider="vllm", question=item["question"], category=item["category"],
                           user_access_level=item["user_access_level"], expected_conclusion=item["expected_conclusion"])
        card = rule_cards.select_card(cards, item["question"])
        if card is None:
            r.error = "no_card"
        else:
            t = time.perf_counter()
            r.answer, d = await rule_cards.answer_with_card(card, item["question"], extractor)
            r.total_latency = r.first_token_latency = time.perf_counter() - t
            first = d.pass_stats[0] if d.pass_stats else extractor.last_stats  # last_stats は直前の 1 回（2 回目）の数
            r.rule_card = {"card_id": d.card_id, "conclusion": d.conclusion, "hold_reason": d.hold_reason, "raw": d.raw,
                           "first_conclusion": d.first_conclusion, "second_conclusion": d.second_conclusion, "second_hold_reason": d.second_hold_reason, "second_raw": d.second_raw,
                           **first}
            tokens = [x.get("completion_tokens") for x in (d.pass_stats or [extractor.last_stats])]
            r.completion_tokens = sum(t for t in tokens if t) if any(tokens) else None  # 時間は 2 回分なので 2 回分を足す
            r.conclusion_ok = metrics.numeric_conclusion_ok(r.answer, item["expected_conclusion"])
        results.append(r)
    return results


def run_direct(out: Path) -> int:
    """カードの抽出だけを流す (検索なし)。N 12 問 × 2 回と開発用 24 問を、別々に起動して。"""
    from app.services import rule_cards

    out.mkdir(parents=True, exist_ok=True)
    arm_dir = out / f"arm_{DIRECT_ARM}"
    if arm_dir.exists() and any(arm_dir.iterdir()):
        raise SystemExit(f"{arm_dir} が空でない (前の結果と混ざるので止める)")
    arm_dir.mkdir(exist_ok=True)
    cards = rule_cards.load_cards(CARDS_DIR, allow_draft=True)
    mem, gpu_log = MemLog(), GpuLog()
    baseline = read_mib()
    rec: dict = {"arm": DIRECT_ARM, "env": "検索なし・カードの抽出だけ", "ctx": DIRECT_CTX, "baseline_mib": baseline,
                 "rounds": {}}
    try:
        for name, dataset, _ in ROUNDS:
            avail = mem.available_mib()
            if avail < ARGS.min_available_mib:
                rec["stopped"] = f"{name} の前の空き {avail} MiB < {ARGS.min_available_mib} MiB"
                log(rec["stopped"])
                return 3
            s5.wait_gpu_released(read_mib, baseline, tol_mib=ARGS.gpu_tol_mib)
            t0 = time.time()
            srv, _gpu = start_server(DIRECT_CTX, arm_dir / f"llama_{name}.log")
            try:
                t1 = time.time()
                results = asyncio.run(_direct_round(dataset, cards))
            finally:
                stop(srv)
            (arm_dir / name).mkdir()
            with open(arm_dir / name / "raw.jsonl", "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(sm.json_safe(asdict(r)), ensure_ascii=False) + "\n")
            scan = s5.scan_llama_log((arm_dir / f"llama_{name}.log").read_text(encoding="utf-8", errors="replace"))
            rec["rounds"][name] = {"rc": 0, "available_before_mib": avail, "launch_seconds": round(t1 - t0, 1),
                                   "eval_seconds": round(time.time() - t1, 1), "oom_lines": len(scan["oom"]),
                                   "overflow_lines": len(scan["preempt"]),
                                   "peak_gpu_mib": s5.peak_in_window(gpu_log.samples, t0, time.time())["peak_mib"]}
            log(f"[{DIRECT_ARM}/{name}] {rec['rounds'][name]}")
        return 0
    finally:
        rec.update(min_available_mib=mem.min_available_mib, peak_rss_mib=mem.peak_rss_mib)
        (arm_dir / "arm.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")


def round_numbers(results: list) -> dict:
    """1 回分の数。失敗した問は不一致のまま数え、時間は無限大にする (step2_local と同じ)。"""
    tally = conclusion_tally(results) or (0, 0, [])
    held = [r.id for r in results if metrics.final_conclusion(r.answer) == HOLD]
    failed = [r.id for r in results if answer_failed(r)]
    totals = [float("inf") if answer_failed(r) else r.total_latency for r in results]
    ftls = [float("inf") if answer_failed(r) else r.first_token_latency for r in results]
    gens = [r.total_latency - r.retrieval_latency for r in results if not answer_failed(r)]
    tps = [v for v in (sm.decode_tps(r) for r in results) if v is not None]
    return {
        "ok": tally[0], "n": tally[1], "wrong": tally[2], "held": held, "failed": failed,
        "wrong_decided": [i for i in tally[2] if i not in held and i not in failed],
        "total_p50": round(metrics.percentile(totals, 50), 3) if totals else None,
        "total_p95": round(metrics.percentile(totals, 95), 3) if totals else None,
        "total_max": round(max(totals), 3) if totals else None,
        "ftl_p50": round(metrics.percentile(ftls, 50), 3) if ftls else None,
        "after_retrieval_p50": round(statistics.median(gens), 3) if gens else None,
        "decode_tps_median": round(statistics.median(tps), 1) if tps else None,
    }


def _same(want: object, have: object) -> bool:
    if isinstance(want, (int, float)) and isinstance(have, (int, float)) and not isinstance(have, bool):
        return float(want) == float(have)
    return want == have


def slot_matches(gold: dict, raw: object) -> dict[str, bool]:
    """枠ごとに、モデルの値が正しい抽出と同じか。正しくは値が無い枠に値を入れた時も違いに数える。"""
    got = raw if isinstance(raw, dict) else {}
    out = {}
    for name, want in gold.items():
        have = got.get(name)
        out[name] = _same(want["value"], have.get("value") if isinstance(have, dict) else None)
    return out


def card_numbers(results: list) -> dict:
    """カードの回の 値の一致率。正しい抽出は backend/tests/test_rule_cards.py の GOLD (開発用の問の文から拾う) を使う。"""
    sys.path.insert(0, str(Path("backend/tests").resolve()))
    from app.services import rule_cards
    import test_rule_cards as t

    cards = {c.id: c for c in rule_cards.load_cards(CARDS_DIR, allow_draft=True)}
    slots_ok = slots_all = whole_ok = whole_all = 0
    wrong: list[dict] = []
    for r in results:
        if r.rule_card is None:
            continue
        card = cards[r.rule_card["card_id"]]
        m = slot_matches(t.gold_extraction(card, r.question), r.rule_card.get("raw"))
        slots_ok, slots_all = slots_ok + sum(m.values()), slots_all + len(m)
        whole_ok, whole_all = whole_ok + all(m.values()), whole_all + 1
        if not all(m.values()):
            wrong.append({"id": r.id, "slots": [k for k, v in m.items() if not v], "hold": r.rule_card["hold_reason"]})
    leaks = [r.id for r in results if r.rule_card and r.rule_card.get("reasoning_chars")]
    cut = [r.id for r in results if r.rule_card and r.rule_card.get("finish_reason") == "length"]
    return {"slots_ok": slots_ok, "slots_all": slots_all, "whole_ok": whole_ok, "whole_all": whole_all,
            "reasoning_leaked": leaks, "cut_by_length": cut,
            "not_card": [r.id for r in results if r.rule_card is None], "wrong": wrong}


def summary(out: Path) -> dict:
    # state.json は索引を作った時だけある (direct だけの回には無い)
    state_path = out / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    doc: dict = {"issue": 16, "ctx": state.get("ctx"), "measure": state.get("measure"), "server_extra": SERVER_EXTRA,
                 "arms": {}}
    for arm in (*ARMS, DIRECT_ARM):
        path = out / f"arm_{arm}" / "arm.json"
        if not path.exists():
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        for name, _, _ in ROUNDS:
            if name not in rec["rounds"]:
                continue
            results = sm.load_results(out / f"arm_{arm}" / name / "raw.jsonl")
            rec["rounds"][name]["numbers"] = round_numbers(results)
            if arm in ("cards", DIRECT_ARM):
                rec["rounds"][name]["card"] = card_numbers(results)
        doc["arms"][arm] = rec
    (out / "summary.json").write_text(json.dumps(sm.json_safe(doc), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return doc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("cmd", choices=("prepare", "run", "direct", "summary"))
    p.add_argument("--out", required=True)
    p.add_argument("--arm", choices=tuple(ARMS))
    p.add_argument("--server", help="llama-server の場所")
    p.add_argument("--gguf", help="重みの場所")
    p.add_argument("--qdrant", help="qdrant の場所")
    p.add_argument("--gpu-name", default="NVIDIA GeForce RTX 3060")
    p.add_argument("--min-available-mib", type=int, default=4096,
                   help="各回の前に要る本体のメモリの空き。足りなければ流さない")
    p.add_argument("--gpu-tol-mib", type=int, default=1024, help="各回の前に GPU がここまで戻るのを待つ幅。残ったサーバーは 4GB を超えるので 1GB でも見分けられる。他のアプリの増減で止まらないよう広げた")
    return p.parse_args(argv)


ARGS: argparse.Namespace


def main(argv: list[str] | None = None) -> int:
    global ARGS
    ARGS = parse_args(argv)
    out = Path(ARGS.out).resolve()
    if ARGS.cmd == "prepare":
        prepare(out)
        return 0
    if ARGS.cmd == "run":
        return run_arm(out, ARGS.arm)
    if ARGS.cmd == "direct":
        return run_direct(out)
    summary(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
