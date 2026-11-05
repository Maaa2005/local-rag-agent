"""検索だけを繰り返して、時間切れ・送り直し・検索の時間を測る（Issue #14 E4）。GPU は使わない。

Qdrant と埋め込み（e5-large・CPU）だけを動かし、回答のモデルは起動しない。v1 と同じ retrieve()
（権限の絞り込み・SQLite の確かめ直しを含む）を問の一覧から順に n 回呼ぶ。本体のメモリを絞る時は
--pressure-cmd で絞る道具を別のプロセスで起動する（Issue #14 E3 の evaluation/gpu_local/mem_pressure.py）。

準備は README.md の「準備」と同じ（.work/qdrant/bin/qdrant・.work/rag-env）。流し方（repo のルートで）:
    .work/rag-env/bin/python -m evaluation.gpu_local.search_bench --work .work --out .work/out-e4-q1-s1 \\
        --n 300 --qdrant-timeout-s 10 --qdrant-retries 1 --pressure-target-mib 1024 \\
        --pressure-cmd ".work/rag-env/bin/python -m evaluation.gpu_local.mem_pressure --target-mib 1024 \\
                        --probe windows --log {log}"
記録には問の id・時間・失敗の型だけを書く（問の文・検索した本文・手元のパスは書かない）。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

from evaluation import metrics

ROOT = Path(__file__).resolve().parents[2]
QUESTION_FILES = ("evaluation/dataset.jsonl", "evaluation/v2/numeric_boundary.jsonl")
QDRANT_URL = "http://127.0.0.1:6333"
# 無認証で起動するので待受を手元に限る。Qdrant 1.12.6 の service.host を環境変数で上書きする
# （config/config.yaml の既定は 0.0.0.0。REST も gRPC もこの値で待ち受け、環境変数は設定ファイルより強い）
QDRANT_HOST = "127.0.0.1"
# Issue #14 E4 の線（測る前に固定・docs/research/2026-11-02/low-memory.md §4 の 3）
E4_N = 300
E4_P95 = 1.0
E4_P99 = 3.0
# 絞る道具の記録の抜けの上限（秒）。mem_pressure.py は既定 1 秒ごとに書く。測定の始まり・終わりとの間も数える
PRESSURE_MAX_GAP_S = 10.0


def load_questions(repo: Path) -> list[dict]:
    """(id, question, user_access_level) だけを読む。"""
    out = []
    for name in QUESTION_FILES:
        for line in (repo / name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out.append({"id": r["id"], "question": r["question"], "level": int(r["user_access_level"])})
    return out


async def run_queries(questions: list[dict], n: int, retrieve, is_timeout) -> list[dict]:
    """questions を順に n 回 retrieve し、1 回ごとの {i, id, seconds, ok, error, timeout} を返す。"""
    rows = []
    for i in range(n):
        q = questions[i % len(questions)]
        t = time.perf_counter()
        try:
            await retrieve(q["question"], q["level"])
            ok, err, to = True, None, False
        except Exception as e:  # noqa: BLE001  失敗は数えて続ける
            ok, err, to = False, type(e).__name__, bool(is_timeout(e))
        rows.append({"i": i, "id": q["id"], "seconds": round(time.perf_counter() - t, 4), "ok": ok,
                     "error": err, "timeout": to})
    return rows


def summarize(rows: list[dict], retry_count: int | None) -> dict:
    """失敗した回の時間は無限大として数える（線を甘くしない）。最初の 1 回（読み込み込み）を除いた値も出す。"""
    times = [r["seconds"] if r["ok"] else float("inf") for r in rows]
    warm = times[1:]

    def pct(xs, p):
        return round(metrics.percentile(xs, p), 4) if xs else float("inf")

    return {
        "n": len(rows),
        "failed": sum(1 for r in rows if not r["ok"]),
        "timeouts": sum(1 for r in rows if r["timeout"]),
        "errors": sorted({r["error"] for r in rows if r["error"]}),
        "retries": retry_count,
        "p50": pct(times, 50), "p95": pct(times, 95), "p99": pct(times, 99),
        "max": round(max(times), 4) if times else float("inf"),
        "first": times[0] if times else None,
        "warm_p95": pct(warm, 95), "warm_p99": pct(warm, 99),
    }


def pressure_window(rows: list[dict], start: float, end: float, target_mib: int, band_mib: int,
                    max_gap_s: float = PRESSURE_MAX_GAP_S, tool_alive: bool = True) -> dict:
    """絞る道具の記録（1 行 = {"t", "free_mib", "action", ...}）から、区間の空きと有効かどうかを出す。

    有効は、空きの中央値が帯の中・全部放し 0 回に加えて、記録が区間の始めから終わりまで max_gap_s より長く
    抜けず、区間の終わりに道具が動いていた回だけ（途中で落ちた回は、前半の中央値が帯の中でも有効にしない）。
    """
    inside = [r for r in rows if start <= r.get("t", -1) <= end and r.get("free_mib") is not None]
    if not inside:
        return {"n": 0, "valid": False}
    free = [r["free_mib"] for r in inside]
    med = statistics.median(free)
    floors = sum(1 for r in inside if r.get("action") == "floor")
    edges = [start, *sorted(r["t"] for r in inside), end]
    gap = max(b - a for a, b in zip(edges, edges[1:]))
    return {"n": len(inside), "target_mib": target_mib, "band_mib": band_mib, "free_min": min(free),
            "free_median": med, "free_max": max(free), "floor_releases": floors,
            "max_gap_s": round(gap, 3), "tool_alive": tool_alive,
            "valid": abs(med - target_mib) <= band_mib and floors == 0 and gap <= max_gap_s and tool_alive}


def e4_line_checks(summary: dict, pressure: dict | None) -> dict:
    """E4 の 1 回分を線と比べる。絞る条件が守れなかった回は合否を出さない（無効）。回数は固定の E4_N と比べる。"""
    valid = pressure is not None and pressure.get("valid") is True
    lines = {
        "L1_no_failure": summary["n"] == E4_N and summary["failed"] == 0,
        "L2_p95": summary["p95"] <= E4_P95,
        "L3_p99": summary["p99"] <= E4_P99,
    }
    lines["all"] = all(lines.values())
    return {"valid": valid, "lines": lines, "verdict": "無効" if not valid else ("通る" if lines["all"] else "落ちる")}


def _read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


# ─── 子のプロセス（app を import するので、索引と同じ環境変数を入れてから起動する） ───

def child_main(a: argparse.Namespace) -> int:
    # backend の app を import する（run_eval.py と同じく repo の backend をパスに足す）
    sys.path.insert(0, str(ROOT / "backend"))
    from app.database import close_db, init_db
    from app.services import qdrant as qmod
    from app.services.retriever import retrieve

    async def go():
        await init_db()
        try:
            return await run_queries(load_questions(Path(a.repo)), a.n, retrieve, qmod.is_timeout)
        finally:
            # aiosqlite 0.20.0 の接続は非 daemon のスレッドなので、閉じないとプロセスが終わらず親が待ち続ける
            try:
                await qmod.close_client()
            finally:
                await close_db()

    rows = asyncio.run(go())
    with open(a.rows, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.write(json.dumps({"retry_count": qmod.retry_count}) + "\n")
    return 0


# ─── 親（Qdrant・索引・絞る道具・子を順に動かす） ───

def parent_main(a: argparse.Namespace) -> int:
    from evaluation.gpu_local.step2_local import kill, launch, wait_up

    repo = Path.cwd()
    out = Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} が空でない (前の結果と混ざるので止める)")
    out.mkdir(parents=True, exist_ok=True)
    work = Path(a.work).resolve()
    py = str(work / "rag-env" / "bin" / "python")
    q_env = dict(os.environ, QDRANT__STORAGE__STORAGE_PATH=a.qdrant_storage or str(out / "qdrant_storage"),
                 QDRANT__SERVICE__HOST=QDRANT_HOST, QDRANT__SERVICE__HTTP_PORT="6333",
                 QDRANT__TELEMETRY_DISABLED="true")
    qdrant = launch([str(work / "qdrant" / "bin" / "qdrant")], out / "qdrant.log", q_env)
    pressure = None
    pressure_alive = False
    summary: dict = {"issue": 14, "trial": "E4", "n": a.n, "qdrant_timeout_s": a.qdrant_timeout_s,
                     "qdrant_retries": a.qdrant_retries, "label": a.label}
    try:
        if not wait_up(f"{QDRANT_URL}/readyz", qdrant, 60):
            raise SystemExit("Qdrant が起動しない")
        p = subprocess.run([py, "evaluation/colab/corpus_index.py", "--repo", str(repo), "--work",
                            str(out / "index_work"), "--qdrant-url", QDRANT_URL, "--collection",
                            f"search_{int(time.time())}"], cwd=repo, capture_output=True, text=True)
        if p.returncode != 0:
            raise SystemExit(f"索引に失敗: {p.stderr[-2000:]}")
        index = json.loads((out / "index_work" / "index.json").read_text())
        plog = out / "pressure.jsonl"
        if a.pressure_cmd:
            pressure = launch(shlex.split(a.pressure_cmd.replace("{log}", str(plog))), out / "pressure.log")
            t0, steady = time.time(), 0
            while time.time() - t0 < a.pressure_settle_s and pressure.poll() is None and steady < 5:
                rows = _read_jsonl(plog)
                ok = bool(rows) and abs(rows[-1]["free_mib"] - a.pressure_target_mib) <= a.pressure_valid_band_mib
                steady = steady + 1 if ok else 0
                time.sleep(1)
        env = dict(os.environ, **index["env"], RERANK_ENABLED="false",
                   QDRANT_TIMEOUT_S=str(a.qdrant_timeout_s), QDRANT_SEARCH_RETRIES=str(a.qdrant_retries))
        t_start = time.time()
        rc = subprocess.run([py, "-m", "evaluation.gpu_local.search_bench", "--child", "--repo", str(repo),
                             "--n", str(a.n), "--rows", str(out / "rows.jsonl")], cwd=repo, env=env,
                            stdout=open(out / "child.log", "w"), stderr=subprocess.STDOUT).returncode
        t_end = time.time()
        pressure_alive = pressure is not None and pressure.poll() is None  # 止める前に見る
    finally:
        if pressure is not None:
            pressure.terminate()
            try:
                pressure.wait(timeout=15)
            except subprocess.TimeoutExpired:
                kill(pressure)
        kill(qdrant)
    rows = _read_jsonl(out / "rows.jsonl")
    tail = rows[-1] if rows and "retry_count" in rows[-1] else {}
    qrows = [r for r in rows if "i" in r]
    summary.update(rc=rc, numbers=summarize(qrows, tail.get("retry_count")))
    if a.pressure_cmd:
        summary["pressure"] = pressure_window(_read_jsonl(plog), t_start, t_end, a.pressure_target_mib,
                                              a.pressure_valid_band_mib, tool_alive=pressure_alive)
        summary["pressure"]["before_free_mib"] = (_read_jsonl(plog) or [{}])[0].get("free_mib")
    summary["lines"] = e4_line_checks(summary["numbers"], summary.get("pressure"))
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps({"numbers": summary["numbers"], "lines": summary["lines"]}, ensure_ascii=False, default=str))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--repo", default=".", help=argparse.SUPPRESS)
    p.add_argument("--rows", help=argparse.SUPPRESS)
    p.add_argument("--work", help="準備物の置き場 (.work)")
    p.add_argument("--out", help="結果の置き場 (空のディレクトリ)")
    p.add_argument("--n", type=int, default=E4_N, help="検索の回数 (E4 は 300。ほかの回数は判定で落ちる)")
    p.add_argument("--label", default="")
    p.add_argument("--qdrant-timeout-s", type=int, default=0, help="QDRANT_TIMEOUT_S (0 は既定の 5 秒)")
    p.add_argument("--qdrant-retries", type=int, default=0, help="QDRANT_SEARCH_RETRIES")
    p.add_argument("--qdrant-storage", help="Qdrant の保存先 (既定は --out の下)")
    p.add_argument("--pressure-cmd", default="", help="絞る道具の起動の命令。{log} を記録の置き場に置き換える")
    p.add_argument("--pressure-target-mib", type=int, default=1024)
    p.add_argument("--pressure-valid-band-mib", type=int, default=256)
    p.add_argument("--pressure-settle-s", type=float, default=180.0)
    a = p.parse_args(argv)
    if not a.child and not (a.work and a.out):
        p.error("--work と --out が要る")
    return a


if __name__ == "__main__":
    args = parse_args()
    sys.exit(child_main(args) if args.child else parent_main(args))
