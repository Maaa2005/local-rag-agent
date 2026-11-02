"""開発用の問の正誤と、試作の選び方（docs/verified-distill-plan.md §5 の 3・4・6）。すべてコードで決める。

- 正誤: 断りの問は断ったか（metrics.is_refusal）、資料の文を答える問は断らずに正解の語句をすべて含むか、
  ほかは最後の「結論: X」が正解と完全一致か。答えが無い・失敗は不正解。重みなし
- 開発用 300 問を種で「選ぶ用 150」「確かめる用 150」に分ける。生成器は割合の丸めで 301 問を返すので、
  生成器の並び（種で混ぜた後）の先頭 300 問を使う（守りの問の除外に使う開発用の集合は変えない）
- 選ぶ: 守りの問の条件（4 種 10/10・完了の時間の中央が対照の 1.5 倍以下）を満たす物のうち、選ぶ用の正答率が
  一番高い物。同点なら学習率の低い方。満たす物が無ければ None（止める条件 5）
"""
from __future__ import annotations

import random
import statistics

from evaluation.distill.gen import make
from evaluation.metrics import final_conclusion, is_refusal

DEV_SEED = 20261102   # evaluation/distill/guard.py と同じ
DEV_N = 300
SPLIT_SEED = 150
TIME_RATIO_MAX = 1.5
GUARD_KINDS = ("refusal", "describe", "two", "three")


def correct(row: dict, answer: str | None) -> bool:
    if not answer:
        return False
    cat = row["category"]
    if cat == "refusal":
        return is_refusal(answer)
    if cat == "describe":
        return not is_refusal(answer) and all(k in answer for k in row["keywords"])
    return final_conclusion(answer) == row["expected"]


def dev_rows() -> list[dict]:
    return make.generate("dev", DEV_N, DEV_SEED)[:DEV_N]


def split_dev(rows: list[dict], seed: int = SPLIT_SEED) -> tuple[list[dict], list[dict]]:
    """(選ぶ用, 確かめる用)。元の並びを変えずに、種で半分ずつに分ける。"""
    ids = [r["id"] for r in rows]
    random.Random(seed).shuffle(ids)
    pick = set(ids[: len(ids) // 2])
    return [r for r in rows if r["id"] in pick], [r for r in rows if r["id"] not in pick]


def accuracy(rows: list[dict], answers: dict[str, str | None], category: str | None = None) -> float:
    use = [r for r in rows if category is None or r["category"] == category]
    return sum(correct(r, answers.get(r["id"])) for r in use) / len(use)


def guard_ok(guard_rows: list[dict], passed: dict[str, bool], seconds: dict[str, float], control_seconds: dict[str, float]) -> dict:
    """守りの問の条件。passed は guard.judge の結果、seconds は完了までの秒（問の id ごと）。"""
    by = {k: [r["id"] for r in guard_rows if r["guard"] == k] for k in GUARD_KINDS}
    kinds = {k: all(passed.get(i, False) for i in ids) for k, ids in by.items()}
    t = statistics.median(seconds[r["id"]] for r in guard_rows)
    tc = statistics.median(control_seconds[r["id"]] for r in guard_rows)
    time_ok = t <= TIME_RATIO_MAX * tc
    return {"kinds": kinds, "median": t, "control_median": tc, "time_ok": time_ok, "ok": all(kinds.values()) and time_ok}


def choose(cands: list[dict]) -> dict | None:
    """cands: {"lr": 学習率, "guard_ok": bool, "select_acc": 選ぶ用の正答率}。"""
    ok = [c for c in cands if c["guard_ok"]]
    if not ok:
        return None
    return min(ok, key=lambda c: (-c["select_acc"], c["lr"]))
