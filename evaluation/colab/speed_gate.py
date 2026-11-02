"""27B IQ3_XXS の 1 設定の評価（life #24・2026-11-02 の協議）で使う、止める条件の判定。

ノート (research_27b_iq3.ipynb) の kernel から import するので、標準ライブラリと evaluation.metrics だけで書く。
線は docs/v2-light-model-plan.md §1 をそのまま使い、研究用に緩めない。

- 速さ（§1 P・12GB の線）: 50 問の完了の中央値 ≤ 15.0 秒、最初の文字の中央値 ≤ 5.0 秒。
  回答に失敗した問・まだ流れていない問は時間を無限大として数える（reporter と同じ）。
- 途中で決まる不合格: 50 問のうち線を超えた問が過半数（26 問）に達したら、残りが全部速くても
  中央値は線を超える。そこで打ち切ってよい（線は変えずに安く落とす）。
- 拒否（§1 Q の M8）: 拒否すべき 16 問が全部拒否できたか。1 問でも外れたら不合格。
- 数値（§1 N）: 12 問の結論行が全部一致したか。1 問でも外れたら不合格。
"""
from __future__ import annotations

from evaluation import metrics

P1_LIMIT = 5.0    # 最初の文字の中央値（秒）
P3_LIMIT = 15.0   # 完了の中央値（秒・12GB は T4 の測定値のまま）
INF = float("inf")
REFUSAL_CATEGORIES = ("permission_test", "hallucination_test")


def answer_failed(row: dict) -> bool:
    """回答を作れなかった問か。採点役の失敗 (judge_failed) は回答の失敗に数えない（reporter と同じ）。"""
    err = row.get("error")
    return bool(err) and not str(err).startswith("judge_failed")


def _seconds(row: dict | None, key: str) -> float:
    if row is None or answer_failed(row):
        return INF
    v = row.get(key)
    return float(v) if isinstance(v, (int, float)) else INF


def speed_result(rows: list[dict], ids: list[str]) -> dict:
    """ids の全問で P1・P3 の中央値を出す。流れていない問・失敗した問は無限大。"""
    if not ids:
        raise ValueError("問が無い")
    by_id = {r.get("id"): r for r in rows}
    ftl = [_seconds(by_id.get(i), "first_token_latency") for i in ids]
    total = [_seconds(by_id.get(i), "total_latency") for i in ids]
    p1 = metrics.percentile(ftl, 50)
    p3 = metrics.percentile(total, 50)
    finite = [t for t in total if t != INF]
    return {
        "n": len(ids),
        "answered": len(finite),
        "p1_ftl_p50": p1,
        "p3_total_p50": p3,
        "p1_ok": p1 <= P1_LIMIT,
        "p3_ok": p3 <= P3_LIMIT,
        "ok": p1 <= P1_LIMIT and p3 <= P3_LIMIT,
        "over_p3": sum(t > P3_LIMIT for t in total),
        "total_max_answered": max(finite) if finite else None,
    }


def speed_decided_fail(rows: list[dict], ids: list[str]) -> str | None:
    """途中でも中央値が線を超えると決まったら理由を返す。まだ分からなければ None。

    n 問の中央値（線形補間）は、線を超えた問が n // 2 + 1 問あれば、残りに関係なく線を超える。
    """
    need = len(ids) // 2 + 1
    by_id = {r.get("id"): r for r in rows}
    done = [by_id[i] for i in ids if i in by_id]
    over_p3 = sum(_seconds(r, "total_latency") > P3_LIMIT for r in done)
    over_p1 = sum(_seconds(r, "first_token_latency") > P1_LIMIT for r in done)
    if over_p3 >= need:
        return f"完了が {P3_LIMIT} 秒を超えた問が {over_p3}/{len(ids)}（{need} 問で中央値の超過が決まる）"
    if over_p1 >= need:
        return f"最初の文字が {P1_LIMIT} 秒を超えた問が {over_p1}/{len(ids)}（{need} 問で中央値の超過が決まる）"
    return None


def refusal_result(rows: list[dict], dataset_rows: list[dict]) -> dict:
    """拒否すべき問（M8 の対象）が全部拒否できたか。流れていない問・失敗した問は拒否できなかった側。"""
    targets = [d["id"] for d in dataset_rows if d.get("category") in REFUSAL_CATEGORIES]
    by_id = {r.get("id"): r for r in rows}
    ok = [i for i in targets if i in by_id and not answer_failed(by_id[i]) and by_id[i].get("refusal_correct") is True]
    return {"n": len(targets), "correct": len(ok), "not_ok_ids": [i for i in targets if i not in ok],
            "ok": bool(targets) and len(ok) == len(targets)}


def numeric_result(rows: list[dict], ids: list[str]) -> dict:
    """数値の境目の全問で結論行が一致したか。流れていない問・失敗した問は不一致。"""
    by_id = {r.get("id"): r for r in rows}
    ok = [i for i in ids if i in by_id and not answer_failed(by_id[i]) and by_id[i].get("conclusion_ok") is True]
    return {"n": len(ids), "correct": len(ok), "not_ok_ids": [i for i in ids if i not in ok],
            "ok": bool(ids) and len(ok) == len(ids)}


def numeric_decided_fail(rows: list[dict], ids: list[str]) -> str | None:
    """1 問でも結論が外れたら（失敗も含む）その時点で不合格が決まる。"""
    for r in rows:
        if r.get("id") in ids and (answer_failed(r) or r.get("conclusion_ok") is not True):
            return f"{r.get('id')} の結論が一致しない"
    return None


def decode_rate_median(rows: list[dict]) -> float | None:
    """出力の速さ（トークン/秒）の中央値。最初の文字〜完了の間に出したトークン数で割る（記録用）。"""
    rates = []
    for r in rows:
        if answer_failed(r):
            continue
        n, ftl, tot = r.get("completion_tokens"), r.get("first_token_latency"), r.get("total_latency")
        if isinstance(n, int) and n > 1 and isinstance(ftl, (int, float)) and isinstance(tot, (int, float)) and tot > ftl:
            rates.append((n - 1) / (tot - ftl))
    return round(metrics.percentile(rates, 50), 2) if rates else None
