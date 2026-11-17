"""判定の記録 → 問ごとの点 → 設定ごとの集計。

V_i        = (vital の事実の点の合計) / (vital の数)   full=1, partial=0.5, none=0
V_i^strict = full だけ 1
C_i        = 相反の文が 1 つ以上あれば 1
s_i        = K 回の V_i の平均（K 回は 1 設定の複数ファイル）
判定できず(undetermined)の回は平均にも数にも入れない（0 点にしない）。
"""
from __future__ import annotations

from typing import Mapping, Sequence

from evaluation.metrics import correctness_keyword_score
from evaluation.nugget.judge import LABELS
from evaluation.nugget.schema import vital_nuggets
from evaluation.nugget import stats

FACT_POINTS = {"full": 1.0, "partial": 0.5, "none": 0.0}


def vital_keywords(row: Mapping) -> list[str]:
    """vital の事実の keywords を重複なしでまとめた一覧（M6 と並べる用）。"""
    seen: list[str] = []
    for n in vital_nuggets(row):
        for k in n.get("keywords") or []:
            if k not in seen:
                seen.append(k)
    return seen


def score_record(rec: Mapping, row: Mapping) -> dict:
    """判定の記録 1 件 + その問の nuggets 行 → 点。"""
    vitals = vital_nuggets(row)
    out = {
        "qid": rec["qid"],
        "run": rec.get("run", ""),
        "determined": False,
        "V": None,
        "V_strict": None,
        "C": None,
        "m6": None,
        "label_counts": {k: 0 for k in LABELS},
        "points": {},
        "contradictions": [],
        "head": (rec.get("answer") or "")[:80],
    }
    if rec.get("status") != "ok" or not vitals:
        return out
    facts = rec["facts"]
    pts = {n["nid"]: FACT_POINTS[facts[n["nid"]]] for n in vitals}
    out["points"] = pts
    out["V"] = sum(pts.values()) / len(vitals)
    out["V_strict"] = sum(1.0 for v in pts.values() if v == 1.0) / len(vitals)
    labels = rec.get("labels") or []
    for lab in labels:
        out["label_counts"][lab] += 1
    out["contradictions"] = [s for s, lab in zip(rec.get("sentences", []), labels) if lab == "contradiction"]
    out["C"] = 1.0 if out["contradictions"] else 0.0
    kws = vital_keywords(row)
    out["m6"] = correctness_keyword_score(rec.get("answer", ""), kws) if kws else None
    out["determined"] = True
    return out


def aggregate_setting(runs: Sequence[Sequence[Mapping]], nuggets_by_qid: Mapping[str, Mapping]) -> dict:
    """1 設定 = K 個の判定ファイル（runs）。{qid: 問の集計} と除外の一覧を返す。

    s は判定できた回だけの平均。1 回も判定できなかった問は items に入れず excluded に積む。
    nuggets に無い qid の記録は unknown に積んで無視する。
    """
    per_q: dict[str, list[dict]] = {}
    unknown: set[str] = set()
    for run in runs:
        for rec in run:
            row = nuggets_by_qid.get(rec["qid"])
            if row is None:
                unknown.add(rec["qid"])
                continue
            per_q.setdefault(rec["qid"], []).append(score_record(rec, row))

    items: dict[str, dict] = {}
    excluded: list[str] = []
    for qid, scored in per_q.items():
        ok = [s for s in scored if s["determined"]]
        if not ok:
            excluded.append(qid)
            continue
        row = nuggets_by_qid[qid]
        lost: dict[str, float] = {}
        for n in vital_nuggets(row):
            mean_pt = stats.mean([s["points"][n["nid"]] for s in ok])
            if mean_pt < 1.0:
                lost[n["nid"]] = mean_pt
        contras: list[str] = []
        for s in ok:
            for c in s["contradictions"]:
                if c not in contras:
                    contras.append(c)
        m6s = [s["m6"] for s in ok if s["m6"] is not None]
        counts = {k: stats.mean([s["label_counts"][k] for s in ok]) for k in LABELS}
        items[qid] = {
            "qid": qid,
            "rule_id": row.get("rule_id"),
            "s": stats.mean([s["V"] for s in ok]),
            "s_strict": stats.mean([s["V_strict"] for s in ok]),
            "c": stats.mean([s["C"] for s in ok]),
            "m6": stats.mean(m6s) if m6s else None,
            "k_total": len(scored),
            "k_used": len(ok),
            "lost": lost,
            "contradictions": contras,
            "label_counts": counts,
            "head": ok[0]["head"],
        }
    missing = sorted(set(nuggets_by_qid) - set(per_q))
    return {
        "items": items,
        "excluded": sorted(excluded),
        "missing": missing,
        "unknown": sorted(unknown),
        "n_runs": len(runs),
    }


def summarize(setting: Mapping, use_cluster: bool = True) -> dict:
    """設定の要約: n・平均 V・SE・V_strict・相反の割合・M6。"""
    items = [setting["items"][q] for q in sorted(setting["items"])]
    s = [i["s"] for i in items]
    groups = [i["rule_id"] for i in items] if use_cluster else None
    m6 = [i["m6"] for i in items if i["m6"] is not None]
    return {
        "n": len(items),
        "n_excluded": len(setting["excluded"]),
        "n_missing": len(setting["missing"]),
        "n_runs": setting["n_runs"],
        "mean_V": stats.mean(s) if s else None,
        "se_clt": stats.se_clt(s),
        "se_cluster": stats.se_clustered(s, groups) if groups else None,
        "mean_V_strict": stats.mean([i["s_strict"] for i in items]) if items else None,
        "contradiction_rate": stats.mean([i["c"] for i in items]) if items else None,
        "m6": stats.mean(m6) if m6 else None,
        "n_m6": len(m6),
        "label_counts": {
            k: (stats.mean([i["label_counts"][k] for i in items]) if items else None) for k in LABELS
        },
    }


def scores_of(setting: Mapping) -> dict[str, float]:
    return {q: i["s"] for q, i in setting["items"].items()}


def clusters_of(nuggets_by_qid: Mapping[str, Mapping]) -> dict[str, str]:
    return {q: r.get("rule_id") for q, r in nuggets_by_qid.items()}
