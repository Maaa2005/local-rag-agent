"""Markdown の表を作る: 1 設定の要約・比べた表・悪化した問の一覧。"""
from __future__ import annotations

from typing import Mapping, Sequence

from evaluation.nugget import stats
from evaluation.nugget.judge import LABELS


def fmt(x, nd: int = 3) -> str:
    if x is None:
        return "-"
    return f"{x:.{nd}f}"


def fmt_signed(x, nd: int = 3) -> str:
    if x is None:
        return "-"
    return f"{x:+.{nd}f}"


def fmt_ci(ci, nd: int = 3) -> str:
    if ci is None:
        return "-"
    return f"[{ci[0]:+.{nd}f}, {ci[1]:+.{nd}f}]"


def cell(s: str) -> str:
    """表のセル用: 改行と | をつぶす。"""
    return str(s).replace("\r", " ").replace("\n", " ").replace("|", "\\|")


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(cell(c) for c in r) + " |")
    return "\n".join(lines)


def summary_table(named_summaries: Sequence[tuple[str, Mapping]]) -> str:
    """設定ごとに 1 行: n・平均 V・SE（まとまり/CLT）・V_strict・相反の割合・M6。"""
    header = ["設定", "回数K", "n", "除外", "平均V", "SE(規程まとまり)", "SE(CLT)", "V_strict", "相反の割合", "M6(キーワード)"]
    rows = []
    for name, s in named_summaries:
        excl = s["n_excluded"] + s["n_missing"]
        rows.append([
            name, str(s["n_runs"]), str(s["n"]), str(excl),
            fmt(s["mean_V"]), fmt(s["se_cluster"]), fmt(s["se_clt"]),
            fmt(s["mean_V_strict"]), fmt(s["contradiction_rate"]),
            f"{fmt(s['m6'])} (n={s['n_m6']})" if s["m6"] is not None else "-",
        ])
    out = _table(header, rows)
    labels = ["設定"] + [f"{jp}" for jp in LABELS.values()]
    lrows = [
        [name] + [fmt(s["label_counts"][k], 2) for k in LABELS] for name, s in named_summaries
    ]
    out += "\n\n1 答えあたりの文の数（ラベル別の平均）\n\n" + _table(labels, lrows)
    return out


def compare_table(cmp: Mapping, aa: Mapping | None, name_a: str = "A", name_b: str = "B") -> str:
    w = stats.aa_width(aa) if aa is not None else None
    v = stats.verdict(cmp, w)
    main = "規程まとまり" if cmp["main"] == "cluster" else "CLT"
    header = [
        "比較", "n", f"{name_b}-{name_a}", "SE(規程まとまり)", "SE(CLT)",
        f"95%区間(主={main})", "A/Aの揺れ幅w", "判定",
    ]
    row = [
        f"{name_b} - {name_a}", str(cmp["n"]), fmt_signed(cmp["dbar"]),
        fmt(cmp["se_cluster"]), fmt(cmp["se_clt"]), fmt_ci(cmp["ci"]),
        fmt(w) if w is not None else "-", v,
    ]
    out = _table(header, [row])
    if aa is not None:
        out += (
            f"\n\nA/A: n={aa['n']}, 差={fmt_signed(aa['dbar'])}, 区間={fmt_ci(aa['ci'])}"
            "（同じ設定を seed だけ替えて流した 2 組の差）"
        )
    else:
        out += "\n\nA/A: 未実施（「信じる差」は判定しない）"
    return out


def worst_table(
    setting_a: Mapping,
    setting_b: Mapping,
    nuggets_by_qid: Mapping[str, Mapping],
    top: int = 10,
) -> str:
    """d_i の小さい順（悪化した順）に上から top 問。"""
    qids, d = stats.paired_diff(
        {q: i["s"] for q, i in setting_a["items"].items()},
        {q: i["s"] for q, i in setting_b["items"].items()},
    )
    order = sorted(zip(qids, d), key=lambda t: (t[1], t[0]))[:top]
    header = ["問", "規程", "Aの点", "Bの点", "差", "Bで落ちたvitalの事実", "Bの相反の文", "Bの答えの頭80字"]
    rows = []
    for q, diff in order:
        a = setting_a["items"][q]
        b = setting_b["items"][q]
        texts = {n["nid"]: n["text"] for n in nuggets_by_qid[q].get("nuggets", [])}
        lost = "; ".join(
            f"{nid} {texts.get(nid, '')}" + ("（一部）" if 0 < pt < 1 else "")
            for nid, pt in b["lost"].items()
        )
        rows.append([
            q, str(b.get("rule_id") or ""), fmt(a["s"]), fmt(b["s"]), fmt_signed(diff),
            lost or "-", " / ".join(b["contradictions"]) or "-", b["head"],
        ])
    if not rows:
        return "（比べられる問が無い）"
    return _table(header, rows)


def full_report(
    *,
    name_a: str,
    sum_a: Mapping,
    name_b: str | None = None,
    sum_b: Mapping | None = None,
    cmp: Mapping | None = None,
    aa: Mapping | None = None,
    worst: str | None = None,
    meta: Mapping | None = None,
) -> str:
    parts = ["# 必須の事実の採点レポート"]
    if meta:
        parts.append("\n".join(f"- {k}: {v}" for k, v in meta.items()))
    named = [(name_a, sum_a)]
    if sum_b is not None:
        named.append((name_b or "B", sum_b))
    parts.append("## 設定ごとの要約\n\n" + summary_table(named))
    if cmp is not None:
        parts.append("## 比較\n\n" + compare_table(cmp, aa, name_a, name_b or "B"))
    if worst is not None:
        parts.append("## 悪化した問（差の小さい順）\n\n" + worst)
    return "\n\n".join(parts) + "\n"
