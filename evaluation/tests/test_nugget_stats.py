"""evaluation.nugget.stats のテスト。式の値は手計算と照合する。

d = [0.5, 0, -0.5, 1]
  d̄ = 0.25, 偏差 = [0.25, -0.25, -0.75, 0.75], 偏差の二乗和 = 1.25
  Var(ddof=1) = 1.25/3, SE_CLT² = (1.25/3)/4 = 20/192
まとまり {1,2},{3,4}: Σ_{j≠i} = (Σa)²-Σa² → (0-0.125)+(0-1.125) = -1.25, /16 = -15/192 → SE² = 5/192
まとまり {1,3},{2,4}: (0.25-0.625)*2 = -0.75, /16 = -9/192 → SE² = 11/192
まとまり {1,4},{2,3}: (1-0.625)*2 = +0.75, /16 = +9/192 → SE² = 29/192
"""
from __future__ import annotations

import math

import pytest

from evaluation.nugget import stats

D = [0.5, 0.0, -0.5, 1.0]


def approx(x):
    return pytest.approx(x, rel=1e-12, abs=1e-12)


def test_mean_var_se_clt_hand_values():
    assert stats.mean(D) == approx(0.25)
    assert stats.var_ddof1(D) == approx(1.25 / 3)
    assert stats.se_clt(D) == approx(math.sqrt(20 / 192))
    assert stats.se_clt(D) == approx(0.32274861218395140)


@pytest.mark.parametrize(
    "groups, var192",
    [
        (["a", "a", "b", "b"], 5),
        (["a", "b", "a", "b"], 11),
        (["a", "b", "b", "a"], 29),
    ],
)
def test_se_clustered_hand_values(groups, var192):
    assert stats.se_clustered(D, groups) == approx(math.sqrt(var192 / 192))


def test_se_clustered_with_one_item_per_group_equals_clt():
    assert stats.se_clustered(D, ["a", "b", "c", "d"]) == approx(stats.se_clt(D))


def test_se_clustered_single_group_closed_form():
    # 全員が 1 つのまとまり: 偏差の和は 0 なので SE² = Σa²/(n²(n-1)) = 1.25/(16*3) = 5/192
    assert stats.se_clustered(D, ["g"] * 4) == approx(math.sqrt(1.25 / (16 * 3)))
    assert stats.se_clustered([1.0, -1.0], ["g", "g"]) == approx(math.sqrt(2 / (4 * 1)))


def test_se_clustered_length_mismatch_and_no_groups():
    assert stats.se_clustered(D, None) is None
    with pytest.raises(ValueError):
        stats.se_clustered(D, ["a"])


def test_ci95():
    lo, hi = stats.ci95(0.25, math.sqrt(20 / 192))
    assert lo == approx(0.25 - 1.96 * math.sqrt(20 / 192))
    assert hi == approx(0.25 + 1.96 * math.sqrt(20 / 192))
    assert stats.ci95(0.25, None) is None


# ─── 小さい n・同じ値 ───────────────────────────────────────────
def test_small_n_and_constant_values():
    assert stats.se_clt([]) is None and stats.se_clt([0.3]) is None
    assert stats.var_ddof1([0.3]) is None
    assert stats.se_clustered([0.3], ["a"]) is None
    assert stats.se_clt([0.25, 0.25, 0.25, 0.25]) == 0.0
    assert stats.se_clustered([0.25] * 4, ["a", "a", "b", "b"]) == 0.0


def test_compare_n0_n1():
    c0 = stats.compare({}, {}, {})
    assert c0["n"] == 0 and c0["dbar"] is None and c0["ci"] is None
    assert stats.verdict(c0, 0.1) == stats.VERDICT_TOO_FEW
    c1 = stats.compare({"q": 0.5}, {"q": 0.9}, {"q": "r"})
    assert c1["n"] == 1 and c1["dbar"] == approx(0.4) and c1["se"] is None and c1["ci"] is None
    assert stats.verdict(c1, 0.1) == stats.VERDICT_TOO_FEW


# ─── 比較 ───────────────────────────────────────────────────────
A = {"q1": 0.5, "q2": 0.5, "q3": 0.5, "q4": 0.5, "only_a": 1.0}
B = {"q1": 1.0, "q2": 0.5, "q3": 0.0, "q4": 1.5, "only_b": 0.0}
CL = {"q1": "x", "q2": "x", "q3": "y", "q4": "y", "only_a": "z", "only_b": "z"}


def test_paired_diff_uses_common_qids_in_order():
    qids, d = stats.paired_diff(A, B)
    assert qids == ["q1", "q2", "q3", "q4"]
    assert d == approx_list(D)


def approx_list(xs):
    return [pytest.approx(x) for x in xs]


def test_compare_with_clusters_main_is_cluster():
    c = stats.compare(A, B, CL)
    assert c["n"] == 4 and c["main"] == "cluster"
    assert c["dbar"] == approx(0.25)
    assert c["se_clt"] == approx(math.sqrt(20 / 192))
    assert c["se_cluster"] == approx(math.sqrt(5 / 192))
    assert c["se"] == c["se_cluster"]
    assert c["ci"][0] == approx(0.25 - 1.96 * math.sqrt(5 / 192))
    assert c["ci_clt"][1] == approx(0.25 + 1.96 * math.sqrt(20 / 192))


def test_compare_without_clusters_uses_clt():
    c = stats.compare(A, B, None)
    assert c["main"] == "clt" and c["se_cluster"] is None and c["se"] == c["se_clt"]


# ─── A/A と信じる差 ─────────────────────────────────────────────
def test_aa_width_is_larger_abs_endpoint():
    assert stats.aa_width({"ci": (-0.08, 0.02)}) == approx(0.08)
    assert stats.aa_width({"ci": (-0.01, 0.05)}) == approx(0.05)
    assert stats.aa_width({"ci": None}) is None


def _cmp(dbar, se):
    """verdict 用の最小の比較結果。"""
    return {"n": 10, "dbar": dbar, "ci": stats.ci95(dbar, se)}


def test_verdict_believed_needs_both_conditions():
    # 区間 [0.2-0.0196*..]: se=0.05 → 区間 [0.102, 0.298]（0 を含まない）
    c = _cmp(0.2, 0.05)
    assert stats.verdict(c, 0.1) == stats.VERDICT_BELIEVE      # |d̄|=0.2 > w=0.1
    assert stats.verdict(c, 0.2) == stats.VERDICT_NOISE        # |d̄| > w でない（等しい）
    assert stats.verdict(c, 0.3) == stats.VERDICT_NOISE        # 揺れの中
    # 区間が 0 を含むなら w が小さくても信じない
    c0 = _cmp(0.2, 0.15)                                       # 区間 [-0.094, 0.494]
    assert stats.verdict(c0, 0.0) == stats.VERDICT_NOISE
    # 悪化側でも対称
    assert stats.verdict(_cmp(-0.2, 0.05), 0.1) == stats.VERDICT_BELIEVE
    assert stats.is_believed(_cmp(-0.2, 0.05), 0.1)
    assert not stats.is_believed(c0, 0.0)


def test_verdict_without_aa_is_withheld():
    assert stats.verdict(_cmp(0.2, 0.05), None) == stats.VERDICT_NO_AA


def test_verdict_constant_difference_has_zero_width_interval():
    a = {f"q{i}": 0.5 for i in range(4)}
    b = {f"q{i}": 0.75 for i in range(4)}
    c = stats.compare(a, b, {q: q for q in a})
    assert c["se"] == 0.0 and c["ci"] == (0.25, 0.25)
    assert stats.verdict(c, 0.1) == stats.VERDICT_BELIEVE
    assert stats.verdict(c, 0.25) == stats.VERDICT_NOISE


def test_end_to_end_aa_then_ab():
    base = {f"q{i}": v for i, v in enumerate([0.5, 0.6, 0.4, 0.55, 0.45, 0.5, 0.6, 0.4])}
    cl = {q: f"r{i // 2}" for i, q in enumerate(base)}
    # A/A: 同じ設定を seed だけ替えて流した → 小さな揺れ
    wobble = [0.02, -0.03, 0.01, 0.0, -0.02, 0.03, -0.01, 0.0]
    aa = stats.compare(base, {q: base[q] + w for q, w in zip(base, wobble)}, cl)
    w = stats.aa_width(aa)
    assert w == approx(max(abs(aa["ci"][0]), abs(aa["ci"][1])))
    # B: 全体に +0.2 してばらつきを足したもの → 信じる差
    shifts = [0.2, 0.25, 0.15, 0.2, 0.22, 0.18, 0.2, 0.2]
    ab = stats.compare(base, {q: base[q] + s for q, s in zip(base, shifts)}, cl)
    assert stats.verdict(ab, w) == stats.VERDICT_BELIEVE
    # B': ほぼ揺れと同じ大きさの差 → 揺れの中
    ab2 = stats.compare(base, {q: base[q] + x for q, x in zip(base, [0.02, -0.02, 0.03, 0.0, 0.01, 0.0, 0.02, -0.01])}, cl)
    assert stats.verdict(ab2, w) == stats.VERDICT_NOISE
