"""27B IQ3_XXS 評価の止める条件 (evaluation/colab/speed_gate.py) のテスト。

確かめること:
- 中央値は reporter と同じく、失敗・未実行の問を無限大として数える
- 途中の打ち切りは、線を超えた問が過半数に達した時だけで、それより前には決めない
- 拒否・数値は 1 問でも外れ（失敗・未実行も含む）たら不合格
- 研究用ノート (research_27b_iq3.ipynb) の全セルが Python として読め、固定した設定が入っている
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from evaluation.colab import speed_gate as sg

IDS = [f"Q{i:03d}" for i in range(1, 51)]


def _row(i: str, ftl: float = 1.0, total: float = 2.0, **kw) -> dict:
    return {"id": i, "first_token_latency": ftl, "total_latency": total, **kw}


# ─── 速さ ──────────────────────────────────────────────────────
def test_speed_passes_when_median_within_limits():
    rows = [_row(i, 4.0, 14.0) for i in IDS]
    res = sg.speed_result(rows, IDS)
    assert res["ok"] and res["p1_ftl_p50"] == 4.0 and res["p3_total_p50"] == 14.0 and res["answered"] == 50


def test_speed_fails_on_p1_even_if_total_ok():
    rows = [_row(i, 5.5, 14.0) for i in IDS]
    res = sg.speed_result(rows, IDS)
    assert res["p3_ok"] and not res["p1_ok"] and not res["ok"]


def test_failed_and_missing_questions_count_as_infinite():
    # 24 問速い・残り 26 問は失敗か未実行 → 中央値は無限大
    rows = [_row(i, 1.0, 2.0) for i in IDS[:24]] + [_row(i, error="run_failed: x") for i in IDS[24:30]]
    res = sg.speed_result(rows, IDS)
    assert res["p3_total_p50"] == float("inf") and not res["ok"] and res["answered"] == 24


def test_judge_failed_is_not_answer_failure():
    rows = [_row(i, 1.0, 2.0, error="judge_failed: x") for i in IDS]
    assert sg.speed_result(rows, IDS)["ok"]


def test_boundary_exactly_at_limit_passes():
    rows = [_row(i, 5.0, 15.0) for i in IDS]
    assert sg.speed_result(rows, IDS)["ok"]


# ─── 途中の打ち切り ────────────────────────────────────────────
def test_not_decided_with_25_slow_questions():
    # 25 問が遅くても、残り 25 問が速ければ中央値は (遅い最小 + 速い最大)/2 になりうるので決めない
    rows = [_row(i, 1.0, 30.0) for i in IDS[:25]]
    assert sg.speed_decided_fail(rows, IDS) is None


def test_decided_with_26_slow_questions():
    rows = [_row(i, 1.0, 30.0) for i in IDS[:26]]
    reason = sg.speed_decided_fail(rows, IDS)
    assert reason and "26/50" in reason
    # 残りが全部 0 秒でも中央値は線を超える（打ち切りが線を緩めも締めもしない）
    full = rows + [_row(i, 0.0, 0.0) for i in IDS[26:]]
    assert not sg.speed_result(full, IDS)["p3_ok"]


def test_25_slow_and_25_fast_can_still_pass():
    rows = [_row(i, 1.0, 15.5) for i in IDS[:25]] + [_row(i, 1.0, 14.0) for i in IDS[25:]]
    assert sg.speed_decided_fail(rows, IDS) is None
    assert sg.speed_result(rows, IDS)["p3_ok"]   # (14.0 + 15.5) / 2 = 14.75


def test_decided_by_first_token():
    rows = [_row(i, 6.0, 8.0) for i in IDS[:26]]
    assert "最初の文字" in sg.speed_decided_fail(rows, IDS)


def test_failures_count_toward_decision():
    rows = [_row(i, error="run_failed: timeout") for i in IDS[:26]]
    assert sg.speed_decided_fail(rows, IDS)


# ─── 拒否・数値 ────────────────────────────────────────────────
DATASET = [{"id": "Q001", "category": "general"}, {"id": "Q040", "category": "permission_test"},
           {"id": "Q045", "category": "hallucination_test"}]


def test_refusal_all_correct():
    rows = [_row("Q040", refusal_correct=True), _row("Q045", refusal_correct=True), _row("Q001", refusal_correct=False)]
    res = sg.refusal_result(rows, DATASET)
    assert res == {"n": 2, "correct": 2, "not_ok_ids": [], "ok": True}


def test_refusal_missing_or_failed_is_not_ok():
    rows = [_row("Q040", refusal_correct=True, error="run_failed: x")]
    res = sg.refusal_result(rows, DATASET)
    assert not res["ok"] and res["not_ok_ids"] == ["Q040", "Q045"]


def test_numeric_result_and_early_decision():
    ids = ["N01", "N02", "N03"]
    rows = [_row("N01", conclusion_ok=True), _row("N02", conclusion_ok=False)]
    assert sg.numeric_decided_fail(rows, ids) == "N02 の結論が一致しない"
    res = sg.numeric_result(rows, ids)
    assert not res["ok"] and res["not_ok_ids"] == ["N02", "N03"]
    assert sg.numeric_decided_fail([_row("N01", conclusion_ok=True)], ids) is None
    assert sg.numeric_result([_row(i, conclusion_ok=True) for i in ids], ids)["ok"]


def test_decode_rate_median_skips_unusable_rows():
    rows = [_row("A", 1.0, 3.0, completion_tokens=21), _row("B", 1.0, 1.0, completion_tokens=5),
            _row("C", error="run_failed")]
    assert sg.decode_rate_median(rows) == 10.0
    assert sg.decode_rate_median([]) is None


# ─── ノート ────────────────────────────────────────────────────
NB = Path(__file__).resolve().parents[1] / "colab" / "research_27b_iq3.ipynb"


def _code_cells() -> list[str]:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _without_shell(src: str) -> str:
    # ! で始まるシェル行（\ で続く行も）は Python ではないので pass に置き換えて読む
    out, in_shell = [], False
    for ln in src.splitlines():
        if in_shell or ln.lstrip().startswith("!"):
            if not in_shell:
                out.append(ln[: len(ln) - len(ln.lstrip())] + "pass")
            in_shell = ln.rstrip().endswith("\\")
        else:
            out.append(ln)
    return "\n".join(out)


def test_notebook_cells_parse_as_python():
    for src in _code_cells():
        ast.parse(_without_shell(src))


def test_notebook_fixes_the_agreed_setting():
    src = "\n".join(_code_cells())
    assert 'GGUF_FILE = "Qwen3.8-27B-UD-IQ3_XXS.gguf"' in src
    assert re.search(r'GGUF_REV = "[0-9a-f]{40}"', src)
    assert 'BUDGET = "12GB"' in src
    assert 'NUMERIC_CHECK = "false"' in src
    # 救済の設定（KV の量子化・文脈の縮小・CPU への退避）を足していない
    assert "EXTRA_ARGS = []" in src
    for bad in ("--cache-type", "-ctk", "-ctv", "--n-cpu-moe", "--override-tensor"):
        assert bad not in src


def test_notebook_uses_only_existing_helpers():
    src = "\n".join(_code_cells())
    for name in set(re.findall(r"\bsg\.(\w+)", src)):
        assert hasattr(sg, name), name
    from evaluation.colab import step05
    for name in set(re.findall(r"\bs5\.(\w+)", src)):
        assert hasattr(step05, name), name
