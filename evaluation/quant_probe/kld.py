"""llama-perplexity の出力（--kl-divergence の末尾の統計）を読む。

出力の並びは tools/perplexity/perplexity.cpp（b11836）の末尾:
Perplexity statistics / KL divergence statistics / Token probability statistics。
"""
from __future__ import annotations

import re

_NUM = r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
_ROW = re.compile(r"^\s*(?P<key>[^:]+?)\s*:\s*" + _NUM + r"\s*%?\s*(?:±\s*" + _NUM + r")?")
_CHUNKS = re.compile(r"over\s+(\d+)\s+chunks")
_PROGRESS = re.compile(r"\[(\d+)\]-?\d+(?:\.\d+)?")  # 途中経過「[n]9.8765」の形だけ
_KLD_ROW = re.compile(r"^\s*(\d+)\s+-?\d+(?:\.\d+)?\s+±", re.M)
_DONE = re.compile(r"Final estimate|KL divergence statistics")
_FINAL = re.compile(r"Final estimate:\s*PPL\s*=\s*" + _NUM + r"\s*\+/-\s*" + _NUM)

KEYS = {
    "Mean PPL(Q)": "ppl_q",
    "Mean PPL(base)": "ppl_base",
    "Mean ln(PPL(Q)/PPL(base))": "ln_ppl_ratio",
    "Mean PPL(Q)/PPL(base)": "ppl_ratio",
    "Mean KLD": "kld_mean",
    "Maximum KLD": "kld_max",
    "99.9% KLD": "kld_p999",
    "99.0% KLD": "kld_p99",
    "95.0% KLD": "kld_p95",
    "Median KLD": "kld_median",
    "RMS Δp": "rms_dp",
    "Same top p": "same_top_p",
}


def parse_log(text: str) -> dict[str, float]:
    """chunks = 予定の区切りの数（「over N chunks」）。runs = その行の数（1 つの記録に 1 回の実行のはず）。
    chunks_done = 済んだ区切りの数（途中経過「[n]数」か、KLD の表の行の番号の最大）。
    complete = 実行が 1 回だけで、終わりの印（Final estimate か KL divergence statistics）が最後の途中経過より後にある。"""
    out: dict[str, float] = {}
    starts = list(_CHUNKS.finditer(text))
    out["runs"] = len(starts)
    if starts:
        out["chunks"] = int(starts[0].group(1))
    marks = [(m.end(), int(m.group(1))) for m in _PROGRESS.finditer(text)]
    marks += [(m.end(), int(m.group(1))) for m in _KLD_ROW.finditer(text)]
    out["chunks_done"] = max((n for _, n in marks), default=0)
    last = max((pos for pos, _ in marks), default=-1)
    done_at = [m.start() for m in _DONE.finditer(text)]
    out["complete"] = len(starts) == 1 and bool(done_at) and max(done_at) > last
    for line in text.splitlines():
        if _KLD_ROW.match(line) or _CHUNKS.search(line):
            continue
        m = _FINAL.search(line)
        if m:
            out["ppl"], out["ppl_err"] = float(m.group(1)), float(m.group(2))
            continue
        m = _ROW.match(line)
        if not m:
            continue
        key = " ".join(m.group("key").split())
        name = KEYS.get(key)
        if name:
            out[name] = float(m.group(2))
            if m.group(3) is not None:
                out[name + "_err"] = float(m.group(3))
    return out
