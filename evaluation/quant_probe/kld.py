"""llama-perplexity の出力（--kl-divergence の末尾の統計）を読む。

出力の並びは tools/perplexity/perplexity.cpp（b11836）の末尾:
Perplexity statistics / KL divergence statistics / Token probability statistics。
"""
from __future__ import annotations

import re

_NUM = r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
_ROW = re.compile(r"^\s*(?P<key>[^:]+?)\s*:\s*" + _NUM + r"\s*%?\s*(?:±\s*" + _NUM + r")?")
_CHUNKS = re.compile(r"over\s+(\d+)\s+chunks")
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
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = _CHUNKS.search(line)
        if m and "chunks" not in out:
            out["chunks"] = int(m.group(1))
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
