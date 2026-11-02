"""プロセスごとの常駐メモリとページ読みの回数を 1 秒ごとに記録する (Linux の /proc だけを読む)。

Issue #14 の E1 (評価プロセスの最大の常駐) と E2 (llama-server の RssAnon + RssFile・majflt) に使う。
名前 → コマンドラインに含まれる文字列 で対象を決める。同じ名前に複数のプロセスが当たれば足す。
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

STATUS_KEYS = ("VmRSS", "VmHWM", "RssAnon", "RssFile")


def parse_status(text: str) -> dict[str, int]:
    """/proc/<pid>/status の VmRSS などを kB の整数で取る。無い欄は入れない。"""
    out: dict[str, int] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key in STATUS_KEYS:
            out[key] = int(rest.split()[0])
    return out


def parse_majflt(stat: str) -> int:
    """/proc/<pid>/stat の 12 番目 (majflt)。コマンド名に空白や括弧があっても ')' の後ろから数える。"""
    return int(stat.rsplit(")", 1)[1].split()[9])


def find_pids(pattern: str, proc: Path = Path("/proc")) -> list[int]:
    pids = []
    for d in proc.iterdir():
        if not d.name.isdigit():
            continue
        try:
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if pattern in cmd:
            pids.append(int(d.name))
    return pids


def sample(patterns: dict[str, str], proc: Path = Path("/proc")) -> dict[str, dict[str, int]]:
    """名前ごとに当たったプロセスの値を足す。当たらない名前は入れない。"""
    out: dict[str, dict[str, int]] = {}
    for name, pattern in patterns.items():
        total: dict[str, int] = {}
        for pid in find_pids(pattern, proc):
            try:
                vals = parse_status((proc / str(pid) / "status").read_text())
                vals["majflt"] = parse_majflt((proc / str(pid) / "stat").read_text())
            except (OSError, ValueError, IndexError):
                continue
            for k, v in vals.items():
                total[k] = total.get(k, 0) + v
        if total:
            out[name] = total
    return out


def window_stats(samples: list[tuple[float, dict]], start: float, end: float) -> dict[str, dict]:
    """区間の中の最大値 (MiB) と majflt の増え方。RssAnon + RssFile の最大も出す。"""
    inside = [s for t, s in samples if start <= t <= end]
    out: dict[str, dict] = {}
    for name in sorted({n for s in inside for n in s}):
        rows = [s[name] for s in inside if name in s]
        mx = {k: round(max(r.get(k, 0) for r in rows) / 1024, 1) for k in STATUS_KEYS}
        mx["RssAnon_plus_File"] = round(max(r.get("RssAnon", 0) + r.get("RssFile", 0) for r in rows) / 1024, 1)
        flt = [r["majflt"] for r in rows if "majflt" in r]
        out[name] = {"max_mib": mx, "majflt_delta": (flt[-1] - flt[0]) if len(flt) > 1 else None,
                     "n_samples": len(rows)}
    return out


class ProcLog:
    def __init__(self, patterns: dict[str, str], interval: float = 1.0) -> None:
        self.patterns = patterns
        self.samples: list[tuple[float, dict]] = []
        self.interval = interval
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while True:
            try:
                self.samples.append((time.time(), sample(self.patterns)))
            except Exception:
                pass
            time.sleep(self.interval)

    def stats(self, start: float, end: float) -> dict[str, dict]:
        return window_stats(self.samples, start, end)
