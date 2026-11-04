"""本体のメモリを絞った状態を作って保つ（Issue #14 E3）。標準ライブラリだけを使う。

学校 PC（WSL2）で「Windows の空き約 1GB」の時の遅れと失敗を測るための道具。WSL の中で匿名メモリを
持ち、空きの読み取り（Windows か Linux）が目標に近づくように、持つ量を 1 秒ごとに増減する。
WSL の中で持ったメモリは WSL の VM（vmmem）の分として Windows の空きを減らす。

- 持つメモリは 1 ページずつ書いて実際に確保し、毎回もう一度書く（触らないと先に追い出されて圧迫にならない）
- 安全のため、空きが floor を下回ったら（Windows か Linux のどちらかでも）持っている分を全部放す
- 空きの記録・持った量・PSI（/proc/pressure/memory）を 1 秒ごとに JSONL へ書く。本文や手元のパスは書かない

単体で使う時（repo のルートで・止める時は Ctrl+C）:
    python -m evaluation.gpu_local.mem_pressure --target-mib 1024 --probe windows --log .work/pressure.jsonl
step2_local.py の --pressure-target-mib からも同じ物を別のプロセスで起動する。
"""
from __future__ import annotations

import argparse
import json
import mmap
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

MIB = 1024 * 1024
PAGE = mmap.PAGESIZE
# Windows の空き（キロバイト）を 1 秒ごとに 1 行ずつ出す。Get-Counter は日本語版で数え方の名前が変わるので使わない。
# Win32_OperatingSystem.FreePhysicalMemory は「使われておらず使える物理メモリ（KB）」
PS_LOOP = ("while ($true) { (Get-CimInstance -ClassName Win32_OperatingSystem).FreePhysicalMemory; "
           "Start-Sleep -Milliseconds {ms} }")


def parse_meminfo_available(text: str) -> int:
    """/proc/meminfo の MemAvailable を MiB で返す。"""
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    raise ValueError("MemAvailable が無い")


def linux_available_mib(path: Path = Path("/proc/meminfo")) -> int:
    return parse_meminfo_available(path.read_text())


def parse_psi(text: str) -> dict[str, dict[str, float]]:
    """/proc/pressure/memory の 2 行（some / full）を {"some": {"avg10":..,"total":..}, ...} にする。total は μs。"""
    out: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        kind, *pairs = line.split()
        if kind not in ("some", "full"):
            continue
        vals = dict(p.split("=", 1) for p in pairs)
        out[kind] = {k: (int(v) if k == "total" else float(v)) for k, v in vals.items()}
    return out


def read_psi(path: Path = Path("/proc/pressure/memory")) -> dict[str, dict[str, float]] | None:
    try:
        return parse_psi(path.read_text())
    except OSError:
        return None


class WindowsFreeProbe:
    """WSL から powershell.exe を 1 本だけ起動し、出てくる Windows の空き（KB）の最新値を MiB で返す。

    毎回 powershell.exe を起動すると、それ自体が数十 MB を使って測る物を乱すので、1 本を流し続ける。
    """

    def __init__(self, interval_ms: int = 1000, exe: str = "powershell.exe",
                 popen: Callable[..., subprocess.Popen] = subprocess.Popen) -> None:
        cmd = [exe, "-NoProfile", "-NonInteractive", "-Command", PS_LOOP.replace("{ms}", str(interval_ms))]
        self._proc = popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self._latest: int | None = None
        self._ready = threading.Event()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in self._proc.stdout:
            line = line.strip()
            if line.isdigit():
                self._latest = int(line) // 1024
                self._ready.set()

    def wait_ready(self, timeout: float = 30.0) -> bool:
        return self._ready.wait(timeout)

    def __call__(self) -> int:
        if self._latest is None:
            raise RuntimeError("Windows の空きがまだ読めない")
        return self._latest

    def close(self) -> None:
        self._proc.terminate()


class Balloon:
    """空きの読み取り probe() が target_mib に近づくように匿名メモリを持つ。

    持つ量は chunk_mib 単位。1 回の adjust で増やすのは step_mib まで（急に空きを潰さない）。
    guard() は安全用の別の空き（WSL の中の MemAvailable など）で、こちらが floor を下回っても全部放す。
    """

    def __init__(self, probe: Callable[[], int], target_mib: int, *, band_mib: int = 128,
                 chunk_mib: int = 32, step_mib: int = 256, floor_mib: int = 256,
                 max_hold_mib: int | None = None, guard: Callable[[], int] | None = None,
                 alloc: Callable[[int], object] | None = None) -> None:
        if target_mib <= floor_mib:
            raise ValueError("target_mib は floor_mib より大きくする")
        if band_mib < chunk_mib:
            raise ValueError("band_mib は chunk_mib 以上にする（小さいと増やす・減らすを繰り返す）")
        self.probe, self.guard = probe, guard
        self.target, self.band, self.chunk, self.step = target_mib, band_mib, chunk_mib, step_mib
        self.floor, self.max_hold = floor_mib, max_hold_mib
        self._alloc = alloc or _touched_chunk
        self.chunks: list[object] = []

    @property
    def held_mib(self) -> int:
        return len(self.chunks) * self.chunk

    def release_all(self) -> None:
        for c in self.chunks:
            close = getattr(c, "close", None)
            if close:
                close()
        self.chunks.clear()

    def touch(self) -> None:
        for c in self.chunks:
            _touch(c)

    def adjust(self) -> dict:
        """1 回分の調整。{"free_mib", "guard_mib", "held_mib", "action"} を返す。"""
        free = self.probe()
        guard = self.guard() if self.guard else None
        action = "keep"
        if free < self.floor or (guard is not None and guard < self.floor):
            self.release_all()
            action = "floor"
        elif free > self.target + self.band:
            n = min(free - self.target, self.step) // self.chunk
            if self.max_hold is not None:
                n = min(n, max(0, (self.max_hold - self.held_mib) // self.chunk))
            # guard 側の空きも floor を割らない範囲までにする
            if guard is not None:
                n = min(n, max(0, (guard - self.floor) // self.chunk))
            for _ in range(int(n)):
                self.chunks.append(self._alloc(self.chunk * MIB))
            action = f"grow {n}" if n else "keep"
        elif free < self.target - self.band:
            n = min(-(-(self.target - free) // self.chunk), len(self.chunks))
            for _ in range(n):
                c = self.chunks.pop()
                close = getattr(c, "close", None)
                if close:
                    close()
            action = f"shrink {n}" if n else "keep"
        self.touch()
        return {"free_mib": free, "guard_mib": guard, "held_mib": self.held_mib, "action": action}


def _touched_chunk(nbytes: int) -> mmap.mmap:
    m = mmap.mmap(-1, nbytes)
    _touch(m)
    return m


def _touch(m: object) -> None:
    """1 ページに 1 バイトずつ書く。bytearray / mmap どちらでも動く。"""
    n = len(m)  # type: ignore[arg-type]
    m[0:n:PAGE] = b"\x01" * len(range(0, n, PAGE))  # type: ignore[index]


def window_summary(rows: list[dict], start: float, end: float, target_mib: int, band_mib: int) -> dict:
    """記録の行（t・free_mib・held_mib・psi）から区間の空き・持った量・PSI の増え方をまとめる。"""
    inside = [r for r in rows if start <= r["t"] <= end and r.get("free_mib") is not None]
    if not inside:
        return {"n": 0, "valid": False}
    free = [r["free_mib"] for r in inside]
    in_band = sum(1 for f in free if abs(f - target_mib) <= band_mib)
    med = statistics.median(free)
    out = {
        "n": len(inside),
        "target_mib": target_mib,
        "band_mib": band_mib,
        "free_min": min(free),
        "free_median": med,
        "free_max": max(free),
        "in_band_share": round(in_band / len(inside), 3),
        "held_max_mib": max(r.get("held_mib", 0) for r in inside),
        "floor_releases": sum(1 for r in inside if r.get("action") == "floor"),
        # 有効な回: 区間の空きの中央値が目標 ± 幅に入っている（E3 で測る前に固定した条件）
        "valid": abs(med - target_mib) <= band_mib,
    }
    psi = [r["psi"] for r in inside if r.get("psi")]
    if len(psi) > 1:
        out["psi"] = {k: {"total_delta_us": psi[-1][k]["total"] - psi[0][k]["total"],
                          "avg10_max": max(p[k]["avg10"] for p in psi)}
                      for k in ("some", "full") if k in psi[0] and k in psi[-1]}
    return out


def load_log(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # 止めた瞬間の書きかけの行
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--target-mib", type=int, required=True, help="保つ空き（MiB）。E3 の S1 は 1024")
    p.add_argument("--probe", choices=["windows", "linux"], default="windows",
                   help="どの空きを目標に合わせるか（WSL では windows）")
    p.add_argument("--band-mib", type=int, default=128)
    p.add_argument("--floor-mib", type=int, default=256, help="これを下回ったら全部放す（Windows・Linux の両方を見る）")
    p.add_argument("--max-hold-mib", type=int, default=None)
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--duration", type=float, default=None, help="秒。無ければ止められるまで")
    p.add_argument("--log", required=True, help="JSONL の記録先")
    a = p.parse_args(argv)

    win = None
    if a.probe == "windows":
        win = WindowsFreeProbe(int(a.interval * 1000))
        if not win.wait_ready():
            print("Windows の空きが読めない（powershell.exe が無い・WSL でない）", file=sys.stderr)
            win.close()
            return 2
        probe, guard = win, linux_available_mib
    else:
        probe, guard = linux_available_mib, None
    balloon = Balloon(probe, a.target_mib, band_mib=a.band_mib, floor_mib=a.floor_mib,
                      max_hold_mib=a.max_hold_mib, guard=guard)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    t_end = None if a.duration is None else time.time() + a.duration
    with open(a.log, "a", encoding="utf-8") as f:
        while not stop.is_set() and (t_end is None or time.time() < t_end):
            row = {"t": time.time(), "probe": a.probe, **balloon.adjust(), "psi": read_psi()}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            stop.wait(a.interval)
    balloon.release_all()
    if win:
        win.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
