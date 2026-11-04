"""mem_pressure（Issue #14 E3 の本体メモリを絞る道具）の試験。GPU も Windows も要らない。"""
from __future__ import annotations

import io
import json
import mmap
import subprocess
import sys

import pytest

from evaluation.gpu_local import mem_pressure as mp

MEMINFO = """MemTotal:       16477184 kB
MemFree:          512000 kB
MemAvailable:    1048576 kB
Buffers:            1024 kB
"""

PSI = """some avg10=1.50 avg60=0.20 avg300=0.05 total=123456
full avg10=0.40 avg60=0.10 avg300=0.00 total=6543
"""


def test_parse_meminfo_available_is_mib():
    assert mp.parse_meminfo_available(MEMINFO) == 1024
    with pytest.raises(ValueError):
        mp.parse_meminfo_available("MemTotal: 1 kB\n")


def test_parse_psi():
    got = mp.parse_psi(PSI)
    assert got["some"] == {"avg10": 1.5, "avg60": 0.2, "avg300": 0.05, "total": 123456}
    assert got["full"]["total"] == 6543 and isinstance(got["full"]["total"], int)


class FakeProbe:
    """持った量だけ空きが減る、を真似る。"""

    def __init__(self, free0: int, balloon_ref: list):
        self.free0, self.ref = free0, balloon_ref

    def __call__(self) -> int:
        return self.free0 - (self.ref[0].held_mib if self.ref else 0)


def small_alloc(nbytes: int) -> bytearray:
    # 試験では本当に何百 MB も取らない。大きさの数え方だけを見る
    return bytearray(mp.PAGE * 2)


def make(free0: int, **kw) -> mp.Balloon:
    ref: list = []
    b = mp.Balloon(FakeProbe(free0, ref), kw.pop("target", 1024), alloc=small_alloc, **kw)
    ref.append(b)
    return b


def test_grows_toward_target_in_steps_and_then_keeps():
    b = make(3000, step_mib=256, chunk_mib=32, band_mib=128)
    actions = [b.adjust()["action"] for _ in range(12)]
    assert actions[0] == "grow 8"  # 256 MiB ずつ
    assert abs((3000 - b.held_mib) - 1024) <= 128
    assert actions[-1] == "keep"


def test_shrinks_when_free_drops_below_band():
    ref: list = []
    free = {"v": 3000}

    def probe():
        return free["v"] - ref[0].held_mib

    b = mp.Balloon(probe, 1024, alloc=small_alloc, step_mib=4096)
    ref.append(b)
    b.adjust()
    held = b.held_mib
    free["v"] -= 600  # ほかのプロセスが 600 MiB 使い始めた
    row = b.adjust()
    assert row["action"].startswith("shrink")
    assert b.held_mib < held
    assert abs(probe() - 1024) <= 128


def test_floor_releases_everything():
    ref: list = []
    free = {"v": 3000}
    b = mp.Balloon(lambda: free["v"] - ref[0].held_mib, 1024, alloc=small_alloc, floor_mib=256)
    ref.append(b)
    b.adjust()
    assert b.held_mib > 0
    free["v"] = 200 + b.held_mib  # 見かけの空きが 200 MiB
    assert b.adjust()["action"] == "floor"
    assert b.held_mib == 0


def test_guard_floor_also_releases_and_limits_growth():
    ref: list = []
    guard = {"v": 400}
    b = mp.Balloon(lambda: 5000 - (ref[0].held_mib if ref else 0), 1024, alloc=small_alloc,
                   floor_mib=256, guard=lambda: guard["v"] - ref[0].held_mib)
    ref.append(b)
    b.adjust()
    # guard 側の空き 400 から floor 256 までの 144 MiB → 32 MiB 単位で 4 個まで
    assert b.held_mib == 128
    guard["v"] = 300
    assert b.adjust()["action"] == "floor"
    assert b.held_mib == 0


def test_max_hold_caps_growth():
    b = make(9000, max_hold_mib=320, step_mib=4096)
    for _ in range(5):
        b.adjust()
    assert b.held_mib == 320


def test_rejects_bad_settings():
    with pytest.raises(ValueError):
        mp.Balloon(lambda: 0, 200, floor_mib=256)
    with pytest.raises(ValueError):
        mp.Balloon(lambda: 0, 1024, band_mib=16, chunk_mib=32)


def test_touched_chunk_writes_every_page():
    m = mp._touched_chunk(mp.PAGE * 4)
    try:
        assert isinstance(m, mmap.mmap)
        assert [m[i * mp.PAGE] for i in range(4)] == [1, 1, 1, 1]
    finally:
        m.close()


def test_window_summary_validity_and_psi():
    rows = [{"t": float(t), "free_mib": f, "held_mib": 900, "action": "keep",
             "psi": {"some": {"avg10": 0.5 * t, "total": 1000 * t}, "full": {"avg10": 0.1, "total": 10 * t}}}
            for t, f in enumerate([1000, 1100, 1050, 900, 990])]
    s = mp.window_summary(rows, 0, 4, 1024, 128)
    assert s["valid"] is True and s["free_median"] == 1000 and s["free_min"] == 900
    assert s["in_band_share"] == 1.0
    assert s["psi"]["some"] == {"total_delta_us": 4000, "avg10_max": 2.0}
    # 区間の外の行は数えない・中央値が幅の外なら無効
    s2 = mp.window_summary(rows + [{"t": 9.0, "free_mib": 5000}], 0, 4, 2048, 128)
    assert s2["n"] == 5 and s2["valid"] is False
    assert mp.window_summary([], 0, 1, 1024, 128) == {"n": 0, "valid": False}


def test_load_log_skips_partial_line(tmp_path):
    p = tmp_path / "p.jsonl"
    p.write_text(json.dumps({"t": 1, "free_mib": 1}) + "\n{\"t\": 2, \"fr")
    assert mp.load_log(p) == [{"t": 1, "free_mib": 1}]
    assert mp.load_log(tmp_path / "none.jsonl") == []


def test_windows_probe_reads_streamed_kb():
    class FakeProc:
        def __init__(self, *a, **k):
            self.args = a[0]
            self.stdout = io.StringIO("\n2097152\n1048576\n")

        def terminate(self):
            pass

    probe = mp.WindowsFreeProbe(popen=FakeProc)
    assert probe.wait_ready(5)
    # 読み取りの糸が最後の行まで読むのを待つ
    for _ in range(100):
        if probe() == 1024:
            break
    assert probe() == 1024
    probe.close()


def test_windows_probe_command_is_locale_independent():
    seen = {}

    class FakeProc:
        def __init__(self, cmd, **k):
            seen["cmd"] = cmd
            self.stdout = io.StringIO("")

        def terminate(self):
            pass

    mp.WindowsFreeProbe(interval_ms=500, popen=FakeProc).close()
    script = seen["cmd"][-1]
    assert "Win32_OperatingSystem" in script and "FreePhysicalMemory" in script
    assert "Get-Counter" not in script and "Milliseconds 500" in script


def test_cli_linux_probe_runs_and_logs(tmp_path):
    log = tmp_path / "p.jsonl"
    # 目標を今の空きより十分大きくして、何も確保せずに 2 行だけ書かせる
    avail = mp.linux_available_mib()
    rc = mp.main(["--target-mib", str(avail + 100_000), "--probe", "linux", "--log", str(log),
                  "--duration", "1.5", "--interval", "0.5"])
    assert rc == 0
    rows = mp.load_log(log)
    assert len(rows) >= 2 and all(r["held_mib"] == 0 for r in rows)
    assert {"t", "free_mib", "held_mib", "action", "psi", "probe"} <= set(rows[0])


def test_cli_module_entry_point(tmp_path):
    log = tmp_path / "p.jsonl"
    avail = mp.linux_available_mib()
    out = subprocess.run([sys.executable, "-m", "evaluation.gpu_local.mem_pressure", "--target-mib",
                          str(avail + 100_000), "--probe", "linux", "--log", str(log), "--duration", "0.5"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert mp.load_log(log)
