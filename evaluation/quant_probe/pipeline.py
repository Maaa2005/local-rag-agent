"""1 モデル分を計画 9 章の順に回す（量子化 → imatrix → KLD → 360 問）。途中で止めても続きから再開する。

    python -m evaluation.quant_probe.pipeline --model qwen3-8b --bf16 <bf16.gguf> --texts <dir> --work <dir> \\
        --bin <llama.cpp の実行ファイルのフォルダ> [--launcher <llama_noble.sh>] [--variants Q4_K_M-en,...]

変種のファイルは <work>/<model>/current.gguf に上書きし、sha256 だけ variants.jsonl に残す（ディスクを増やさない）。
重い処理は優先度を下げ、1 度に 1 つだけ動かす。

揺れを測る組（判定の線 1）も流す: Q4_K_M-en をもう 1 回（Q4_K_M-en-r2）。基準が GPU に載り切らなかった時は、
Q6_K-none を基準と同じ載せ方（-ngl）でもう 1 回（Q6_K-none-place）。--bf16-imatrix-check を付けると、
BF16 から作った imatrix で Q4_K_M を作り KLD だけ測る（Q4_K_M-en-bf16imx・判定には使わない）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from . import run_answers

QUANTS = ("Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K")
CALIBS = ("none", "en", "ja")
ALL_VARIANTS = tuple(f"{q}-{c}" for q in QUANTS for c in CALIBS)
TEXTS = ("en", "ja", "dom")
REF = "ref-Q8_0"
RERUN = "Q4_K_M-en-r2"
PLACE = "Q6_K-none-place"
BF16IMX = "Q4_K_M-en-bf16imx"
_OFFLOAD = re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers to GPU")


def offload_from_log(text: str) -> tuple[int, int] | None:
    """llama-server の起動の記録から、GPU に載せた層の数と全部の層の数。"""
    m = _OFFLOAD.search(text)
    return (int(m.group(1)), int(m.group(2))) if m else None


def variant_parts(name: str) -> tuple[str, str]:
    q, _, c = name.rpartition("-")
    if q not in QUANTS or c not in CALIBS:
        raise ValueError(f"変種の名前が違う: {name}")
    return q, c


class Tools:
    def __init__(self, bin_dir: Path | None, launcher: list[str] | None, dry: bool = False):
        self.bin_dir, self.launcher, self.dry = bin_dir, launcher, dry

    def cmd(self, tool: str, *args: str) -> list[str]:
        if self.launcher:
            base = [*self.launcher, tool]
        else:
            exe = self.bin_dir / (tool + (".exe" if os.name == "nt" else ""))
            base = [str(exe)]
        return [*base, *map(str, args)]

    def _low(self) -> dict:
        if os.name == "nt":
            return {"creationflags": subprocess.BELOW_NORMAL_PRIORITY_CLASS}
        return {"preexec_fn": lambda: os.nice(19)}

    def run(self, cmd: list[str], log: Path) -> None:
        print("$ " + " ".join(cmd), file=sys.stderr)
        if self.dry:
            return
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, **self._low())
        if r.returncode != 0:
            raise RuntimeError(f"失敗 rc={r.returncode}: {log}")

    def popen(self, cmd: list[str], log: Path) -> subprocess.Popen | None:
        print("$ " + " ".join(cmd) + " &", file=sys.stderr)
        if self.dry:
            return None
        log.parent.mkdir(parents=True, exist_ok=True)
        f = log.open("w", encoding="utf-8")
        return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, **self._low())


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


class Pipeline:
    def __init__(self, a: argparse.Namespace, tools: Tools):
        self.a, self.t = a, tools
        self.dir = a.work / a.model
        self.logs = self.dir / "logs"
        self.runs = self.dir / "runs"
        self.ref = self.dir / f"{REF}.gguf"
        self.cur = self.dir / "current.gguf"
        self.timing = self.dir / "timing.jsonl"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _time(self, step: str, t0: float) -> None:
        if self.t.dry:
            return
        with self.timing.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, "sec": round(time.time() - t0, 1),
                                "at": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")

    def _ngl(self) -> list[str]:
        return [] if self.a.ngl == "auto" else ["-ngl", self.a.ngl]

    def ref_offload(self) -> tuple[int, int] | None:
        log = self.logs / f"{REF}-r1-server.log"
        return offload_from_log(log.read_text(encoding="utf-8", errors="replace")) if log.exists() else None

    def quantize(self, out: Path, qtype: str, imatrix: Path | None, name: str) -> None:
        t0 = time.time()
        args = ["--imatrix", str(imatrix)] if imatrix else []
        self.t.run(self.t.cmd("llama-quantize", *args, self.a.bf16, out, qtype), self.logs / f"{name}-quantize.log")
        self._time(f"{name}:quantize", t0)

    def step_ref(self) -> None:
        if not self.ref.exists():
            self.quantize(self.ref, "Q8_0", None, REF)
        for lang in ("en", "ja"):
            out = self.dir / f"imatrix-{lang}.gguf"
            if out.exists():
                continue
            t0 = time.time()
            self.t.run(self.t.cmd("llama-imatrix", "-m", self.ref, "-f", self.a.texts / f"calib_{lang}.txt",
                                  "-c", "512", "--chunks", str(self.a.imatrix_chunks), *self._ngl(), "-o", out),
                       self.logs / f"imatrix-{lang}.log")
            self._time(f"imatrix-{lang}", t0)
        for x in TEXTS:
            base = self.dir / f"base-{x}.kld"
            if base.exists():
                continue
            t0 = time.time()
            self.t.run(self.t.cmd("llama-perplexity", "-m", self.ref, "-f", self.a.texts / f"eval_{x}.txt",
                                  "-c", "512", "--chunks", str(self.a.kld_chunks), *self._ngl(),
                                  "--kl-divergence-base", base), self.logs / f"{REF}-base-{x}.log")
            self._time(f"base-{x}", t0)
        for r in range(1, self.a.ref_runs + 1):
            self.answers(self.ref, f"{REF}-r{r}")
        if not self.t.dry:
            off = self.ref_offload()
            (self.dir / "ref_offload.json").write_text(json.dumps({"offloaded": off}), encoding="utf-8")

    def kld(self, model: Path, name: str) -> None:
        for x in TEXTS:
            log = self.logs / f"{name}-kld-{x}.log"
            if log.exists() and "KL divergence statistics" in log.read_text(encoding="utf-8", errors="replace"):
                continue
            t0 = time.time()
            self.t.run(self.t.cmd("llama-perplexity", "-m", model, "-c", "512", "--chunks", str(self.a.kld_chunks),
                                  *self._ngl(), "--kl-divergence-base", self.dir / f"base-{x}.kld",
                                  "--kl-divergence"), log)
            self._time(f"{name}:kld-{x}", t0)

    def answers(self, model: Path, name: str, ngl: list[str] | None = None) -> None:
        out = self.runs / f"{name}.jsonl"
        n_done = sum(1 for _ in out.open(encoding="utf-8")) if out.exists() else 0
        if n_done >= self.a.expect_questions:
            return
        t0 = time.time()
        port = str(self.a.port)
        srv = self.t.popen(self.t.cmd("llama-server", "-m", model, "-c", str(self.a.ctx), "-np", "1",
                                      *(self._ngl() if ngl is None else ngl), "--seed", "1", "--cache-ram", "0", "-lv", "4", "--host", "127.0.0.1",
                                      "--port", port), self.logs / f"{name}-server.log")
        try:
            if not self.t.dry:
                argv = ["--server", f"http://127.0.0.1:{port}", "--out", str(out),
                        "--max-tokens", str(self.a.max_tokens)]
                if self.a.ids:
                    argv += ["--ids", str(self.a.ids)]
                run_answers.main(argv)
        finally:
            if srv is not None:
                srv.terminate()
                try:
                    srv.wait(60)
                except subprocess.TimeoutExpired:
                    srv.kill()
        self._time(f"{name}:answers", t0)

    def step_variant(self, name: str) -> None:
        q, c = variant_parts(name)
        done_ans = (self.runs / f"{name}.jsonl")
        n_done = sum(1 for _ in done_ans.open(encoding="utf-8")) if done_ans.exists() else 0
        kld_done = all((self.logs / f"{name}-kld-{x}.log").exists() for x in TEXTS)
        if n_done >= self.a.expect_questions and kld_done:
            return
        marker = self.dir / "current.name"
        if self.t.dry or not (self.cur.exists() and marker.exists() and marker.read_text().strip() == name):
            imx = None if c == "none" else self.dir / f"imatrix-{c}.gguf"
            self.quantize(self.cur, q, imx, name)
            if not self.t.dry:
                marker.write_text(name)
                with (self.dir / "variants.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"name": name, "sha256": sha256(self.cur),
                                        "bytes": self.cur.stat().st_size}) + "\n")
        self.kld(self.cur, name)
        self.answers(self.cur, name)
        if name == "Q4_K_M-en" and self.a.noise_reruns:
            self.answers(self.cur, RERUN)
        if name == "Q6_K-none" and self.a.noise_reruns:
            off = self.ref_offload()
            if self.t.dry or (off and off[0] < off[1]):
                self.answers(self.cur, PLACE, ["-ngl", str(off[0] if off else 0)])

    def step_bf16_check(self) -> None:
        """BF16 から作った imatrix（英語）で Q4_K_M を作り、KLD だけ測る。"""
        if all((self.logs / f"{BF16IMX}-kld-{x}.log").exists() for x in TEXTS):
            return
        imx = self.dir / "imatrix-en-bf16.gguf"
        if not imx.exists():
            t0 = time.time()
            self.t.run(self.t.cmd("llama-imatrix", "-m", self.a.bf16, "-f", self.a.texts / "calib_en.txt",
                                  "-c", "512", "--chunks", str(self.a.imatrix_chunks), *self._ngl(), "-o", imx),
                       self.logs / "imatrix-en-bf16.log")
            self._time("imatrix-en-bf16", t0)
        self.quantize(self.cur, "Q4_K_M", imx, BF16IMX)
        if not self.t.dry:
            (self.dir / "current.name").write_text(BF16IMX)
            with (self.dir / "variants.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"name": BF16IMX, "sha256": sha256(self.cur),
                                    "bytes": self.cur.stat().st_size}) + "\n")
        self.kld(self.cur, BF16IMX)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="作業フォルダの名前（例 qwen3-8b）")
    ap.add_argument("--bf16", required=True, type=Path)
    ap.add_argument("--texts", required=True, type=Path)
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--bin", type=Path)
    ap.add_argument("--launcher", nargs="*", help="実行ファイルの前に付ける物（WSL の llama_noble.sh など）")
    ap.add_argument("--variants", default=",".join(ALL_VARIANTS))
    ap.add_argument("--kld-chunks", type=int, default=80)
    ap.add_argument("--imatrix-chunks", type=int, default=128)
    ap.add_argument("--ref-runs", type=int, default=2)
    ap.add_argument("--ngl", default="auto")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--ids", type=Path, help="この id だけ流す（試し用）")
    ap.add_argument("--expect-questions", type=int, default=360)
    ap.add_argument("--only", choices=("ref", "variants", "all"), default="all")
    ap.add_argument("--no-noise-reruns", dest="noise_reruns", action="store_false",
                    help="揺れを測る組（Q4_K_M-en-r2・Q6_K-none-place）を流さない")
    ap.add_argument("--bf16-imatrix-check", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if not a.launcher and not a.bin:
        ap.error("--bin か --launcher が要る")
    # WSL の起動役は実行ファイルのフォルダへ移るので、道は全部絶対にする
    a.bf16, a.texts, a.work = a.bf16.resolve(), a.texts.resolve(), a.work.resolve()
    if a.ids:
        a.ids = a.ids.resolve()
    p = Pipeline(a, Tools(a.bin, a.launcher, a.dry_run))
    names = [v for v in a.variants.split(",") if v]
    for n in names:
        variant_parts(n)
    if a.only in ("ref", "all"):
        p.step_ref()
    if a.only in ("variants", "all"):
        for n in names:
            p.step_variant(n)
        if a.bf16_imatrix_check:
            p.step_bf16_check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
