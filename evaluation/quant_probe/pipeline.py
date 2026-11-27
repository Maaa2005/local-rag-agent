"""1 モデル分を計画 9 章の順に回す（量子化 → imatrix → KLD → 360 問）。途中で止めても続きから再開する。

    python -m evaluation.quant_probe.pipeline --model qwen3-8b --bf16 <bf16.gguf> --texts <dir> --work <dir> \\
        --bin <llama.cpp の実行ファイルのフォルダ> [--launcher <llama_noble.sh>] [--variants Q4_K_M-en,...]

変種のファイルは <work>/<model>/current.gguf に上書きし、sha256 だけ variants.jsonl に残す（ディスクを増やさない）。
重い処理は優先度を下げ、1 度に 1 つだけ動かす。

揺れを測る組（判定の線 1）も流す: Q4_K_M-en をもう 1 回（Q4_K_M-en-r2）。--bf16-imatrix-check を付けると、
BF16 から作った imatrix で Q4_K_M を作り KLD だけ測る（Q4_K_M-en-bf16imx・判定には使わない）。

載せ方（版 8 の計画 6 章）: --ngl に数 n を渡すと、答えの回は全部（基準の 1 回目も）`-ngl n --fit off` で流す。
本番（--trial なし、または --strict-place）は --ngl が数でないと答えの回を始めない。各答えの回は、サーバーが立ち上がった後・
問を送る前に、記録から載った層の数を読み、n（n が全部の層より大きければ全部の層）と違えば止める（その回の jsonl を作らない）。
KLD と imatrix は今までどおり自動（-ngl を付けない）。--print-offload は自動の載せ方で基準 Q8_0 のサーバーを立ち上げ、
載った層の数だけを書き出して終わる（問は送らない・n を決める動作確認用）。基準 1 回目の載った層は ref_offload.json に残す。

kld_chat の確かめ（計画 5 章）: 基準 1 回目は --probe-self、基準 2 回目・変種・Q4_K_M-en-r2 は --probe-ref <基準 1 回目の jsonl>。
--no-probe で全部の回から外せる（挟む・挟まないを比べる動作確認用）。
取り違えを防ぐ: 答えの jsonl の行の数が足りていても、meta の指紋（問・渡す文の版・生成の設定・最大語数）が今と同じ時だけ飛ばす。
違えば run_answers が終了コード 2 で止まり、ここも止まる。
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

    def _log_offload(self, name: str) -> tuple[int, int] | None:
        log = self.logs / f"{name}-server.log"
        return offload_from_log(log.read_text(encoding="utf-8", errors="replace")) if log.exists() else None

    def _save_ref_offload(self) -> None:
        """基準 1 回目の記録から載った層の数を ref_offload.json に残す（report の点検の元）。"""
        if self.t.dry:
            return
        off = self._log_offload(f"{REF}-r1")
        if off is None and (self.dir / "ref_offload.json").exists():
            return  # 記録が読めないが前に残した数がある（上書きして消さない）
        (self.dir / "ref_offload.json").write_text(json.dumps({"offloaded": off}), encoding="utf-8")

    def strict_place(self) -> bool:
        return bool(self.a.strict_place or not self.a.trial)

    def ngl_value(self) -> int | None:
        """--ngl の数（auto なら None）。"""
        return int(self.a.ngl) if str(self.a.ngl).isdigit() else None

    def answer_ngl(self, name: str) -> list[str]:
        """答えの回の載せ方。数なら全部の回が `-ngl n --fit off`。本番で数でなければ答えの回を始めない。"""
        n = self.ngl_value()
        if n is None:
            if self.strict_place():
                raise SystemExit(f"{name}: 本番の答えの回は --ngl に数が要る（--ngl {self.a.ngl}）。"
                                 "自動の載せ方で載る層の数は --print-offload で調べる（試しは --trial）")
            return []
        return ["-ngl", str(n), "--fit", "off"]

    def check_offload(self, name: str) -> None:
        """サーバーが立ち上がった後・問を送る前に、記録の載った層の数が指定と同じか確かめる。違えば止める。"""
        n = self.ngl_value()
        if n is None:
            return
        off = self._log_offload(name)
        if off is None:
            raise RuntimeError(f"{name}: サーバーの記録に載った層の数が無い（logs/{name}-server.log）")
        if off[0] != min(n, off[1]):
            raise RuntimeError(f"{name}: 載った層が {off[0]}/{off[1]}（指定は -ngl {n}）。問を送らずに止める")

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
                                  "-c", "512", "--chunks", str(self.a.imatrix_chunks), "-o", out),
                       self.logs / f"imatrix-{lang}.log")
            self._time(f"imatrix-{lang}", t0)
        for x in TEXTS:
            base = self.dir / f"base-{x}.kld"
            if base.exists():
                continue
            t0 = time.time()
            self.t.run(self.t.cmd("llama-perplexity", "-m", self.ref, "-f", self.a.texts / f"eval_{x}.txt",
                                  "-c", "512", "--chunks", str(self.a.kld_chunks),
                                  "--kl-divergence-base", base), self.logs / f"{REF}-base-{x}.log")
            self._time(f"base-{x}", t0)
        for r in range(1, self.a.ref_runs + 1):
            self.answers(self.ref, f"{REF}-r{r}")
            if r == 1:
                self._save_ref_offload()  # report が全部の答えの回の載った層と照らす元

    def kld(self, model: Path, name: str) -> None:
        for x in TEXTS:
            log = self.logs / f"{name}-kld-{x}.log"
            if log.exists() and "KL divergence statistics" in log.read_text(encoding="utf-8", errors="replace"):
                continue
            t0 = time.time()
            self.t.run(self.t.cmd("llama-perplexity", "-m", model, "-c", "512", "--chunks", str(self.a.kld_chunks),
                                  "--kl-divergence-base", self.dir / f"base-{x}.kld",
                                  "--kl-divergence"), log)
            self._time(f"{name}:kld-{x}", t0)

    def answers_done(self, name: str) -> bool:
        """360 問の答えがそろっていて、meta の指紋が今と同じ時だけ True（行の数だけで飛ばさない）。"""
        out = self.runs / f"{name}.jsonl"
        n_done = sum(1 for _ in out.open(encoding="utf-8")) if out.exists() else 0
        if n_done < self.a.expect_questions:
            return False
        meta = run_answers.read_meta(run_answers.meta_path(out))
        return meta is not None and not run_answers.meta_mismatch(meta, self.a.max_tokens)

    def probe_args(self, name: str) -> list[str]:
        """確かめの引数。基準 1 回目は --probe-self、ほかは基準 1 回目の jsonl を --probe-ref で渡す。--no-probe なら無し。"""
        if self.a.no_probe:
            return []
        if name == f"{REF}-r1":
            return ["--probe-self"]
        return ["--probe-ref", str(self.runs / f"{REF}-r1.jsonl")]

    def answers(self, model: Path, name: str, ngl: list[str] | None = None) -> None:
        """360 問の答えの回。ngl を渡さなければ answer_ngl の決まり（--ngl の数で全部の回を同じに）。"""
        out = self.runs / f"{name}.jsonl"
        if self.answers_done(name):
            return
        t0 = time.time()
        port = str(self.a.port)
        place = self.answer_ngl(name) if ngl is None else ngl
        srv = self.t.popen(self.t.cmd("llama-server", "-m", model, "-c", str(self.a.ctx), "-np", "1",
                                      *place, "--seed", "1", "--cache-ram", "0",
                                      "-lv", "4", "--host", "127.0.0.1", "--port", port),
                           self.logs / f"{name}-server.log")
        try:
            if not self.t.dry:
                url = f"http://127.0.0.1:{port}"
                run_answers.wait_ready(url)
                if ngl is None:
                    self.check_offload(name)
                argv = ["--server", url, "--out", str(out), "--max-tokens", str(self.a.max_tokens),
                        *self.probe_args(name)]
                if self.a.ids:
                    argv += ["--ids", str(self.a.ids)]
                rc = run_answers.main(argv)
                if rc:
                    raise RuntimeError(f"{name}: run_answers が終了コード {rc} で止まった（meta の指紋が今と違うなど）")
        finally:
            if srv is not None:
                srv.terminate()
                try:
                    srv.wait(60)
                except subprocess.TimeoutExpired:
                    srv.kill()
        self._time(f"{name}:answers", t0)

    def print_offload(self) -> tuple[int, int] | None:
        """自動の載せ方で基準 Q8_0 のサーバーを立ち上げ、載った層の数だけを書き出す（問は送らない）。n を決める動作確認用。"""
        if not self.ref.exists():
            self.quantize(self.ref, "Q8_0", None, REF)
        name = f"{REF}-print-offload"
        srv = self.t.popen(self.t.cmd("llama-server", "-m", self.ref, "-c", str(self.a.ctx), "-np", "1",
                                      "--seed", "1", "--cache-ram", "0", "-lv", "4", "--host", "127.0.0.1",
                                      "--port", str(self.a.port)), self.logs / f"{name}-server.log")
        try:
            if not self.t.dry:
                run_answers.wait_ready(f"http://127.0.0.1:{self.a.port}")
        finally:
            if srv is not None:
                srv.terminate()
                try:
                    srv.wait(60)
                except subprocess.TimeoutExpired:
                    srv.kill()
        off = None if self.t.dry else self._log_offload(name)
        if not self.t.dry:
            print(json.dumps({"offloaded": list(off) if off else None, "ctx": self.a.ctx}))
        return off

    def step_variant(self, name: str) -> None:
        q, c = variant_parts(name)
        kld_done = all((self.logs / f"{name}-kld-{x}.log").exists() for x in TEXTS)
        if self.answers_done(name) and kld_done:
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

    def step_bf16_check(self) -> None:
        """BF16 から作った imatrix（英語）で Q4_K_M を作り、KLD だけ測る。"""
        if all((self.logs / f"{BF16IMX}-kld-{x}.log").exists() for x in TEXTS):
            return
        imx = self.dir / "imatrix-en-bf16.gguf"
        if not imx.exists():
            t0 = time.time()
            self.t.run(self.t.cmd("llama-imatrix", "-m", self.a.bf16, "-f", self.a.texts / "calib_en.txt",
                                  "-c", "512", "--chunks", str(self.a.imatrix_chunks), "-o", imx),
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
    ap.add_argument("--ngl", default="auto", help="答えの回の載せ方。数 n なら全部の回を `-ngl n --fit off` で流す（本番は数が要る）")
    ap.add_argument("--trial", action="store_true", help="試し。--ngl が数でなくても答えの回を流す")
    ap.add_argument("--strict-place", action="store_true", help="--trial でも、--ngl が数でないと答えの回を始めない")
    ap.add_argument("--no-probe", action="store_true", help="kld_chat の確かめの依頼を全部の回で挟まない")
    ap.add_argument("--print-offload", action="store_true",
                    help="自動の載せ方で基準 Q8_0 を立ち上げ、載った層の数だけを書き出して終わる（問は送らない）")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--ids", type=Path, help="この id だけ流す（試し用）")
    ap.add_argument("--expect-questions", type=int, default=360)
    ap.add_argument("--only", choices=("ref", "variants", "all"), default="all")
    ap.add_argument("--no-noise-reruns", dest="noise_reruns", action="store_false",
                    help="揺れを測る組（Q4_K_M-en-r2）を流さない")
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
    if a.print_offload:
        p.print_offload()
        return 0
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
