"""家の PC から Colab を動かす（計画 版 8 の 12 章）。画面の端末から python で呼ぶ（Bash の囲いの中では Colab のログインが見えない）。

    python home.py bundle            測るコード（固定したコミット）・文章・台本を 1 つにまとめる（Colab は使わない）
    python home.py up <名前>         L4 を 1 台借り、まとめた物（と前の台の続き）を上げる
    python home.py launch <名前>     流し始めて終わるまで待つ（端末を閉じても VM の中は進む）。やり直しは最後に force
    python home.py fetch <名前>      出来た段の結果を取り、~/qp-data/work8/<名前>/ に広げる
    python home.py stop <名前>       VM を返し、残りの枠と動いている台を表示する

ユニットは買わない（`colab pay` は呼ばない）。表示は ~/qp-data/colab/home.log にも残す。
"""
from __future__ import annotations

import hashlib
import io
import subprocess
import sys
import tarfile
import time
from pathlib import Path

PIN = "26e7052975f12397fa9ffef3d9884cf24d06d3a1"  # 測るコードを固定したコミット（life#24 の固定のコメント）
FINGERPRINT = "d82935b365b30004e262826293e6b81e984fe0d530c4b7ab50586b40c801f667"
MODELS = {
    "ministral3-8b": {"repo": "mistralai/Ministral-3-8B-Instruct-2512-BF16", "ngl": 35, "session": "qp8-min",
                      "sha": "602fe402c57af8102b3cd96a9c829cb2237a667b33e6894b788cdc82486352fb"},
    "nemotron-ja-9b": {"repo": "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese", "ngl": 57, "session": "qp8-nem",
                       "sha": "149997134183017de1696252c370ec6dbab0c424e22c7871462f3d06470e3174"},
}
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
D = Path.home() / "qp-data"
CD = D / "colab"
COLAB = [str(Path.home() / ".local" / "bin" / "colab.exe"), "--config", str(CD / "sessions.json")]
BUNDLE = CD / "qp8-bundle.tgz"
SKIP_RESTORE = ("setup-logs", "fail", "smoke-")


def log(*a: object) -> None:
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    CD.mkdir(parents=True, exist_ok=True)
    with (CD / "home.log").open("a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + s + "\n")


def colab(*a: str, tee: Path | None = None) -> int:
    log("$ colab", *a)
    p = subprocess.Popen([*COLAB, *a], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace")
    assert p.stdout is not None
    for line in p.stdout:
        line = line.rstrip()
        if line:
            log("  " + line)
            if tee:
                with tee.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
    rc = p.wait()
    if rc:
        log(f"  rc={rc}")
    return rc


def bundle() -> None:
    full = subprocess.run(["git", "-C", str(REPO), "rev-parse", PIN], capture_output=True, text=True, check=True).stdout.strip()
    src = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", full, "evaluation/__init__.py",
                          "evaluation/quant_probe"], capture_output=True, check=True).stdout
    CD.mkdir(parents=True, exist_ok=True)

    def add(t: tarfile.TarFile, name: str, data: bytes) -> None:
        i = tarfile.TarInfo(name); i.size = len(data); i.mode = 0o644; i.mtime = 0
        t.addfile(i, io.BytesIO(data))

    with tarfile.open(BUNDLE, "w:gz") as out:
        with tarfile.open(fileobj=io.BytesIO(src)) as t:
            for m in t.getmembers():
                if m.isfile() and not m.name.startswith("evaluation/quant_probe/colab/"):
                    add(out, m.name, t.extractfile(m).read())
        sums = []
        for p in sorted((D / "texts8").glob("*.txt")):
            b = p.read_bytes()
            add(out, f"texts8/{p.name}", b)
            sums.append(f"{hashlib.sha256(b).hexdigest()}  texts8/{p.name}\n")
        add(out, "texts8.sha256", "".join(sums).encode())
        add(out, "FINGERPRINT", (FINGERPRINT + "\n").encode())
        add(out, "COMMIT", (full + "\n").encode())
        for n in ("job.sh", "smoke_colab.py", "prefetch.py", "convert-requirements.txt"):
            add(out, f"colab/{n}", (HERE / n).read_bytes().replace(b"\r\n", b"\n"))
    log("bundle", BUNDLE, BUNDLE.stat().st_size, "bytes", "code", full,
        "sha256", hashlib.sha256(BUNDLE.read_bytes()).hexdigest())


def up(name: str) -> int:
    m = MODELS[name]; s = m["session"]
    if colab("new", "-s", s, "--gpu", "L4"):
        return 1
    colab("status", "-s", s)
    if colab("upload", "-s", s, str(BUNDLE), "/content/qp8-bundle.tgz"):
        return 1
    for p in sorted((CD / name).glob("*.tgz")):
        if not p.name.startswith(SKIP_RESTORE):
            if colab("upload", "-s", s, str(p), f"/content/in-{name}-{p.name}"):
                return 1
    return 0


def launch(name: str, force: bool) -> int:
    m = MODELS[name]
    env = {"QP_NAME": name, "QP_REPO": m["repo"], "QP_NGL": str(m["ngl"]), "QP_SHA": m["sha"]}
    if force:
        env["QP_FORCE"] = "1"
    args = [x for k, v in env.items() for x in ("--env", f"{k}={v}")]
    (CD / name).mkdir(parents=True, exist_ok=True)
    return colab("exec", "-s", m["session"], "--timeout", "0", *args, "-f", str(HERE / "launch.py"),
                 tee=CD / name / "launch.log")


def fetch(name: str) -> int:
    m = MODELS[name]; s = m["session"]; loc = CD / name; loc.mkdir(parents=True, exist_ok=True)
    rem = f"/content/out/{name}"
    for f in ("status.txt", "index.txt", "smoke.json", "job.log", "setup-logs.tgz", "pipeline.log"):
        subprocess.run([*COLAB, "download", "-s", s, f"{rem}/{f}", str(loc / f)], capture_output=True)
    idx = loc / "index.txt"
    names = list(dict.fromkeys(idx.read_text(encoding="utf-8").split())) if idx.exists() else []
    got = 0
    for n in names:
        dst = loc / n
        if dst.exists() and n != "fail.tgz":
            continue
        part = loc / (n + ".part")
        if colab("download", "-s", s, f"{rem}/{n}", str(part)) or not part.exists():
            continue
        try:
            with tarfile.open(part) as t:
                t.extractall(D / "work8" / name, filter="data")
        except Exception as e:  # 途中で切れた物は次に取り直す
            log("  壊れている（取り直す）:", n, e)
            continue
        part.replace(dst); got += 1
    st = loc / "status.txt"
    tail = st.read_text(encoding="utf-8").strip().splitlines()[-4:] if st.exists() else ["(status.txt が取れない)"]
    log(f"fetch {name}: 新しく {got} 個 / 全部で {sum((loc / n).exists() for n in names)}/{len(names)} 個")
    for line in tail:
        log("  " + line)
    return 0


def stop(name: str) -> int:
    rc = colab("stop", "-s", MODELS[name]["session"])
    colab("usage"); colab("sessions")
    return rc


def main(a: list[str]) -> int:
    if a[:1] == ["bundle"]:
        bundle(); return 0
    if len(a) >= 2 and a[1] in MODELS:
        if a[0] == "up":
            return up(a[1])
        if a[0] == "launch":
            return launch(a[1], "force" in a[2:])
        if a[0] == "fetch":
            return fetch(a[1])
        if a[0] == "stop":
            return stop(a[1])
    print(__doc__); return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
