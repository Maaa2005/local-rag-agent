# Entry point sent with `colab exec -f` (keep this file ASCII: the Windows CLI reads it as cp932).
# Starts job.sh detached, then waits until it finishes so the kernel stays busy and the VM is kept.
# Pass with --env: QP_NAME, QP_REPO, QP_NGL, QP_SHA. Add QP_FORCE=1 only to start over.
# Upload /content/qp8-bundle.tgz (pinned code, texts, job scripts) first.
import os, pathlib, subprocess, tarfile, time

name, repo, ngl, sha = (os.environ[k] for k in ("QP_NAME", "QP_REPO", "QP_NGL", "QP_SHA"))
force = os.environ.pop("QP_FORCE", "") == "1"
B = pathlib.Path("/content/qp"); O = pathlib.Path("/content/out") / name
code = B / "code"; started = O / "started"; done = O / "done"; status = O / "status.txt"
O.mkdir(parents=True, exist_ok=True); code.mkdir(parents=True, exist_ok=True)
running = subprocess.run(["pgrep", "-f", f"job.sh {name} "], capture_output=True).returncode == 0
if running:
    print("job.sh is already running; waiting only", flush=True)
elif force or not started.exists():
    with tarfile.open("/content/qp8-bundle.tgz") as t:
        t.extractall(code)
    done.unlink(missing_ok=True)
    subprocess.Popen(["bash", str(code / "colab" / "job.sh"), name, repo, ngl, sha], cwd=str(code),
                     stdout=open(O / "job.log", "ab"), stderr=subprocess.STDOUT, start_new_session=True)
    started.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
    print("job.sh started", flush=True)
last, n = "", 0
while not done.exists():
    time.sleep(30); n += 1
    s = status.read_text(encoding="utf-8").strip().splitlines()[-1] if status.exists() else ""
    if s != last or n % 10 == 0:
        print(time.strftime("%H:%M:%S"), s.encode("ascii", "replace").decode(), flush=True); last = s
print("DONE", done.read_text().strip(), flush=True)
