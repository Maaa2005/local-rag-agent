"""版 8 の動作確認を Colab の上でやり直す（見済みの版 7 の 10 問だけ・家の時と同じ手順と合格の線）。
    python3 colab/smoke_colab.py <モデル名> <計画に書いた載せる層の数> <work8> <llama-server>
1) 自動の載せ方で基準 Q8_0 を立ち上げ、載った層を読む  2) -ngl n --fit off で、確かめなし・ありの 2 回  3) 比べる
線を 1 つでも外れたら終了コード 1（新しい問を流さない）。"""
import json, pathlib, subprocess, sys, time
from evaluation.quant_probe import run_answers
from evaluation.quant_probe.pipeline import offload_from_log

name, ngl, work, exe = sys.argv[1], int(sys.argv[2]), pathlib.Path(sys.argv[3]), sys.argv[4]
port = 8792
ref = work / name / "ref-Q8_0.gguf"
out = work / "_smoke" / name; out.mkdir(parents=True, exist_ok=True)
url = f"http://127.0.0.1:{port}"


def serve(tag, extra):
    log = out / f"{tag}-server.log"
    f = log.open("w", encoding="utf-8")
    p = subprocess.Popen([exe, "-m", str(ref), "-c", "4096", "-np", "1", *extra, "--seed", "1", "--cache-ram", "0", "-lv", "4",
                          "--host", "127.0.0.1", "--port", str(port)], stdout=f, stderr=subprocess.STDOUT)
    return p, log


def stop(p):
    p.terminate()
    try:
        p.wait(60)
    except Exception:
        p.kill()
    time.sleep(3)


res = {"model": name, "planned_ngl": ngl}
p, log = serve("auto", [])
try:
    run_answers.wait_ready(url)
finally:
    stop(p)
off = offload_from_log(log.read_text(encoding="utf-8", errors="replace")); res["auto_offload"] = off
for tag, extra in (("noprobe", []), ("probe", ["--probe-self"])):
    for f in (out / f"{tag}.jsonl", out / f"{tag}.meta.json"):
        if f.exists():
            f.rename(f.with_name(f.name + f".old{int(time.time())}"))
    p, log = serve(tag, ["-ngl", str(ngl), "--fit", "off"])
    try:
        run_answers.wait_ready(url)
        rc = run_answers.main(["--server", url, "--out", str(out / f"{tag}.jsonl"), "--v7-smoke", *extra])
    finally:
        stop(p)
    res[f"{tag}_rc"] = rc; res[f"{tag}_offload"] = offload_from_log(log.read_text(encoding="utf-8", errors="replace"))
a = run_answers.read_rows(out / "noprobe.jsonl"); b = run_answers.read_rows(out / "probe.jsonl")
res["n"] = (len(a), len(b)); res["same_text"] = sum(a[i]["text"] == b[i]["text"] for i in a if i in b)
res["reasoning"] = sum(bool(r.get("reasoning")) or "<think>" in r["text"] for r in b.values())
res["has_mark"] = sum("結論:" in r["text"] for r in b.values())
res["finish"] = sorted({r["finish_reason"] for r in b.values()})
pr = [r for r in b.values() if r.get("probe")]
res["probe_n"] = len(pr); res["probe_len"] = sorted({len(r["probe"]) for r in pr})
res["tok_equal"] = sum(r["probe_prompt_tokens"] == r["usage"]["prompt_tokens"] for r in pr)
res["sec_per_q"] = round(sum(r["elapsed"] for r in b.values()) / max(1, len(b)), 1)
full = [ngl, ngl]
bad = [k for k, ok in {
    "載った層が計画の数（全部の層）でない": list(off or []) == full and list(res["noprobe_offload"] or []) == full
                                          and list(res["probe_offload"] or []) == full,
    "答えの回が終わらない": res["noprobe_rc"] == 0 and res["probe_rc"] == 0 and list(res["n"]) == [10, 10],
    "挟む・挟まないで答えの文が違う": res["same_text"] == 10,
    "考える文が出た": res["reasoning"] == 0,
    "「結論:」が無い": res["has_mark"] == 10,
    "確かめが取れない": res["probe_n"] == 10,
    "入力の語の数が違う": res["tok_equal"] == 10,
}.items() if not ok]
res["bad"] = bad
print(json.dumps(res, ensure_ascii=False))
(out / "smoke.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
sys.exit(1 if bad else 0)
