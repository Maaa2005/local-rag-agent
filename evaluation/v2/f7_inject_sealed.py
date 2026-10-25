import json, shutil, subprocess, sys
from pathlib import Path
py, repo, S, src, work = sys.argv[1:6]
src, work = Path(src), Path(work)
if work.exists(): shutil.rmtree(work)
work.mkdir(parents=True)
def mut_reverse(r):
    o = r["outcomes"]; a, b = o.get("all_true"), o.get("any_false")
    if not (a and b) or a.get("value") == b.get("value"): return False
    a["value"], b["value"] = b["value"], a["value"]; return True
def mut_number(r):
    for c in r.get("conditions") or []:
        t = c.get("threshold") or {}
        if isinstance(t.get("value"), (int, float)): t["value"] = t["value"] + 1; return True
        if isinstance(t.get("n"), (int, float)): t["n"] = t["n"] + 1; return True
    return False
def mut_version(r):
    v = r.get("doc_version") or ""; r["doc_version"] = ("0" if v[:1] != "0" else "1") + v[1:]; return True
kinds = {"reverse": mut_reverse, "number": mut_number, "version": mut_version}
res = []
ids = sorted(p.stem for p in src.glob("R-*.json"))
cases = [(rid, k) for rid in ids for k in kinds] + [("+unstamped", "add")]
for rid, k in cases:
    d = work / f"{rid}.{k}"; recs = d / "records"
    shutil.copytree(src, recs)
    if k == "add":
        r = json.loads((recs / f"{ids[0]}.json").read_text(encoding="utf-8")); r["rule_id"] += "-copy"
        (recs / f"{ids[0]}-copy.json").write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8"); ok = True
    else:
        p = recs / f"{rid}.json"; r = json.loads(p.read_text(encoding="utf-8")); ok = kinds[k](r)
        p.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    if not ok: res.append((rid, k, "n/a", "")); continue
    out, summ = d / "rows.jsonl", d / "summary.json"
    p = subprocess.run([py, "-I", "evaluation/v2/run_decide_sealed.py",
        "--questions", f"{S}/questions_boundary.jsonl", "--questions", f"{S}/questions_more.jsonl",
        "--docs-root", S, "--records", str(recs), "--out", str(out), "--summary", str(summ), "--label", "inject"],
        cwd=repo, capture_output=True, text=True)
    stopped = p.returncode != 0 and not out.exists() and not summ.exists()
    msg = (p.stderr.strip().splitlines() or [""])[-1][:90]
    res.append((rid, k, "STOP" if stopped else f"RAN rc={p.returncode}", msg))
for x in res: print(*x, sep="\t")
c = {}
for _, k, s, _m in res: c.setdefault(k, {}).setdefault(s.split()[0], 0); c[k][s.split()[0]] += 1
print(json.dumps(c))
