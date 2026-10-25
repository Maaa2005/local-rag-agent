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
def mut_relation(r):
    # 7.2 (e): any_false に結論を持ったまま、その側を許さない if_then にする。表のハッシュも付け直し、表を通った後の型の検査で止まるかを見る
    if r.get("relation") != "if_else" or (r["outcomes"].get("any_false") or {}).get("value") in (None, "unknown"): return False
    r["relation"] = "if_then"; return True
kinds = {"reverse": mut_reverse, "number": mut_number, "version": mut_version, "relation": mut_relation}
expect = {"reverse": "確認済みの表と違う", "number": "確認済みの表と違う", "version": "確認済みの表と違う",
          "add": "確認済みの表に無い", "relation": "この側に結論を持てない"}
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
        if ok and k == "relation":
            sys.path.insert(0, repo); from v2.decide.records import record_sha256
            vt = recs / "verified.json"; t = json.loads(vt.read_text(encoding="utf-8"))
            t["records"][rid] = record_sha256(r); vt.write_text(json.dumps(t, ensure_ascii=False), encoding="utf-8")
    if not ok: res.append((rid, k, "n/a", "")); continue
    out, summ = d / "rows.jsonl", d / "summary.json"
    p = subprocess.run([py, "-I", "evaluation/v2/run_decide_sealed.py",
        "--questions", f"{S}/questions_boundary.jsonl", "--questions", f"{S}/questions_more.jsonl",
        "--docs-root", S, "--records", str(recs), "--out", str(out), "--summary", str(summ), "--label", "inject"],
        cwd=repo, capture_output=True, text=True)
    stopped = p.returncode != 0 and not out.exists() and not summ.exists()
    both = p.stdout + p.stderr
    why = expect[k] in both
    msg = next((l for l in both.splitlines() if expect[k] in l), (both.strip().splitlines() or [""])[-1])[:90]
    res.append((rid, k, ("STOP" if why else "STOP-other") if stopped else f"RAN rc={p.returncode}", msg))
for x in res: print(*x, sep="\t")
c = {}
for _, k, s, _m in res: c.setdefault(k, {}).setdefault(s.split()[0], 0); c[k][s.split()[0]] += 1
print(json.dumps(c))
