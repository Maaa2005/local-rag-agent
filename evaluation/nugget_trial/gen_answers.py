"""試しの答えを作る（PR #32 の渡す文 qp-1 の写し。B は理由を 1 文にする所だけ替える）。"""
import json, sys, time, urllib.request
from pathlib import Path

D = Path(r"C:\Users\mtkzi\dev\local-rag-agent\evaluation\nugget_trial")
HOLD = "保留"
SYSTEM = ("あなたは社内規程の問い合わせに答える担当です。与えられた規程の文だけを根拠にします。\n"
          "質問に書かれていない事実で結論が変わる時は、推測で決めずに保留します。")
USER = {"A": "## 規程\n{rule}\n\n## 質問\n{body}\n\n理由を3文以内で書き、最後の行には「結論: {a}」「結論: {b}」「結論: " + HOLD + "」のどれか1つだけを書いてください。",
        "B": "## 規程\n{rule}\n\n## 質問\n{body}\n\n理由を1文で書き、最後の行には「結論: {a}」「結論: {b}」「結論: " + HOLD + "」のどれか1つだけを書いてください。"}

cfg, seed, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
rules = {p.stem: p.read_text(encoding="utf-8").strip() for p in (D / "data/numeric_dev90/corpus").glob("*.md")}
done = set()
if out.exists():
    done = {json.loads(l)["id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
with out.open("a", encoding="utf-8") as f:
    for line in (D / "questions.jsonl").read_text(encoding="utf-8").splitlines():
        q = json.loads(line)
        if q["id"] in done:
            continue
        body = q["question"].rstrip("\n").split("\n")
        body = "\n".join(body[:-1]).rstrip()
        a, b = q["conclusion_options"]
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER[cfg].format(rule=rules[q["rule_id"]], body=body, a=a, b=b)}]
        req = {"messages": msgs, "max_tokens": 400, "temperature": 0.7, "top_p": 0.95, "seed": seed, "cache_prompt": False}
        t0 = time.time()
        r = json.loads(urllib.request.urlopen(urllib.request.Request(
            "http://127.0.0.1:8080/v1/chat/completions", data=json.dumps(req).encode(),
            headers={"Content-Type": "application/json"}), timeout=900).read())
        ch = r["choices"][0]
        f.write(json.dumps({"id": q["id"], "text": ch["message"].get("content") or "", "config": cfg, "seed": seed,
                            "finish_reason": ch.get("finish_reason"), "elapsed": round(time.time() - t0, 1)},
                           ensure_ascii=False) + "\n")
        f.flush()
        print(q["id"], round(time.time() - t0, 1), file=sys.stderr)
