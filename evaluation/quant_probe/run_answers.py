"""起動済みの llama-server に 360 問を順に投げ、生の答えを 1 問 1 行で残す（途中から再開できる）。

    python -m evaluation.quant_probe.run_answers --server http://127.0.0.1:8080 --out runs/<名前>.jsonl

同じ答えを毎回出すため: 温度 0・top_k 1・seed 固定・cache_prompt false（llama-server の README:
cache_prompt は処理の大きさで logits が変わり非決定になりうる）。サーバーは -np 1 で起動する。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

from .questions import PROMPT_VERSION, build_messages, load_questions, load_rules, manifest

GEN = {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "min_p": 0.0, "seed": 1, "cache_prompt": False,
       "chat_template_kwargs": {"enable_thinking": False}}


def _post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get(url: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def wait_ready(server: str, timeout: float = 600) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if _get(f"{server}/health", 5).get("status") == "ok":
                return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError("llama-server が準備できない")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--ids", type=Path, help="この id だけ流す（1 行 1 id）。試し用")
    ap.add_argument("--limit", type=int, help="先頭からこの数だけ。試し用")
    a = ap.parse_args(argv)

    qs = load_questions()
    if a.ids:
        keep = set(a.ids.read_text(encoding="utf-8").split())
        qs = [q for q in qs if q.id in keep]
    if a.limit:
        qs = qs[: a.limit]
    rules = load_rules()
    wait_ready(a.server)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    meta = a.out.with_suffix(".meta.json")
    if not meta.exists():
        try:
            props = _get(f"{a.server}/props")
        except Exception as e:  # 古い版では無い事がある
            props = {"error": str(e)}
        meta.write_text(json.dumps({
            "prompt_version": PROMPT_VERSION, "manifest": manifest(), "gen": GEN, "max_tokens": a.max_tokens,
            "server_props": {k: props.get(k) for k in ("model_path", "build_info", "n_ctx", "total_slots",
                                                       "chat_template") if k in props},
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "n_questions": len(qs),
        }, ensure_ascii=False, indent=1), encoding="utf-8")
    done = set()
    if a.out.exists():
        for line in a.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    todo = [q for q in qs if q.id not in done]
    print(f"{len(done)} 済み・残り {len(todo)}", file=sys.stderr)
    with a.out.open("a", encoding="utf-8") as f:
        for i, q in enumerate(todo, 1):
            body = {"messages": build_messages(q, rules), "max_tokens": a.max_tokens, **GEN}
            t0 = time.time()
            r = _post(f"{a.server}/v1/chat/completions", body, a.timeout)
            ch = r["choices"][0]
            row = {"id": q.id, "text": ch["message"].get("content") or "",
                   "reasoning": ch["message"].get("reasoning_content"),
                   "finish_reason": ch.get("finish_reason"), "usage": r.get("usage"),
                   "timings": r.get("timings"), "elapsed": round(time.time() - t0, 3)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 20 == 0 or i == len(todo):
                print(f"{i}/{len(todo)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
