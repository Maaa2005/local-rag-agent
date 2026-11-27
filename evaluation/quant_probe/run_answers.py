"""起動済みの llama-server に 360 問を順に投げ、生の答えを 1 問 1 行で残す（途中から再開できる）。

    python -m evaluation.quant_probe.run_answers --server http://127.0.0.1:8080 --out runs/<名前>.jsonl \\
        [--probe-ref <基準 1 回目の jsonl> | --probe-self] [--v7-smoke]

同じ答えを毎回出すため: 温度 0・top_k 1・seed 固定・cache_prompt false（llama-server の README:
cache_prompt は処理の大きさで logits が変わり非決定になりうる）。サーバーは -np 1 で起動する。
答えの依頼は版 7 と 1 字も同じ（確率を頼まない）。

確かめの依頼（版 8 の計画 5 章・kld_chat）: --probe-ref か --probe-self を付けた時だけ、各問で答えの依頼の直後に 1 つ挟む。
- 書き出し = 基準 1 回目の答えの文を、最後の「結論:」の直後で切った物（「結論:」まで含む・後ろの空白は付けない）。
  --probe-self はその場で出た答えの文を使う（基準 1 回目の回）。--probe-ref <jsonl> は保存してある基準 1 回目の文を使う。
  「結論:」が無い問は確かめを送らず、行の "probe" を null にする。
- POST /apply-template（messages・chat_template_kwargs は答えの依頼と同じ）で型を当てた入力の文を受け取り、その後ろに書き出しを
  付けて POST /completion（n_predict 1・n_probs 100・post_sampling_probs false・temperature 0・top_k 1・seed 1・
  cache_prompt false・grammar PROBE_GRAMMAR）。返った最初の 1 語の上位 100 を、確率に直して "probe": [{"id", "p"}, ...] に残す。
  文法を足す理由（実機 b11836 で確かめた）: 出た 1 語が漢字の途中のバイトで終わると、サーバーの process_token
  （tools/server/server-context.cpp 2102〜2135 行）が、文が UTF-8 として途中なので slot.add_token を呼ばず、その語の確率が
  捨てられて completion_probabilities が空で返る（Qwen は 10 問中 8 問）。出す語を ASCII の 1 文字（root ::= [ -~]）に縛れば
  出る語は必ず完全な文字になり、確率が全問返る。確率はサンプラーを通す前の全語彙の物（同 2247 行 get_token_probabilities）
  なので、文法で縛っても返る上位 100 の確率は変わらない。cache_prompt を偽にする理由: 真だと前の依頼の使い回す量で、
  同じ入力でも確率が少し変わった（Swallow で 0.9133 と 0.9113）。確かめの依頼の設定は meta の "probe_gen" に残す。
  確率は post_sampling_probs が偽なので、サンプラーを通す前の、全語彙の softmax
  （b11836: tools/server/server-context.cpp populate_token_probs → server-common.cpp get_token_probabilities）。
- 型を当てた入力の語の数（書き出しを付ける前・/tokenize で add_special・parse_special とも真で数える。
  答えの依頼の usage.prompt_tokens と同じ作りの数）を "probe_prompt_tokens" に残す。書き出しの sha256 を "probe_prefix_sha256" に残す。

取り違えを防ぐ: 出力の meta がもうあって、prompt_version・manifest（問の指紋）・gen・max_tokens・smoke が今と違えば、
答えを足さずに終了コード 2 で止まる。途中まである jsonl の id が今の問の集まりに無い時も止まる。
--v7-smoke: 版 7 の見済みの 10 問（questions.smoke_ids_v7）だけを版 7 の問として流す（meta に "smoke": true）。
--out は本番の出力先と混ざらない別の場所にする。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

from .questions import (DATA_DIR_V7, PROMPT_VERSION, SETS_V7, build_messages, load_questions, load_rules, manifest,
                        smoke_ids_v7)

GEN = {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "min_p": 0.0, "seed": 1, "cache_prompt": False,
       "chat_template_kwargs": {"enable_thinking": False}}
PROBE_N = 100
PROBE_GRAMMAR = "root ::= [ -~]"  # ASCII の 1 文字だけを許す（頭の説明を見よ）
MARK = "結論:"
EXIT_MISMATCH = 2


def request_body(q, rules, max_tokens: int) -> dict:
    """1 問の答えの依頼（版 7 と同じ。確率は頼まない）。"""
    return {"messages": build_messages(q, rules), "max_tokens": max_tokens, **GEN}


def probe_prefix(text: str | None) -> str | None:
    """答えの文を、最後の「結論:」の直後で切る（「結論:」まで含む）。無ければ None。"""
    if not text:
        return None
    k = text.rfind(MARK)
    return None if k < 0 else text[: k + len(MARK)]


def apply_template_body(q, rules) -> dict:
    """/apply-template の依頼。messages・chat_template_kwargs は答えの依頼と同じ。"""
    return {"messages": build_messages(q, rules), "chat_template_kwargs": GEN["chat_template_kwargs"]}


def completion_body(prompt: str) -> dict:
    """/completion の依頼（確かめ）。1 語だけ・サンプラーを通す前の確率の上位 100。文法と cache_prompt の理由は頭の説明を見よ。"""
    return {"prompt": prompt, "n_predict": 1, "n_probs": PROBE_N, "post_sampling_probs": False,
            "temperature": 0.0, "top_k": 1, "seed": 1, "cache_prompt": False, "grammar": PROBE_GRAMMAR}


def probe_gen() -> dict:
    """meta に残す確かめの依頼の設定（completion_body から prompt を除いた物）。"""
    return {k: v for k, v in completion_body("").items() if k != "prompt"}


def probe_top(resp: dict) -> list[dict] | None:
    """/completion の返事から、最初の 1 語の上位を [{"id", "p"}, ...]（確率）にする。無ければ None。
    post_sampling_probs が偽なら "logprob"（自然対数）、真なら "prob"。"""
    cp = resp.get("completion_probabilities") or []
    if not cp:
        return None
    first = cp[0]
    top = first.get("top_logprobs")
    if top is None:
        top = first.get("top_probs")
    out = []
    for t in top or []:
        if "logprob" in t:
            lp = t["logprob"]
            p = math.exp(lp) if isinstance(lp, (int, float)) and lp < 700 else float("nan")
        else:
            p = t.get("prob")
        if t.get("id") is None or not isinstance(p, (int, float)) or isinstance(p, bool) or not math.isfinite(p):
            raise ValueError(f"確かめの返事の形が違う: {t!r}")
        out.append({"id": int(t["id"]), "p": float(p)})
    return out or None


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


def probe_one(server: str, q, rules, prefix: str, timeout: float) -> dict:
    """1 問の確かめ。返り値: 行に足す {probe, probe_prompt_tokens, probe_prefix_sha256}。"""
    templ = _post(f"{server}/apply-template", apply_template_body(q, rules), timeout)["prompt"]
    n_tok = len(_post(f"{server}/tokenize", {"content": templ, "add_special": True, "parse_special": True},
                      timeout)["tokens"])
    r = _post(f"{server}/completion", completion_body(templ + prefix), timeout)
    return {"probe": probe_top(r), "probe_prompt_tokens": n_tok,
            "probe_prefix_sha256": hashlib.sha256(prefix.encode("utf-8")).hexdigest()}


def current_meta_fields(max_tokens: int, smoke: bool) -> dict:
    """この回が今の問・設定で流している事を示す、meta の固定の欄。"""
    man = manifest(DATA_DIR_V7, SETS_V7) if smoke else manifest()
    return {"prompt_version": PROMPT_VERSION, "manifest": man, "gen": GEN, "max_tokens": max_tokens,
            "smoke": bool(smoke)}


def meta_mismatch(meta: dict, max_tokens: int, smoke: bool = False) -> list[str]:
    """記録の meta が今と違う欄の名前（空なら同じ）。smoke の欄が無い古い meta は偽として比べる。"""
    now = current_meta_fields(max_tokens, smoke)
    have = {**meta, "smoke": bool(meta.get("smoke", False))}
    return [k for k in now if have.get(k) != json.loads(json.dumps(now[k]))]


def read_meta(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def meta_path(out: Path) -> Path:
    return out.with_suffix(".meta.json")


def read_rows(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = r
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--ids", type=Path, help="この id だけ流す（1 行 1 id）。試し用")
    ap.add_argument("--limit", type=int, help="先頭からこの数だけ。試し用")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--probe-ref", type=Path, help="確かめの書き出しに使う、基準 1 回目の答えの jsonl")
    g.add_argument("--probe-self", action="store_true", help="確かめの書き出しに、その場で出た答えを使う（基準 1 回目の回）")
    ap.add_argument("--v7-smoke", action="store_true",
                    help="版 7 の見済みの 10 問だけを版 7 の問として流す（動作確認用・--out は別の場所にする）")
    a = ap.parse_args(argv)

    if a.v7_smoke:
        keep = set(smoke_ids_v7())
        qs = [q for q in load_questions(DATA_DIR_V7, SETS_V7) if q.id in keep]
        rules = load_rules(DATA_DIR_V7, SETS_V7)
    else:
        qs, rules = load_questions(), load_rules()
    universe = {q.id for q in qs}  # 今の問の集まり（--ids・--limit で絞る前）
    if a.ids:
        keep = set(a.ids.read_text(encoding="utf-8").split())
        qs = [q for q in qs if q.id in keep]
    if a.limit:
        qs = qs[: a.limit]
    ref_text: dict[str, str] | None = None
    if a.probe_ref:
        ref_text = {i: r.get("text") or "" for i, r in read_rows(a.probe_ref).items()}

    meta = meta_path(a.out)
    have = read_meta(meta)
    if have is not None:
        bad = meta_mismatch(have, a.max_tokens, a.v7_smoke)
        if bad:
            print(f"止まる: {meta} の {bad} が今と違う（別の問・設定の答えに足さない）", file=sys.stderr)
            return EXIT_MISMATCH
    rows_done = read_rows(a.out) if a.out.exists() else {}
    unknown = sorted(set(rows_done) - universe)
    if unknown:
        print(f"止まる: {a.out} に今の問の集まりに無い id がある（{unknown[:5]}…）", file=sys.stderr)
        return EXIT_MISMATCH

    wait_ready(a.server)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if have is None:
        try:
            props = _get(f"{a.server}/props")
        except Exception as e:  # 古い版では無い事がある
            props = {"error": str(e)}
        meta.write_text(json.dumps({
            **current_meta_fields(a.max_tokens, a.v7_smoke),
            "probe": "self" if a.probe_self else ("ref" if a.probe_ref else None),
            "probe_ref": str(a.probe_ref) if a.probe_ref else None, "probe_top": PROBE_N if (a.probe_self or a.probe_ref) else None,
            "probe_gen": probe_gen() if (a.probe_self or a.probe_ref) else None,
            "server_props": {k: props.get(k) for k in ("model_path", "build_info", "n_ctx", "total_slots",
                                                       "chat_template") if k in props},
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "n_questions": len(qs),
        }, ensure_ascii=False, indent=1), encoding="utf-8")
    todo = [q for q in qs if q.id not in rows_done]
    print(f"{len(rows_done)} 済み・残り {len(todo)}", file=sys.stderr)
    with a.out.open("a", encoding="utf-8") as f:
        for i, q in enumerate(todo, 1):
            body = request_body(q, rules, a.max_tokens)
            t0 = time.time()
            r = _post(f"{a.server}/v1/chat/completions", body, a.timeout)
            ch = r["choices"][0]
            row = {"id": q.id, "text": ch["message"].get("content") or "",
                   "reasoning": ch["message"].get("reasoning_content"),
                   "finish_reason": ch.get("finish_reason"), "usage": r.get("usage"),
                   "timings": r.get("timings"), "elapsed": round(time.time() - t0, 3)}
            if a.probe_self or ref_text is not None:
                prefix = probe_prefix(row["text"] if a.probe_self else ref_text.get(q.id))
                if prefix is None:
                    row.update({"probe": None, "probe_prompt_tokens": None, "probe_prefix_sha256": None})
                else:
                    row.update(probe_one(a.server, q, rules, prefix, a.timeout))
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 20 == 0 or i == len(todo):
                print(f"{i}/{len(todo)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
