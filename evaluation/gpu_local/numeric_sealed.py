"""封の問 (Issue #16) を流す入口。カード方式だけを、検索なし・封用のカードの置き場で流す。

手順 (docs/v2-rule-card-procedure.md) の流れ 3・6 に当たる。numeric_arms の direct と違う点:
- 流す前に freeze.json の指紋 (カードの全ファイル・unable.json・プロンプト・rule_cards.py) を全部照合し、違えば止める
- 封の本体を verify_numeric240 で照合し、draft のカードがあれば止める (--dev の時だけ緩める)
- カードに合わない問はモデルに回さず「答えなかった」と記録する (C には戻さない)
- 標準出力には件数と線の合否だけを出す。問の文・正解・値は出さない (raw.jsonl は --out に置く)

流し方 (repo のルートで・PYTHONUTF8=1・PYTHONPATH=backend):
    python -m evaluation.gpu_local.numeric_sealed freeze --cards <dir> --prompt <file> --out <freeze.json> [--allow-draft]
    python -m evaluation.gpu_local.numeric_sealed run --set <questions.jsonl> --freeze <freeze.json> --server <exe> \\
        --gguf <file> --out <dir> [--runs 2] [--dev] [--sealed-dir <dir>]
    python -m evaluation.gpu_local.numeric_sealed summary --out <dir>
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from evaluation.gpu_local import numeric_arms as na
from evaluation.gpu_local import step2_local as s2
from evaluation.colab import step05 as s5

REPO = Path(__file__).resolve().parents[2]
RULE_CARDS_PY = "backend/app/services/rule_cards.py"
UNABLE = "unable.json"
FREEZE_FORMAT = "numeric-sealed-freeze@1"

# Issue #16 本文の線。変えない
DECIDED_OVERALL = (9, 10)  # 回答できる問のうちカードが結論を出した割合 (切り上げ)
DECIDED_PER_RULE = (8, 10)
PAIR_RATIO = (9, 10)
WRONG_ASSERTIONS_MAX = 0

# 止まった理由 (answer_with_card の保留の文頭と対応する)
NO_CARD, EXTRACT_FAILED, VALIDATION_FAILED, UNKNOWN_HOLD = "no_card", "extract_failed", "validation_failed", "unknown_hold"
STOP_REASONS = (NO_CARD, EXTRACT_FAILED, VALIDATION_FAILED, UNKNOWN_HOLD)
_PREFIX_EXTRACT = "抽出に失敗した"
_PREFIX_VALIDATION = "抽出の検査に通らない"


# ---------------------------------------------------------------- 指紋 (freeze)


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO / q


def _store(p: Path) -> str:
    """repo の中なら repo からの相対 (/ 区切り)、外なら絶対パスで残す。"""
    p = Path(p).resolve()
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        return str(p)


def _dir_hashes(directory: Path) -> dict[str, str]:
    return {f.relative_to(directory).as_posix(): sha256_file(f) for f in sorted(directory.rglob("*")) if f.is_file()}


def _load_all(cards_dir: Path | str) -> list:
    """rule_cards.load_cards と同じ読み方 (形の検査・id の重複)。ただし unable.json はカードでないので除く。"""
    from app.services import rule_cards

    cards = [rule_cards.load_card(p) for p in sorted(Path(cards_dir).glob("*.json")) if p.name != UNABLE]
    if len({c.id for c in cards}) != len(cards):
        raise rule_cards.CardError(f"{cards_dir}: duplicate card ids")
    return cards


def load_cards_checked(cards_dir: Path | str, allow_draft: bool) -> list:
    """カードを読む。allow_draft=False で draft が 1 枚でもあれば止める (load_cards は黙って外すので、先に数える)。"""
    from app.services import rule_cards

    try:
        every = _load_all(cards_dir)
    except rule_cards.CardError as e:
        raise SystemExit(f"カードの読み込みで止まった: {e}") from e
    drafts = [c.id for c in every if c.status != "confirmed"]
    if drafts and not allow_draft:
        raise SystemExit(f"draft のカードがある ({len(drafts)} 枚): {', '.join(drafts)}")
    cards = [c for c in every if allow_draft or c.status == "confirmed"]  # load_cards(allow_draft=...) と同じ
    if not cards:
        raise SystemExit("カードが 1 枚も無い")
    return cards


def read_unable(cards_dir: Path) -> list[dict] | None:
    """作れなかった規程の記録 (unable.json)。無ければ None。形が違えば止める。"""
    p = cards_dir / UNABLE
    if not p.is_file():
        return None
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"{UNABLE} が JSON でない: {e}") from e
    if not (isinstance(doc, list) and all(isinstance(x, dict) and isinstance(x.get("rule"), str) and x["rule"]
                                          and isinstance(x.get("reason"), str) and x["reason"] for x in doc)):
        raise SystemExit(f'{UNABLE} は [{{"rule": 規程名, "reason": 理由}}] の形')
    return doc


def build_freeze(cards_dir: Path, prompt: Path, allow_draft: bool) -> dict:
    cards = load_cards_checked(cards_dir, allow_draft)
    unable = read_unable(cards_dir)
    return {
        "format": FREEZE_FORMAT,
        "allow_draft": allow_draft,
        "cards_dir": _store(cards_dir),
        "cards": [{"id": c.id, "status": c.status} for c in cards],
        "files": _dir_hashes(cards_dir),
        "unable": {"present": unable is not None, "rules": [x["rule"] for x in unable or []],
                   "sha256": sha256_file(cards_dir / UNABLE) if unable is not None else None},
        "prompt": {"path": _store(prompt), "sha256": sha256_file(prompt)},
        "rule_cards_py": {"path": RULE_CARDS_PY, "sha256": sha256_file(REPO / RULE_CARDS_PY)},
    }


def verify_freeze(freeze: dict) -> list[str]:
    """freeze の指紋と今のファイルを照合し、違う点の一覧を返す (空なら一致)。中身は返さない。"""
    problems: list[str] = []
    if freeze.get("format") != FREEZE_FORMAT:
        return ["freeze の形式が違う"]
    d = _resolve(freeze["cards_dir"])
    if not d.is_dir():
        problems.append("カードの置き場が無い")
    else:
        now = _dir_hashes(d)
        want = freeze["files"]
        for name in sorted(set(want) | set(now)):
            if name not in now:
                problems.append(f"カードの置き場に {name} が無い")
            elif name not in want:
                problems.append(f"カードの置き場に freeze に無い {name} がある")
            elif now[name] != want[name]:
                problems.append(f"{name} の sha256 が違う")
        u = freeze["unable"]
        if (d / UNABLE).is_file() != u["present"]:
            problems.append(f"{UNABLE} の有無が違う")
        elif u["present"] and sha256_file(d / UNABLE) != u["sha256"]:
            problems.append(f"{UNABLE} の sha256 が違う")
    for label, rec in (("プロンプト", freeze["prompt"]), (RULE_CARDS_PY, freeze["rule_cards_py"])):
        p = _resolve(rec["path"])
        if not p.is_file():
            problems.append(f"{label} のファイルが無い")
        elif sha256_file(p) != rec["sha256"]:
            problems.append(f"{label} の sha256 が違う")
    return problems


def verify_sealed(sealed_dir: str | None, set_path: Path) -> list[str]:
    """封の本体を verify_numeric240 で照合し、--set がその questions.jsonl と同じかも見る。"""
    from evaluation.v2.sealed import verify_numeric240 as v

    if not sealed_dir:
        return ["--dev でない時は --sealed-dir が要る"]
    manifest = v.load_manifest()
    problems = v.verify(Path(sealed_dir), manifest)
    if sha256_file(set_path) != manifest["questions_sha256"]:
        problems.append("--set の sha256 が封の questions_sha256 と違う")
    return problems


# ---------------------------------------------------------------- 1 回分を流す


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def stop_reason(decision: Any) -> str | None:
    """結論を出さなかった理由。結論が出ていれば None。"""
    if decision.conclusion is not None:
        return None
    reason = decision.hold_reason or ""
    if reason.startswith(_PREFIX_EXTRACT):
        return EXTRACT_FAILED
    if reason.startswith(_PREFIX_VALIDATION):
        return VALIDATION_FAILED
    return UNKNOWN_HOLD


def make_record(item: dict, index: int, run: int, card: Any, decision: Any, stats: dict, seconds: float) -> dict:
    """問 1 つの記録。欄が無い問でも落ちない。decision が None はカードに合わなかった問。"""
    rec = {"run": run, "id": str(item.get("id", f"row{index}")), "rule_id": item.get("rule_id"),
           "should_hold": bool(item.get("should_hold")), "expected_conclusion": item.get("expected_conclusion"),
           "pair_id": item.get("pair_id"), "group": item.get("group")}
    if decision is None:
        rec.update(outcome=NO_CARD, stop_reason=NO_CARD, card_id=None, conclusion=None, hold_reason=None)
        return rec
    reason = stop_reason(decision)
    rec.update(outcome="decided" if reason is None else "held", stop_reason=reason, card_id=card.id,
               conclusion=decision.conclusion, hold_reason=decision.hold_reason, raw=decision.raw,
               seconds=round(seconds, 3), **(stats or {}))
    return rec


async def sealed_round(rows: list[dict], cards: list, extractor: Any, run: int) -> list[dict]:
    """select_card → answer_with_card だけ。カードに合わない問はモデルに回さない。"""
    from app.services import rule_cards

    out = []
    for i, item in enumerate(rows):
        card = rule_cards.select_card(cards, str(item.get("question", "")))
        if card is None:
            out.append(make_record(item, i, run, None, None, {}, 0.0))
            continue
        t = time.perf_counter()
        _, d = await rule_cards.answer_with_card(card, item["question"], extractor)
        out.append(make_record(item, i, run, card, d, dict(getattr(extractor, "last_stats", {}) or {}),
                               time.perf_counter() - t))
    return out


# ---------------------------------------------------------------- 線の計算 (純粋な関数)


def ceil_ratio(n: int, ratio: tuple[int, int]) -> int:
    """n × 分子/分母 の切り上げ (小数を使わない)。"""
    return -(-n * ratio[0] // ratio[1])


def _correct(r: dict) -> bool:
    """正しい: 回答できる問はカードの結論が正解と同じ。保留すべき問はカードが保留した。答えなかった問は正しくない。"""
    if r.get("should_hold"):
        return r.get("outcome") == "held"
    return r.get("outcome") == "decided" and r.get("conclusion") == r.get("expected_conclusion")


def evaluate_round(records: list[dict], expected_n: int | None = None) -> dict:
    """1 回分の記録から線を計算する。答えなかった問は保留に数えず、分母からも外さない。"""
    answerable = [r for r in records if not r.get("should_hold")]
    must_hold = [r for r in records if r.get("should_hold")]
    wrong = [r for r in answerable if r.get("outcome") == "decided" and r.get("conclusion") != r.get("expected_conclusion")]
    wrong += [r for r in must_hold if r.get("outcome") == "decided"]
    held = sum(r.get("outcome") == "held" for r in must_hold)
    decided = sum(r.get("outcome") == "decided" for r in answerable)
    need = ceil_ratio(len(answerable), DECIDED_OVERALL)

    per_rule: dict[str, dict] = {}
    for r in records:
        s = per_rule.setdefault(str(r.get("rule_id") or "?"), {"answerable": 0, "decided": 0, "correct": 0, "wrong": 0,
                                                               "should_hold": 0, "held": 0})
        if r.get("should_hold"):
            s["should_hold"] += 1
            s["held"] += r.get("outcome") == "held"
            s["wrong"] += r.get("outcome") == "decided"
        else:
            s["answerable"] += 1
            s["decided"] += r.get("outcome") == "decided"
            s["correct"] += _correct(r)
            s["wrong"] += r.get("outcome") == "decided" and r.get("conclusion") != r.get("expected_conclusion")
    for s in per_rule.values():
        s["required"] = ceil_ratio(s["answerable"], DECIDED_PER_RULE)
        s["ok"] = s["decided"] >= s["required"]

    pairs: dict[str, list[dict]] = {}
    for r in records:
        if r.get("pair_id") not in (None, ""):
            pairs.setdefault(str(r["pair_id"]), []).append(r)
    pairs_ok = sum(len(m) >= 2 and all(_correct(r) for r in m) for m in pairs.values())
    pairs_need = ceil_ratio(len(pairs), PAIR_RATIO)

    reasons = {k: 0 for k in STOP_REASONS}
    for r in records:
        if r.get("stop_reason") in reasons:
            reasons[r["stop_reason"]] += 1

    lines = {
        "wrong_assertions_zero": len(wrong) <= WRONG_ASSERTIONS_MAX,
        "all_held": bool(must_hold) and held == len(must_hold),
        "decided_overall": bool(answerable) and decided >= need,
        "decided_per_rule": bool(per_rule) and all(s["ok"] for s in per_rule.values()),
        "pairs": bool(pairs) and pairs_ok >= pairs_need,
    }
    complete = expected_n is None or len(records) == expected_n
    return {"n": len(records), "complete": complete, "answerable": len(answerable), "should_hold": len(must_hold),
            "wrong_assertions": len(wrong), "held": held, "decided": decided, "decided_required": need,
            "pairs": len(pairs), "pairs_both_correct": pairs_ok, "pairs_required": pairs_need,
            "stop_reasons": reasons, "per_rule": dict(sorted(per_rule.items())), "lines": lines,
            "ok": complete and all(lines.values())}


def evaluate_all(records: list[dict], runs: int, expected_n: int | None = None) -> dict:
    """全部の回で満たして合格。回が足りない・途中で止まった回は不合格。"""
    rounds = {}
    for k in range(1, runs + 1):
        rounds[str(k)] = evaluate_round([r for r in records if r.get("run") == k], expected_n)
    return {"runs": runs, "rounds": rounds, "ok": runs > 0 and all(x["ok"] for x in rounds.values())}


# ---------------------------------------------------------------- コマンド


def cmd_freeze(a: argparse.Namespace) -> int:
    freeze = build_freeze(_resolve(a.cards), _resolve(a.prompt), a.allow_draft)
    Path(a.out).write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"freeze: カード {len(freeze['cards'])} 枚・ファイル {len(freeze['files'])} 個・"
          f"unable {len(freeze['unable']['rules'])} 規程・allow_draft={freeze['allow_draft']}")
    return 0


def _stop(problems: list[str]) -> None:
    raise SystemExit("止める:\n" + "\n".join(f" - {p}" for p in problems))


def cmd_run(a: argparse.Namespace) -> int:
    freeze_path = Path(a.freeze)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    problems = verify_freeze(freeze)
    if problems:
        _stop(problems)
    set_path = Path(getattr(a, "set"))
    if not a.dev:
        if freeze["allow_draft"]:
            _stop(["freeze が --allow-draft で作られている (--dev でない時は使えない)"])
        problems = verify_sealed(a.sealed_dir, set_path)
        if problems:
            _stop(problems)
    cards = load_cards_checked(_resolve(freeze["cards_dir"]), allow_draft=a.dev)
    rows = read_jsonl(set_path)
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit(f"{out} が空でない (前の結果と混ざるので止める)")

    na.ARGS = argparse.Namespace(server=a.server, gguf=a.gguf)  # start_server が読む
    mem, gpu_log = na.MemLog(), s2.GpuLog()
    baseline = s2.read_mib()
    rec: dict = {"format": "numeric-sealed-run@1", "dev": a.dev, "runs": a.runs, "n": len(rows),
                 "set_sha256": sha256_file(set_path), "freeze_sha256": sha256_file(freeze_path),
                 "unable_rules": freeze["unable"]["rules"], "ctx": na.DIRECT_CTX, "server_extra": na.SERVER_EXTRA,
                 "conditions": {"conclusion_options_given_first": True, "retrieval": False,
                                "no_card_goes_to_model": False},
                 "baseline_mib": baseline, "rounds": {}}
    code = 0
    try:
        for k in range(1, a.runs + 1):
            avail = mem.available_mib()
            if avail < a.min_available_mib:
                rec["stopped"] = f"{k} 回目の前の空き {avail} MiB < {a.min_available_mib} MiB"
                s2.log(rec["stopped"])
                code = 3
                return code
            s5.wait_gpu_released(s2.read_mib, baseline)
            t0 = time.time()
            log_path = out / f"llama_{k}.log"
            srv, _gpu = na.start_server(na.DIRECT_CTX, log_path)
            try:
                t1 = time.time()
                from app.services import rule_cards

                records = asyncio.run(sealed_round(rows, cards, rule_cards.OpenAICompatExtractor(na.BASE_URL, "local-llm"), k))
            finally:
                na.stop(srv)
            with open(out / "raw.jsonl", "a", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            scan = s5.scan_llama_log(log_path.read_text(encoding="utf-8", errors="replace"))
            rec["rounds"][str(k)] = {"available_before_mib": avail, "launch_seconds": round(t1 - t0, 1),
                                     "eval_seconds": round(time.time() - t1, 1), "oom_lines": len(scan["oom"]),
                                     "overflow_lines": len(scan["preempt"]),
                                     "peak_gpu_mib": s5.peak_in_window(gpu_log.samples, t0, time.time())["peak_mib"]}
            s2.log(f"[{k}/{a.runs}] {len(records)} 問・カードに合わず答えなかった "
                   f"{sum(r['outcome'] == NO_CARD for r in records)} 問")
        return code
    finally:
        rec.update(min_available_mib=mem.min_available_mib, peak_rss_mib=mem.peak_rss_mib)
        (out / "run.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_summary(a: argparse.Namespace) -> int:
    out = Path(a.out)
    run_path = out / "run.json"
    if not run_path.is_file():
        raise SystemExit(f"{run_path} が無い")
    rec = json.loads(run_path.read_text(encoding="utf-8"))
    raw = out / "raw.jsonl"
    records = read_jsonl(raw) if raw.is_file() else []
    doc = evaluate_all(records, int(rec["runs"]), int(rec["n"]))
    doc.update(issue=16, dev=rec.get("dev"), stopped=rec.get("stopped"), unable_rules=rec.get("unable_rules", []),
               conditions=rec.get("conditions"), set_sha256=rec.get("set_sha256"), freeze_sha256=rec.get("freeze_sha256"),
               lines_definition={"wrong_assertions_max": WRONG_ASSERTIONS_MAX, "hold": "全部 (答えなかった問は数えない)",
                                 "decided_overall": "9/10 以上 (切り上げ)", "decided_per_rule": "8/10 以上 (切り上げ)",
                                 "pairs": "9/10 以上 (切り上げ)"})
    if rec.get("stopped"):
        doc["ok"] = False
    (out / "summary.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for k, r in doc["rounds"].items():
        ng = [name for name, ok in r["lines"].items() if not ok]
        print(f"回 {k}: {r['n']} 問{'' if r['complete'] else '(不足)'}・誤った断定 {r['wrong_assertions']}・"
              f"保留 {r['held']}/{r['should_hold']}・結論 {r['decided']}/{r['answerable']} (要 {r['decided_required']})・"
              f"対 {r['pairs_both_correct']}/{r['pairs']} → {'合格' if r['ok'] else '不合格 ' + ','.join(ng)}")
    print("全体: " + ("合格" if doc["ok"] else "不合格"))
    return 0 if doc["ok"] else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--cards", required=True)
    f.add_argument("--prompt", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--allow-draft", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--set", required=True, help="問の jsonl")
    r.add_argument("--freeze", required=True)
    r.add_argument("--server", required=True, help="llama-server の場所")
    r.add_argument("--gguf", required=True, help="重みの場所")
    r.add_argument("--out", required=True)
    r.add_argument("--runs", type=int, default=2)
    r.add_argument("--dev", action="store_true", help="開発用の問。封の照合と draft の禁止を外す")
    r.add_argument("--sealed-dir", help="封の本体のフォルダ (--dev でない時に要る)")
    r.add_argument("--min-available-mib", type=int, default=4096, help="各回の前に要る本体のメモリの空き。足りなければ流さない")
    s = sub.add_parser("summary")
    s.add_argument("--out", required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    return {"freeze": cmd_freeze, "run": cmd_run, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
