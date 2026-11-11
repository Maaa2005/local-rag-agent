"""作業フォルダの答えと KLD の記録を集め、表と判定（judge.py）を出す。

    python -m evaluation.quant_probe.report --work <dir> --models qwen3-8b gemma4-12b swallow-8b --out <dir> [--trial]

本番（--trial なし）は厳しく見る（計画 5 章の 11）。モデルごとに先に状態を決める:
- 変換不能: 作業フォルダに unconvertible.txt（理由）がある。答えを流す前に決めた物だけ。替え（qwen3-14b）を 1 回使える。
- 基準不適格: 基準 2 回が 360 問そろい、線 0 を割った。そのモデルは「測れない」で、変種は見ない・替えない。
- 適格: 基準が線 0 を満たした。変種・揺れの組・KLD・imatrix・基準の載りを全部点検し、欠けがあれば「データ欠損」。
データ欠損のモデルが 1 つでもあれば、全体の答えは出さない。決めた 3 モデル（替えを含む）でない時も出さない。
本番は区切りの下限（KLD 80・imatrix 128）を下げられない。区切りは予定の数と済んだ数と終わりの印を全部見る。
--trial を付けると表の頭に「試し・結論にしない」と書き、厳しく見ず、全体の判定は出さない。
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from .judge import (JUDGE_VERSION, METRICS, MIN_DELTA, PRIMARY, StrictError, judge_model,
                    judge_overall, quality, ref_ok)
from .kld import parse_log
from .pipeline import ALL_VARIANTS, BF16IMX, PLACE, REF, RERUN, TEXTS
from .questions import PROMPT_VERSION, load_questions
from .score import score_rows, summarize

NOISE_OF = {RERUN: "Q4_K_M-en", PLACE: "Q6_K-none"}
PLANNED = ("qwen3-8b", "swallow-8b", "gemma4-12b")
SPARE = "qwen3-14b"
UNCONVERTIBLE = "unconvertible.txt"
MAIN_MIN_CHUNKS = 80
MAIN_MIN_IMATRIX_CHUNKS = 128


def _rows(p: Path) -> dict[str, dict]:
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = r
    return out


def _log(path: Path) -> dict | None:
    return parse_log(path.read_text(encoding="utf-8", errors="replace")) if path.exists() else None


def _metrics(mdir: Path, name: str) -> dict:
    m = {}
    for x in TEXTS:
        d = _log(mdir / "logs" / f"{name}-kld-{x}.log")
        if d is not None:
            m[f"kld_{x}"] = d
            if x in ("en", "ja"):
                m[f"ppl_{x}"] = d
    return m


def _done(d: dict | None) -> str | None:
    """区切りが予定どおり最後まで済んだか。済んでいなければ理由。"""
    if not d or d.get("chunks") is None:
        return "区切りが読めない"
    if not d.get("complete") or d.get("chunks_done") != d["chunks"]:
        return f"最後まで済んでいない（予定 {d['chunks']}・済み {d.get('chunks_done')}）"
    return None


def _strict_tokens(res: dict, min_chunks: int, min_imatrix_chunks: int) -> str | None:
    """処理した語の数の点検: 英・日の比べ元は min_chunks 以上、各変種は比べ元と同じ区切りの数、imatrix は所定の数。
    どれも予定の区切りが最後まで済み、終わりの印があること。"""
    base = res["base_logs"]
    for x in TEXTS:
        why = _done(base.get(x))
        if why:
            return f"比べ元 {x} の{why}"
        if x in ("en", "ja") and base[x]["chunks"] < min_chunks:
            return f"比べ元 {x} の区切りが {base[x]['chunks']}（{min_chunks} 未満）"
    for n, m in res["metrics"].items():
        for x in TEXTS:
            d = m.get(f"kld_{x}")
            why = _done(d)
            if why or d["chunks"] != base[x]["chunks"]:
                return f"{n} の KLD {x} の{why or '区切りが ' + str(d['chunks'])}（比べ元は {base[x]['chunks']}）"
    for lang in ("en", "ja"):
        d = res["imatrix_logs"].get(lang)
        why = _done(d)
        if why or d["chunks"] < min_imatrix_chunks:
            return f"imatrix {lang} の{why or '区切りが ' + str(d['chunks'])}（{min_imatrix_chunks} 以上が要る）"
    return None


READ_ERRORS = (OSError, ValueError, KeyError, TypeError)


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _offload(x) -> list[int]:
    """基準の載りの記録 [載った層・全部の層] の形を確かめる。違えば ValueError。"""
    if not (isinstance(x, list) and len(x) == 2 and all(type(v) is int for v in x) and x[1] > 0 and 0 <= x[0] <= x[1]):
        raise ValueError(f"基準の載りの記録の形が違う（{x!r}・整数 2 つで 0 <= 載った <= 全部・全部 > 0 のはず）")
    return x


def _items(rows: list, need: dict[str, tuple], name: str) -> list[dict]:
    """時間・変種の記録の行の形を確かめる。違えば ValueError。"""
    for r in rows:
        if not isinstance(r, dict) or any(not check(r.get(k)) for k, check in need.items()):
            raise ValueError(f"{name} の行の形が違う（{r!r}）")
    return rows


# 秒は有限で 0 以上・指紋は 16 進 64 文字・大きさは正の整数（Astra の 5 回目の点検）。
# この 2 つは判定には使わない記録なので、無いことは止める理由にしない（形を確かめた ≠ 全部そろった）。
NEED = {"timing.jsonl": {"step": lambda v: isinstance(v, str) and bool(v),
                         "sec": lambda v: _num(v) and math.isfinite(v) and v >= 0},
        "variants.jsonl": {"name": lambda v: isinstance(v, str) and bool(v),
                           "sha256": lambda v: isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) is not None,
                           "bytes": lambda v: type(v) is int and v > 0}}


def _read_rest(mdir: Path, qs, res: dict, skip: set[str]) -> dict[str, dict]:
    """基準の 2 回のほかの記録（変種の答え・KLD・比べ元・imatrix・載り・時間）を読む。変種の採点を返す。"""
    runs = {p.stem: _rows(p) for p in sorted((mdir / "runs").glob("*.jsonl")) if p.stem not in skip}
    scored = {n: score_rows(qs, r) for n, r in runs.items()}
    res["summaries"].update({n: summarize(qs, sc) for n, sc in scored.items()})
    res["metrics"] = {n: _metrics(mdir, n) for n in [*ALL_VARIANTS, BF16IMX]
                      if (mdir / "logs" / f"{n}-kld-en.log").exists() or n in runs}
    for x in TEXTS:
        d = _log(mdir / "logs" / f"{REF}-base-{x}.log")
        if d is not None:
            res["base_logs"][x] = d
    for lang in ("en", "ja", "en-bf16"):
        d = _log(mdir / "logs" / f"imatrix-{lang}.log")
        if d is not None:
            res["imatrix_logs"][lang] = d
    res["base_chunks"] = {x: d.get("chunks") for x, d in res["base_logs"].items()}
    res["imatrix_chunks"] = {x: d.get("chunks") for x, d in res["imatrix_logs"].items()}
    off = mdir / "ref_offload.json"
    if off.exists():
        res["ref_offload"] = _offload(json.loads(off.read_text(encoding="utf-8"))["offloaded"])
    for f in ("timing.jsonl", "variants.jsonl"):
        fp = mdir / f
        if fp.exists():
            rows = [json.loads(l) for l in fp.read_text(encoding="utf-8").splitlines() if l.strip()]
            res[f.split(".")[0]] = _items(rows, NEED[f], f)  # 確かめてから入れる（壊れた行は表に使わない）
    return scored


def collect_model(mdir: Path, ids: set[str] | None = None, strict: bool = False, min_chunks: int = 0,
                  min_imatrix_chunks: int = 0) -> dict:
    """読む順番（計画 5 章の 11）: 変換不能の印 → 基準の 2 回だけ → 360 問 → 基準不適格なら返す → 残りの記録 →
    区切りと終わりの印 → 載り → 判定。読めない記録はデータ欠損にする。本番（strict）は区切りの下限を下げない。"""
    qs = [q for q in load_questions() if ids is None or q.id in ids]
    if strict:
        min_chunks = max(min_chunks, MAIN_MIN_CHUNKS)
        min_imatrix_chunks = max(min_imatrix_chunks, MAIN_MIN_IMATRIX_CHUNKS)
    answered = sorted((mdir / "runs").glob("*.jsonl"))
    if strict and (mdir / UNCONVERTIBLE).exists():
        try:
            reason = (mdir / UNCONVERTIBLE).read_text(encoding="utf-8").strip()
        except READ_ERRORS as e:
            return {"status": "データ欠損", "summaries": {}, "metrics": {},
                    "judge_error": f"変換不能の印が読めない（{type(e).__name__}）"}
        if answered:
            return {"status": "データ欠損", "reason": reason, "summaries": {}, "metrics": {},
                    "judge_error": f"変換不能の印があるのに答えの記録が {len(answered)} 本ある（答えを流した後の替えは受け付けない）"}
        return {"status": "変換不能", "reason": reason}
    res = {"summaries": {}, "metrics": {}, "base_logs": {}, "imatrix_logs": {}}

    def lost(why: str) -> dict:
        res["status"], res["judge_error"] = "データ欠損", why
        return res

    r1, r2 = f"{REF}-r1", f"{REF}-r2"
    hold = {q.id: q.should_hold for q in qs}
    try:
        ref_rows = {r: _rows(mdir / "runs" / f"{r}.jsonl") for r in (r1, r2) if (mdir / "runs" / f"{r}.jsonl").exists()}
        if len(ref_rows) < 2:
            return lost("基準の 2 回の答えが無い")
        scored = {r: score_rows(qs, rows) for r, rows in ref_rows.items()}
    except READ_ERRORS as e:
        return lost(f"基準の答えが読めない（{type(e).__name__}: {e}）")
    res["summaries"] = {r: summarize(qs, sc) for r, sc in scored.items()}
    outcomes = {n: {k: v["outcome"] for k, v in sc.items()} for n, sc in scored.items()}
    if strict:
        for r in (r1, r2):
            miss = sum(v == "missing" for v in outcomes[r].values())
            if len(outcomes[r]) != 360 or miss:
                return lost(f"{r} の答えが 360 問そろっていない（欠け {miss}）")
        q12 = [quality(outcomes[r1], hold), quality(outcomes[r2], hold)]
        if not (ref_ok(q12[0]) and ref_ok(q12[1])):
            # 基準不適格: 変種の記録は読まない。全体ではこのモデルを「測れない」に数える
            res["status"] = "基準不適格"
            res["judge"] = {"version": JUDGE_VERSION, "valid": False, "ref_quality": q12,
                            "by_metric": {s: {"verdict": "測れない"} for s in METRICS}}
            return res
    try:
        scored_v = _read_rest(mdir, qs, res, {r1, r2})
    except READ_ERRORS as e:
        return lost(f"記録が読めない（{type(e).__name__}: {e}）")
    outcomes.update({n: {k: v["outcome"] for k, v in sc.items()} for n, sc in scored_v.items()})
    if strict:
        err = _strict_tokens(res, min_chunks, min_imatrix_chunks)
        if err is None and not res.get("ref_offload"):
            err = "基準の GPU への載りが読めない（ref_offload.json）"
        if err:
            return lost(err)
    need_place = bool(res.get("ref_offload")) and res["ref_offload"][0] < res["ref_offload"][1]
    variants = [{"name": n, "outcomes": outcomes[n], "metrics": res["metrics"].get(n, {})}
                for n in ALL_VARIANTS if n in outcomes]
    noise_pairs = [(n, outcomes[base], outcomes[n]) for n, base in NOISE_OF.items()
                   if n in outcomes and base in outcomes]
    try:
        res["judge"] = judge_model(outcomes[r1], outcomes[r2], variants, hold,
                                   {q.id: f"{q.set}/{q.rule_id}" for q in qs}, noise_pairs, strict, need_place)
        res["status"] = "適格" if res["judge"]["valid"] else "基準不適格"
    except (StrictError, ValueError) as e:
        return lost(str(e))
    return res


def _fmt(x, nd=4):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def markdown(model: str, res: dict, trial: bool) -> str:
    L = [f"## {model}", ""]
    if trial:
        L += ["**試し。結論にしない。**", ""]
    st = res.get("status")
    if st == "変換不能":
        return "\n".join(L + [f"状態: 変換不能（{res.get('reason', '')}）・替えを使う"]) + "\n"
    if st:
        L += [f"状態: {st}", ""]
    weak = st == "基準不適格" and "variants" not in (res.get("judge") or {})
    L += ["| 回 | 誤った断定 | うち保留すべき問 | 正答 | 保留できた | 保留しすぎ | 形式違反 | 欠け | 打ち切り | 対の両方正答 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for n, s in res.get("summaries", {}).items():
        if weak and not n.startswith(REF):
            continue  # 基準不適格: 変種の答えは見ない
        L.append(f"| {n} | {s['wrong_assert']} | {s['wrong_assert_hold']} | {s['correct']}/{s['answerable']} | "
                 f"{s['held_ok']}/{s['should_hold']} | {s['over_hold']} | {s['format']} | {s['missing']} | "
                 f"{s['truncated']} | {s['pairs_both_correct']}/{s['pairs']} |")
    if weak:
        q = res["judge"]["ref_quality"]
        L += ["", "基準が線 0 を割った（変種は見ない）: " + "・".join(
            f"{i + 1} 回目 正答 {x['correct']:.3f} 保留 {x['held']:.3f} 形式 {x['format']:.3f}" for i, x in enumerate(q))]
        return "\n".join(L) + "\n"
    L += ["", "| 変種 | KLD 英 | KLD 日 | KLD 規程 | ln PPL比 英 | ln PPL比 日 | Same top p 英 | 区切り 英 |",
          "|---|---|---|---|---|---|---|---|"]
    for n, m in res.get("metrics", {}).items():
        g = lambda k, f: (m.get(k) or {}).get(f)
        L.append(f"| {n} | {_fmt(g('kld_en', 'kld_mean'))} | {_fmt(g('kld_ja', 'kld_mean'))} | "
                 f"{_fmt(g('kld_dom', 'kld_mean'))} | {_fmt(g('ppl_en', 'ln_ppl_ratio'))} | "
                 f"{_fmt(g('ppl_ja', 'ln_ppl_ratio'))} | {_fmt(g('kld_en', 'same_top_p'), 2)} | "
                 f"{_fmt(g('kld_en', 'chunks'))} |")
    j = res.get("judge")
    if j:
        q1 = j["ref_quality"][0]
        L += ["", f"基準の誤った断定 {j['ref_wa']}・揺れ {j['noise']}（{j['noise_each']}）・基準 1 回目 正答 {q1['correct']:.3f}"
              f" 保留 {q1['held']:.3f} 形式 {q1['format']:.3f}・測れる {'はい' if j['valid'] else 'いいえ'}"
              f"・悪化した変種 {', '.join(j['worse']) or 'なし'}・応答不能が増えた {', '.join(j['unresponsive']) or 'なし'}", ""]
        L += ["| 変種 | 誤った断定 | 増え | p（規程で入れ替え） | Holm | 悪化 | 小さい（<" + str(MIN_DELTA) + "） | 答えている | 95%上側（参考） |",
              "|---|---|---|---|---|---|---|---|---|"]
        for n, d in j["variants"].items():
            L.append(f"| {n} | {d['wa']} | {d['delta']} | {d['p']:.4g} | {d['p_holm']:.4g} | "
                     f"{'はい' if d['worse'] else '-'} | {'はい' if d['small'] else 'いいえ'} | "
                     f"{'はい' if d['responsive'] else 'いいえ'} | {_fmt(d['upper95_ref'], 0)} |")
        L += ["", "| 物差し | 判定 | 型 A | 型 B | 安全域で残る | 近い対で差が残る |", "|---|---|---|---|---|---|"]
        for s in METRICS:
            b = j["by_metric"][s]
            tag = "" if s == PRIMARY else "（探索）"
            L.append(f"| {s}{tag} | {b['verdict']} | {', '.join(b.get('pattern_a', [])) or '-'} | "
                     f"{', '.join(b.get('pattern_b', [])) or '-'} | {', '.join(b.get('safe_left', [])) or '-'} | "
                     f"{', '.join(b.get('near_left', [])) or '-'} |")
    elif res.get("judge_error"):
        L += ["", f"判定なし（データ欠損）: {res['judge_error']}"]
    if res.get("ref_offload"):
        L += ["", f"基準の GPU への載り: {res['ref_offload'][0]}/{res['ref_offload'][1]} 層"]
    if res.get("timing"):
        L += ["", "所要時間（秒）: " + "・".join(f"{t['step']} {t['sec']}" for t in res["timing"])]
    return "\n".join(L) + "\n"


def overall_block(allres: dict[str, dict]) -> str | None:
    """全体の答えを出してよいか。出せない理由を返す（出せる時は None）。"""
    names = set(allres)
    if not set(PLANNED) <= names or names - set(PLANNED) - {SPARE}:
        return f"決めたモデルと違う（{sorted(names)}・決めたのは {list(PLANNED)}、替えは {SPARE} だけ）"
    gone = [m for m in PLANNED if allres[m].get("status") == "変換不能"]
    if len(gone) > 1:
        return f"変換不能が 2 つ以上（{gone}）"
    if bool(gone) != (SPARE in names):
        return f"{SPARE} は、決めた 3 つのうち変換不能がちょうど 1 つの時だけ使う（変換不能 {gone}）"
    use = [m for m in (*PLANNED, SPARE) if m in names and m not in gone]
    bad = [f"{m}: {allres[m].get('judge_error', '判定が無い')}" for m in use
           if allres[m].get("status") not in ("適格", "基準不適格") or "judge" not in allres[m]]
    if bad:
        return "データ欠損（" + "・".join(bad) + "）"
    return None


def overall_models(allres: dict[str, dict]) -> dict[str, dict]:
    """全体の答えに使うモデル（変換不能の 1 つがあれば替えを入れる）。基準不適格は「測れない」として残す。"""
    return {m: allres[m]["judge"] for m in (*PLANNED, SPARE)
            if m in allres and allres[m].get("status") != "変換不能"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--ids", type=Path)
    ap.add_argument("--trial", action="store_true")
    ap.add_argument("--min-chunks", type=int, default=80)
    ap.add_argument("--min-imatrix-chunks", type=int, default=128)
    a = ap.parse_args(argv)
    if not a.trial and (a.min_chunks < MAIN_MIN_CHUNKS or a.min_imatrix_chunks < MAIN_MIN_IMATRIX_CHUNKS):
        ap.error(f"本番は --min-chunks {MAIN_MIN_CHUNKS}・--min-imatrix-chunks {MAIN_MIN_IMATRIX_CHUNKS} より下げられない")
    ids = set(a.ids.read_text(encoding="utf-8").split()) if a.ids else None
    a.out.mkdir(parents=True, exist_ok=True)
    allres = {m: collect_model(a.work / m, ids, strict=not a.trial, min_chunks=a.min_chunks,
                               min_imatrix_chunks=a.min_imatrix_chunks) for m in a.models}
    md = [f"# 量子化の物差し: 集計（{'試し' if a.trial else '本番'}）", "",
          f"渡す文 {PROMPT_VERSION}・判定 {JUDGE_VERSION}", ""]
    if not a.trial:
        why = overall_block(allres)
        if why:
            md += [f"全体の答え: 出さない（{why}）", ""]
        else:
            judged = overall_models(allres)
            md += ["| 物差し | 全体の答え |", "|---|---|"]
            md += [f"| {s}{'' if s == PRIMARY else '（探索）'} | {judge_overall(judged, s)} |" for s in METRICS]
            md += ["", "基準不適格（線 0 を割った・測れない）: " + (", ".join(m for m, j in judged.items() if not j["valid"]) or "なし"),
                   "変換不能（替えを使った）: " + (", ".join(m for m in PLANNED if allres[m].get("status") == "変換不能") or "なし"),
                   f"主の物差し（{PRIMARY}）の「はい」だけ、計画 5 章の前提（各比較の帰無の下で、固定したモデル・文章・KLD を条件に、規程ごとの差の揺れが 0 の回りで対称・規程の間は独立）の下で、"
                   "誤りの確率 0.05 以下。探索の物差しには保証が無い。「いいえ」はこの 360 問の上での記述で、保証は無い。", ""]
    md += [markdown(m, r, a.trial) for m, r in allres.items()]
    (a.out / "summary.json").write_text(json.dumps(allres, ensure_ascii=False, indent=1), encoding="utf-8")
    (a.out / "README.md").write_text("\n".join(md), encoding="utf-8")
    print((a.out / "README.md").read_text(encoding="utf-8")[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
