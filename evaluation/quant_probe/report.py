"""作業フォルダの答えと KLD の記録を集め、表と判定（judge.py）を出す。

    python -m evaluation.quant_probe.report --work <dir> --models qwen3-8b gemma4-12b swallow-8b --out <dir> [--trial]

本番（--trial なし）は厳しく見る: 答えの欠け・変種の不足・ずれが数でない・英日の比べ元の区切りが --min-chunks 未満・
変種の区切りが比べ元と違う・imatrix の区切りが --min-imatrix-chunks 未満なら、判定せずに止める。
--trial を付けると表の頭に「試し・結論にしない」と書き、厳しく見ず、全体の判定は出さない。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .judge import (EQUIV_MARGIN, JUDGE_VERSION, METRICS, PRIMARY, StrictError, judge_model,
                    judge_overall)
from .kld import parse_log
from .pipeline import ALL_VARIANTS, BF16IMX, PLACE, REF, RERUN, TEXTS
from .questions import PROMPT_VERSION, load_questions
from .score import score_rows, summarize

NOISE_OF = {RERUN: "Q4_K_M-en", PLACE: "Q6_K-none"}


def _rows(p: Path) -> dict[str, dict]:
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = r
    return out


def _metrics(mdir: Path, name: str) -> dict:
    m = {}
    for x in TEXTS:
        log = mdir / "logs" / f"{name}-kld-{x}.log"
        if log.exists():
            d = parse_log(log.read_text(encoding="utf-8", errors="replace"))
            m[f"kld_{x}"] = d
            if x in ("en", "ja"):
                m[f"ppl_{x}"] = d
    return m


def _strict_tokens(res: dict, min_chunks: int, min_imatrix_chunks: int) -> str | None:
    """処理した語の数の点検: 英・日の比べ元は min_chunks 以上、各変種は比べ元と同じ区切りの数、imatrix は所定の数。"""
    base = res["base_chunks"]
    for x in TEXTS:
        if base.get(x) is None:
            return f"比べ元 {x} の区切りが読めない"
        if x in ("en", "ja") and base[x] < min_chunks:
            return f"比べ元 {x} の区切りが {base[x]}（{min_chunks} 未満）"
    for n, m in res["metrics"].items():
        for x in TEXTS:
            c = (m.get(f"kld_{x}") or {}).get("chunks")
            if c != base[x]:
                return f"{n} の KLD {x} の区切りが {c}（比べ元は {base[x]}）"
    for lang in ("en", "ja"):
        c = res["imatrix_chunks"].get(lang)
        if c is None or c < min_imatrix_chunks:
            return f"imatrix {lang} の区切りが {c}（{min_imatrix_chunks} 未満か読めない）"
    return None


def collect_model(mdir: Path, ids: set[str] | None = None, strict: bool = False, min_chunks: int = 0,
                  min_imatrix_chunks: int = 0) -> dict:
    qs = [q for q in load_questions() if ids is None or q.id in ids]
    runs = {p.stem: _rows(p) for p in sorted((mdir / "runs").glob("*.jsonl"))}
    scored = {n: score_rows(qs, r) for n, r in runs.items()}
    summaries = {n: summarize(qs, s) for n, s in scored.items()}
    metrics = {n: _metrics(mdir, n) for n in [*ALL_VARIANTS, BF16IMX]
               if (mdir / "logs" / f"{n}-kld-en.log").exists() or n in runs}
    res = {"summaries": summaries, "metrics": metrics, "base_chunks": {}, "imatrix_chunks": {}}
    for x in TEXTS:
        log = mdir / "logs" / f"{REF}-base-{x}.log"
        if log.exists():
            res["base_chunks"][x] = parse_log(log.read_text(encoding="utf-8", errors="replace")).get("chunks")
    for lang in ("en", "ja", "en-bf16"):
        log = mdir / "logs" / f"imatrix-{lang}.log"
        if log.exists():
            res["imatrix_chunks"][lang] = parse_log(log.read_text(encoding="utf-8", errors="replace")).get("chunks")
    if strict:
        err = _strict_tokens(res, min_chunks, min_imatrix_chunks)
        if err:
            res["judge_error"] = err
            return res
    r1, r2 = f"{REF}-r1", f"{REF}-r2"
    if r1 in scored and r2 in scored:
        outcomes = {n: {k: v["outcome"] for k, v in s.items()} for n, s in scored.items()}
        variants = [{"name": n, "outcomes": outcomes[n], "metrics": metrics.get(n, {})}
                    for n in ALL_VARIANTS if n in outcomes]
        noise_pairs = [(n, outcomes[base], outcomes[n]) for n, base in NOISE_OF.items()
                       if n in outcomes and base in outcomes]
        try:
            res["judge"] = judge_model(outcomes[r1], outcomes[r2], variants,
                                       {q.id: q.should_hold for q in qs},
                                       {q.id: f"{q.set}/{q.rule_id}" for q in qs}, noise_pairs, strict)
        except (StrictError, ValueError) as e:
            res["judge_error"] = str(e)
    for f in ("timing.jsonl", "variants.jsonl"):
        p = mdir / f
        if p.exists():
            res[f.split(".")[0]] = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    off = mdir / "ref_offload.json"
    if off.exists():
        res["ref_offload"] = json.loads(off.read_text(encoding="utf-8"))["offloaded"]
    return res


def _fmt(x, nd=4):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def markdown(model: str, res: dict, trial: bool) -> str:
    L = [f"## {model}", ""]
    if trial:
        L += ["**試し。結論にしない。**", ""]
    L += ["| 回 | 誤った断定 | うち保留すべき問 | 正答 | 保留できた | 保留しすぎ | 形式違反 | 欠け | 打ち切り | 対の両方正答 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for n, s in res["summaries"].items():
        L.append(f"| {n} | {s['wrong_assert']} | {s['wrong_assert_hold']} | {s['correct']}/{s['answerable']} | "
                 f"{s['held_ok']}/{s['should_hold']} | {s['over_hold']} | {s['format']} | {s['missing']} | "
                 f"{s['truncated']} | {s['pairs_both_correct']}/{s['pairs']} |")
    L += ["", "| 変種 | KLD 英 | KLD 日 | KLD 規程 | ln PPL比 英 | ln PPL比 日 | Same top p 英 | 区切り 英 |",
          "|---|---|---|---|---|---|---|---|"]
    for n, m in res["metrics"].items():
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
        L += ["| 変種 | 誤った断定 | 増え | p（規程で入れ替え） | Holm | 悪化 | 95%上側 | 同じ（<" + str(EQUIV_MARGIN) + "） |",
              "|---|---|---|---|---|---|---|---|"]
        for n, d in j["variants"].items():
            L.append(f"| {n} | {d['wa']} | {d['delta']} | {d['p']:.4g} | {d['p_holm']:.4g} | "
                     f"{'はい' if d['worse'] else '-'} | {d['upper95']:.0f} | {'はい' if d['same'] else 'いいえ'} |")
        L += ["", "| 物差し | 判定 | 型 A | 型 B | 安全域で同じと言えない |", "|---|---|---|---|---|"]
        for s in METRICS:
            b = j["by_metric"][s]
            tag = "" if s == PRIMARY else "（探索）"
            L.append(f"| {s}{tag} | {b['verdict']} | {', '.join(b.get('pattern_a', [])) or '-'} | "
                     f"{', '.join(b.get('pattern_b', [])) or '-'} | {', '.join(b.get('safe_not_same', [])) or '-'} |")
    elif res.get("judge_error"):
        L += ["", f"判定なし: {res['judge_error']}"]
    if res.get("ref_offload"):
        L += ["", f"基準の GPU への載り: {res['ref_offload'][0]}/{res['ref_offload'][1]} 層"]
    if res.get("timing"):
        L += ["", "所要時間（秒）: " + "・".join(f"{t['step']} {t['sec']}" for t in res["timing"])]
    return "\n".join(L) + "\n"


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
    ids = set(a.ids.read_text(encoding="utf-8").split()) if a.ids else None
    a.out.mkdir(parents=True, exist_ok=True)
    allres = {m: collect_model(a.work / m, ids, strict=not a.trial, min_chunks=a.min_chunks,
                               min_imatrix_chunks=a.min_imatrix_chunks) for m in a.models}
    md = [f"# 量子化の物差し: 集計（{'試し' if a.trial else '本番'}）", "",
          f"渡す文 {PROMPT_VERSION}・判定 {JUDGE_VERSION}", ""]
    if not a.trial:
        judged = {m: r["judge"] for m, r in allres.items() if "judge" in r}
        md += ["| 物差し | 全体の答え |", "|---|---|"]
        md += [f"| {s}{'' if s == PRIMARY else '（探索）'} | {judge_overall(judged, s)} |" for s in METRICS] + [""]
    md += [markdown(m, r, a.trial) for m, r in allres.items()]
    (a.out / "summary.json").write_text(json.dumps(allres, ensure_ascii=False, indent=1), encoding="utf-8")
    (a.out / "README.md").write_text("\n".join(md), encoding="utf-8")
    print((a.out / "README.md").read_text(encoding="utf-8")[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
