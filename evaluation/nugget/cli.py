"""入口: python -m evaluation.nugget.cli {draft,freeze,judge,score,compare,report} ...

詳細は evaluation/NUGGET.md。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Sequence

from evaluation.nugget import NUGGET_VERSION_DEFAULT, draft as draft_mod, judge as judge_mod
from evaluation.nugget import report as report_mod, score as score_mod, stats
from evaluation.nugget.schema import (
    FreezeError,
    SchemaError,
    _read_jsonl,
    check_freeze,
    default_corpus_dir,
    load_answers,
    load_corpus,
    load_nuggets,
    sha256_lf,
    unreviewed_qids,
    validate_nuggets,
    write_freeze,
)


class CliError(RuntimeError):
    pass


# ─── 共通 ───────────────────────────────────────────────────────
def _corpus_for(args) -> dict[str, str]:
    d = Path(args.corpus) if getattr(args, "corpus", None) else default_corpus_dir(args.nuggets)
    if not d.is_dir():
        raise CliError(f"規程の corpus ディレクトリが無い: {d}")
    return load_corpus(d)


def _frozen_nuggets(nuggets_path: str) -> tuple[dict[str, dict], dict]:
    """freeze と合っていることを確かめて {qid: 行} を返す。合わなければ止める。"""
    info = check_freeze(nuggets_path)
    rows = load_nuggets(nuggets_path)
    return {r["qid"]: r for r in rows}, info


def load_judged_groups(
    groups: dict[str, Sequence[str]], nuggets_path: str
) -> tuple[dict[str, list[list[dict]]], dict]:
    """判定ファイル群を読む。nuggets の sha256 が現在と違う記録、判定の版・モデルの混在は止める。"""
    sha = sha256_lf(nuggets_path)
    out: dict[str, list[list[dict]]] = {}
    versions: set[tuple[str, str]] = set()
    for gname, paths in groups.items():
        runs = []
        for p in paths:
            recs = _read_jsonl(Path(p))
            for r in recs:
                if r.get("nuggets_sha256") != sha:
                    raise CliError(f"{p}: nuggets の sha256 が現在の nuggets.jsonl と違う（別の版で判定された記録）")
                versions.add((str(r.get("judge_version")), str(r.get("judge_model"))))
            runs.append(recs)
        out[gname] = runs
    if len(versions) > 1:
        raise CliError(f"判定の版・判定役のモデルが混在している: {sorted(versions)}")
    meta = {}
    if versions:
        v, m = next(iter(versions))
        meta = {"judge_version": v, "judge_model": m}
    return out, meta


def _emit(text: str, out: str | None) -> None:
    if out:
        Path(out).write_text(text, encoding="utf-8")
        print(f"書き出し: {out}")
    else:
        print(text)


# ─── draft ──────────────────────────────────────────────────────
def cmd_draft(args, send) -> int:
    corpus = load_corpus(args.corpus)
    questions = _read_jsonl(Path(args.questions))
    if send is None:
        send = judge_mod.make_http_send(args.base_url, args.model, args.seed, args.max_tokens)
    n_flag = 0
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for q in questions:
            qid = str(q.get("qid", q.get("id")))
            rule_id = q.get("rule_id")
            if rule_id not in corpus:
                raise CliError(f"{qid}: 規程 {rule_id} が corpus に無い")
            row = draft_mod.draft_question(
                send,
                qid=qid,
                question=q["question"],
                rule_id=rule_id,
                rule_text=corpus[rule_id],
                set_name=str(q.get("set", args.set or "")),
                drafted_by=args.model,
                version=args.version,
            )
            n_flag += sum(1 for n in row["nuggets"] if n.get("flags")) + (1 if row.get("flags") else 0)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"下書き {len(questions)} 問 → {args.out}（要確認の印つき {n_flag} 件。reviewed_by は空）")
    return 0


# ─── freeze ─────────────────────────────────────────────────────
def cmd_freeze(args, send) -> int:
    corpus = _corpus_for(args)
    rows = load_nuggets(args.nuggets)
    errors = validate_nuggets(rows, corpus)
    if errors:
        print("検査に通らない。freeze しない:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    un = unreviewed_qids(rows)
    if un:
        print(f"注意: reviewed_by が空の問が {len(un)} ある（{', '.join(un[:5])}…）", file=sys.stderr)
    info = write_freeze(args.nuggets, args.version)
    print(f"freeze: version={info['version']} sha256={info['sha256'][:16]}… {info['frozen_at']} ({info['n_questions']} 問)")
    return 0


# ─── judge ──────────────────────────────────────────────────────
def cmd_judge(args, send) -> int:
    by_qid, finfo = _frozen_nuggets(args.nuggets)
    corpus = _corpus_for(args)
    sha = sha256_lf(args.nuggets)
    if send is None:
        send = judge_mod.make_http_send(args.base_url, args.model, args.seed, args.max_tokens)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = bad = 0
    for ap in args.answers:
        answers = load_answers(ap)
        out_path = out_dir / (Path(ap).stem + ".judged.jsonl")
        skipped = 0
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            for a in answers:
                row = by_qid.get(a["qid"])
                if row is None:
                    skipped += 1
                    continue
                question = _question_of(row, args)
                if not question:
                    raise CliError(f"{a['qid']}: 質問文が無い（nuggets の question か --questions で渡す）")
                res = judge_mod.judge_answer(
                    send,
                    rule_text=corpus[row["rule_id"]],
                    question=question,
                    answer=a["text"],
                    nuggets=row["nuggets"],
                )
                rec = {
                    "qid": a["qid"],
                    "run": a["run"],
                    "answer": a["text"],
                    **res,
                    "judge_version": judge_mod.JUDGE_VERSION,
                    "judge_model": args.model,
                    "judge_seed": args.seed,
                    "nuggets_sha256": sha,
                    "nuggets_version": finfo.get("version"),
                    "source": str(ap),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total += 1
                bad += rec["status"] != "ok"
        print(f"{ap} → {out_path}（{len(answers) - skipped} 件判定, nuggets に無い qid {skipped} 件を飛ばした）")
    print(f"判定 {total} 件, 判定できず {bad} 件")
    return 2 if total and bad == total else 0


def _question_of(row: dict, args) -> str:
    """質問文: nuggets 行の question、無ければ --questions のファイルから。"""
    if row.get("question"):
        return row["question"]
    qs = getattr(args, "_questions_cache", None)
    if qs is None:
        qs = {}
        if getattr(args, "questions", None):
            for q in _read_jsonl(Path(args.questions)):
                qs[str(q.get("qid", q.get("id")))] = q.get("question", "")
        args._questions_cache = qs
    return qs.get(row["qid"], "")


# ─── score / compare / report ──────────────────────────────────
def _settings(args, groups: dict[str, Sequence[str]]):
    by_qid, finfo = _frozen_nuggets(args.nuggets)
    loaded, meta = load_judged_groups(groups, args.nuggets)
    settings = {g: score_mod.aggregate_setting(runs, by_qid) for g, runs in loaded.items()}
    meta = {**meta, "nuggets_version": finfo.get("version"), "nuggets_sha256": finfo["sha256"][:16] + "…"}
    return by_qid, settings, meta


def cmd_score(args, send) -> int:
    by_qid, settings, meta = _settings(args, {"A": args.judged})
    st = settings["A"]
    lines = ["| 問 | 規程 | s(V) | V_strict | 相反 | M6 | 使った回/K |", "|---|---|---|---|---|---|---|"]
    for q in sorted(st["items"]):
        i = st["items"][q]
        lines.append(
            f"| {q} | {report_mod.cell(i['rule_id'])} | {report_mod.fmt(i['s'])} | {report_mod.fmt(i['s_strict'])} "
            f"| {report_mod.fmt(i['c'], 2)} | {report_mod.fmt(i['m6'])} | {i['k_used']}/{i['k_total']} |"
        )
    summ = score_mod.summarize(st)
    text = report_mod.summary_table([(args.name or "A", summ)]) + "\n\n" + "\n".join(lines)
    if st["excluded"]:
        text += f"\n\n判定できず（点から外した問）: {', '.join(st['excluded'])}"
    if st["unknown"]:
        text += f"\n\nnuggets に無い qid（無視）: {', '.join(st['unknown'])}"
    _emit(text + "\n", args.out)
    return 0


def _compare_core(args):
    groups = {"A": args.a, "B": args.b}
    if args.aa:
        groups["AA"] = args.aa
    by_qid, settings, meta = _settings(args, groups)
    clusters = score_mod.clusters_of(by_qid) if args.cluster == "rule" else None
    sa, sb = settings["A"], settings["B"]
    cmp = stats.compare(score_mod.scores_of(sa), score_mod.scores_of(sb), clusters)
    aa = None
    if "AA" in settings:
        aa = stats.compare(score_mod.scores_of(sa), score_mod.scores_of(settings["AA"]), clusters)
    return by_qid, settings, meta, cmp, aa


def cmd_compare(args, send) -> int:
    by_qid, settings, meta, cmp, aa = _compare_core(args)
    if args.json:
        w = stats.aa_width(aa) if aa is not None else None
        payload = {
            "n": cmp["n"], "dbar": cmp["dbar"], "se_clt": cmp["se_clt"], "se_cluster": cmp["se_cluster"],
            "ci": cmp["ci"], "aa_width": w, "verdict": stats.verdict(cmp, w), **meta,
        }
        _emit(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", args.out)
    else:
        _emit(report_mod.compare_table(cmp, aa, args.name_a, args.name_b) + "\n", args.out)
    return 0


def cmd_report(args, send) -> int:
    if args.b:
        by_qid, settings, meta, cmp, aa = _compare_core(args)
    else:
        by_qid, settings, meta = _settings(args, {"A": args.a})
        cmp = aa = None
    use_cluster = args.cluster == "rule"
    sum_a = score_mod.summarize(settings["A"], use_cluster)
    sum_b = score_mod.summarize(settings["B"], use_cluster) if args.b else None
    worst = (
        report_mod.worst_table(settings["A"], settings["B"], by_qid, args.top) if args.b else None
    )
    text = report_mod.full_report(
        name_a=args.name_a, sum_a=sum_a, name_b=args.name_b, sum_b=sum_b,
        cmp=cmp, aa=aa, worst=worst, meta=meta,
    )
    _emit(text, args.out)
    return 0


# ─── 引数 ───────────────────────────────────────────────────────
def _add_http(p: argparse.ArgumentParser) -> None:
    p.add_argument("--base-url", default=judge_mod.DEFAULT_BASE_URL, help="OpenAI 互換の窓口（既定は localhost）")
    p.add_argument("--model", default="local", help="判定役のモデル名（記録に残る）")
    p.add_argument("--seed", type=int, default=judge_mod.DEFAULT_SEED)
    p.add_argument("--max-tokens", type=int, default=judge_mod.DEFAULT_MAX_TOKENS)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m evaluation.nugget.cli", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("draft", help="必須の事実の下書きを作る")
    p.add_argument("--questions", required=True, help='JSONL: {"qid"|"id","question","rule_id",任意"set"}')
    p.add_argument("--corpus", required=True, help="規程の .md が入ったディレクトリ")
    p.add_argument("--out", required=True, help="書き出す nuggets.jsonl")
    p.add_argument("--set", default="")
    p.add_argument("--version", default=NUGGET_VERSION_DEFAULT)
    _add_http(p)
    p.set_defaults(fn=cmd_draft)

    p = sub.add_parser("freeze", help="nuggets.jsonl を検査して sha256・時刻・版を freeze.json に残す")
    p.add_argument("--nuggets", required=True)
    p.add_argument("--corpus", help="既定は nuggets.jsonl と同じ階層の corpus/")
    p.add_argument("--version", help="既定は nuggets の version")
    p.set_defaults(fn=cmd_freeze)

    p = sub.add_parser("judge", help="答えのファイル → 判定の JSONL")
    p.add_argument("--nuggets", required=True)
    p.add_argument("--corpus")
    p.add_argument("--answers", nargs="+", required=True, help="答えの JSONL（1 ファイル = 1 回分）")
    p.add_argument("--questions", help="質問文の JSONL（nuggets の行に question が無い時）")
    p.add_argument("--out-dir", required=True)
    _add_http(p)
    p.set_defaults(fn=cmd_judge)

    p = sub.add_parser("score", help="1 設定（判定ファイル K 個）の点")
    p.add_argument("--nuggets", required=True)
    p.add_argument("--judged", nargs="+", required=True)
    p.add_argument("--name")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_score)

    for name, helptext in (("compare", "A と B の対応ありの比較"), ("report", "要約・比較・悪化した問をまとめて出す")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--nuggets", required=True)
        p.add_argument("--a", nargs="+", required=True, help="設定 A の判定ファイル群（K 回分）")
        p.add_argument("--b", nargs="+", required=(name == "compare"), help="設定 B の判定ファイル群")
        p.add_argument("--aa", nargs="+", help="設定 A を seed だけ替えて流し直した判定ファイル群（A/A）")
        p.add_argument("--cluster", choices=("rule", "none"), default="rule")
        p.add_argument("--name-a", default="A")
        p.add_argument("--name-b", default="B")
        p.add_argument("--out")
        if name == "compare":
            p.add_argument("--json", action="store_true")
            p.set_defaults(fn=cmd_compare)
        else:
            p.add_argument("--top", type=int, default=10, help="悪化した問を上から何問出すか")
            p.set_defaults(fn=cmd_report)
    return ap


def main(argv: Sequence[str] | None = None, send: Callable | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args, send)
    except (FreezeError, SchemaError, CliError) as e:
        print(f"止めた: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8")
    sys.exit(main())
