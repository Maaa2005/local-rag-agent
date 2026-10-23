"""人が、保存済みの RAG 回答を「文ごとに 0/1」で採点するための表の生成と集計 (M4-S)。

本人が、モデル名・速さ・既存の点を見ずに、順を混ぜた回答を採点する。
- make : raw.jsonl から sheet.csv (1 行 1 文) と read.md (質問・取得した本文・回答の文) を作る。
         N の問 (expected_conclusion あり) は表に入れず、対応表の excluded_numeric_boundary に残す。
         run_label・モデル名・候補名・時間・点・問 ID・ファイル名は表にも read.md にも出さない。
         対応表 (--key-out) は --out-dir の外に置く。空/失敗の回答は auto_zero として対応表にだけ入れる。
- score: 記入済みの sheet.csv と対応表から run_label ごとの平均を出す (auto_zero は 0 点として含める)。

定義: 1 問の点は文の判定 (0/1) の平均。1 回 (run_label) の点はその回の本番の問 (50 問) の平均。0.90 以上で合格。

CLI (repo ルートから):
    python -m evaluation.human_sheet make --raw A/raw.jsonl [--raw B/raw.jsonl] --seed 1 \\
        --out-dir sheet_dir --key-out key.json
    python -m evaluation.human_sheet score --sheet sheet_dir/sheet.csv --key key.json --out score.json
終了コード: score は 0=すべて pass / 1=どれかが fail / 3=採点未完了 / 2=表や対応表の不整合・引数の誤り。
make は 0=作成 / 2=拒否。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import unicodedata
from fractions import Fraction
from pathlib import Path

from evaluation import rejudge  # noqa: E402
from evaluation.metrics import split_sentences  # noqa: E402

PASS_THRESHOLD = Fraction(9, 10)
COLUMNS = ["回答ID", "文番号", "文", "判定", "理由"]
PENDING = "保留"
MAX_ANSWERS = 200
SHEET_NAME = "sheet.csv"
READ_NAME = "read.md"
INCOMPLETE = "採点未完了"
EXIT = {"pass": 0, "fail": 1, INCOMPLETE: 3}
REFUSED = 2


class SheetError(Exception):
    """入力の不整合。終了コード 2。"""


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cell(sentence: str) -> str:
    """表の「文」列の値。「」で囲み、表計算ソフトが「1.」を数に、「=…」を式に変えないようにする。"""
    return f"「{sentence}」"


def _norm(text: str) -> str:
    """Numbers/Excel を通すと変わりうる改行の違いだけをならす。"""
    return text.replace("\r\n", "\n").replace("\r", "\n")


# ─── make ──────────────────────────────────────────────────────
def collect(raws: list[Path]) -> tuple[list[dict], list[dict]]:
    """(本番の回答, 外した N の問) を、raw の順・ファイル内の順に集める。"""
    items: list[dict] = []
    excluded: list[dict] = []
    seen: dict[tuple[str, str], str] = {}
    for raw in raws:
        for label, all_results in rejudge.load_saved_results(raw).items():
            for r in all_results:
                k = (label, r.id)
                if k in seen:
                    raise SheetError(f"run_label {label!r} の問 {r.id} が {seen[k]} と {raw} の両方にある")
                seen[k] = str(raw)
            results, n_ids = rejudge.split_numeric_boundary(all_results)
            excluded += [{"raw": str(raw), "run_label": label, "question_id": q} for q in n_ids]
            for r in results:
                sentences = split_sentences(r.answer or "")
                reason = None
                if r.error:
                    reason = f"run_failed: {r.error}"
                elif not sentences:
                    reason = "empty_answer"
                items.append({
                    "raw": str(raw), "run_label": label, "question_id": r.id,
                    "question": r.question, "chunks": r.retrieved_chunks,
                    "sentences": sentences, "auto_zero_reason": reason,
                })
    return items, excluded


def _read_md(shown: list[tuple[str, dict]]) -> str:
    lines = [
        "# 回答の採点",
        "",
        "記入のしかた (sheet.csv の「判定」「理由」列に書く):",
        "",
        "- 判定は 0 か 1。その文の事実・条件・数値・結論のすべてが「取得した本文」に書かれていれば 1。",
        "  一つでも本文と矛盾する部分、または本文に根拠がない部分があれば 0。一般知識や推測で補わない。",
        "- 「分かりません」「お答えできません」のように断る文は、本文にその答えがあれば 0、なければ 1。",
        "- 0 にした文には理由が必須。",
        "- 迷ったら「保留」と書く。保留が残る間は採点未完了。",
        "- 回答ID と 文番号 と 文 は書き換えない (行の追加・削除もしない)。文は「」で囲んである。",
        "- 表計算ソフトで書いたら、文字コード UTF-8 の CSV で保存する。",
        "",
    ]
    for aid, it in shown:
        lines += [f"## {aid}", "", "### 質問", "", it["question"], "", "### 取得した本文", ""]
        if not it["chunks"]:
            lines += ["(取得した本文なし)", ""]
        for i, c in enumerate(it["chunks"], 1):
            lines += [f"[{i}] 出典: {c.get('source_file', '?')}", "", c.get("content", ""), ""]
        lines += ["### 回答の文", ""]
        lines += [f"{n}. {s}" for n, s in enumerate(it["sentences"], 1)]
        lines.append("")
    return "\n".join(lines)


def make(raws: list[Path], seed: int, out_dir: Path, key_out: Path, max_answers: int = MAX_ANSWERS) -> dict:
    out_dir, key_out = Path(out_dir), Path(key_out)
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise SheetError(f"--out-dir {out_dir} が既にあり中身がある")
    if key_out.exists():
        raise SheetError(f"--key-out {key_out} が既にある (上書きしないため拒否)")
    if key_out.resolve().is_relative_to(out_dir.resolve()):
        raise SheetError("--key-out は --out-dir の外に置く (本人が対応表を見ないため)")

    items, excluded = collect(raws)
    if not items:
        raise SheetError("本番の問が 1 つも無い (N の問だけの raw は人手採点の対象外)")
    auto = [it for it in items if it["auto_zero_reason"]]
    rest = [it for it in items if not it["auto_zero_reason"]]
    if len(rest) > max_answers:
        raise SheetError(f"採点する回答が {len(rest)} 件ある (上限 {max_answers})")
    random.Random(seed).shuffle(rest)
    width = max(3, len(str(len(rest))))
    shown = [(f"H{i:0{width}d}", it) for i, it in enumerate(rest, 1)]

    out_dir.mkdir(parents=True, exist_ok=True)
    sheet, read = out_dir / SHEET_NAME, out_dir / READ_NAME
    with sheet.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for aid, it in shown:
            for n, s in enumerate(it["sentences"], 1):
                w.writerow([aid, n, _cell(s), "", ""])
    read.write_text(_read_md(shown), encoding="utf-8")

    key = {
        "seed": seed,
        "answers": {
            aid: {"raw": it["raw"], "run_label": it["run_label"],
                  "question_id": it["question_id"], "sentences": it["sentences"]}
            for aid, it in shown
        },
        "auto_zero": [
            {"raw": it["raw"], "run_label": it["run_label"], "question_id": it["question_id"],
             "reason": it["auto_zero_reason"]}
            for it in auto
        ],
        "excluded_numeric_boundary": excluded,
        "sheet_sha256": _sha(sheet),
        "read_sha256": _sha(read),
        "raws": [{"path": str(r), "sha256": _sha(r)} for r in raws],
    }
    key_out.parent.mkdir(parents=True, exist_ok=True)
    key_out.write_text(json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return key


# ─── score ─────────────────────────────────────────────────────
def _load_key(path: Path) -> dict:
    try:
        key = json.loads(Path(path).read_text(encoding="utf-8"))
        answers, auto = key["answers"], key["auto_zero"]
        if not isinstance(answers, dict) or not isinstance(auto, list):
            raise ValueError("形が違う")
        for a in answers.values():
            a["run_label"], a["sentences"]  # noqa: B018
        for z in auto:
            z["run_label"]  # noqa: B018
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SheetError(f"--key を読めない: {exc!r}") from exc
    return key


def _load_sheet(path: Path) -> list[dict]:
    try:
        with Path(path).open(encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None or [h.strip() for h in header] != COLUMNS:
                raise SheetError(f"sheet の列が {COLUMNS} と違う: {header}")
            rows = []
            for lineno, rec in enumerate(reader, 2):
                if not any(c.strip() for c in rec):
                    continue  # 末尾などの空行
                if len(rec) != len(COLUMNS):
                    raise SheetError(f"sheet {lineno} 行目の列数が {len(rec)} (期待 {len(COLUMNS)})")
                rows.append(dict(zip(COLUMNS, rec)))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise SheetError(f"--sheet を読めない: {exc!r}") from exc
    return rows


def _verify_rows(rows: list[dict], key: dict) -> dict[tuple[str, int], dict]:
    """行の追加・欠け・回答ID/文番号/文の書き換えがあれば SheetError。並べ替えは許す。"""
    expected = {
        (aid, n): _cell(s)
        for aid, a in key["answers"].items()
        for n, s in enumerate(a["sentences"], 1)
    }
    got: dict[tuple[str, int], dict] = {}
    problems: list[str] = []
    for row in rows:
        aid = row["回答ID"].strip()
        try:
            n = int(row["文番号"].strip())
        except ValueError:
            problems.append(f"文番号が整数でない: {aid} {row['文番号']!r}")
            continue
        k = (aid, n)
        if k in got:
            problems.append(f"行が重複: {aid} 文{n}")
        elif k not in expected:
            problems.append(f"対応表に無い行: {aid} 文{n}")
        elif _norm(row["文"]) != _norm(expected[k]):
            problems.append(f"文が書き換えられている: {aid} 文{n}")
        got[k] = row
    for k in expected:
        if k not in got:
            problems.append(f"行が欠けている: {k[0]} 文{k[1]}")
    if problems:
        raise SheetError("表が対応表と合わない: " + " / ".join(problems[:20]) + (" ..." if len(problems) > 20 else ""))
    return got


def _parse_verdict(cell: str) -> str:
    """'0' / '1' / 'pending' / 'blank' / 'invalid'。全角数字は半角にそろえる。"""
    v = unicodedata.normalize("NFKC", cell).strip()
    if v == "":
        return "blank"
    if v == PENDING:
        return "pending"
    return v if v in ("0", "1") else "invalid"


def score(sheet: Path, key_path: Path) -> dict:
    key = _load_key(key_path)
    got = _verify_rows(_load_sheet(sheet), key)

    incomplete: list[dict] = []
    verdicts: dict[str, list[int]] = {aid: [] for aid in key["answers"]}
    for (aid, n), row in sorted(got.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        kind = _parse_verdict(row["判定"])
        if kind in ("0", "1"):
            if kind == "0" and not row["理由"].strip():
                incomplete.append({"回答ID": aid, "文番号": n, "問題": "0 なのに理由が空"})
            else:
                verdicts[aid].append(int(kind))
        else:
            incomplete.append({"回答ID": aid, "文番号": n, "問題": {
                "blank": "判定が空欄", "pending": "保留", "invalid": f"判定が 0/1 でない: {row['判定']!r}",
            }[kind]})
    if incomplete:
        return {"result": INCOMPLETE, "incomplete": incomplete, "runs": None}

    per_label: dict[str, list[Fraction]] = {}
    n_auto: dict[str, int] = {}
    for aid, a in key["answers"].items():
        v = verdicts[aid]
        per_label.setdefault(a["run_label"], []).append(Fraction(sum(v), len(v)))
    for z in key["auto_zero"]:
        per_label.setdefault(z["run_label"], []).append(Fraction(0))
        n_auto[z["run_label"]] = n_auto.get(z["run_label"], 0) + 1
    runs = {}
    for label, scores in per_label.items():
        mean = sum(scores) / len(scores)
        runs[label] = {
            "mean": float(mean),
            "n_questions": len(scores),
            "n_auto_zero": n_auto.get(label, 0),
            "result": "pass" if mean >= PASS_THRESHOLD else "fail",
        }
    result = "pass" if runs and all(r["result"] == "pass" for r in runs.values()) else "fail"
    return {"result": result, "incomplete": [], "runs": runs}


# ─── CLI ───────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="人による文ごと 0/1 採点の表の生成と集計 (M4-S)")
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make", help="sheet.csv と read.md と対応表を作る")
    m.add_argument("--raw", type=Path, action="append", required=True, help="raw.jsonl (複数可)")
    m.add_argument("--seed", type=int, required=True)
    m.add_argument("--out-dir", type=Path, required=True, help="既にあり中身があれば拒否")
    m.add_argument("--key-out", type=Path, required=True, help="対応表 JSON (--out-dir の外・既にあれば拒否)")
    m.add_argument("--max-answers", type=int, default=MAX_ANSWERS)
    s = sub.add_parser("score", help="記入済みの表を集計する")
    s.add_argument("--sheet", type=Path, required=True)
    s.add_argument("--key", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True, help="結果の JSON (既にあれば拒否)")
    a = p.parse_args(argv)

    try:
        if a.cmd == "make":
            key = make(a.raw, a.seed, a.out_dir, a.key_out, a.max_answers)
            print(
                f"作成: 回答 {len(key['answers'])} 件 (auto_zero {len(key['auto_zero'])} 件は表に出さない) "
                f"→ {a.out_dir} / 対応表 {a.key_out}",
                flush=True,
            )
            return 0
        if a.out.exists():
            raise SheetError(f"--out {a.out} が既にある (上書きしないため拒否)")
        res = score(a.sheet, a.key)
    except (SheetError, ValueError, OSError) as exc:
        print(f"拒否: {exc}", flush=True)
        return REFUSED

    res = {**res, "sheet": {"path": str(a.sheet), "sha256": _sha(a.sheet)},
           "key": {"path": str(a.key), "sha256": _sha(a.key)}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if res["result"] == INCOMPLETE:
        print(f"{INCOMPLETE}: {len(res['incomplete'])} 行")
        for it in res["incomplete"]:
            print(f"  {it['回答ID']} 文{it['文番号']}: {it['問題']}")
    else:
        for label, v in res["runs"].items():
            print(f"  {label}: {v['result']} 平均={v['mean']:.4f} 問={v['n_questions']} auto_zero={v['n_auto_zero']}")
    print(f"人手採点: {res['result']} → {a.out}", flush=True)
    return EXIT[res["result"]]


if __name__ == "__main__":
    sys.exit(main())
