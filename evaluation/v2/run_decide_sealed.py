"""封をした新しい問（96 問）を「コードが決める」方式（v2/decide）だけに流す道具（plan 7.5）。

流す段は 記録・計算と結論・答え全体。検索の代わりに docs/ の全文書を 1 文書 1 塊で渡す。
取り出し（検索）・閲覧レベル・モデルの生成（C1 を含む実際の経路）は流さず、未測定。
モデルだけの回答に回った問（no_record）は、モデルを動かさず不正解として数える。

- 起動の時点で止まる（問のファイルを開く前）: 記録が読めない・確認済みの表に無い／ハッシュが違う・記録が 0 件
- 答えを作る側は正解欄を見ない。engine が返した後で初めて正解欄を読む
- --out（問ごとの判断の記録）と --summary は、どちらもリポジトリの外。問の文・正解・問ごとの行は画面に出さない

使い方:
  python evaluation/v2/run_decide_sealed.py --questions Q.jsonl [--questions Q2.jsonl] --docs-root DIR \\
      --records v2/decide/records --out /外/rows.jsonl --summary /外/summary.json --label 1回目
  DIR は docs/ の下に平らな *.md を持つ（文書の id は /watched/docs/<ファイル名>）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "v2") not in sys.path:
    sys.path.insert(0, str(ROOT / "v2"))

from decide.engine import DecisionEngine, FileDocLookup  # noqa: E402
from decide.records import RecordError, load_records  # noqa: E402
from orchestrator.local_answer import Chunk  # noqa: E402

DECIDE_DIR = ROOT / "v2/decide"
HOLD_EXPECTED = "保留"
EXPECTED_FIELDS = ("expected_conclusion", "expected")  # 正解欄の名前（先に見つかった方）


class StartupError(Exception):
    pass


def inside_repo(p: Path) -> bool:
    p = p.resolve()
    return p == ROOT or ROOT in p.parents


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def code_identity(decide_dir: Path = DECIDE_DIR, runner: Path | None = None) -> str:
    """v2/decide/**/*.py と、この道具自身の中身（と相対パス）の SHA-256。凍結のハッシュに含める。"""
    runner = Path(runner) if runner is not None else Path(__file__).resolve()
    h = hashlib.sha256()
    items = [("v2/decide/" + p.relative_to(decide_dir).as_posix(), p) for p in decide_dir.rglob("*.py")]
    items.append(("evaluation/v2/run_decide_sealed.py", runner))
    for rel, p in sorted(items, key=lambda t: t[0]):
        h.update(rel.encode("utf-8") + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def doc_chunks(docs_root: Path) -> list[Chunk]:
    """docs_root/docs/*.md を 1 文書 1 塊で。閲覧レベルは測らない（全部 1）。"""
    d = Path(docs_root) / "docs"
    files = sorted(d.glob("*.md")) if d.is_dir() else []
    if not files:
        raise StartupError(f"文書が無い: {d} に *.md が要る")
    return [Chunk(content=f.read_text(encoding="utf-8"), source=f.name, access_level=1,
                  document_id=f"/watched/docs/{f.name}", chunk_index=0) for f in files]


def build_engine(records_dir: Path, docs_root: Path) -> tuple[DecisionEngine, int]:
    """確認済みの記録だけで engine を作る。読めない・表と合わない・0 件なら StartupError。"""
    lookup = FileDocLookup([docs_root])
    try:
        records = load_records(records_dir, lookup, require_verified=True)
    except RecordError as e:
        raise StartupError(f"記録を読めない: {e}") from e
    if not records:
        raise StartupError(f"記録が 0 件: {records_dir}")
    # 文書が見つからない・版が違う記録は load_records が黙って stale にする。判定が繋がっていないので起動で止める
    ids = {c.document_id for c in doc_chunks(docs_root)}
    bad = sorted(r.rule_id for r in records if r.stale or r.doc_path not in ids)
    if bad:
        raise StartupError(f"文書に繋がっていない記録（文書が無い・版が違う）: {len(bad)} / {len(records)} 件: {', '.join(bad)}")
    return DecisionEngine(records, lookup), len(records)


def load_questions(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        try:
            lines = Path(p).read_text(encoding="utf-8").splitlines()
        except OSError as e:
            raise StartupError(f"問のファイルを読めない: {Path(p).name}（{type(e).__name__}）") from e
        for n, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                raise StartupError(f"{Path(p).name} の {n} 行目: JSON が読めない") from None
            if not isinstance(row, dict) or not isinstance(row.get("question"), str):
                raise StartupError(f"{Path(p).name} の {n} 行目: question 欄が無い")
            rows.append(row)
    return rows


def final_line(text: str) -> str:
    return text.strip().split("\n")[-1]


def grade(res, expected) -> bool:
    if expected == HOLD_EXPECTED:
        return res.kind == "held"
    return res.kind == "decided" and final_line(res.answer_text) == f"結論: {expected}"


def route_final_of(kind: str) -> str:
    return {"decided": "code", "held": "held", "no_record": "model_only_not_run"}.get(kind, "other")


def run_row(engine: DecisionEngine, row: dict, chunks: list[Chunk]) -> tuple[dict, dict]:
    """1 問を流す。(--out の行, 集計用の行) を返す。正解欄は engine が返した後で初めて読む。"""
    res = engine.decide(row["question"], chunks)
    # ---- ここまでが答えを作る側。ここから先が採点
    expected = next((row[k] for k in EXPECTED_FIELDS if k in row), None)
    missing = expected is None
    correct = grade(res, expected)
    answer = res.answer_text if res.kind == "decided" else None
    out = dict(res.decision_record)
    out.update(id=row.get("id"), kind=res.kind, route_final=route_final_of(res.kind), final_answer=answer,
               correct=correct)
    dr = res.decision_record
    tally = {"kind": res.kind, "stage": dr.get("hold_code") or dr.get("stage") or "-", "correct": correct,
             "set": row.get("set"), "qkind": row.get("kind"), "revised": row.get("revised_to_hold") is True,
             "missing_expected": missing}
    return out, tally


def _grouped(tallies: list[dict], key: str) -> dict:
    g: dict[str, dict[str, int]] = {}
    for t in tallies:
        if t[key] is None:
            continue
        slot = g.setdefault(str(t[key]), {"correct": 0, "total": 0})
        slot["total"] += 1
        slot["correct"] += int(t["correct"])
    return dict(sorted(g.items()))


def summarize(tallies: list[dict], *, label: str, records_dir: Path, n_records: int) -> dict:
    revised = [t for t in tallies if t["revised"]]
    return {
        "label": label,
        "total": len(tallies),
        "by_kind": dict(Counter(t["kind"] for t in tallies)),
        "by_hold_code_or_stage": dict(Counter(t["stage"] for t in tallies)),
        "correct": sum(t["correct"] for t in tallies),
        "by_set": _grouped(tallies, "set"),
        "by_question_kind": _grouped(tallies, "qkind"),
        "revised_to_hold": {"total": len(revised), "correct": sum(t["correct"] for t in revised)},
        # 正解欄が見つからなかった行の数。0 でなければ欄の名前の取り違えを疑う（その行は不正解に数えている）
        "missing_expected": sum(t["missing_expected"] for t in tallies),
        "code_identity": code_identity(),
        "verified_sha256": sha256_file(Path(records_dir) / "verified.json"),
        "n_records": n_records,
        "unmeasured": ["取り出し（検索）", "閲覧レベル", "実際の経路（C1・検索・モデルの生成）"],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--questions", action="append", required=True, help="問の JSONL（繰り返し可）")
    ap.add_argument("--docs-root", required=True, help="docs/ を持つ dir")
    ap.add_argument("--records", required=True, help="確認済みの記録の dir（verified.json 付き）")
    ap.add_argument("--out", required=True, help="問ごとの判断の記録の JSONL（リポジトリの外）")
    ap.add_argument("--summary", required=True, help="数だけの JSON（リポジトリの外）")
    ap.add_argument("--label", required=True)
    args = ap.parse_args(argv)
    out, summary_path = Path(args.out), Path(args.summary)
    for name, p in (("--out", out), ("--summary", summary_path)):
        if inside_repo(p):
            print(f"{name} はリポジトリの外に指定してください")
            return 2
    try:
        # 起動の検査は、問のファイルを開く前に済ませる
        engine, n_records = build_engine(Path(args.records), Path(args.docs_root))
        chunks = doc_chunks(Path(args.docs_root))
        rows = load_questions([Path(q) for q in args.questions])
    except StartupError as e:
        print(str(e))
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    tallies = []
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            line, tally = run_row(engine, row, chunks)
            f.write(json.dumps(line, ensure_ascii=False) + "\n")  # 1 問ごとに、終わった時点で 1 回
            f.flush()
            tallies.append(tally)
    summary = summarize(tallies, label=args.label, records_dir=Path(args.records), n_records=n_records)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
