"""開発用の問（N12 + D24）を「コードが決める」方式（v2/decide）に流し、段ごとの結果を出す。

- 読む問: numeric_boundary.jsonl と numeric_dev24.jsonl だけ（holdout48・corpus_nx_holdout は開かない）
- 検索の代わり: 利用者の閲覧レベル（問の user_access_level）以下の文書を、1 文書 1 chunk（本文全体）で渡す。
  閲覧レベルは general=1・manager=2・executive=3（evaluation/README.md）。expected_sources は使わない
- 順番: 先に engine が答え、その後で expected_conclusion と照らす（答えを作る側は正解を見ない）
- 判断の記録（plan §4）は engine が答えた時に 1 回だけ作ったものに id と correct を足した写しを --out に書く
- --out の既定は /tmp/decide-dev.jsonl。リポジトリの中は拒否する

使い方: python evaluation/v2/run_decide_dev.py [--out /tmp/decide-dev.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "v2") not in sys.path:
    sys.path.insert(0, str(ROOT / "v2"))

from decide.engine import DecisionEngine, FileDocLookup  # noqa: E402
from decide.records import load_records  # noqa: E402
from orchestrator.local_answer import Chunk  # noqa: E402

QUESTION_FILES = (ROOT / "evaluation/v2/numeric_boundary.jsonl", ROOT / "evaluation/v2/numeric_dev24.jsonl")
CORPUS_DIRS = (ROOT / "evaluation/corpus", ROOT / "evaluation/v2/corpus_nx_dev")
RECORDS_DIR = ROOT / "v2/decide/records"
LEVELS = {"general": 1, "manager": 2, "executive": 3}
DEFAULT_OUT = Path("/tmp/decide-dev.jsonl")


def load_rows(paths=QUESTION_FILES) -> list[dict]:
    rows = []
    for p in paths:
        rows += [json.loads(line) for line in Path(p).read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows


def visible_chunks(level: int, corpus_dirs=CORPUS_DIRS) -> list[Chunk]:
    """閲覧レベル level 以下の文書を 1 文書 1 chunk で。document_id は /watched/<区分>/<ファイル名>。"""
    chunks, seen = [], set()
    for d in corpus_dirs:
        for sub, lv in LEVELS.items():
            if lv > level or not (Path(d) / sub).is_dir():
                continue
            for f in sorted((Path(d) / sub).glob("*.md")):
                doc_id = f"/watched/{sub}/{f.name}"
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                chunks.append(Chunk(content=f.read_text(encoding="utf-8"), source=f.name, access_level=lv,
                                    document_id=doc_id, chunk_index=0))
    return chunks


def make_engine(records_dir=RECORDS_DIR, corpus_dirs=CORPUS_DIRS, on_record=None) -> DecisionEngine:
    lookup = FileDocLookup(corpus_dirs)
    return DecisionEngine(load_records(records_dir, lookup), lookup, on_record=on_record)


def final_line(text: str) -> str:
    return text.strip().split("\n")[-1]


def run_rows(rows: list[dict], engine: DecisionEngine | None = None) -> list[dict]:
    """各問を engine に流して結果の行を返す。engine が答えを返した後で初めて正解欄を見る。"""
    records: list[dict] = []
    engine = engine or make_engine(on_record=records.append)
    out = []
    for row in rows:
        n_before = len(records)
        res = engine.decide(row["question"], visible_chunks(int(row["user_access_level"])))
        # ---- ここまでが答えを作る側。ここから先が採点
        dr = dict(res.decision_record) if len(records) == n_before else dict(records[-1])
        expected = row.get("expected_conclusion")
        if res.kind == "decided":
            got = final_line(res.answer_text)
            correct = got == f"結論: {expected}"
        else:
            got, correct = None, False
        dr["id"] = row["id"]
        dr["correct"] = correct
        out.append({"id": row["id"], "kind": res.kind, "stage": dr.get("stage"), "hold_code": dr.get("hold_code"),
                    "rule_id": dr.get("rule_id"), "got": got, "expected": expected, "correct": correct,
                    "decision_record": dr})
    return out


def write_jsonl(results: list[dict], out: Path) -> None:
    with out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({**r["decision_record"], "kind": r["kind"]}, ensure_ascii=False) + "\n")


def print_table(results: list[dict]) -> None:
    print(f"{'id':<5} {'種別':<9} {'段':<22} {'記録':<24} {'結論':<12} {'正解':<12} 正誤")
    for r in results:
        stage = r["hold_code"] or r["stage"] or "-"
        print(f"{r['id']:<5} {r['kind']:<9} {stage:<22} {r['rule_id'] or '-':<24} {r['got'] or '-':<12} "
              f"{r['expected'] or '-':<12} {'OK' if r['correct'] else 'NG'}")
    c = Counter(r["kind"] for r in results)
    print()
    print(f"全 {len(results)} 問: 答えた {c['decided']}・保留 {c['held']}・モデルだけの回答へ {c['no_record']}")
    print("段ごと:", dict(Counter((r["hold_code"] or r["stage"] or "-") for r in results)))
    ok = sum(r["correct"] for r in results)
    print(f"正解 {ok}/{len(results)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="判断の記録の JSONL（リポジトリの外）")
    args = ap.parse_args(argv)
    out = Path(args.out).resolve()
    if ROOT == out or ROOT in out.parents:
        print(f"--out はリポジトリの外に指定してください: {out}", file=sys.stderr)
        return 2
    results = run_rows(load_rows())
    write_jsonl(results, out)
    print_table(results)
    print(f"判断の記録: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
