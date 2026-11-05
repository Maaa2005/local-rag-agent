"""確認済みの表（verified.json）を作る道具（計画 7.2）。

使い方（v2/ で）: python -m decide.stamp --records DIR --grade GRADE_FILE [--graded-by NAME]

- 採点結果のファイルは、{"rule_id": ..., "verdict": ..., "problems": [...]} の JSON の行と、それ以外の行（無視する）からなり、
  「===END===」だけの行で終わる。その行が無ければ何も書かずに終了コード 2。
- 表に載せるのは、verdict が "OK" で problems が空の rule_id のうち、DIR に記録のファイルが在る物だけ。
  採点の行が無い記録・「可」でない記録・同じ rule_id に「可」でない行が1つでもある記録は載せない。
- 表の grade_report_sha256 は採点結果のファイルのバイト列の SHA-256。records は記録の中身の SHA-256（record_sha256）。
- 出力は標準出力だけ。実装役が手で表を書かないための道具で、「可」の行しか載せない。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from decide.records import VERIFIED_FILE, record_sha256

END_MARK = "===END==="


def read_grades(text: str) -> dict[str, tuple[str, list]] | None:
    """採点結果の本文から {rule_id: (verdict, problems)} を取る。終端の行が無ければ None。
    同じ rule_id の行が複数あれば、「可」で problems が空の行だけが揃った時に限り「可」。"""
    lines = text.splitlines()
    end = next((i for i, ln in enumerate(lines) if ln.strip() == END_MARK), None)
    if end is None:
        return None
    grades: dict[str, tuple[str, list]] = {}
    for ln in lines[:end]:
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("rule_id"), str) or "verdict" not in obj:
            continue
        rid, verdict, problems = obj["rule_id"], obj["verdict"], obj.get("problems")
        problems = problems if isinstance(problems, list) else ["problems が配列でない"]
        if rid in grades:  # 重複: 1つでも「可」でなければ「可」にしない
            prev_v, prev_p = grades[rid]
            if prev_v != "OK" or prev_p:
                continue
        grades[rid] = (verdict if isinstance(verdict, str) else str(verdict), problems)
    return grades


def is_ok(grade: tuple[str, list]) -> bool:
    return grade[0] == "OK" and not grade[1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m decide.stamp")
    ap.add_argument("--records", required=True, help="記録（*.json）のあるディレクトリ")
    ap.add_argument("--grade", required=True, help="採点結果のファイル")
    ap.add_argument("--graded-by", default="codex", help="採点した役の名前（表に残す）")
    a = ap.parse_args(argv)

    d = Path(a.records)
    grade_path = Path(a.grade)
    if not d.is_dir() or not grade_path.is_file():
        print("記録のディレクトリか採点結果のファイルが無い。何も書かない")
        return 2
    raw = grade_path.read_bytes()
    grades = read_grades(raw.decode("utf-8", errors="replace"))
    if grades is None:
        print(f"採点結果に {END_MARK} の行が無い（途中で切れた恐れ）。何も書かない")
        return 2

    on_disk: dict[str, dict] = {}
    for p in sorted(d.glob("*.json")):
        if p.name == VERIFIED_FILE:
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"読めない記録: {p.name}（載せない）")
            continue
        if isinstance(data, dict) and isinstance(data.get("rule_id"), str):
            on_disk[data["rule_id"]] = data

    stamped: dict[str, str] = {}
    refused: list[str] = []
    for rid in sorted(set(on_disk) | set(grades)):
        if rid not in on_disk:
            refused.append(f"{rid}: 記録のファイルが無い（採点: {grades[rid][0]}）")
        elif rid not in grades:
            refused.append(f"{rid}: 採点の行が無い")
        elif not is_ok(grades[rid]):
            v, pr = grades[rid]
            refused.append(f"{rid}: 載せない（採点: {v}、問題 {len(pr)} 件）")
        else:
            stamped[rid] = record_sha256(on_disk[rid])

    for rid in sorted(stamped):
        print(f"載せた: {rid}")
    for line in refused:
        print(f"載せない: {line}")
    if not stamped:
        print("載せる記録が無い。何も書かない")
        return 1

    table = {"grade_report_sha256": hashlib.sha256(raw).hexdigest(), "graded_by": a.graded_by, "records": stamped}
    (d / VERIFIED_FILE).write_text(json.dumps(table, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
