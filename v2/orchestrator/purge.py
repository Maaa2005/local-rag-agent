"""監査ストアの保持期限（設計書 N7: メタデータ 90 日）の削除を 1 回実行する CLI（外部 cron 用）。

  python -m orchestrator.purge --db PATH
  python -m orchestrator.purge --db PATH --dry-run   # 削除せず件数だけ（同じ DELETE を実行して ROLLBACK）

Orchestrator の常駐プロセスはまだ無い（deploy/compose.yml の internal-app はプレースホルダ）ため、
定期実行はこの CLI を cron 等から 1 日 1 回程度呼ぶ。常駐プロセスができたら、そのライフサイクル内で
gateway.purge.purge_periodically と同様に AuditStore.purge を回す。

件数だけを JSON で標準出力に出す（run_id・本文は出さない）。DB ファイルが無いときは作らずにエラー終了する。
append と同時に実行してよい（どちらも BEGIN IMMEDIATE で直列化され、削除は DB トリガーでも
90 日超の行に限られる）。
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import asdict

from orchestrator.audit_store import AuditStore


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m orchestrator.purge")
    ap.add_argument("--db", required=True, help="監査ストアの SQLite パス")
    ap.add_argument("--dry-run", action="store_true", help="削除せず対象件数だけを出す")
    args = ap.parse_args(argv)
    if not os.path.isfile(args.db):
        print(json.dumps({"error": "DatabaseNotFound"}), file=sys.stderr)
        return 1
    store = None
    try:
        store = AuditStore(args.db)
        result = store.purge(dry_run=args.dry_run)
    except sqlite3.Error as e:
        print(json.dumps({"error": type(e).__name__}), file=sys.stderr)
        return 1
    finally:
        if store is not None:
            store.close()
    out = asdict(result)
    if args.dry_run:
        out = {"dry_run": True, **out}
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
