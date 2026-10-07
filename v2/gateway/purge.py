"""保持期限の削除を 1 回実行する CLI（外部 cron 用）。

  python -m gateway.purge            # GATEWAY_DB（既定 /data/gateway.db）を対象にする
  python -m gateway.purge --db PATH

件数だけを JSON で標準出力に出す（本文・request_id は出さない）。Gateway 起動中に実行してよい
（対象は終端状態と期限切れ PREPARED のみで、BEGIN IMMEDIATE の 1 トランザクションで消す）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

from gateway.store import SendStore, StoreError


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m gateway.purge")
    ap.add_argument("--db", default=os.environ.get("GATEWAY_DB", "/data/gateway.db"))
    args = ap.parse_args(argv)
    try:
        result = SendStore(args.db).purge_expired(time.time())
    except StoreError as e:
        print(json.dumps({"error": type(e.__cause__ or e).__name__}), file=sys.stderr)
        return 1
    print(json.dumps(asdict(result)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
