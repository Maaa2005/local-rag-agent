"""保持期限（設計書 N7: 本文 30 日・メタデータ 90 日）の削除。

1. 常駐中の定期実行: gateway.app の lifespan が purge_periodically をバックグラウンドで回す。
   間隔は環境変数 GATEWAY_PURGE_INTERVAL_SECONDS（秒、既定 86400 = 24 時間）。
2. 1 回だけ実行する CLI（外部 cron・手動用）:

  python -m gateway.purge            # GATEWAY_DB（既定 /data/gateway.db）を対象にする
  python -m gateway.purge --db PATH
  python -m gateway.purge --dry-run  # 削除せず件数だけ（同じ文を実行して ROLLBACK）

件数だけを JSON で標準出力に出す（本文・request_id は出さない）。Gateway 起動中に実行してよい
（対象は終端状態と期限切れ PREPARED のみで、BEGIN IMMEDIATE の 1 トランザクションで消す）。
DB ファイルが無いときは作らずにエラー終了する（パスの打ち間違いで空 DB を作らない）。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict
from typing import Callable

from gateway.store import SendStore, StoreError

log = logging.getLogger("gateway")

PURGE_INTERVAL_ENV = "GATEWAY_PURGE_INTERVAL_SECONDS"
DEFAULT_PURGE_INTERVAL_SECONDS = 86400.0
# purge は BEGIN IMMEDIATE で書き込みを止めるので、極端に短い間隔で送信処理を詰まらせない下限
MIN_PURGE_INTERVAL_SECONDS = 60.0


def parse_purge_interval(raw: str | None) -> float:
    """環境変数の値を間隔（秒）にする。

    未設定・空は既定値。数値でない・NaN/無限大・0 以下も既定値にする（警告ログ）。無効化を選ばない
    のは、設定ミスで定期削除が黙って止まると本文が 30 日を超えて残り N7 に反するため。
    0 < 値 < 下限 は下限に切り上げる。
    """
    if raw is None or not raw.strip():
        return DEFAULT_PURGE_INTERVAL_SECONDS
    try:
        value = float(raw)
    except ValueError:
        log.warning("%s is not a number; using default %.0fs",
                    PURGE_INTERVAL_ENV, DEFAULT_PURGE_INTERVAL_SECONDS)
        return DEFAULT_PURGE_INTERVAL_SECONDS
    if not math.isfinite(value) or value <= 0:
        log.warning("%s must be a positive finite number; using default %.0fs",
                    PURGE_INTERVAL_ENV, DEFAULT_PURGE_INTERVAL_SECONDS)
        return DEFAULT_PURGE_INTERVAL_SECONDS
    if value < MIN_PURGE_INTERVAL_SECONDS:
        log.warning("%s below minimum; using %.0fs", PURGE_INTERVAL_ENV, MIN_PURGE_INTERVAL_SECONDS)
        return MIN_PURGE_INTERVAL_SECONDS
    return value


def purge_interval_from_env() -> float:
    return parse_purge_interval(os.environ.get(PURGE_INTERVAL_ENV))


async def purge_periodically(purge: Callable[[], object], interval: float, stop: asyncio.Event) -> None:
    """stop が立つまで interval 秒ごとに purge を呼ぶ（初回は interval 後。起動時の 1 回は lifespan 側）。

    purge はブロッキング（SQLite）なのでスレッドで実行する。例外は型名だけログに出して続ける。
    停止は stop で行い、実行中の purge は終わるまで待つ（途中で打ち切らない。1 トランザクションなので
    どちらにしても半端には残らないが、終了後に接続が残らないようにする）。
    """
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
            return
        except asyncio.TimeoutError:
            pass
        try:
            await asyncio.to_thread(purge)
        except Exception as e:  # noqa: BLE001  定期削除の失敗で Gateway を止めない
            log.error("periodic purge failed kind=%s", type(e.__cause__ or e).__name__)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m gateway.purge")
    ap.add_argument("--db", default=os.environ.get("GATEWAY_DB", "/data/gateway.db"))
    ap.add_argument("--dry-run", action="store_true", help="削除せず対象件数だけを出す")
    args = ap.parse_args(argv)
    if not os.path.isfile(args.db):
        print(json.dumps({"error": "DatabaseNotFound"}), file=sys.stderr)
        return 1
    try:
        result = SendStore(args.db).purge_expired(time.time(), dry_run=args.dry_run)
    except StoreError as e:
        print(json.dumps({"error": type(e.__cause__ or e).__name__}), file=sys.stderr)
        return 1
    out = asdict(result)
    if args.dry_run:
        out = {"dry_run": True, **out}
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
