"""Gateway の送信記録（SQLite, WAL + synchronous=FULL）。"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from common.schemas import FailureKind, SendState

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sends (
    request_id   TEXT PRIMARY KEY,
    digest       TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    expires_at   REAL NOT NULL,
    state        TEXT NOT NULL,
    failure      TEXT,
    output_text  TEXT,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL
)
"""


@dataclass(frozen=True)
class SendRecord:
    request_id: str
    digest: str
    payload_json: str
    expires_at: float
    state: SendState
    failure: FailureKind | None
    output_text: str | None


class StoreError(RuntimeError):
    """記録に失敗した。"""


class SendStore:
    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=30, isolation_level=None)
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    @staticmethod
    def _row(row: tuple | None) -> SendRecord | None:
        if row is None:
            return None
        return SendRecord(
            request_id=row[0],
            digest=row[1],
            payload_json=row[2],
            expires_at=row[3],
            state=SendState(row[4]),
            failure=FailureKind(row[5]) if row[5] else None,
            output_text=row[6],
        )

    def get(self, request_id: str) -> SendRecord | None:
        conn = self._connect()
        try:
            cur = conn.execute(
                "SELECT request_id, digest, payload_json, expires_at, state, failure, output_text "
                "FROM sends WHERE request_id=?",
                (request_id,),
            )
            return self._row(cur.fetchone())
        finally:
            conn.close()

    def insert_prepared(self, request_id: str, digest: str, payload_json: str, expires_at: float) -> bool:
        """新規なら True。同じ ID が既にあれば False（内容比較は呼び出し側）。"""
        now = time.time()
        conn = self._connect()
        try:
            cur = conn.execute(
                "INSERT OR IGNORE INTO sends (request_id, digest, payload_json, expires_at, state, "
                "failure, output_text, created_at, updated_at) VALUES (?,?,?,?,?,NULL,NULL,?,?)",
                (request_id, digest, payload_json, expires_at, SendState.PREPARED.value, now, now),
            )
            return cur.rowcount == 1
        except sqlite3.Error as e:
            raise StoreError("insert failed") from e
        finally:
            conn.close()

    def claim_attempt(self, request_id: str) -> bool:
        """PREPARED→ATTEMPTING を条件付き UPDATE で 1 件だけ通す。"""
        conn = self._connect()
        try:
            cur = conn.execute(
                "UPDATE sends SET state=?, updated_at=? WHERE request_id=? AND state=?",
                (SendState.ATTEMPTING.value, time.time(), request_id, SendState.PREPARED.value),
            )
            return cur.rowcount == 1
        except sqlite3.Error as e:
            raise StoreError("claim failed") from e
        finally:
            conn.close()

    def finish(
        self,
        request_id: str,
        state: SendState,
        failure: FailureKind | None = None,
        output_text: str | None = None,
    ) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE sends SET state=?, failure=?, output_text=?, updated_at=? "
                "WHERE request_id=? AND state=?",
                (
                    state.value,
                    failure.value if failure else None,
                    output_text,
                    time.time(),
                    request_id,
                    SendState.ATTEMPTING.value,
                ),
            )
        except sqlite3.Error as e:
            raise StoreError("finish failed") from e
        finally:
            conn.close()

    def fail_attempting_as_unknown(self) -> int:
        conn = self._connect()
        try:
            cur = conn.execute(
                "UPDATE sends SET state=?, failure=?, updated_at=? WHERE state=?",
                (
                    SendState.FAILED.value,
                    FailureKind.unknown.value,
                    time.time(),
                    SendState.ATTEMPTING.value,
                ),
            )
            return cur.rowcount
        finally:
            conn.close()
