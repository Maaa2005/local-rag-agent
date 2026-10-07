"""Gateway の送信記録（SQLite, WAL + synchronous=FULL）。"""
from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator
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

# 状態遷移履歴。本文（payload）・出力は入れない。遷移と同じトランザクションで追記する。
_HISTORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS send_history (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id  TEXT NOT NULL,
    from_state  TEXT,
    to_state    TEXT NOT NULL,
    failure     TEXT,
    at          REAL NOT NULL
)
"""
_HISTORY_INDEX = "CREATE INDEX IF NOT EXISTS send_history_rid ON send_history (request_id, seq)"
_HISTORY_INSERT = (
    "INSERT INTO send_history (request_id, from_state, to_state, failure, at) VALUES (?,?,?,?,?)"
)


@dataclass(frozen=True)
class SendRecord:
    request_id: str
    digest: str
    payload_json: str
    expires_at: float
    state: SendState
    failure: FailureKind | None
    output_text: str | None


@dataclass(frozen=True)
class Transition:
    request_id: str
    from_state: SendState | None
    to_state: SendState
    failure: FailureKind | None
    at: float


class StoreError(RuntimeError):
    """記録に失敗した。"""


class SendStore:
    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(_SCHEMA)
            conn.execute(_HISTORY_SCHEMA)
            conn.execute(_HISTORY_INDEX)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=30, isolation_level=None)
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """書き込みトランザクション（BEGIN IMMEDIATE）。例外時は ROLLBACK。"""
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")
        finally:
            conn.close()

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
        try:
            with self._tx() as conn:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO sends (request_id, digest, payload_json, expires_at, state, "
                    "failure, output_text, created_at, updated_at) VALUES (?,?,?,?,?,NULL,NULL,?,?)",
                    (request_id, digest, payload_json, expires_at, SendState.PREPARED.value, now, now),
                )
                if cur.rowcount != 1:
                    return False
                conn.execute(_HISTORY_INSERT, (request_id, None, SendState.PREPARED.value, None, now))
                return True
        except sqlite3.Error as e:
            raise StoreError("insert failed") from e

    def claim_attempt(self, request_id: str) -> bool:
        """PREPARED→ATTEMPTING を条件付き UPDATE で 1 件だけ通す。"""
        now = time.time()
        try:
            with self._tx() as conn:
                cur = conn.execute(
                    "UPDATE sends SET state=?, updated_at=? WHERE request_id=? AND state=?",
                    (SendState.ATTEMPTING.value, now, request_id, SendState.PREPARED.value),
                )
                if cur.rowcount != 1:
                    return False
                conn.execute(_HISTORY_INSERT, (request_id, SendState.PREPARED.value,
                                               SendState.ATTEMPTING.value, None, now))
                return True
        except sqlite3.Error as e:
            raise StoreError("claim failed") from e

    def finish(
        self,
        request_id: str,
        state: SendState,
        failure: FailureKind | None = None,
        output_text: str | None = None,
    ) -> int:
        """ATTEMPTING の行だけを終端状態にする。更新行数を返す（正常なら 1）。"""
        now = time.time()
        fv = failure.value if failure else None
        try:
            with self._tx() as conn:
                cur = conn.execute(
                    "UPDATE sends SET state=?, failure=?, output_text=?, updated_at=? "
                    "WHERE request_id=? AND state=?",
                    (state.value, fv, output_text, now, request_id, SendState.ATTEMPTING.value),
                )
                if cur.rowcount == 1:
                    # output_text は履歴に入れない
                    conn.execute(_HISTORY_INSERT, (request_id, SendState.ATTEMPTING.value,
                                                   state.value, fv, now))
                return cur.rowcount
        except sqlite3.Error as e:
            raise StoreError("finish failed") from e

    def fail_attempting_as_unknown(self) -> int:
        now = time.time()
        with self._tx() as conn:
            # 更新対象と同じ行に履歴を追記してから更新する（同一トランザクション）
            conn.execute(
                "INSERT INTO send_history (request_id, from_state, to_state, failure, at) "
                "SELECT request_id, ?, ?, ?, ? FROM sends WHERE state=? ORDER BY request_id",
                (SendState.ATTEMPTING.value, SendState.FAILED.value, FailureKind.unknown.value,
                 now, SendState.ATTEMPTING.value),
            )
            cur = conn.execute(
                "UPDATE sends SET state=?, failure=?, updated_at=? WHERE state=?",
                (SendState.FAILED.value, FailureKind.unknown.value, now, SendState.ATTEMPTING.value),
            )
            return cur.rowcount

    def history(self, request_id: str) -> list[Transition]:
        """request_id の遷移を記録順に返す（本文・出力は含まない）。"""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT request_id, from_state, to_state, failure, at FROM send_history "
                "WHERE request_id=? ORDER BY seq",
                (request_id,),
            ).fetchall()
        finally:
            conn.close()
        return [
            Transition(
                request_id=r[0],
                from_state=SendState(r[1]) if r[1] else None,
                to_state=SendState(r[2]),
                failure=FailureKind(r[3]) if r[3] else None,
                at=r[4],
            )
            for r in rows
        ]
