"""SQLite (aiosqlite) のシングル接続管理。

aiosqlite の Connection は内部で 1 つのワーカスレッドを持つため、
全リクエストで 1 接続を共有しても安全。短命接続を毎回開閉する従来実装は
bcrypt 込みのログインや watcher → タスク投入のたびに無視できないオーバーヘッドが発生していた。
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import aiosqlite

from app.config import settings

DB_PATH = Path(settings.data_dir) / "app.db"
MIGRATION_PATH = Path(__file__).parent.parent / "migrations" / "init.sql"


class Database:
    """単一の aiosqlite.Connection をシングルトンとして保持するラッパー。"""

    def __init__(self) -> None:
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        if self._conn is not None:
            return
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(DB_PATH)
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA synchronous=NORMAL")
        await conn.execute("PRAGMA foreign_keys=ON")
        sql = MIGRATION_PATH.read_text(encoding="utf-8")
        await conn.executescript(sql)
        await conn.commit()
        self._conn = conn

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    def _require(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database is not connected; call connect() first.")
        return self._conn

    async def fetchone(self, sql: str, params: tuple = ()) -> aiosqlite.Row | None:
        conn = self._require()
        async with self._lock:
            async with conn.execute(sql, params) as cur:
                row = await cur.fetchone()
            await conn.commit()  # UPDATE/INSERT/DELETE ... RETURNING でも安全に永続化
            return row

    async def fetchall(self, sql: str, params: tuple = ()) -> list[aiosqlite.Row]:
        conn = self._require()
        async with self._lock:
            async with conn.execute(sql, params) as cur:
                return await cur.fetchall()

    async def execute(self, sql: str, params: tuple = ()) -> int:
        conn = self._require()
        async with self._lock:
            async with conn.execute(sql, params) as cur:
                last = cur.lastrowid or 0
            await conn.commit()
            return last

    async def executemany(self, sql: str, params_list: list[tuple]) -> None:
        conn = self._require()
        async with self._lock:
            await conn.executemany(sql, params_list)
            await conn.commit()


db = Database()


async def init_db() -> None:
    """旧 API 互換: 接続を確立してスキーマを初期化する。"""
    await db.connect()


async def close_db() -> None:
    await db.close()
