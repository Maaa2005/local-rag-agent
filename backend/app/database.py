import aiosqlite
from pathlib import Path
from app.config import settings

DB_PATH = Path(settings.data_dir) / "app.db"
MIGRATION_PATH = Path(__file__).parent.parent / "migrations" / "init.sql"


async def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA synchronous=NORMAL")
        sql = MIGRATION_PATH.read_text(encoding="utf-8")
        await conn.executescript(sql)
        await conn.commit()


class Database:
    """シンプルな aiosqlite ラッパー。"""

    def __init__(self):
        self._path = str(DB_PATH)

    async def fetchone(self, sql: str, params: tuple = ()) -> aiosqlite.Row | None:
        async with aiosqlite.connect(self._path) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute(sql, params) as cur:
                row = await cur.fetchone()
            # UPDATE/INSERT/DELETE ... RETURNING でも安全に永続化する
            await conn.commit()
            return row

    async def fetchall(self, sql: str, params: tuple = ()) -> list[aiosqlite.Row]:
        async with aiosqlite.connect(self._path) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute(sql, params) as cur:
                return await cur.fetchall()

    async def execute(self, sql: str, params: tuple = ()) -> int:
        async with aiosqlite.connect(self._path) as conn:
            async with conn.execute(sql, params) as cur:
                await conn.commit()
                return cur.lastrowid or 0

    async def executemany(self, sql: str, params_list: list[tuple]) -> None:
        async with aiosqlite.connect(self._path) as conn:
            await conn.executemany(sql, params_list)
            await conn.commit()


db = Database()
