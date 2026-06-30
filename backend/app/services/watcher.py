"""
監視フォルダ内のファイル追加・更新・削除を検知して SQLite タスクキューに登録する。
"""
import hashlib
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".xlsm", ".txt", ".md"}

_observer: Observer | PollingObserver | None = None


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


class SyncDatabase:
    """watchdog は別スレッドで動くため同期 sqlite3 を使用。"""

    def __init__(self, db_path: str):
        self._path = db_path

    def fetchone(self, sql: str, params: tuple = ()):
        import sqlite3
        with sqlite3.connect(self._path) as conn:
            conn.row_factory = sqlite3.Row
            return conn.execute(sql, params).fetchone()

    def fetchall(self, sql: str, params: tuple = ()):
        import sqlite3
        with sqlite3.connect(self._path) as conn:
            conn.row_factory = sqlite3.Row
            return conn.execute(sql, params).fetchall()

    def execute(self, sql: str, params: tuple = ()) -> None:
        import sqlite3
        with sqlite3.connect(self._path) as conn:
            conn.execute(sql, params)
            conn.commit()

    def execute_many_atomic(self, statements: list[tuple[str, tuple]]) -> None:
        """複数 SQL を単一トランザクションで実行。途中失敗時は全てロールバック。"""
        import sqlite3
        with sqlite3.connect(self._path) as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                for sql, params in statements:
                    conn.execute(sql, params)
                conn.commit()
            except Exception:
                conn.rollback()
                raise


def _get_access_level_for_path(path: str, db_sync: SyncDatabase) -> int:
    """ファイルパスが属する監視フォルダの access_level を返す (最長一致)。

    一致はパス境界で判定する。`/watched/exec` の設定が `/watched/exec-public/...`
    のような**兄弟ディレクトリ**へ誤って適用されるのを防ぐため、完全一致または
    `フォルダパス + "/"` で始まる場合のみ一致とみなす。どの監視フォルダにも属さない
    パスは既定 1 (一般)。
    """
    folders = db_sync.fetchall(
        "SELECT path, access_level FROM watch_folders WHERE is_active=1"
    )
    best_level, best_len = 1, -1
    for folder in folders:
        fp = folder["path"].rstrip("/")
        if (path == fp or path.startswith(fp + "/")) and len(fp) > best_len:
            best_len = len(fp)
            best_level = folder["access_level"]
    return best_level


class _Handler(FileSystemEventHandler):
    def __init__(self, db_path: str):
        self._db = SyncDatabase(db_path)

    def _handle_file(self, path: str) -> None:
        p = Path(path)
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS or not p.is_file():
            return
        try:
            file_hash = _file_hash(path)
        except OSError:
            return

        existing = self._db.fetchone(
            "SELECT id, file_hash FROM documents WHERE source_path=?", (path,)
        )
        now = datetime.now(timezone.utc).isoformat()

        if existing is None:
            doc_id = str(uuid.uuid4())
            access_level = _get_access_level_for_path(path, self._db)
            self._db.execute_many_atomic([
                (
                    "INSERT INTO documents (id, source_path, file_hash, access_level, file_type, "
                    "status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)",
                    (doc_id, path, file_hash, access_level, p.suffix.lower(), now, now),
                ),
                (
                    "INSERT INTO tasks (document_id, status, created_at, updated_at) "
                    "VALUES (?, 'pending', ?, ?)",
                    (doc_id, now, now),
                ),
            ])
            logger.info("New file queued: %s", path)
        elif existing["file_hash"] != file_hash:
            doc_id = existing["id"]
            self._db.execute_many_atomic([
                (
                    "UPDATE documents SET file_hash=?, status='pending', error_msg=NULL, "
                    "updated_at=? WHERE id=?",
                    (file_hash, now, doc_id),
                ),
                (
                    "INSERT INTO tasks (document_id, status, created_at, updated_at) "
                    "VALUES (?, 'pending', ?, ?)",
                    (doc_id, now, now),
                ),
            ])
            logger.info("Updated file queued: %s", path)

    def _handle_delete(self, path: str) -> None:
        existing = self._db.fetchone(
            "SELECT id FROM documents WHERE source_path=?", (path,)
        )
        if existing:
            now = datetime.now(timezone.utc).isoformat()
            self._db.execute(
                "UPDATE documents SET status='deleted', updated_at=? WHERE id=?",
                (now, existing["id"]),
            )
            logger.info("File deleted: %s", path)

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._handle_file(event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._handle_file(event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._handle_delete(event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._handle_delete(event.src_path)
            self._handle_file(event.dest_path)


def start_watcher(watched_path: str, db_path: str, use_polling: bool = False) -> None:
    global _observer
    Path(watched_path).mkdir(parents=True, exist_ok=True)
    _observer = PollingObserver() if use_polling else Observer()
    _observer.schedule(_Handler(db_path), watched_path, recursive=True)
    _observer.start()
    logger.info("Watching: %s (polling=%s)", watched_path, use_polling)


def stop_watcher() -> None:
    global _observer
    if _observer:
        _observer.stop()
        _observer.join()
        _observer = None
