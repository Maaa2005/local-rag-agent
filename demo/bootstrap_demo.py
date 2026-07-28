from pathlib import Path
import sqlite3

from app.config import settings
from app.core.security import hash_password
from app.services.watcher import _Handler


DEMO_USERS = [
    ('demo_employee', 'DemoEmployee2026!', 1),
    ('demo_manager', 'DemoManager2026!', 2),
    ('demo_executive', 'DemoExecutive2026!', 3),
]

WATCH_FOLDERS = [
    ('/watched/general', 1),
    ('/watched/managers', 2),
    ('/watched/executives', 3),
]


def main() -> None:
    db_path = str(Path(settings.data_dir) / 'app.db')
    with sqlite3.connect(db_path) as connection:
        for path, level in WATCH_FOLDERS:
            connection.execute(
                'INSERT INTO watch_folders(path, access_level, is_active) VALUES (?, ?, 1) '
                'ON CONFLICT(path) DO UPDATE SET access_level=excluded.access_level, is_active=1',
                (path, level),
            )
        for username, password, level in DEMO_USERS:
            row = connection.execute(
                'SELECT id FROM users WHERE username=?', (username,)
            ).fetchone()
            password_hash = hash_password(password)
            if row:
                connection.execute(
                    'UPDATE users SET password_hash=?, access_level=?, is_admin=0, '
                    'must_change_password=0, is_active=1 WHERE username=?',
                    (password_hash, level, username),
                )
            else:
                connection.execute(
                    'INSERT INTO users(username, password_hash, access_level, is_admin, '
                    'must_change_password, is_active) VALUES (?, ?, ?, 0, 0, 1)',
                    (username, password_hash, level),
                )
        connection.commit()

    handler = _Handler(db_path)
    files = list(Path(settings.watched_path).rglob('*'))
    indexed = 0
    for path in files:
        if path.is_file():
            handler._handle_file(str(path))
            indexed += 1

    print(f'Demo ready: {indexed} files checked, {len(DEMO_USERS)} users provisioned.')
    for username, password, level in DEMO_USERS:
        print(f'  Lv{level}: {username} / {password}')


if __name__ == '__main__':
    main()
