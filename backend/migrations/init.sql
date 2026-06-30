CREATE TABLE IF NOT EXISTS users (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT    UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    -- 1=一般, 2=管理職, 3=役員
    access_level INTEGER NOT NULL DEFAULT 1,
    is_active    INTEGER NOT NULL DEFAULT 1,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS watch_folders (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    path         TEXT    UNIQUE NOT NULL,
    access_level INTEGER NOT NULL DEFAULT 1,
    is_active    INTEGER NOT NULL DEFAULT 1,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS documents (
    id           TEXT    PRIMARY KEY,  -- UUID
    source_path  TEXT    UNIQUE NOT NULL,
    file_hash    TEXT    NOT NULL,
    access_level INTEGER NOT NULL DEFAULT 1,
    file_type    TEXT    NOT NULL,
    -- pending | processing | done | failed
    status       TEXT    NOT NULL DEFAULT 'pending',
    chunk_count  INTEGER NOT NULL DEFAULT 0,
    error_msg    TEXT,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS tasks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT    NOT NULL REFERENCES documents(id),
    -- pending | processing | done | failed
    status      TEXT    NOT NULL DEFAULT 'pending',
    attempts    INTEGER NOT NULL DEFAULT 0,
    error_msg   TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 初期管理者ユーザー (パスワード: admin / 本番では必ず変更)
-- bcrypt hash of "admin" (rounds=12)
INSERT OR IGNORE INTO users (username, password_hash, access_level)
VALUES ('admin', '$2b$12$dgnPVyT8LOrpHv7VUEfL..SEBc1k5MyueqlZhIsveoQR7kXPx0L7e', 3);
