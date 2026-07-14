CREATE TABLE IF NOT EXISTS users (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT    UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    -- 1=一般, 2=管理職, 3=役員。「文書を閲覧できる上限レベル」であり、
    -- システム管理者権限 (is_admin) とは独立。admin だからといって
    -- 自動的に全文書 (Lv3) を読めるわけではない。
    access_level INTEGER NOT NULL DEFAULT 1,
    -- ユーザー管理・監視フォルダ設定・監査ログ閲覧等の「システム管理者」権限。
    -- 文書アクセスレベルとは独立したロールで、is_admin=1 でも access_level が
    -- 低ければその文書は読めない (項目7: 権限分離)。
    is_admin     INTEGER NOT NULL DEFAULT 0,
    -- デフォルト資格情報 (admin/admin) のまま残っている等、パスワード変更が
    -- 強制されている間は 1。/api/auth/password と /api/auth/me 以外の API を
    -- 403 でブロックする。
    must_change_password INTEGER NOT NULL DEFAULT 0,
    -- パスワードを最後に変更した UTC ISO8601 時刻。change_password 成功時に更新する。
    -- create_access_token が発行する iat (発行時刻) がこれより厳密に古いトークンは
    -- get_current_user で無効化する (項目高3: パスワード変更で既存 JWT を失効)。
    password_changed_at TEXT,
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
    -- どの監視フォルダにも一致せず既定 (Lv1=全員閲覧可) でインデックスされた場合 1。
    -- フォルダが追加されて一致するようになった resync 時に 0 へ戻る。
    unclassified INTEGER NOT NULL DEFAULT 0,
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

-- チャット監査ログ (誰が・何を聞き・何を根拠に・何を回答したか)
-- question は不正利用調査のために保存を継続するが、利用者がここに機密情報を
-- 書き込みうる残余リスクがある (項目高1: 設計判断として確定)。
-- answer (回答本文) はもう書き込まない。桁数だけを answer_chars に保存し、
-- 本文そのものは監査ログに残さない。既存カラムは後方互換のため残すのみ。
CREATE TABLE IF NOT EXISTS audit_logs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          INTEGER,
    username         TEXT    NOT NULL,
    question         TEXT    NOT NULL,
    -- JSON配列: [{"document_id", "chunk_index", "source_file" (basename化), "score", "access_level"}, ...]
    retrieved_chunks TEXT    NOT NULL DEFAULT '[]',
    answer           TEXT,
    answer_chars     INTEGER,
    error            TEXT,
    created_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

-- システム管理者操作の監査ログ (項目高4: 自己昇格等の検知可能化)。
CREATE TABLE IF NOT EXISTS admin_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER,
    username   TEXT,
    action     TEXT    NOT NULL,
    detail     TEXT,
    created_at TEXT    NOT NULL
);

-- 一度きりのデータマイグレーション適用済みフラグ (冪等性のため)。
CREATE TABLE IF NOT EXISTS schema_migrations (
    key        TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Phase3: 会話履歴の永続化
CREATE TABLE IF NOT EXISTS conversations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id),
    title      TEXT    NOT NULL DEFAULT '',
    created_at TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id),
    -- user | assistant
    role            TEXT    NOT NULL,
    content         TEXT    NOT NULL,
    -- JSON配列 (現行形式): [{"id", "document_id", "chunk_index", "source_file" (basename化), "score"}, ...]
    -- 本文 (content) は複製保存しない (項目5)。表示時に document_id から
    -- 権限再チェックの上で解決する。旧形式 (content キー入り) は起動時の
    -- 一度きりのマイグレーションで content を除去する (項目高2)。
    sources         TEXT    NOT NULL DEFAULT '[]',
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);

-- 初期管理者ユーザー (パスワード: admin / 本番では必ず変更)
-- bcrypt hash of "admin" (rounds=12)
-- must_change_password=1: このデフォルト資格情報のままでは /api/auth/password と
-- /api/auth/me 以外の API を使えない (項目6)。
INSERT OR IGNORE INTO users (username, password_hash, access_level, is_admin, must_change_password)
VALUES ('admin', '$2b$12$dgnPVyT8LOrpHv7VUEfL..SEBc1k5MyueqlZhIsveoQR7kXPx0L7e', 3, 1, 1);
