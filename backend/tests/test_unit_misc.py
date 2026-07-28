"""残りの軽量ユニットテスト: LLM コンテキスト整形 / テキストパース / アクセスレベル判定。"""
import sqlite3
import tempfile
from pathlib import Path

import pytest

from app.services.llm import _build_context, SYSTEM_PROMPT
from app.services.parser import _parse_text
from app.services.watcher import SyncDatabase, _get_access_level_for_path


# ─── LLM プロンプト整形 ──────────────────────────────────────
def test_build_context_lists_sources_with_index():
    chunks = [
        {"content": "規定A", "source_file": "/watched/rules.pdf"},
        {"content": "規定B", "source_file": "/watched/handbook.docx"},
    ]
    ctx = _build_context(chunks)
    assert "[1] rules.pdf" in ctx
    assert "[2] handbook.docx" in ctx
    assert "規定A" in ctx and "規定B" in ctx


def test_build_context_uses_unknown_when_missing():
    ctx = _build_context([{"content": "本文"}])
    assert "[1] 不明" in ctx


def test_build_context_strips_full_path_to_basename():
    """LLM コンテキストへの機密ディレクトリパス漏洩防止 (フルパス → ファイル名)。"""
    chunks = [{"content": "本文", "source_file": "/watched/exec/secret-project/plan.docx"}]
    ctx = _build_context(chunks)
    assert "[1] plan.docx" in ctx
    assert "exec" not in ctx
    assert "secret-project" not in ctx


def test_build_context_strips_windows_style_path_to_basename():
    chunks = [{"content": "本文", "source_file": r"\\smbserver\exec\budget.xlsx"}]
    ctx = _build_context(chunks)
    assert "[1] budget.xlsx" in ctx
    assert "smbserver" not in ctx


def test_system_prompt_forbids_external_knowledge():
    """ハルシネーション抑止の文言が含まれていること。"""
    assert "参考情報" in SYSTEM_PROMPT
    assert "資料には含まれていません" in SYSTEM_PROMPT
    assert "日本語" in SYSTEM_PROMPT


# ─── テキストパース ──────────────────────────────────────
def test_parse_text_utf8():
    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as f:
        f.write("こんにちは世界\nABC")
        path = Path(f.name)
    try:
        assert "こんにちは世界" in _parse_text(path)
    finally:
        path.unlink()


def test_parse_text_cp932_fallback():
    """UTF-8 でデコード失敗 → cp932 へフォールバック。"""
    with tempfile.NamedTemporaryFile("wb", suffix=".txt", delete=False) as f:
        f.write("会議議事録".encode("cp932"))
        path = Path(f.name)
    try:
        decoded = _parse_text(path)
        assert "会議議事録" in decoded
    finally:
        path.unlink()


# ─── watch_folders のアクセスレベル判定 ──────────────────────
@pytest.fixture()
def sync_db(tmp_path):
    db_path = tmp_path / "test.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE watch_folders ("
            "id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL,"
            " access_level INTEGER NOT NULL DEFAULT 1, is_active INTEGER NOT NULL DEFAULT 1,"
            " created_at TEXT NOT NULL DEFAULT (datetime('now')))"
        )
        conn.executemany(
            "INSERT INTO watch_folders (path, access_level, is_active) VALUES (?, ?, ?)",
            [
                ("/watched/general", 1, 1),
                ("/watched/general/exec", 3, 1),
                ("/watched/manager", 2, 1),
                ("/watched/inactive", 3, 0),
            ],
        )
        conn.commit()
    return SyncDatabase(str(db_path))


def test_longest_prefix_wins(sync_db):
    """/watched/general/exec の方が長いので役員レベル (3) を返す。"""
    lv = _get_access_level_for_path("/watched/general/exec/report.pdf", sync_db)
    assert lv == 3


def test_general_path_returns_1(sync_db):
    lv = _get_access_level_for_path("/watched/general/notes.md", sync_db)
    assert lv == 1


def test_manager_path_returns_2(sync_db):
    lv = _get_access_level_for_path("/watched/manager/kpi.xlsx", sync_db)
    assert lv == 2


def test_unknown_path_defaults_to_1(sync_db):
    lv = _get_access_level_for_path("/somewhere/else/file.txt", sync_db)
    assert lv == 1


def test_inactive_folder_ignored(sync_db):
    """is_active=0 のフォルダに一致した場合は None (インデックス対象外) を返す。

    以前はデフォルト 1 (一般) にフォールバックしていたが、これは無効化した
    フォルダ配下のファイルが更新されると全員閲覧可で再インデックスされてしまう
    下方向リークだった。fail-closed にするため None を返す仕様に変更。
    """
    lv = _get_access_level_for_path("/watched/inactive/file.txt", sync_db)
    assert lv is None


def test_sibling_dir_does_not_inherit_level(sync_db):
    """境界一致: /watched/manager-public は /watched/manager の権限を継承しない。

    生の startswith では `/watched/manager-public/...` が `/watched/manager` に
    誤マッチして Lv2 になっていた (権限の誤適用)。境界一致で既定 1 に落ちること。
    """
    lv = _get_access_level_for_path("/watched/manager-public/notice.pdf", sync_db)
    assert lv == 1


def test_sibling_of_confidential_falls_back_to_parent(sync_db):
    """/watched/general/exec-archive は exec(3) でなく親 general(1) 扱い。"""
    lv = _get_access_level_for_path("/watched/general/exec-archive/old.pdf", sync_db)
    assert lv == 1


def test_exact_folder_path_matches(sync_db):
    """フォルダパスと完全一致するパスも一致と見なす。"""
    lv = _get_access_level_for_path("/watched/manager", sync_db)
    assert lv == 2


def test_trailing_slash_in_folder_is_normalized(sync_db, tmp_path):
    """登録パスに末尾スラッシュがあっても境界一致が壊れない。"""
    import sqlite3 as _sqlite
    db_path = tmp_path / "ts.db"
    with _sqlite.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE watch_folders (id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL,"
            " access_level INTEGER NOT NULL DEFAULT 1, is_active INTEGER NOT NULL DEFAULT 1,"
            " created_at TEXT NOT NULL DEFAULT (datetime('now')))"
        )
        conn.execute(
            "INSERT INTO watch_folders (path, access_level, is_active) VALUES (?, ?, ?)",
            ("/watched/hr/", 3, 1),
        )
        conn.commit()
    db = SyncDatabase(str(db_path))
    assert _get_access_level_for_path("/watched/hr/salary.pdf", db) == 3
    assert _get_access_level_for_path("/watched/hr-team/x.pdf", db) == 1
