"""検索の権限プレフィルタ (build_access_filter) の検証。

実際の Qdrant 検索は GPU/サービスを要するためここでは行わず、リリース基準
「権限フィルタ正答率 100%」の中核であるフィルタ構造が正しいことを単体検証する。
`lte` の取り違えや is_active 条件の欠落といった回帰を防ぐ。

加えて、Qdrant のヒット結果を SQLite で再検証する _revalidate_against_sqlite
(項目2: SQLite/Qdrant の権限不整合防止、項目1: 未分類文書の除外、
項目3a: 削除直後の即時失効) も検証する。
"""
from __future__ import annotations

import asyncio
import sqlite3

import pytest

from app.services import retriever as retriever_module
from app.services.retriever import build_access_filter


def _conditions(user_level: int) -> dict:
    """FieldCondition を {key: condition} に整理する。"""
    f = build_access_filter(user_level)
    return {c.key: c for c in f.must}


def _not_conditions(user_level: int) -> dict:
    f = build_access_filter(user_level)
    return {c.key: c for c in (f.must_not or [])}


def test_filter_has_both_conditions():
    conds = _conditions(2)
    assert set(conds) == {"access_level", "is_active"}


def test_access_level_uses_lte_user_level():
    """access_level <= ユーザー権限 (権限外文書を除外)。gte 等の取り違え防止。"""
    cond = _conditions(2)["access_level"]
    assert cond.range.lte == 2
    assert cond.range.gte is None
    assert cond.range.lt is None
    assert cond.range.gt is None


def test_is_active_must_be_true():
    """無効化チャンク (is_active=False) を除外する条件が必ず存在する。"""
    cond = _conditions(3)["is_active"]
    assert cond.match.value is True


def test_filter_tracks_user_level():
    assert _conditions(1)["access_level"].range.lte == 1
    assert _conditions(3)["access_level"].range.lte == 3


def test_filter_excludes_unclassified_documents():
    """未分類 (unclassified=True) はどのユーザー権限でも検索対象外 (項目1)。"""
    cond = _not_conditions(2)["unclassified"]
    assert cond.match.value is True


# ─── _revalidate_against_sqlite ────────────────────────────────
@pytest.fixture()
def fake_documents_db(tmp_path, monkeypatch):
    import aiosqlite

    db_path = tmp_path / "revalidate.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE documents (
                id TEXT PRIMARY KEY,
                access_level INTEGER NOT NULL DEFAULT 1,
                status TEXT NOT NULL DEFAULT 'done',
                unclassified INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        conn.executemany(
            "INSERT INTO documents (id, access_level, status, unclassified) VALUES (?, ?, ?, ?)",
            [
                ("ok-doc", 1, "done", 0),
                ("downgraded-doc", 3, "done", 0),  # Qdrant はまだ古い access_level=1 のまま
                ("deleted-doc", 1, "deleted", 0),
                ("unclassified-doc", 1, "done", 1),
            ],
        )
        conn.commit()

    class _DB:
        async def fetchall(self, sql, params=()):
            async with aiosqlite.connect(str(db_path)) as conn:
                conn.row_factory = aiosqlite.Row
                async with conn.execute(sql, params) as cur:
                    return await cur.fetchall()

    monkeypatch.setattr(retriever_module, "db", _DB())


def test_revalidate_keeps_only_authorized_chunks(fake_documents_db):
    """Qdrant のペイロードが古く access_level=1 のままヒットしても、
    SQLite 側で access_level が上がっていれば (権限外になっていれば) 除外する。
    削除済み・未分類の文書も同様に除外し、正当な文書だけ残す。
    """
    chunks = [
        {"document_id": "ok-doc", "content": "a"},
        {"document_id": "downgraded-doc", "content": "b"},
        {"document_id": "deleted-doc", "content": "c"},
        {"document_id": "unclassified-doc", "content": "d"},
        {"document_id": "unknown-doc", "content": "e"},
    ]
    result = asyncio.run(retriever_module._revalidate_against_sqlite(chunks, user_access_level=2))
    assert [c["document_id"] for c in result] == ["ok-doc"]


def test_revalidate_returns_empty_when_no_document_ids():
    """document_id を1つも持たないヒットは SQLite で真偽を検証できないため、
    fail-open で通すのではなく fail-closed で全て弾く (項目高8)。
    """
    chunks = [{"content": "no document_id here"}]
    result = asyncio.run(retriever_module._revalidate_against_sqlite(chunks, user_access_level=3))
    assert result == []
