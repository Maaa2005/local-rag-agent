"""検索の権限プレフィルタ (build_access_filter) の検証。

実際の Qdrant 検索は GPU/サービスを要するためここでは行わず、リリース基準
「権限フィルタ正答率 100%」の中核であるフィルタ構造が正しいことを単体検証する。
`lte` の取り違えや is_active 条件の欠落といった回帰を防ぐ。
"""
from __future__ import annotations

from app.services.retriever import build_access_filter


def _conditions(user_level: int) -> dict:
    """FieldCondition を {key: condition} に整理する。"""
    f = build_access_filter(user_level)
    return {c.key: c for c in f.must}


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
