"""indexer._chunk_text の挙動を検証する。"""
import pytest

from app.services.indexer import _chunk_text


def test_short_text_returns_single_chunk():
    text = "短いテキストです。"
    chunks = _chunk_text(text, size=500, overlap=50)
    assert chunks == [text]


def test_long_text_is_split_with_overlap():
    text = "あ" * 1200
    chunks = _chunk_text(text, size=500, overlap=50)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) <= 500 + 30


def test_japanese_sentence_boundary_preferred():
    """500 文字を超えた直後の。で区切られることを期待。"""
    text = "前段の本文がしばらく続きます。" + "あ" * 480 + "ここで区切る。" + "後段の話が始まります。"
    chunks = _chunk_text(text, size=500, overlap=50)
    assert len(chunks) >= 2
    assert chunks[0].rstrip().endswith("。")


def test_chunks_are_non_empty():
    text = "\n\n\n\n" + "テキスト" * 200 + "\n\n"
    chunks = _chunk_text(text, size=300, overlap=30)
    assert all(c.strip() for c in chunks)


def test_empty_text_returns_empty_or_single_blank():
    chunks = _chunk_text("   \n\n\n  ", size=500, overlap=50)
    assert chunks == [""] or chunks == []


def test_progress_each_iteration():
    """size > overlap であれば必ず前進し無限ループしない。"""
    text = "ab" * 5000
    chunks = _chunk_text(text, size=200, overlap=50)
    assert len(chunks) > 1
    assert sum(len(c) for c in chunks) >= len(text) - 200
