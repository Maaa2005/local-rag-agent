"""build_sparse の挙動を検証する。"""
from app.services.sparse import build_sparse, _VOCAB_SIZE


def test_empty_string_returns_empty():
    idx, val = build_sparse("")
    assert idx == []
    assert val == []


def test_japanese_bigram_present():
    """CJK bigram を生成すること。"""
    idx, val = build_sparse("社内規定")
    assert len(idx) == 3
    assert len(val) == 3
    assert abs(sum(val) - 1.0) < 1e-6


def test_ascii_lowercased():
    idx_upper, _ = build_sparse("Hello World")
    idx_lower, _ = build_sparse("hello world")
    assert idx_upper == idx_lower


def test_mixed_japanese_ascii():
    idx, val = build_sparse("Qdrantは社内DB")
    assert len(idx) > 0
    assert abs(sum(val) - 1.0) < 1e-6


def test_indices_sorted_unique():
    idx, val = build_sparse("ああああああいいい")
    assert idx == sorted(idx)
    assert len(idx) == len(set(idx))
    assert all(0 <= i < _VOCAB_SIZE for i in idx)


def test_tf_normalization_sums_to_one():
    idx, val = build_sparse("社内規定の改訂版です。改訂版です改訂版です。")
    assert abs(sum(val) - 1.0) < 1e-6


def test_repeated_token_higher_weight():
    """繰り返し出現するトークンは TF が高くなる。"""
    idx, val = build_sparse("社員社員社員社員社員")
    assert max(val) > 0.5
