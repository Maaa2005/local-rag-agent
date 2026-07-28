"""
重い外部依存 (sentence-transformers / docling) をスタブして
ピュアロジックのみテスト可能にする。

実体モデルは Docker 環境で動かす想定で、ここでは
- チャンク分割
- スパースベクトル生成
- プロンプト整形
- bcrypt / JWT
- アクセスレベル判定
- API スキーマ
を中心に検証する。
"""
from __future__ import annotations

import os
import sys
import types
import tempfile
from pathlib import Path


def _stub_module(name: str, **attrs) -> types.ModuleType:
    """指定モジュールを sys.modules に挿入してインポート時のロードを回避する。"""
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# ─── sentence_transformers をスタブ ─────────────────────────────────────
class _StubSentenceTransformer:
    """SentenceTransformer の差し替え。encode は決定的ベクトルを返す。"""

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, normalize_embeddings: bool = True, show_progress_bar: bool = False):
        import numpy as np

        out = []
        for t in texts:
            seed = sum(ord(c) for c in t[:32]) or 1
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(1024).astype("float32")
            if normalize_embeddings:
                v /= max(float(np.linalg.norm(v)), 1e-8)
            out.append(v)
        return np.stack(out)


class _StubCrossEncoder:
    """CrossEncoder の差し替え。テストでは monkeypatch で predict を上書きする想定。"""

    def __init__(self, *args, **kwargs):
        pass

    def predict(self, pairs):
        return [0.0 for _ in pairs]


_stub_module(
    "sentence_transformers",
    SentenceTransformer=_StubSentenceTransformer,
    CrossEncoder=_StubCrossEncoder,
)


# ─── docling のスタブ ────────────────────────────────────────────────
_stub_module("docling")


class _StubDoclingResult:
    class document:  # type: ignore[misc]
        @staticmethod
        def export_to_markdown() -> str:
            return "# stub\n"


class _StubDoclingConverter:
    def convert(self, path: str):
        return _StubDoclingResult()


_stub_module(
    "docling.document_converter", DocumentConverter=_StubDoclingConverter
)


# ─── テスト用のデータディレクトリ・環境変数 ─────────────────────────────
_TMP_DATA = Path(tempfile.mkdtemp(prefix="rag_test_"))
_TMP_WATCH = Path(tempfile.mkdtemp(prefix="rag_watched_"))
os.environ.setdefault("DATA_DIR", str(_TMP_DATA))
os.environ.setdefault("WATCHED_PATH", str(_TMP_WATCH))
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("VLLM_BASE_URL", "http://localhost:8001/v1")
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-pytest")

HERE = Path(__file__).resolve()
BACKEND_DIR = HERE.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
