"""embed_device / rerank_device 設定のテスト。"""

import pytest
from pydantic import ValidationError

from app.config import Settings, settings
from app.services import embedder, reranker


def setup_function():
    reranker._load_model.cache_clear()
    embedder._load_model.cache_clear()


def teardown_function():
    reranker._load_model.cache_clear()
    embedder._load_model.cache_clear()


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, model_name, **kwargs):
        self.calls.append((model_name, kwargs))
        return object()


def test_defaults_are_cpu():
    assert Settings.model_fields["embed_device"].default == "cpu"
    assert Settings.model_fields["rerank_device"].default == "cpu"


def test_rerank_device_is_passed_to_cross_encoder(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(reranker, "CrossEncoder", recorder)
    monkeypatch.setattr(settings, "rerank_device", "cuda")

    reranker._load_model()

    assert recorder.calls == [(settings.rerank_model, {"device": "cuda"})]


def test_embed_device_is_passed_to_sentence_transformer(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(embedder, "SentenceTransformer", recorder)
    monkeypatch.setattr(settings, "embed_device", "cuda:1")

    embedder._load_model()

    assert recorder.calls == [(settings.embed_model, {"device": "cuda:1"})]


@pytest.mark.parametrize("value", ["cpu", "mps", "cuda", "cuda:0", "cuda:12"])
def test_valid_devices_are_accepted(value):
    s = Settings(secret_key="x", embed_device=value, rerank_device=value)
    assert s.embed_device == value
    assert s.rerank_device == value


@pytest.mark.parametrize("value", ["gpu", "CUDA", "cuda:", "cuda:a", " cpu", ""])
def test_invalid_devices_are_rejected(value):
    with pytest.raises(ValidationError):
        Settings(secret_key="x", rerank_device=value)
    with pytest.raises(ValidationError):
        Settings(secret_key="x", embed_device=value)
