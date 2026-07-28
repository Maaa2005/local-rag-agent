"""LLM 推論の公開エントリポイント。

プロバイダ自体は app.services.providers パッケージに分離。ここでは
- プロンプト整形
- DB から資格情報をロードしてプロバイダを取得
- ストリーミング応答を yield
を担う。
"""
from __future__ import annotations

import ntpath
import posixpath
import re
from typing import AsyncIterator

from app.config import settings
from app.services.providers import LLMProvider, get_meta, get_provider, reset_instances

SYSTEM_PROMPT = """あなたは社内情報専門のアシスタントです。
以下の【参考情報】のみを根拠として質問に回答してください。
参考情報に含まれていない情報については「その情報は現在の資料には含まれていません」と回答してください。
回答は必ず日本語で行ってください。
結論と必要な手続きを3文以内で簡潔に回答してください。
回答中で根拠にした参考情報には、該当箇所の文末に [1] のようにその番号を付けて引用してください。"""


def _basename(path: str) -> str:
    """フルパスからファイル名のみを取り出す。

    LLM コンテキストに機密ディレクトリ構成 (組織名・部署名等を含む監視フォルダ
    のフルパス) が漏れないよう、ここでファイル名だけに縮約する。SSE で UI に
    返す sources 側は認証済みユーザー向けなので従来どおりフルパスのまま。
    Windows 区切り (\\) / POSIX 区切り (/) のどちらでも対応する。
    """
    name = ntpath.basename(path)
    return posixpath.basename(name)


def _build_context(chunks: list[dict]) -> str:
    parts = []
    remaining_chars = 240
    for i, c in enumerate(chunks, 1):
        if remaining_chars <= 0:
            break
        source = c.get("source_file", "不明")
        if source != "不明":
            source = _basename(source)
        prefix = f"[{i}] {source}\n"
        available = max(0, remaining_chars - len(prefix))
        content = str(c['content'])[:available]
        parts.append(prefix + content)
        remaining_chars -= len(prefix) + len(content)
    return "\n\n".join(parts)


def _relevant_excerpt(question: str, content: str, limit: int = 190) -> str:
    normalized_question = re.sub(r'\W+', '', question)
    qgrams = {
        normalized_question[index:index + 2]
        for index in range(max(0, len(normalized_question) - 1))
    }
    sentences = [
        item.strip()
        for item in re.split(r'(?<=[。！？])|\n+', content)
        if item.strip() and not item.lstrip().startswith('#')
    ]
    if not sentences:
        return content[:limit]
    scored = []
    for index, sentence in enumerate(sentences):
        normalized = re.sub(r'\W+', '', sentence)
        grams = {
            normalized[offset:offset + 2]
            for offset in range(max(0, len(normalized) - 1))
        }
        scored.append((len(qgrams & grams), index, sentence))
    selected = sorted(scored, key=lambda item: (-item[0], item[1]))[:3]
    selected.sort(key=lambda item: item[1])
    excerpt = ''
    for _score, _index, sentence in selected:
        separator = '\n' if excerpt else ''
        if len(excerpt) + len(separator) + len(sentence) <= limit:
            excerpt += separator + sentence
    return excerpt or max(scored, key=lambda item: item[0])[2][:limit]


def _build_messages(question: str, chunks: list[dict]) -> list[dict]:
    focused_chunks = []
    if chunks:
        first = dict(chunks[0])
        first['content'] = _relevant_excerpt(question, str(first.get('content', '')))
        focused_chunks.append(first)
    context = _build_context(focused_chunks)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"【参考情報】\n{context}\n\n【質問】\n{question}"},
    ]


async def _resolve_provider(name: str) -> LLMProvider:
    # ローカル LLM 固定アプリのため資格情報は不要 (vLLM は requires_credentials=False)。
    meta = get_meta(name)
    if meta.requires_credentials:
        raise ValueError(f"Provider '{name}' は資格情報が必要ですが本アプリは未対応です")
    return await get_provider(name)


async def stream_answer(
    question: str, chunks: list[dict], provider_name: str | None = None
) -> AsyncIterator[str]:
    name = provider_name or settings.llm_provider
    provider = await _resolve_provider(name)
    messages = _build_messages(question, chunks)
    async for token in provider.stream(messages):
        yield token


# 後方互換: テストで provider を直差し替えするためのフック
def set_provider(provider: LLMProvider | None) -> None:
    """グローバル既定プロバイダを差し替える (None でリセット)。"""
    if provider is None:
        reset_instances()
        return
    # 既存 vLLM スロットを上書きしてしまうのが最も簡単
    from app.services.providers import base as _base
    _base._instances[settings.llm_provider] = provider


def get_provider_sync(name: str | None = None) -> LLMProvider:
    """同期版 (テスト用)。インスタンスが未生成ならエラー。"""
    from app.services.providers import base as _base
    key = name or settings.llm_provider
    if key not in _base._instances:
        raise RuntimeError(f"Provider '{key}' not initialised")
    return _base._instances[key]
