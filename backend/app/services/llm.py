from typing import AsyncIterator

from openai import AsyncOpenAI

from app.config import settings

_client = AsyncOpenAI(base_url=settings.vllm_base_url, api_key="NONE")

SYSTEM_PROMPT = """あなたは社内情報専門のアシスタントです。
以下の【参考情報】のみを根拠として質問に回答してください。
参考情報に含まれていない情報については「その情報は現在の資料には含まれていません」と回答してください。
回答は必ず日本語で行ってください。"""


def _build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        source = c.get("source_file", "不明")
        parts.append(f"[{i}] {source}\n{c['content']}")
    return "\n\n".join(parts)


async def stream_answer(question: str, chunks: list[dict]) -> AsyncIterator[str]:
    context = _build_context(chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"【参考情報】\n{context}\n\n【質問】\n{question}"},
    ]
    stream = await _client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
