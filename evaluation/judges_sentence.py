"""文ごとに 0/1 で採点する採点役 (M4-S)。

回答を metrics.split_sentences で分け、番号を付けて採点役に見せる。忠実さ (文ごとの 0/1) と
関連性 (回答全体の 0/1) は別々の 2 回の呼び出しで取る。期待キーワードは渡さない。
取得本文は切り詰めず全文を渡す。

出力の読み取りは厳格: 要素数・文番号・型が 1 つでも期待と違えば ScoringFailure。
切り詰め・補完・既定値での穴埋めはしない (旧 judges は欠けた値を黙って 0.0 にしていた)。
プロンプトの文面は定数で、prompt_sha256() で指示文が変わっていないことを記録できる。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from evaluation.judges import _JSON_RE
from evaluation.metrics import split_sentences


class ScoringFailure(Exception):
    """採点役の出力が期待の形でない (読み取れない・件数や型が違う)。raw に生テキストを持つことがある。"""

    def __init__(self, message: str, raw: dict | None = None):
        super().__init__(message)
        self.raw = raw or {}


@dataclass
class SentenceJudgement:
    sentences: list[str]      # split_sentences(answer) の結果
    verdicts: list[int]       # 文ごとの 0/1。sentences と同じ長さ・同じ順
    reasons: list[str]        # 文ごとの根拠
    faithfulness: float       # verdicts の平均。文が 0 個なら 0.0
    relevance: int            # 0/1
    raw: dict                 # {"faithfulness": 採点役の生テキスト, "relevance": 生テキスト}


# ─── プロンプト (文面は定数。変えると prompt_sha256 が変わる) ───────────────
FAITHFULNESS_SYSTEM = "あなたは公平で厳格な RAG 評価者です。必ず JSON のみで答えてください。"

FAITHFULNESS_PROMPT = """回答の各文が、取得した本文に基づいているかを 1 文ずつ判定してください。

【質問】
{question}

【取得した本文】
{chunks}

【回答の文】(全 {n} 文)
{numbered}

判定の基準:
- 各文について、その文に含まれる事実・条件・数値・結論のすべてが取得した本文に書かれていれば 1。
- 一つでも本文と矛盾する部分、または本文に根拠がない部分があれば 0。
- 一般知識や推測で補ってはいけません。本文に書かれていることだけを根拠にしてください。
- 「資料に記載がない」「お答えできない」のような文も主張として扱います。本文にその答えが書かれていれば 0、本当に書かれていなければ 1。
- 文番号 1 から {n} まで、すべての文を 1 回ずつ判定してください。文を飛ばしたり、まとめたり、増やしたりしてはいけません。

出力は JSON のみ。次の形式で、判定は {n} 個にしてください:
{{"判定": [{{"文番号": 1, "判定": 1, "根拠": "本文の該当箇所。なければ「なし」"}}, {{"文番号": 2, "判定": 0, "根拠": "..."}}]}}"""

RELEVANCE_SYSTEM = "あなたは公平な RAG 評価者です。必ず JSON のみで答えてください。"

RELEVANCE_PROMPT = """回答が質問に答えているかを判定してください。

【質問】
{question}

【回答】
{answer}

判定の基準:
- 回答が、質問の求める情報を答えていれば 1。
- 回答が答えることを断る、はぐらかす、または質問とは別の話をしているなら 0。
- 回答の内容が正しいかどうかはここでは問いません。質問に答える形になっているかだけを見てください。

出力は JSON のみ: {{"関連": 1, "根拠": "判定の理由"}}"""


def prompt_sha256() -> str:
    """指示文 (システム文と本文の定数) の sha256。後から指示文を直していないことの記録用。"""
    h = hashlib.sha256()
    for part in (FAITHFULNESS_SYSTEM, FAITHFULNESS_PROMPT, RELEVANCE_SYSTEM, RELEVANCE_PROMPT):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _format_chunks_full(chunks: list[dict]) -> str:
    """本文は切り詰めず全文。出典名を付ける。"""
    if not chunks:
        return "(なし)"
    return "\n\n".join(
        f"[{i + 1}] 出典: {c.get('source_file', '?')}\n{c.get('content', '')}"
        for i, c in enumerate(chunks)
    )


def _numbered(sentences: list[str]) -> str:
    return "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))


# ─── 厳格なパーサ ────────────────────────────────────────────────
def _load_json_object(text: str, what: str) -> dict:
    m = _JSON_RE.search(text or "")
    if not m:
        raise ScoringFailure(f"{what}: JSON が取り出せない")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        raise ScoringFailure(f"{what}: JSON が壊れている ({exc.msg})") from exc
    if not isinstance(obj, dict):
        raise ScoringFailure(f"{what}: JSON がオブジェクトでない")
    return obj


def _is_int(v) -> bool:
    """int そのものだけ。bool・float・文字列は不可。"""
    return type(v) is int


def parse_sentence_verdicts(text: str, n: int) -> tuple[list[int], list[str]]:
    """忠実さの出力を (verdicts, reasons) にする。文番号順に並べ直す以外の補正はしない。"""
    obj = _load_json_object(text, "忠実さ")
    items = obj.get("判定")
    if not isinstance(items, list):
        raise ScoringFailure("忠実さ: \"判定\" が無い、またはリストでない")
    if len(items) != n:
        raise ScoringFailure(f"忠実さ: 判定の要素数 {len(items)} が文の数 {n} と違う")
    by_no: dict[int, tuple[int, str]] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ScoringFailure(f"忠実さ: 判定の要素がオブジェクトでない: {item!r}")
        no = item.get("文番号")
        if not _is_int(no):
            raise ScoringFailure(f"忠実さ: 文番号が int でない: {no!r}")
        if not 1 <= no <= n:
            raise ScoringFailure(f"忠実さ: 文番号 {no} が範囲 1〜{n} の外")
        if no in by_no:
            raise ScoringFailure(f"忠実さ: 文番号 {no} が重複")
        v = item.get("判定")
        if not _is_int(v) or v not in (0, 1):
            raise ScoringFailure(f"忠実さ: 文 {no} の判定が int の 0/1 でない: {v!r}")
        reason = item.get("根拠")
        if not isinstance(reason, str):
            raise ScoringFailure(f"忠実さ: 文 {no} の根拠が文字列でない: {reason!r}")
        by_no[no] = (v, reason)
    missing = [i for i in range(1, n + 1) if i not in by_no]
    if missing:
        raise ScoringFailure(f"忠実さ: 文番号 {missing} の判定が無い")
    ordered = [by_no[i] for i in range(1, n + 1)]
    return [v for v, _ in ordered], [r for _, r in ordered]


def parse_relevance(text: str) -> int:
    """関連性の出力を 0/1 にする。"""
    obj = _load_json_object(text, "関連性")
    v = obj.get("関連")
    if not _is_int(v) or v not in (0, 1):
        raise ScoringFailure(f"関連性: \"関連\" が int の 0/1 でない: {v!r}")
    return v


async def _ask(provider, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    buf: list[str] = []
    async for tok in provider.stream(messages):
        buf.append(tok)
    return "".join(buf)


async def judge_sentences(provider, question: str, chunks: list[dict], answer: str) -> SentenceJudgement:
    """回答を文ごとに 0/1 で採点する。文が 0 個なら採点役を呼ばない。"""
    sentences = split_sentences(answer or "")
    if not sentences:
        return SentenceJudgement(
            sentences=[], verdicts=[], reasons=[], faithfulness=0.0, relevance=0,
            raw={"faithfulness": "", "relevance": ""},
        )
    n = len(sentences)
    faith_prompt = FAITHFULNESS_PROMPT.format(
        question=question, chunks=_format_chunks_full(chunks), n=n, numbered=_numbered(sentences),
    )
    rel_prompt = RELEVANCE_PROMPT.format(question=question, answer=answer)
    raw_faith = await _ask(provider, FAITHFULNESS_SYSTEM, faith_prompt)
    raw_rel = await _ask(provider, RELEVANCE_SYSTEM, rel_prompt)
    raw = {"faithfulness": raw_faith, "relevance": raw_rel}
    try:
        verdicts, reasons = parse_sentence_verdicts(raw_faith, n)
        relevance = parse_relevance(raw_rel)
    except ScoringFailure as exc:
        exc.raw = raw
        raise
    return SentenceJudgement(
        sentences=sentences,
        verdicts=verdicts,
        reasons=reasons,
        faithfulness=sum(verdicts) / n,
        relevance=relevance,
        raw=raw,
    )
