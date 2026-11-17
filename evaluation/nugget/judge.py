"""ローカルの判定役（OpenAI 互換 /v1/chat/completions）で回答を判定する。

呼び出し 1: 文ごとのラベル 6 種。呼び出し 2: vital の事実ごとに full/partial/none。
標準ライブラリ (urllib) だけで呼ぶ。テストでは send を差し替える。
"""
from __future__ import annotations

import json
import re
import urllib.request
from typing import Callable, Mapping, Sequence

from evaluation.metrics import split_sentences

JUDGE_VERSION = "NJ-1"
DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_SEED = 0
DEFAULT_MAX_TOKENS = 1024

# 英語の内部名 → 日本語の表示名
LABELS = {
    "vital": "必須",
    "okay": "補足",
    "boilerplate": "定型",
    "irrelevant": "不要",
    "not_stated": "記載なし",
    "contradiction": "相反",
}
_LABEL_ALIASES = {**{v: k for k, v in LABELS.items()}, **{k: k for k in LABELS}}

FACT_LEVELS = ("full", "partial", "none")
_FACT_ALIASES = {
    "full": "full", "partial": "partial", "none": "none",
    "すべて": "full", "全て": "full", "全部": "full",
    "一部": "partial",
    "述べない": "none", "なし": "none", "無し": "none",
    "1": "full", "1.0": "full", "0.5": "partial", "0": "none", "0.0": "none",
}

# messages -> 返答の本文
Send = Callable[[list], str]


# ─── 送る関数（HTTP） ───────────────────────────────────────────
def make_http_send(
    base_url: str = DEFAULT_BASE_URL,
    model: str = "local",
    seed: int = DEFAULT_SEED,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: float = 300.0,
) -> Send:
    url = base_url.rstrip("/") + "/v1/chat/completions"

    def send(messages: list) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "seed": seed,
            "max_tokens": max_tokens,
            "stream": False,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (localhost 既定)
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"] or ""

    return send


# ─── 文の分割 ───────────────────────────────────────────────────
def split_answer(text: str) -> list[str]:
    """split_sentences に加えて改行でも分ける（「結論: …」の行を 1 文にする）。"""
    out: list[str] = []
    for s in split_sentences(text):
        for line in s.splitlines():
            line = line.strip()
            if line:
                out.append(line)
    return out


# ─── JSON を寛容に読む ──────────────────────────────────────────
def extract_json(text: str):
    """```json の囲みや前後の文が付いていても最初の JSON を読む。読めなければ ValueError。"""
    s = (text or "").strip()
    if not s:
        raise ValueError("返答が空")
    candidates = []
    for m in re.finditer(r"```[a-zA-Z]*\s*(.*?)```", s, re.S):
        candidates.append(m.group(1))
    candidates.append(s)
    dec = json.JSONDecoder()
    for cand in candidates:
        cand = cand.strip()
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass
        for opener in ("{", "["):  # 辞書を優先（文中の [1] などを拾わない）
            for i, ch in enumerate(cand):
                if ch != opener:
                    continue
                try:
                    obj, _ = dec.raw_decode(cand, i)
                except json.JSONDecodeError:
                    continue
                return obj
    raise ValueError("JSON が読めない")


# ─── プロンプト ─────────────────────────────────────────────────
def _fact_lines(nuggets: Sequence[Mapping]) -> str:
    return "\n".join(
        f"- {n['nid']}: {n['text']}（{'必須' if n.get('importance') == 'vital' else '補足'}）"
        for n in nuggets
    )


def build_prompt1(rule_text: str, question: str, sentences: Sequence[str], nuggets: Sequence[Mapping]) -> list:
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, start=1))
    labels = " / ".join(LABELS.values())
    user = f"""[判定手順 {JUDGE_VERSION} / 呼び出し1: 文のラベル]
次の規程本文・質問・番号付きの回答文・事実の一覧を読み、回答の各文に次の 6 つのラベルのどれか 1 つだけを付けてください。
- 必須: 「必須」とされた事実のどれかを述べている文
- 補足: 規程に合っているが、必須の事実ではない文
- 定型: 決まり文句（挨拶・「以上です」など）
- 不要: 質問と関係のない内容の文
- 記載なし: 規程にその情報が無いと述べる文
- 相反: 規程本文と食い違う断定をしている文

## 規程本文
{rule_text}

## 質問
{question}

## 事実の一覧
{_fact_lines(nuggets)}

## 回答の文（全 {len(sentences)} 文）
{numbered}

## 出力
JSON だけを返してください。説明は不要です。ラベルは文の番号順に、ちょうど {len(sentences)} 個。
ラベルは {labels} のいずれか。
{{"labels": ["必須", "補足"]}}"""
    return [
        {"role": "system", "content": "あなたは社内規程の回答を採点する厳密な判定役です。JSON だけを返します。"},
        {"role": "user", "content": user},
    ]


def build_prompt2(rule_text: str, question: str, answer: str, vitals: Sequence[Mapping]) -> list:
    facts = "\n".join(f"- {n['nid']}: {n['text']}" for n in vitals)
    user = f"""[判定手順 {JUDGE_VERSION} / 呼び出し2: 必須の事実ごとの充足]
次の回答が、下の「必須の事実」をそれぞれどこまで述べているかを判定してください。
- full: 事実の中身（数値・条件・境目を含む）をすべて述べている
- partial: 一部だけ述べている、または数値や条件が欠けている
- none: 述べていない（食い違っている場合も none）

## 規程本文
{rule_text}

## 質問
{question}

## 回答
{answer}

## 必須の事実
{facts}

## 出力
JSON だけを返してください。説明は不要です。事実はすべて、上の一覧の順で。
{{"facts": [{{"nid": "{vitals[0]['nid']}", "score": "full"}}]}}"""
    return [
        {"role": "system", "content": "あなたは社内規程の回答を採点する厳密な判定役です。JSON だけを返します。"},
        {"role": "user", "content": user},
    ]


# ─── 返答の解釈 ─────────────────────────────────────────────────
def parse_labels(raw: str, n_sentences: int) -> list[str]:
    obj = extract_json(raw)
    items = obj.get("labels") if isinstance(obj, dict) else obj
    if not isinstance(items, list):
        raise ValueError("labels の配列が無い")
    if len(items) != n_sentences:
        raise ValueError(f"ラベル数 {len(items)} が文の数 {n_sentences} と合わない")
    out: list[str] = []
    for it in items:
        if isinstance(it, dict):
            it = it.get("label", it.get("ラベル"))
        key = str(it).strip() if it is not None else ""
        if key not in _LABEL_ALIASES:
            raise ValueError(f"知らないラベル: {it!r}")
        out.append(_LABEL_ALIASES[key])
    return out


def _fact_level(v) -> str:
    key = str(v).strip().lower() if v is not None else ""
    if key not in _FACT_ALIASES:
        raise ValueError(f"知らない判定: {v!r}")
    return _FACT_ALIASES[key]


def parse_facts(raw: str, vital_nids: Sequence[str]) -> dict[str, str]:
    obj = extract_json(raw)
    items = obj.get("facts") if isinstance(obj, dict) else obj
    if not isinstance(items, list):
        raise ValueError("facts の配列が無い")
    if len(items) != len(vital_nids):
        raise ValueError(f"事実の数 {len(items)} が vital の数 {len(vital_nids)} と合わない")
    out: dict[str, str] = {}
    for pos, it in enumerate(items):
        if isinstance(it, dict):
            nid = it.get("nid", it.get("id", vital_nids[pos]))
            level = _fact_level(it.get("score", it.get("verdict", it.get("label"))))
        else:
            nid = vital_nids[pos]
            level = _fact_level(it)
        if nid in out:
            raise ValueError(f"事実 {nid} が重複")
        out[nid] = level
    if set(out) != set(vital_nids):
        raise ValueError("事実の nid が vital と合わない")
    return out


# ─── 1 回だけ頼み直す ───────────────────────────────────────────
def _ask(send: Send, messages: list, parse: Callable[[str], object]):
    """(結果 or None, エラー文 or None, 頼み直した回数)。"""
    msgs = list(messages)
    err: str | None = None
    for attempt in range(2):
        raw = None
        try:
            raw = send(msgs)
        except Exception as e:  # noqa: BLE001 通信の失敗も「判定できず」に回す
            err = f"送信に失敗: {type(e).__name__}: {e}"
        else:
            try:
                return parse(raw), None, attempt
            except ValueError as e:
                err = str(e)
        if attempt == 0 and raw is not None:
            msgs = list(messages) + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": f"形式が正しくありません（{err}）。指定した JSON だけを、もう一度返してください。",
                },
            ]
    return None, err, 1


# ─── 1 つの回答を判定する ───────────────────────────────────────
def judge_answer(
    send: Send,
    *,
    rule_text: str,
    question: str,
    answer: str,
    nuggets: Sequence[Mapping],
) -> dict:
    """{"sentences","labels","facts","status","error","retries"}。

    status は "ok" か "undetermined"（判定できず。点から外す。0 点にしない）。
    labels は内部名 (vital/okay/...) の配列、facts は {nid: full|partial|none}。
    """
    sentences = split_answer(answer)
    vitals = [n for n in nuggets if n.get("importance") == "vital"]
    vital_nids = [n["nid"] for n in vitals]
    result = {
        "sentences": sentences,
        "labels": None,
        "facts": None,
        "status": "undetermined",
        "error": None,
        "retries": 0,
    }
    if not sentences:  # 空の答え: 何も述べていない
        result.update(labels=[], facts={nid: "none" for nid in vital_nids}, status="ok")
        return result

    labels, err, r1 = _ask(
        send,
        build_prompt1(rule_text, question, sentences, nuggets),
        lambda raw: parse_labels(raw, len(sentences)),
    )
    result["retries"] += r1
    if labels is None:
        result["error"] = f"呼び出し1: {err}"
        return result

    facts, err, r2 = _ask(
        send,
        build_prompt2(rule_text, question, answer, vitals),
        lambda raw: parse_facts(raw, vital_nids),
    )
    result["retries"] += r2
    if facts is None:
        result["error"] = f"呼び出し2: {err}"
        return result

    result.update(labels=labels, facts=facts, status="ok")
    return result
