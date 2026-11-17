"""ローカルの判定役（OpenAI 互換 /v1/chat/completions）で回答を判定する。

呼び出し 1: 文ごとのラベル 6 種。呼び出し 2: vital の事実ごとに full/partial/none と根拠。
NJ-2: 根拠は答えからの写し（evidence）。空、または回答に無い写しは none に下げ、downgraded に残す。
NJ-3（既定）: 回答を番号付きの文で渡し、根拠は文の番号（sentences）。番号が空・範囲外だけなら none に下げる。
標準ライブラリ (urllib) だけで呼ぶ。テストでは send を差し替える。
"""
from __future__ import annotations

import json
import re
import urllib.request
from typing import Callable, Mapping, Sequence

from evaluation.metrics import split_sentences

JUDGE_VERSIONS = ("NJ-2", "NJ-3")
JUDGE_VERSION = "NJ-3"  # 既定
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


def build_prompt1(
    rule_text: str, question: str, sentences: Sequence[str], nuggets: Sequence[Mapping], version: str = JUDGE_VERSION
) -> list:
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, start=1))
    labels = " / ".join(LABELS.values())
    user = f"""[判定手順 {version} / 呼び出し1: 文のラベル]
次の規程本文・質問・番号付きの回答文・事実の一覧を読み、回答の各文に次の 6 つのラベルのどれか 1 つだけを付けてください。
- 必須: 「必須」とされた事実のどれかを述べている文
- 補足: 規程に合っているが、必須の事実ではない文
- 定型: 決まり文句（挨拶・「以上です」など）
- 不要: 質問と関係のない内容の文
- 記載なし: 規程にその情報が無いと述べる文
- 相反: 規程本文と食い違う断定をしている文

注意: 「結論: …」の行は定型ではない。規程に合う結論なら 必須、規程と食い違う結論なら 相反 とする。

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


def build_prompt2(
    rule_text: str, question: str, answer: str, vitals: Sequence[Mapping], version: str = "NJ-2"
) -> list:
    """NJ-2 の呼び出し 2（答えからの写し）。"""
    facts = "\n".join(f"- {n['nid']}: {n['text']}" for n in vitals)
    user = f"""[判定手順 {version} / 呼び出し2: 必須の事実ごとの充足]
次の回答が、下の「必須の事実」をそれぞれどこまで述べているかを判定してください。
- full: 事実の中身（数値・条件・境目を含む）をすべて述べている
- partial: 一部だけ述べている、または数値や条件が欠けている
- none: 述べていない（食い違っている場合も none）

evidence は回答の文から一字一句そのまま写す。事実の数値・条件・境目が回答に書かれていなければ full にしない。結論だけ合っていて理由の数値が無い時は partial か none。
full か partial の時は、その根拠になる回答の部分を evidence に必ず写す。none の時は evidence は空でよい。

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
{{"facts": [{{"nid": "{vitals[0]['nid']}", "score": "full", "evidence": "回答からそのまま写した部分"}}]}}"""
    return [
        {"role": "system", "content": "あなたは社内規程の回答を採点する厳密な判定役です。JSON だけを返します。"},
        {"role": "user", "content": user},
    ]


def build_prompt2_nj3(
    rule_text: str, question: str, sentences: Sequence[str], vitals: Sequence[Mapping], version: str = "NJ-3"
) -> list:
    """NJ-3 の呼び出し 2（回答の文の番号で根拠を示す）。"""
    facts = "\n".join(f"- {n['nid']}: {n['text']}" for n in vitals)
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, start=1))
    user = f"""[判定手順 {version} / 呼び出し2: 必須の事実ごとの充足]
次の回答（番号付きの文）が、下の「必須の事実」をそれぞれどこまで述べているかを判定してください。
- full: 事実の中身（数値・条件・境目を含む）をすべて述べている
- partial: 一部だけ述べている、または数値や条件が欠けている
- none: 述べていない（食い違っている場合も none）

sentences にはその事実を述べている回答の文の番号を書く（規程本文や事実の文ではなく、回答の文の番号）。事実の数値・条件・境目が書かれた文が無ければ full にしない。結論だけ合っていて理由の数値が無い時は partial か none。
full か partial の時は、根拠になる回答の文の番号を sentences に必ず書く。none の時は空でよい。

## 規程本文
{rule_text}

## 質問
{question}

## 回答の文（全 {len(sentences)} 文）
{numbered}

## 必須の事実
{facts}

## 出力
JSON だけを返してください。説明は不要です。事実はすべて、上の一覧の順で。sentences は 1 から {len(sentences)} までの番号の配列。
{{"facts": [{{"nid": "{vitals[0]['nid']}", "score": "full", "sentences": [1, 2]}}]}}"""
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


def parse_facts_detailed(raw: str, vital_nids: Sequence[str]) -> tuple[dict[str, str], dict[str, str]]:
    """({nid: full|partial|none}, {nid: evidence})。evidence が無い形式は空文字。"""
    obj = extract_json(raw)
    items = obj.get("facts") if isinstance(obj, dict) else obj
    if not isinstance(items, list):
        raise ValueError("facts の配列が無い")
    if len(items) != len(vital_nids):
        raise ValueError(f"事実の数 {len(items)} が vital の数 {len(vital_nids)} と合わない")
    out: dict[str, str] = {}
    evid: dict[str, str] = {}
    for pos, it in enumerate(items):
        if isinstance(it, dict):
            nid = it.get("nid", it.get("id", vital_nids[pos]))
            level = _fact_level(it.get("score", it.get("verdict", it.get("label"))))
            ev = it.get("evidence")
            ev = "" if ev is None else str(ev)
        else:
            nid = vital_nids[pos]
            level = _fact_level(it)
            ev = ""
        if nid in out:
            raise ValueError(f"事実 {nid} が重複")
        out[nid] = level
        evid[nid] = ev
    if set(out) != set(vital_nids):
        raise ValueError("事実の nid が vital と合わない")
    return out, evid


def parse_facts(raw: str, vital_nids: Sequence[str]) -> dict[str, str]:
    return parse_facts_detailed(raw, vital_nids)[0]


def _squash(text: str) -> str:
    """空白・改行・全角空白をすべて除く（正規表現の空白は全角空白も含む）。"""
    return re.sub(r"\s+", "", text or "")


def check_evidence(
    facts: Mapping[str, str], evidence: Mapping[str, str], answer: str
) -> tuple[dict[str, str], list[str]]:
    """full/partial で evidence が空、または回答に（空白を除いて）部分一致しない事実を none に下げる。

    (下げた後の facts, 下げた nid の一覧) を返す。none の事実は evidence が無くてよい。
    """
    haystack = _squash(answer)
    checked: dict[str, str] = {}
    downgraded: list[str] = []
    for nid, level in facts.items():
        if level != "none":
            ev = _squash(evidence.get(nid, ""))
            if not ev or ev not in haystack:
                level = "none"
                downgraded.append(nid)
        checked[nid] = level
    return checked, downgraded


def _sentence_numbers(v) -> list[int]:
    """sentences の値を整数の番号の並びにする。読めない要素は捨てる（範囲の検査は別）。"""
    if v is None:
        return []
    if not isinstance(v, (list, tuple)):
        v = [v]
    out: list[int] = []
    for x in v:
        if isinstance(x, bool):
            continue
        if isinstance(x, int):
            out.append(x)
        elif isinstance(x, float) and x == int(x):
            out.append(int(x))
        elif isinstance(x, str) and re.fullmatch(r"\s*\d+\s*", x):
            out.append(int(x))
    return out


def parse_facts_sentences(raw: str, vital_nids: Sequence[str]) -> tuple[dict[str, str], dict[str, list[int]]]:
    """NJ-3: ({nid: full|partial|none}, {nid: [文の番号]})。sentences が無ければ空。"""
    obj = extract_json(raw)
    items = obj.get("facts") if isinstance(obj, dict) else obj
    if not isinstance(items, list):
        raise ValueError("facts の配列が無い")
    if len(items) != len(vital_nids):
        raise ValueError(f"事実の数 {len(items)} が vital の数 {len(vital_nids)} と合わない")
    out: dict[str, str] = {}
    nums: dict[str, list[int]] = {}
    for pos, it in enumerate(items):
        if isinstance(it, dict):
            nid = it.get("nid", it.get("id", vital_nids[pos]))
            level = _fact_level(it.get("score", it.get("verdict", it.get("label"))))
            ns = _sentence_numbers(it.get("sentences"))
        else:
            nid = vital_nids[pos]
            level = _fact_level(it)
            ns = []
        if nid in out:
            raise ValueError(f"事実 {nid} が重複")
        out[nid] = level
        nums[nid] = ns
    if set(out) != set(vital_nids):
        raise ValueError("事実の nid が vital と合わない")
    return out, nums


def check_sentences(
    facts: Mapping[str, str], numbers: Mapping[str, Sequence[int]], sentences: Sequence[str]
) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """NJ-3 の機械の検査。full/partial で、1..文の数 に入る番号が 1 つも無い事実を none に下げる。

    (下げた後の facts, {nid: 引いた文の本文をつないだ文字列}, 下げた nid の一覧)。
    """
    n = len(sentences)
    checked: dict[str, str] = {}
    evidence: dict[str, str] = {}
    downgraded: list[str] = []
    for nid, level in facts.items():
        valid = sorted({i for i in numbers.get(nid, ()) if 1 <= i <= n})
        evidence[nid] = "".join(sentences[i - 1] for i in valid)
        if level != "none" and not valid:
            level = "none"
            downgraded.append(nid)
        checked[nid] = level
    return checked, evidence, downgraded


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
    version: str = JUDGE_VERSION,
) -> dict:
    """{"sentences","labels","facts","evidence","downgraded","status","error","retries"}。

    version は "NJ-3"（既定。文の番号で根拠を示す）か "NJ-2"（答えからの写し）。呼び出し 1 は同じ。

    status は "ok" か "undetermined"（判定できず。点から外す。0 点にしない）。
    labels は内部名 (vital/okay/...) の配列、facts は {nid: full|partial|none}（evidence の検査後）。
    evidence は {nid: 判定役が写した部分}（NJ-3 は引いた文の本文をつないだもの）、
    downgraded は検査で none に下げた nid の一覧。
    """
    if version not in JUDGE_VERSIONS:
        raise ValueError(f"知らない判定の版: {version!r}（{', '.join(JUDGE_VERSIONS)}）")
    sentences = split_answer(answer)
    vitals = [n for n in nuggets if n.get("importance") == "vital"]
    vital_nids = [n["nid"] for n in vitals]
    result = {
        "sentences": sentences,
        "labels": None,
        "facts": None,
        "evidence": None,
        "downgraded": None,
        "status": "undetermined",
        "error": None,
        "retries": 0,
    }
    if not sentences:  # 空の答え: 何も述べていない
        result.update(
            labels=[], facts={nid: "none" for nid in vital_nids}, evidence={}, downgraded=[], status="ok"
        )
        return result

    labels, err, r1 = _ask(
        send,
        build_prompt1(rule_text, question, sentences, nuggets, version),
        lambda raw: parse_labels(raw, len(sentences)),
    )
    result["retries"] += r1
    if labels is None:
        result["error"] = f"呼び出し1: {err}"
        return result

    if version == "NJ-3":
        parsed, err, r2 = _ask(
            send,
            build_prompt2_nj3(rule_text, question, sentences, vitals, version),
            lambda raw: parse_facts_sentences(raw, vital_nids),
        )
    else:
        parsed, err, r2 = _ask(
            send,
            build_prompt2(rule_text, question, answer, vitals, version),
            lambda raw: parse_facts_detailed(raw, vital_nids),
        )
    result["retries"] += r2
    if parsed is None:
        result["error"] = f"呼び出し2: {err}"
        return result

    if version == "NJ-3":
        facts, numbers = parsed
        facts, evidence, downgraded = check_sentences(facts, numbers, sentences)
    else:
        facts, evidence = parsed
        facts, downgraded = check_evidence(facts, evidence, answer)
    result.update(labels=labels, facts=facts, evidence=evidence, downgraded=downgraded, status="ok")
    return result
