"""規程本文と質問から、必須の事実の下書きを作る（nuggets.jsonl の 1 行の形）。

下書きは人か強いモデルが直してから freeze する。quote が規程に逐語で無い事実は
落とさず "flags": ["quote_not_verbatim"] を付けて残す（直す手がかりにする）。
"""
from __future__ import annotations

from typing import Mapping

from evaluation.nugget import NUGGET_VERSION_DEFAULT
from evaluation.nugget.judge import Send, _ask, extract_json
from evaluation.nugget.schema import quote_in_text

_IMPORTANCE_ALIASES = {"vital": "vital", "必須": "vital", "okay": "okay", "補足": "okay"}


def build_draft_prompt(rule_text: str, question: str) -> list:
    user = f"""[必須の事実の下書き]
次の規程本文を読み、質問に答えるために「無いと答えにならない事実（vital）」と「補足の事実（okay）」を挙げてください。
- 事実は 1 つずつ、短い 1 文で。数値・条件・境目は省かない。
- 各事実に、規程本文からそのままコピーした 1 文 (quote) を付ける。言い換えない。
- keywords は、回答にその事実が書かれていれば出てくるはずの短い語（数値・条件の語）。
- 質問と関係ない定めは excluded に理由つきで挙げる。
- vital は 1 つ以上。

## 規程本文
{rule_text}

## 質問
{question}

## 出力
JSON だけを返してください。
{{"nuggets": [{{"text": "...", "importance": "vital", "quote": "...", "keywords": ["..."]}}],
 "excluded": [{{"text": "...", "why": "..."}}]}}"""
    return [
        {"role": "system", "content": "あなたは社内規程から採点用の事実を抜き出す担当です。JSON だけを返します。"},
        {"role": "user", "content": user},
    ]


def _parse_draft(raw: str) -> dict:
    obj = extract_json(raw)
    if not isinstance(obj, dict) or not isinstance(obj.get("nuggets"), list) or not obj["nuggets"]:
        raise ValueError("nuggets の配列が無い")
    for n in obj["nuggets"]:
        if not isinstance(n, dict) or not str(n.get("text", "")).strip():
            raise ValueError("事実に text が無い")
    return obj


def draft_question(
    send: Send,
    *,
    qid: str,
    question: str,
    rule_id: str,
    rule_text: str,
    set_name: str = "",
    drafted_by: str = "",
    version: str = NUGGET_VERSION_DEFAULT,
) -> dict:
    obj, err, _ = _ask(send, build_draft_prompt(rule_text, question), _parse_draft)
    row: dict = {
        "qid": qid,
        "set": set_name,
        "rule_id": rule_id,
        "question": question,
        "nuggets": [],
        "excluded": [],
        "drafted_by": drafted_by,
        "reviewed_by": "",
        "version": version,
    }
    if obj is None:
        row["flags"] = ["draft_failed"]
        row["draft_error"] = err
        return row
    for i, n in enumerate(obj["nuggets"], start=1):
        imp = _IMPORTANCE_ALIASES.get(str(n.get("importance", "")).strip().lower())
        flags: list[str] = []
        if imp is None:
            imp = "okay"
            flags.append("importance_unknown")
        quote = str(n.get("quote", ""))
        if not quote_in_text(quote, rule_text):
            flags.append("quote_not_verbatim")
        kws = n.get("keywords", [])
        if not isinstance(kws, list):
            kws = []
        item = {
            "nid": f"{qid}-n{i}",
            "text": str(n["text"]).strip(),
            "importance": imp,
            "quote": quote,
            "keywords": [str(k) for k in kws],
        }
        if flags:
            item["flags"] = flags
        row["nuggets"].append(item)
    excluded = obj.get("excluded", [])
    if isinstance(excluded, list):
        row["excluded"] = [e for e in excluded if isinstance(e, Mapping)]
    return row
