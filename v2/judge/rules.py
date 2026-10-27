"""GPU なしで動くルールベースの C1 / C2（比較用の基準線）。

これは本命ではない。判断モデル（Laya-multilingual / Clef-flash）を C6 で比べるときの
「最低限これより良いか」を測る基準線として置く。
過学習を避けるため、C2 の規則は docs/v2-contract-examples.md の「禁止する情報」リスト
（個人名・社員番号、取引先名、金額・日付の実データ、社内のテーブル名・列名、M&A 等の
未公開情報、指示の注入）だけから作り、evaluation/v2/cases.jsonl を見ながら調整していない。
"""
from __future__ import annotations

import re
from typing import Any

from common.schemas import SendPayload
from judge.base import C1Decision, C2Verdict, question_version
from judge.destination import RAW_SOURCE_KEY, contract_input_problem, lookup_contract

MODEL = "rules"
REVISION = "rules-v1"

# 宛先: C1 は契約キーを社内の契約定義（policies/contracts.json）から完全一致で引き、
# その契約の経路と唯一の許可宛先を返す（judge.destination）。宛先の推論・代替選択はしない。
# destination_hint は利用者の希望という参考情報で、宛先の決定には使わない（理由欄に記録だけする）。

# C1: 契約を使わない依頼の素朴なキーワード振り分け（契約例ドキュメントの経路定義から）
_C1_REJECT = re.compile(r"(指示は?無視|全文を(出力|送|貼)|外部に(送|出)|持ち出|漏ら)")
_C1_HUMAN = re.compile(r"(人事|評価|処分|例外|判断して|相談|承認して)")
_C1_CODE = re.compile(r"(計算|合計|集計|何日|日数|件数|平均|変換)")
# ルールベースは判断モデルに質問しないので、「質問の版」= 判定規則そのものの版（F9）。
# 規則の定義から計算するので、規則を変えれば版も変わる
C1_QUESTION_VERSION = question_version("rules-c1@3", {
    "contract": "exact key -> contracts.json route + sole destination",
    "contract_input": (f"{RAW_SOURCE_KEY} (raw source reference) -> reject; keys outside "
                       "{contract, input_kind, options, destination_hint, free_text if contract.free_text} "
                       "or input_kind missing/empty/non-string -> human"),
    "reject": _C1_REJECT.pattern, "human": _C1_HUMAN.pattern, "code": _C1_CODE.pattern,
})


class RuleC1Router:
    question_version = C1_QUESTION_VERSION

    def route(self, user: str, request: str, input: dict[str, Any] | None) -> C1Decision:
        if input and input.get("contract"):
            key = input["contract"]
            c, why = lookup_contract(key)
            if c is None:
                return C1Decision(route="human", reason=f"contract not usable: {why}", model=MODEL, revision=REVISION, question_version=C1_QUESTION_VERSION)
            # 契約入力の中身を見る。元資料の直接指定は契約の外（承認済みテキストを通らない持ち出し）なので拒否、
            # それ以外の形の崩れ（許容外キー・入力種別の欠落/空/非文字列）は人の判断へ
            if input.get(RAW_SOURCE_KEY) is not None:
                return C1Decision(route="reject", reason=f"contract {key}: raw source reference is not accepted ({RAW_SOURCE_KEY})",
                                  model=MODEL, revision=REVISION, question_version=C1_QUESTION_VERSION)
            problem = contract_input_problem(c, input)
            if problem is not None:
                return C1Decision(route="human", reason=f"contract input not usable: {problem}",
                                  model=MODEL, revision=REVISION, question_version=C1_QUESTION_VERSION)
            hint = input.get("destination_hint")
            note = f"; hint {hint} is reference only" if hint is not None else ""
            # 権限は C1 では与えない（送信資格で別に検査する）。契約の経路と唯一の宛先を返すだけ
            return C1Decision(route=c.route, destination=c.destinations[0], reason=f"contract {key}; sole destination{note}",
                              model=MODEL, revision=REVISION, question_version=C1_QUESTION_VERSION)
        if _C1_REJECT.search(request):
            return C1Decision(route="reject", reason="exfiltration/injection phrase", model=MODEL, revision=REVISION, question_version=C1_QUESTION_VERSION)
        if _C1_HUMAN.search(request):
            return C1Decision(route="human", reason="needs human judgement", model=MODEL, revision=REVISION, question_version=C1_QUESTION_VERSION)
        if _C1_CODE.search(request):
            return C1Decision(route="code", reason="deterministic computation", model=MODEL, revision=REVISION, question_version=C1_QUESTION_VERSION)
        return C1Decision(route="local", reason="default: answer locally", model=MODEL, revision=REVISION, question_version=C1_QUESTION_VERSION)


# C2: 禁止情報リストに対応する正規表現（追加の拒否検査。許可の根拠にはしない）
_C2_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("employee_id", re.compile(r"(?<![A-Za-z0-9])[EＥ]\d{3,}")),
    ("amount", re.compile(r"(\d[\d,，.]*\s*(億|万|千)?円|[億万]円)")),
    ("real_date", re.compile(r"\d{4}\s*[-/年]\s*\d{1,2}\s*[-/月]\s*\d{1,2}")),
    ("table_name", re.compile(r"(?<![\w.])[A-Za-z_][A-Za-z0-9_]*\.(?!(?:csv|py|txt|json|md|xlsx?)\b)[A-Za-z_][A-Za-z0-9_]*")),
    ("injection", re.compile(r"(前の指示(は|を)?無視|指示(は|を)無視|ignore (all )?(the )?previous|システムプロンプト|全文を出力)", re.I)),
    ("m_and_a", re.compile(r"(M&A|Ｍ＆Ａ|買収|合併|TOB|出資比率)")),
    # 「仕様」「同様」など敬称でない「様」は除く（契約 B の承認済み仕様文に「仕様」が入るため）
    ("person_name", re.compile(r"[一-龥]{1,4}(さん|氏|(?<![仕同模多各異])様)")),
    ("counterparty", re.compile(r"([A-ZＡ-Ｚ]社|株式会社|㈱)")),
]
C2_QUESTION_VERSION = question_version("rules-c2@1", [(n, p.pattern, p.flags) for n, p in _C2_PATTERNS])


class RuleC2Gate:
    question_version = C2_QUESTION_VERSION

    def check(self, payload: SendPayload) -> C2Verdict:
        text = "\n".join(m.content for m in payload.messages)
        hits = [name for name, pat in _C2_PATTERNS if pat.search(text)]
        if hits:
            return C2Verdict(decision="block", reason="matched: " + ",".join(hits), model=MODEL, revision=REVISION, prob_block=1.0, question_version=C2_QUESTION_VERSION)
        # 正規表現に当たらない＝安全、ではない。基準線として allow を返すだけ
        return C2Verdict(decision="allow", reason="no rule matched", model=MODEL, revision=REVISION, prob_block=0.0, question_version=C2_QUESTION_VERSION)
