"""開発用の問 360 問（4 組 × 90 問・24 規程）を読み、モデルに渡す文を作る。

問は local-rag-agent の枝 feat/numeric-method3（a20f3c9）の evaluation/v2/numeric_dev90{,b,c,d} の写し。
封をした 240 問は使わない（このリポにも入っていない）。

文の形は測る前に固定する（PROMPT_VERSION）。変えたら版を上げ、前の版の結果と混ぜない。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
SETS = ("numeric_dev90", "numeric_dev90b", "numeric_dev90c", "numeric_dev90d")
HOLD = "保留"
PROMPT_VERSION = "qp-1"

SYSTEM = (
    "あなたは社内規程の問い合わせに答える担当です。与えられた規程の文だけを根拠にします。\n"
    "質問に書かれていない事実で結論が変わる時は、推測で決めずに保留します。"
)
USER = (
    "## 規程\n{rule}\n\n## 質問\n{body}\n\n"
    "理由を3文以内で書き、最後の行には「結論: {a}」「結論: {b}」「結論: " + HOLD + "」のどれか1つだけを書いてください。"
)

# 問の種類を 6 つの見出しにまとめる（集計用。合否の線には使わない）
GROUP_KIND = {
    "below": "境目", "at": "境目", "above": "境目", "boundary": "境目", "paraphrase": "境目",
    "compute": "計算", "derived": "計算",
    "exception_hit": "例外", "exception_near_miss": "例外", "exception": "例外", "exception_near": "例外",
    "premise_unmet": "前提", "second_condition": "前提", "premise": "前提", "multi": "前提",
    "irrelevant_number": "無関係な数", "distractor": "無関係な数",
    "determined_unstated": "書かれていない事", "unknown_determined": "書かれていない事",
    "hold_missing_value": "保留", "hold_unknown_condition": "保留", "hold_range_straddle": "保留",
    "hold_missing": "保留", "hold_unknown": "保留", "hold_range": "保留",
}


@dataclass(frozen=True)
class Question:
    id: str
    set: str
    rule_id: str
    body: str
    options: tuple[str, str]
    expected: str  # 保留の問は "保留"
    should_hold: bool
    group: str
    pair_id: str | None

    @property
    def kind(self) -> str:
        return GROUP_KIND[self.group]


def _strip_instruction(question: str) -> str:
    lines = question.rstrip("\n").split("\n")
    if not lines[-1].startswith("最後の行には"):
        raise ValueError(f"最後の行が指定の 1 文ではない: {lines[-1][:30]}")
    return "\n".join(lines[:-1]).rstrip()


def load_questions(data_dir: Path = DATA_DIR, sets: tuple[str, ...] = SETS) -> list[Question]:
    out: list[Question] = []
    for s in sets:
        for line in (data_dir / s / "questions.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            opts = tuple(r["conclusion_options"])
            if len(opts) != 2 or HOLD in opts:
                raise ValueError(f"{r['id']}: 結論の語が 2 つでない、または保留を含む")
            exp = r["expected_conclusion"]
            if r["should_hold"] != (exp == HOLD) or (exp != HOLD and exp not in opts):
                raise ValueError(f"{r['id']}: 正解と保留の印が合わない")
            out.append(Question(
                id=r["id"], set=s, rule_id=r["rule_id"], body=_strip_instruction(r["question"]),
                options=opts, expected=exp, should_hold=bool(r["should_hold"]), group=r["group"],
                pair_id=(f"{s}:{r['pair_id']}" if r.get("pair_id") else None),
            ))
    ids = [q.id for q in out]
    if len(ids) != len(set(ids)):
        raise ValueError("問の id が重なっている")
    return out


def load_rules(data_dir: Path = DATA_DIR, sets: tuple[str, ...] = SETS) -> dict[tuple[str, str], str]:
    rules = {}
    for s in sets:
        for p in sorted((data_dir / s / "corpus").glob("*.md")):
            rules[(s, p.stem)] = p.read_text(encoding="utf-8").replace("\r\n", "\n").strip()
    return rules


def build_messages(q: Question, rules: dict[tuple[str, str], str]) -> list[dict[str, str]]:
    a, b = q.options
    user = USER.format(rule=rules[(q.set, q.rule_id)], body=q.body, a=a, b=b)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def manifest(data_dir: Path = DATA_DIR, sets: tuple[str, ...] = SETS) -> dict[str, str]:
    """問と規程の指紋（改行を LF にそろえた sha256）。結果の記録に残し、別の問で測っていない事を確かめる。"""
    h: dict[str, str] = {}
    for s in sets:
        for p in [data_dir / s / "questions.jsonl", *sorted((data_dir / s / "corpus").glob("*.md"))]:
            h[f"{s}/{p.relative_to(data_dir / s).as_posix()}"] = hashlib.sha256(
                p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    total = hashlib.sha256("".join(f"{k}={v}\n" for k, v in sorted(h.items())).encode()).hexdigest()
    return {"total": total, "prompt_version": PROMPT_VERSION, **h}


def domain_text(data_dir: Path = DATA_DIR, sets: tuple[str, ...] = SETS) -> str:
    """KLD を測る 3 つ目の文章（規程 24 本と問の本文）。モデルの答えは入れない。"""
    rules = load_rules(data_dir, sets)
    qs = load_questions(data_dir, sets)
    parts = [rules[k] for k in sorted(rules)]
    parts += [q.body for q in sorted(qs, key=lambda x: (x.set, x.id))]
    return "\n\n".join(parts) + "\n"
