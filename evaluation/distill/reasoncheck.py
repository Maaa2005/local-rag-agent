"""教師の答えの理由の抜き取り（docs/verified-distill-plan.md §3）。確認役（Codex）に正解を見せて、
理由のつながり（条件・比べた結果・結論）が正しいかを 1 件ずつ判定させ、誤りの数を数える。

    python -m evaluation.distill.reasoncheck --blind <dir> --sample sample210.jsonl   # 確認役に渡すフォルダを作る
    python -m evaluation.distill.reasoncheck --count sample210.jsonl answer.md          # 誤りを数える（4 件以上で止める）
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

STOP_AT = 4
_LINE = re.compile(r"^\s*(S\d{3})\s*\|\s*(正|誤)\s*(?:\|\s*(.*))?$")

PROMPT = """あなたは学習データの点検役です。読み取り専用で、このフォルダの `sample.jsonl` だけを見て、全件を判定してください。

- 各行は 1 件です。`input` はモデルに渡した【参考情報】と【質問】、`answer` は教師モデルの答え、`expected` は正しい結論
  （断りの問と資料の文を答える問は null。`keywords` は答えに入るべき語句）です。
- 結論そのものはコードで照合済みです。見るのは理由のつながりだけです: 答えが使った条件（規程のどの決まりか）、
  問の値と境目の値の比べ方（以上・未満・超える等の向き、ちょうどの扱い、単位の換算）、例外の当てはめ、そこから結論への流れ。
- 理由のどこかに、規程や問の事実と食い違う所、比べる向きの誤り、換算の誤り、結論と逆の事を言う所があれば「誤」。
  言葉が足りないだけで、書いた事に誤りが無ければ「正」。
- 断りの問は、資料に無い事を断っていて、資料に無い値を作っていなければ「正」。

出力は次の形だけにしてください。全件を sid の順に、1 件 1 行で。「誤」の時だけ 3 つ目に短い理由を書きます。
```
S001|正
S002|誤|超えるの向きを逆に書いている
```
"""


def write_blind(sample: Path, out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in sample.read_text(encoding="utf-8").splitlines() if x.strip()]
    keep = ("sid", "category", "input", "answer", "expected", "keywords")
    (out / "sample.jsonl").write_text("".join(json.dumps({k: r.get(k) for k in keep}, ensure_ascii=False) + "\n" for r in rows),
                                      encoding="utf-8")
    (out / "prompt.md").write_text(PROMPT, encoding="utf-8")
    return len(rows)


def count(sample: Path, answer_text: str) -> dict:
    sids = [json.loads(x)["sid"] for x in sample.read_text(encoding="utf-8").splitlines() if x.strip()]
    got = {}
    for line in answer_text.splitlines():
        m = _LINE.match(line)
        if m:
            got[m.group(1)] = (m.group(2), (m.group(3) or "").strip())
    missing = [s for s in sids if s not in got]
    wrong = {s: got[s][1] for s in sids if s in got and got[s][0] == "誤"}
    return {"n": len(sids), "judged": len(sids) - len(missing), "missing": missing, "wrong": wrong,
            "stop": len(wrong) >= STOP_AT or bool(missing)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blind", type=Path)
    ap.add_argument("--sample", type=Path)
    ap.add_argument("--count", nargs=2, type=Path)
    a = ap.parse_args()
    if a.blind:
        print(write_blind(a.sample, a.blind))
    if a.count:
        print(json.dumps(count(a.count[0], a.count[1].read_text(encoding="utf-8")), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
