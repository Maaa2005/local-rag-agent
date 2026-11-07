"""入力のゆらぎ用の問を作る。意味を変えない受付番号を問の先頭に足す。引数: 元の問 出力先の場所"""
import json
import pathlib
import random
import sys

src, out = sys.argv[1:3]
rows = [json.loads(line) for line in open(src, encoding="utf-8") if line.strip()]
for n in (1, 2):
    rng = random.Random(f"ask90b-{n}")
    with open(pathlib.Path(out) / f"questions_n{n}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            prefix = f"受付番号 {rng.randint(1000, 9999)} の問い合わせです。"
            f.write(json.dumps({**r, "question": prefix + r["question"]}, ensure_ascii=False) + "\n")
print(len(rows))
