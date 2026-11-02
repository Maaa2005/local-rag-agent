"""試し回しの実測から、計画 §5 の全体の計算量を出し直す（life #45・evaluation/v2/results/2026-11-02-distill-trial-l4-4d769b9）。

使い方: python -m evaluation.distill.estimate
"""
RATE = 1.94            # L4 の使用率（Colab の表示、CU/時）
TEACH_S_PER_Q = 765 / 100          # 教師 2 回答え・並列 8
EX_PER_Q = 187 / 100               # 照合と重複除きの後に残る件数
TRAIN_S_PER_PASS = 632 / (187 * 2)   # 1 件を 1 回学ぶ秒（train_runtime 632 秒）
TRAIN_LOAD_S = 774 - 632              # 学習 1 本ごとの読み込み
CONV_S = 82 + (257 + 581) / 2       # 結合 82 秒＋変換と量子化（学習後 257 秒・対照 581 秒の平均）
ANS_S = 3.0                        # E4B が 1 問答える秒（L4。守りの問の中央は学習後 1.45 秒。開発の問は長いので 3 秒と置く）
SETUP_S = 600          # 1 つのランタイムの入れる・落とす


def h(s):
    return s / 3600


rows = []
# 試作: 1,000 問・学習率 3 本・開発(選ぶ)150+守り40 を 3 本と対照・確かめ 150 を選んだ 1 本と対照
pilot_teach = 1000 * TEACH_S_PER_Q
pilot_train = 3 * (1000 * EX_PER_Q * 2 * TRAIN_S_PER_PASS + TRAIN_LOAD_S)
pilot_conv = 4 * CONV_S
pilot_eval = (4 * 190 + 2 * 150) * ANS_S
rows.append(("試作", pilot_teach, pilot_train, pilot_conv, pilot_eval, 2 * SETUP_S))
# 本学習: 教師で 2,000 問を足す・3,000 問ぶん 1 本・守り40+確かめ150
full_teach = 2000 * TEACH_S_PER_Q
full_train = 3000 * EX_PER_Q * 2 * TRAIN_S_PER_PASS + TRAIN_LOAD_S
full_conv = CONV_S
full_eval = 190 * ANS_S
rows.append(("本学習", full_teach, full_train, full_conv, full_eval, 2 * SETUP_S))
# 封 A 2 回（T4・起動ごとに G0 → 62 問）と封 B 96 問。T4 の使用率は L4 以下として L4 で数える
seal_eval = (2 * 62 + 96) * 8.0
rows.append(("封", 0, 0, 0, seal_eval, 3 * SETUP_S))

tot = 0
for name, *parts in rows:
    s = sum(parts)
    tot += s
    print(f"{name}: 教師 {h(parts[0]):.1f} 時・学習 {h(parts[1]):.1f} 時・変換 {h(parts[2]):.1f} 時・測定 {h(parts[3]):.1f} 時・"
          f"準備 {h(parts[4]):.1f} 時 = {h(s):.1f} 時 = {h(s) * RATE:.1f} CU")
print(f"合計 {h(tot):.1f} 時 = {h(tot) * RATE:.1f} CU（L4 {RATE} CU/時）")
