# 送信してよいかを小さく判定する方法: 文献と実装の調査

調査日: 2026-11-02。対象: local-rag-agent v2 の C2（外に出してよいか）と C1（どこへ送るか）。
原文は `sources/` に 154 件（1 件 1 ファイル・先頭行が URL と取得日）。本文中の `[名前]` は `sources/名前.md`。
調べ役（サブエージェント）の報告を親セッションが保存した。調べ役はファイルを書けない決まりのため。方式ごとの細かい記録（第 3 章）は調べ役の返信にあり、ここでは数字の要点だけ残す。実機の測定はしていない。

## 0. 先に結論

1. 既製のガードモデルは、そのままでは C2 に使えない。自前の方針を渡すと崩れる。DynaBench（自前方針の評価）の F1: Llama Guard 3 13.1・WildGuard 20.9・NemoGuard 23.7・ShieldGemma 38.2 [dynaguard_paper]。
2. Laya の「同じ手がかり語が許可文にも禁止文にも出る」失敗は、文献では語への近道学習（shortcut）。Prompt Guard 系が「ignore」だけで誤検知するのと同じ型 [injecguard_piguard_paper]。
3. 「単独では無害、組み合わせると禁止」（D08）はプロンプトでは直らない。区分見出しと集約ルールを足しても禁止の結論が 32.4% 出た [agent_mosaic_paper]。コードの関所は漏えいを 52.2% → 0.5% に下げた [flowseal_ifc_privacy_paper]。
4. 注入の検出器は固定データの点数ほど強くない。適応攻撃で 12 個の防御の大半が 90% 超で破られた [attacker_moves_second_paper]。守りの本体は設計（添付物を要求文に左右させない）に置く。
5. 日本語で「社内の機密か」を測った公開研究は見つからなかった。ここが一番薄い。

## 1. おすすめの順

| 順 | 何をするか | 判定 | 費用 |
|---|---|---|---|
| 1 | 送信物を「固定文（ID）」と「要求文」に分け、手がかり語の検査は要求文だけに掛ける。組み合わせ禁止は表で止める。添付の集合が要求文で変わらないことをテストで固定 | 採る | コードのみ 1〜2 日 |
| 2 | 要求文だけを見る小型の方針適応モデルを Clef-flash T と比べる。第一候補 Shieldstral-1.0-3B（日本語あり・Q4 約 2.15GB・yes/no の確率）。次点 Nemotron-Content-Safety-Reasoning-4B・DynaGuard-4B・Granite Guardian 4.1 8B | 試す | 1〜2 日 |
| 3 | 手がかり語を共有する「双子データ」で小型分類器を鍛える（ModernBERT-Ja-130M / ruri-v3 + SetFit）。150〜500 対 | 試す | 2〜3 日 |
| 4 | 注入の前段に Prompt Guard 2 86M（本体にしない）+ NFKC・ゼロ幅除去 | 試す | 半日 |
| 5 | 社内文書チャンクとの類似度・n-gram 重なりで漏えい元を照合（モデル判定なし）。論文の定量評価なし | 試す | 1 日 |

## 2. 方式の一覧（判定つき・要点）

### A. ガード・方針適応モデル
| 名前 | 判定 | 要点 |
|---|---|---|
| Shieldstral-1.0-3B（Mistral, 2026-08）[shieldstral_hfcard, shieldstral_techreport] | 試す（第一候補） | 方針を 1 つの yes/no 質問で渡し 1 トークンの確率を読む。新しい分類表での適応性 F1 91.3%（開発元測定）。12 言語に日本語。Apache-2.0。第三者の検証なし |
| Nemotron-Content-Safety-Reasoning-4B [nemotron_content_safety_reasoning_4b_hfcard] | 試す | 自前方針 F1 0.857（推論なし）/0.868。英語表記。独自ライセンス |
| DynaGuard 1.7B/4B/8B [dynaguard_paper] | 試す | DynaBench F1 65.2/72.0/73.1。英語。方針の並べ替えで最大 7.4 点ぶれる追試あり [lpg_latent_policy_guardrails_paper] |
| Granite Guardian 4.1 8B [granite_guardian_41_hfcard] | 試す（中） | 任意の基準を yes/no（BYOC）。英語のみ。8B で同居しにくい |
| gpt-oss-safeguard 20B [gptoss_safeguard_techreport] | 本番は採らない・作業用に試す | Q4 で約 11.6GB。ラベル監査・双子データ作りに使える |
| Llama Guard / ShieldGemma / WildGuard / Aegis・NemoGuard / Qwen3Guard / Nemotron Safety Guard | 採らない | 有害表現の固定表。機密の概念がない。ShieldGemma は日本語の有害文で拒否率 11.1% |
| 日本語ガード（ARISE・chakoshi・KARAKURI・Granite ja） | 採らない（作り方は参考） | 有害表現向け。ARISE は gemma-2-2b-jpn + LoRA・約 9 千件で実用域 |

### B. PII・機密検出と DLP
| 名前 | 判定 | 要点 |
|---|---|---|
| 正規化（NFKC・ゼロ幅除去）+ 辞書 | 採る | RuleC2Gate の入口に足す。全角・ゼロ幅での回避を止める |
| Presidio | 部分採用 | 日本語の組み込み認識器なし。正規化と辞書認識器の考え方だけ借りる |
| gitleaks / detect-secrets | 採る（鍵の形のみ） | trufflehog は AGPL-3.0 |
| GLiNER 系（DataSign gliner-ja-pii-v1） | 試す（低） | 日本語 PII 8 種で F1 0.924（合成 600 件のみ） |
| OpenAI Privacy Filter | 採らない | 主に英語・固定 8 ラベル |
| LLM Guard | 採らない | 2026-07 アーカイブ |
| NeMo Guardrails / Guardrails AI / LiteLLM | 枠として参考 | 判定そのものは持たない |
| 社内文書索引との照合 | 試す | 既存の索引を流用。定量評価なし |

### C. 注入の検出
| 名前 | 判定 | 要点 |
|---|---|---|
| Prompt Guard 2 86M | 試す（前段のみ） | 多言語 mDeBERTa。AUC 0.995（多言語）・A100 で 92ms。日本語は公式評価なし |
| InjecGuard / PIGuard の MOF | 訓練法を採る | 語を 1 つずつ入れて誤検知の語を探し、その語を含む良性文を足す。過剰防御 64.31 → 87.32 |
| ProtectAI v2 / Sentinel / NemoGuard JailbreakDetect | 採らない | 英語のみ |
| 既知答え検出・パープレキシティ | 採らない | 自然文の注入に効かない |
| Spotlighting | 採る（外部モデルへ渡す側） | 検出でなく弱める策 |

### D. 判定の信頼性・小型分類器
- P(True) の読み出しと較正: 採る。閾値は自前 dev で決め、1 判定 1 方針 [kadavath_ptrue_paper]
- 小型の微調整はゼロショットの大型 LLM を一貫して上回る例（50〜1000 件で頭打ち）[finetuned_small_llm_vs_zeroshot_paper]。型の決まった課題では BERT 系が強い [bert_vs_llm_classification_emnlp2025]
- SetFit: クラスあたり 8 件で通常の微調整より平均 19.3 点高い [setfit_paper]
- 双子データ・コントラストセット・反事実拡張: 採る [contrast_sets_paper, counterfactually_augmented_data_paper]
- 評価の注意: ラベル定義で指標の幅が最大 23 倍開く・公開データの訓練混入 82〜88% [policy_is_not_capability_paper]。分割で AUC が最大 8.4 点水増し [guardnet_paper]

### E. 情報の流れ・来歴
- FlowSeal: 出口の関所をコードで固定し、LLM は保護レコードと出力案だけを見る。共同作業の装いで漏えい 52.2% → 0.5%、効用 72.4% [flowseal_ifc_privacy_paper]。原則を採る
- CaMeL: 制御とデータを分離。AgentDojo 77%（無防備 84%）[camel_paper]。原則を採る
- Dual-LLM / Rule of Two / lethal trifecta: 確認表として採る
- ConfAIde・PrivacyLens: 強いモデルでも「出してよいか」の判断が 25〜57% 割れる。コードを主役にする根拠
- 組み合わせ漏えい（Agent Mosaic・TOP-Bench）: モデルの賢さでは防げない。禁止の組み合わせを列挙して表で止める

### F. C1（振り分け）
- 埋め込み + 軽い分類器 / SetFit / semantic-router 型: 試す。迷ったらローカル側に倒す
- RouteLLM / Arch-Router: 採らない（品質・コストの振り分け）
- PAPILLON 等の「機密を取り除いて送る」第三の道: 参考（書き換え自体が漏えい源になりうる）

## 4. 最初の実験と、実験前に宣言する合格ライン（調べ役の案）

共通: 全項目満たして合格。データはテンプレート家族ごとに分け、test は 1 回だけ。対照に語の袋 + ロジスティック回帰を置き、それで dev の AUC が 0.8 を超えるなら dev が手がかり語だけで解けている。

- 実験 1（採る・来歴で分けて組み合わせは表）: 組み合わせ禁止 30 件で見逃し 0／許可 30 件で誤ブロック 2 以下／既存 E2E で V055・V062 が止まり V015 が通り、今の許可を落とさない／添付集合の不変テスト全部通過
- 実験 2（方針適応モデル）: 各 60 件以上で test AUC 0.90 以上／見逃し 2/60 以下／誤ブロック 6/60 以下／双子の反転 90% 以上／順序入れ替えで一致 95% 以上／回答モデルと同居して 1 件 1.5 秒以内／Clef-flash T より AUC +0.03 以上
- 実験 3（双子で分類器）: 家族別 test で AUC 0.95 以上／語を移した敵対セットで低下 0.05 以下／語の袋 LR より +0.10 以上／CPU で 1 件 50ms 以下
- 実験 4（注入の前段）: 日本語注入 60 件で検出 90% 以上／良性 100 件で誤検知 2% 以下／1 件 200ms 以下
- 実験 5（漏えい元照合）: 言い換え漏えい検出 90% 以上／承認文由来の誤検知 5% 以下／手がかり語共有の誤検知 5% 以下

## 5. 採らないもの

| 対象 | 理由 |
|---|---|
| 固定表のガード（Llama Guard 3/4・ShieldGemma・WildGuard・Aegis/NemoGuard・Qwen3Guard・Nemotron Safety Guard・Granite の事前定義基準） | 機密の概念がない。自前方針の F1 13〜38 |
| gpt-oss-safeguard 20B のオンライン使用 | Q4 で約 11.6GB で同居できない |
| LLM Guard | アーカイブ済み |
| ProtectAI v2 / Sentinel / NemoGuard JailbreakDetect | 英語のみ |
| パープレキシティ・既知答え検出 | 自然文の注入に効かない |
| 外部サービス前提の物（Lakera・Azure・Cisco・Private AI・chakoshi・KARAKURI） | 外部送信が前提か重み非公開 |
| CaMeL / FIDES の全体導入 | エージェント向けで重い。原則だけ使う |
| 「判定を全部 LLM」の再挑戦 | 同じ失敗型を繰り返す |

## 6. 証拠が薄い所

1. 日本語で社内の機密かを測った公開研究・ベンチは無い
2. Shieldstral・Nemotron・Granite 4.1・LPG の数字は開発元か著者の測定で、独立の再現が無い
3. 8〜12GB の GPU で回答モデルと同居した遅延の資料が無い
4. 撤回論文（PolicyGuard の DLP 版）・要旨だけの日本語注入（JSAI2026）・合成データだけの日本語 PII は参考止まり
5. 埋め込みによる DLP は定量評価が無い
6. 組み合わせ漏えいの研究は英語・エージェント中心
7. 「何件で足りるか」を本件で測った研究は無い

## 7. 標本の大きさ（今の手持ちで言える事）

| 手持ち | 点推定 | 95% 区間（Wilson） |
|---|---|---|
| C2 規則: 禁止 14 件中 7 件を許可 | 50% | 27〜73% |
| C2 規則: 許可 14 件中 2 件を遮断 | 14% | 4〜40% |
| E2E 誤送 2/22 | 9% | 2.5〜28% |
| C1 規則 34/39 | 87% | 73〜94% |

見逃し 0 件のときの見逃し率の 95% 上限は約 3/n（n=30 で 9.5%、60 で 4.9%、100 で 3.0%）。14 件ずつの dev では AUC 0.8 と 0.9 を区別できない。比べるには各クラス 60 件以上、できれば 100 件。
