# 資料案内

このディレクトリを、操作方法・検証結果・ファインチューニング資料の入口とする。

ローカルの資料ルート:

`C:\Users\suika\hr-assistant-ai\local-rag-agent\docs`

## 最初に読む資料

| 知りたいこと | 資料 | リポジトリ内パス |
|---|---|---|
| 起動方法、ログイン、5分デモ、Word操作 | [デモ・操作ガイド](../DEMO_GUIDE_20260728.md) | `DEMO_GUIDE_20260728.md` |
| システム全体の構成とインストール | [トップREADME](../README.md) | `README.md` |
| Gemma 4 E2BをFTした意図、データ設計、結果 | [FT設計・データセット・結果](FT_DESIGN_AND_RESULTS.md) | `docs/FT_DESIGN_AND_RESULTS.md` |
| 実際のFT用JSONLとハッシュ | [FTデータセット説明](ft-data/README.md) | `docs/ft-data/README.md` |
| RAG検索の精度・権限・整合性検証 | [RAG検証レポート](../RAG_VALIDATION_REPORT_20260728.md) | `RAG_VALIDATION_REPORT_20260728.md` |
| Word編集の安全設計と検証 | [Word検証レポート](../WORD_EDIT_VALIDATION_REPORT_20260728.md) | `WORD_EDIT_VALIDATION_REPORT_20260728.md` |
| RAGの要求仕様 | [RAG要件定義](../RAG要件定義.md) | `RAG要件定義.md` |
| 企画の背景と目的 | [企画書](../企画書.md) | `企画書.md` |
| RAG評価データと採点条件 | [評価README](../evaluation/README.md) / [評価基準](../evaluation/CRITERIA.md) | `evaluation/README.md`, `evaluation/CRITERIA.md` |

## 実演用ファイル

| 用途 | パス |
|---|---|
| 証憑紛失理由書の原本 | `demo/assets/word_edit_sample.docx` |
| 承諾後の編集例 | `demo/assets/word_edit_sample_edited.docx` |
| Word API E2Eデモ | `demo/word_edit_api_demo.py` |
| Qdrant権限別検索デモ | `demo/qdrant_access_demo.py` |
| Gemma司令塔デモ | `demo/ft_orchestrator_demo.py` |
| RAG回答品質評価 | `demo/rag_answer_eval.py`, `demo/rag_quality_eval.py` |

## FT証跡の構成

- `docs/ft-data/`: 学習・validation・test・人手OODの固定スナップショット
- `docs/ft-evidence/training-summary.json`: 最終160 step学習の集計
- `docs/ft-evidence/ood-comparison.md`: 改修前後のOOD比較
- `models/hr-orchestrator/manifest.json`: ベースモデル、アダプター、データセットの系譜とSHA-256
- `training/`: データ生成、Schema検証、Unsloth学習、OOD評価の再現コード

LoRAアダプター重みはHugging Face Hubで配布し、インストーラーが `models/hr-orchestrator/` へ取得してSHA-256を検証する。Gitには重みを含めない。約7.58GBのGemmaベースモデルもGitへ保存せず、vLLM初回起動時にHugging Face cacheへ取得する。詳細は [アダプター配布方針](ADAPTER_DISTRIBUTION.md) と同ディレクトリのREADME・manifestに記載する。
