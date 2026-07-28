# Word自動編集機能 検証レポート

検証日: 2026年7月28日  
対象: `local-rag-agent` Word編集プロトタイプ

## 結論

DOCX原本をアップロードし、完全一致文字列の置換案を作成、変更点を確認し、利用者が明示承諾した場合だけ編集済みDOCXを別名保存する機能を実装した。システムは原本上書き、文書の確定、電子署名、社内提出を行わない。

## 実装範囲

- 認証済み利用者向けWord編集画面
- 本文、表、ヘッダー、フッターの文字列置換
- 最大20件の置換指示と変更件数・位置のプレビュー
- 利用者ごとのメモリ内下書き、15分の有効期限、最大50下書き
- 承諾拒否時の非保存、明示承諾後の一回限りダウンロード
- 原本を変更せず、`*_edited.docx` としてブラウザダウンロード
- 10MB、展開後30MB、ZIP項目1000件の安全上限
- 変更対象外のrunに設定された太字、斜体、下線等の書式保持

## 検証結果

| 検証 | 結果 |
|---|---|
| Python/JavaScript構文 | 成功 |
| Word固有ユニット/APIテスト | 5/5成功 |
| 全回帰テスト | 239/239成功 |
| 承諾前の非保存 | 成功 |
| `confirmed: false` の拒否 | HTTP 400、保存なし |
| `confirmed: true` の保存 | 成功、DOCX 39,383 bytes |
| 同一下書きの再取得防止 | HTTP 404 |
| 原本非変更 | 成功 |
| 書式保持 | 太字・斜体・下線の回帰テスト成功 |
| Docker実接続 | backend/frontend/Qdrant/vLLM稼働下で成功 |
| UI配信 | Word画面、承諾チェック、保存ボタンを確認 |
| Word描画 | 原本・編集後ともA4一ページ、切れ・はみ出しなし |

全回帰テストの警告は既存のStarlette TestClientとhttpxに関する非推奨警告1件のみで、失敗はない。

## 実演用成果物

- 原本: `demo/assets/word_edit_sample.docx`
- 承諾後の編集例: `demo/assets/word_edit_sample_edited.docx`
- 原本生成: `demo/create_word_edit_sample.py`
- API E2E: `demo/word_edit_api_demo.py`
- Word/PDF描画補助: `demo/render_docx_with_word.ps1`

E2Eデモでは、証憑紛失理由書の所属、氏名、紛失日、証憑内容、金額、紛失理由、再発防止策の7項目を置換した。編集後文書はMicrosoft WordでPDF化し、PNGへ変換して全ページを目視確認した。

## 責任分界と制約

システムが提供するのは編集支援と下書きファイルの生成までである。記載内容の事実確認、社内規程への適合判断、承認、確定、署名、提出は利用者本人と所属組織が行う。

次は対象外である。

- 旧Word形式 `.doc`、マクロ付き `.docm`、パスワード保護文書
- 画像内文字、図形・テキストボックス内文字、複雑なフィールド
- 変更履歴の意味を保った校閲、電子署名、ワークフロー提出
- 複数バックエンドプロセス間での下書き共有

現状の下書きは単一バックエンドのメモリ内に保持される。水平分散する場合は、暗号化・TTL・所有者検証を備えたRedis等へ移行する必要がある。

## 再現手順

```powershell
docker compose up -d
python demo\create_word_edit_sample.py
python demo\word_edit_api_demo.py
$env:ORCHESTRATOR_ENABLED='false'
$env:ORCHESTRATOR_REQUIRED='false'
python -m pytest backend\tests evaluation\tests -q
```

画面デモは `http://localhost:3000` に `demo_employee` でログインし、左メニューの「Word編集」から実施する。
