"""「モデルは読む、コードが決める」方式（F7・設計 B）。docs/code-decides-plan.md の 1〜4 節の実装。

- parse:   質問の値の取り出しと、規程の引用の読み取り（標準ライブラリのみ）
- records: 確認済みの構造化記録（JSON）の読み込みと検証
- engine:  記録の選択 → 事実の取り出し → 計算 → 結論の選択（コードだけで決める）
- render:  答えの文の組み立て（型は固定。モデルは何も書かない）

この package は orchestrator を import しない（検索結果は content / document_id / access_level / source を持つ物として扱う）。
"""
