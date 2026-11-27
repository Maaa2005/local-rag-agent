# 小さいファイル（設定・トークナイザ）だけを1本ずつ先に取る。
# huggingface_hub 0.36.2 は Windows で並列に取ると、symlink が使えるかの確かめが競合して落ちる（WinError 1314）。
# 取る物は convert_hf_to_gguf.py --remote の allow_patterns に *.jinja を足した物（Gemma 4 は会話の型が chat_template.jinja にあり、--remote は取らない）。
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id=sys.argv[1], allow_patterns=["LICENSE", "*.json", "*.md", "*.txt", "tokenizer.model", "*.jinja"], max_workers=1))
