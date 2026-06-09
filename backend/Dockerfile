FROM python:3.11-slim

WORKDIR /app

# システム依存ライブラリ (Docling の libGL 等) + Node.js (Claude/Codex CLI 用)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 git curl ca-certificates gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# エージェント CLI (Phase 3 プロバイダで subprocess 起動)
# claude_code_provider が claude CLI を、codex_provider が codex CLI を呼び出す
RUN npm install -g @anthropic-ai/claude-code @openai/codex

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY migrations/ ./migrations/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
