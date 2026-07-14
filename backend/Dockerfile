FROM python:3.11-slim

WORKDIR /app

# システム依存ライブラリ (Docling の libGL 等)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY migrations/ ./migrations/

# 項目10: コンテナ最小権限化。非rootユーザーで実行する。
# /app/data (SQLite) と HuggingFace キャッシュ用ホームディレクトリを
# 事前に作成し所有権を app に渡しておく (named volume のマウント先も含む)。
RUN useradd --uid 1000 --create-home --home-dir /home/app --shell /usr/sbin/nologin app \
    && mkdir -p /app/data /home/app/.cache/huggingface \
    && chown -R app:app /app /home/app

USER app
ENV HOME=/home/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
