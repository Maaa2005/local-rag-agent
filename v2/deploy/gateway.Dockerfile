# Gateway ゾーンのイメージ。build context は v2/（compose.yml の build.context: ..）。
# v2/common と v2/gateway だけを含める（社内側コード・評価データ・ポリシーは入れない）。
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN pip install "pydantic>=2.7" "fastapi>=0.110" "uvicorn>=0.29" "httpx>=0.27" "anthropic>=0.40" "openai>=1.50" \
    && groupadd --gid 10001 gateway \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin gateway \
    && mkdir -p /app /data /run/gateway \
    && chown 10001:10001 /data \
    && chown 10001:10001 /run/gateway && chmod 2770 /run/gateway

WORKDIR /app
COPY common/ /app/common/
COPY gateway/ /app/gateway/

USER 10001:10001
CMD ["python", "-m", "gateway.app"]
