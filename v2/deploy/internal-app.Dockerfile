# 社内ゾーン internal-app のイメージ。build context は v2/（compose.yml の build.context: ..）。
# Orchestrator（FastAPI）本体は未実装。今は常駐せずに待つだけで、監査 DB の purge（python -m orchestrator.purge）だけが動く。
# 依存パッケージは入れていない。動くのは標準ライブラリだけで書いた orchestrator.audit_store / orchestrator.purge のみ。
#
# 書込先の volume（audit-data / internal-data）の所有者をここで決める。名前付き volume は空のとき、
# イメージ内の同じパスの所有者と権限を写して使われる。イメージにパスが無いと root 所有 755 のままになり、
# 10002 番で動く internal-app が監査 DB を作れなかった（10/26 OrbStack で確認・修正）。
# /run/gateway（gateway-sock）は作らない。作ると先に起動した側の所有者が写り、gateway がソケットを作れなくなり得る。
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

RUN mkdir -p /app /audit-data /internal-data \
    && chown 10002:10001 /audit-data /internal-data \
    && chmod 0700 /audit-data /internal-data

WORKDIR /app
COPY common/ /app/common/
COPY orchestrator/ /app/orchestrator/

USER 10002:10001
CMD ["python", "-c", "import time; print('internal-app placeholder (Orchestrator 未実装)', flush=True); time.sleep(10**9)"]
