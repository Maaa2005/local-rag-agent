import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import auth, chat, admin
from app.config import settings
from app.database import DB_PATH, close_db, init_db
from app.services.embedder import embed_query
from app.services.indexer import ensure_collection
from app.services.qdrant import close_client
from app.services.reranker import warmup as warmup_reranker
from app.services.task_processor import run_task_processor
from app.services.watcher import start_watcher, stop_watcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


async def _warmup_embedder() -> None:
    try:
        await embed_query("ウォームアップ")
        logging.getLogger(__name__).info("Embedder warmed up")
    except Exception:
        logging.getLogger(__name__).exception("Embedder warmup failed")


async def _warmup_reranker() -> None:
    try:
        await warmup_reranker()
        logging.getLogger(__name__).info("Reranker warmed up")
    except Exception:
        logging.getLogger(__name__).exception("Reranker warmup failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await ensure_collection()
    start_watcher(settings.watched_path, str(DB_PATH), use_polling=settings.use_polling_watcher)
    task = asyncio.create_task(run_task_processor())
    # 初回チャットでのモデルロード待ちを避けるため、起動をブロックせずウォームアップする。
    # 短命タスクなので shutdown 時の cancel は不要だが、GC 回収を防ぐため参照だけ保持。
    _warmup_task = asyncio.create_task(_warmup_embedder())
    _warmup_rerank_task = None
    if settings.rerank_enabled:
        _warmup_rerank_task = asyncio.create_task(_warmup_reranker())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        stop_watcher()
        await close_client()
        await close_db()


app = FastAPI(title="社内 RAG エージェント API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _security_headers(request, call_next):
    """項目中5: クリックジャッキング・MIME スニッフィング対策の基本ヘッダ。

    HTML 配信 (frontend/nginx) 側は別途 CSP を付与する。ここでは API/静的
    問わず全レスポンス共通で付けられる最小限のヘッダのみ扱う。
    """
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(admin.router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
