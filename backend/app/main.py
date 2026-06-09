import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import auth, chat, admin, providers as providers_api
from app.config import settings
from app.database import DB_PATH, close_db, init_db
from app.services.indexer import ensure_collection
from app.services.qdrant import close_client
from app.services.task_processor import run_task_processor
from app.services.watcher import start_watcher, stop_watcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await ensure_collection()
    start_watcher(settings.watched_path, str(DB_PATH), use_polling=settings.use_polling_watcher)
    task = asyncio.create_task(run_task_processor())
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

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(admin.router)
app.include_router(providers_api.router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
