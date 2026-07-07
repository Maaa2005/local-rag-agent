import json
import logging
import secrets
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"
    vector_size: int = 1024  # multilingual-e5-large

    # 推論プロバイダ: "vllm" | "openai" (将来 "transformers" 等を追加可能)
    llm_provider: str = "vllm"
    vllm_base_url: str = "http://localhost:8001/v1"
    llm_model: str = "local-llm"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.1

    embed_model: str = "intfloat/multilingual-e5-large"

    watched_path: str = "/watched"
    data_dir: str = "/app/data"

    secret_key: str = ""
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 480

    @field_validator("secret_key", mode="before")
    @classmethod
    def _ensure_secret_key(cls, v: Any) -> Any:
        if not v or v == "change-me-in-production":
            logger.warning(
                "SECRET_KEY が未設定のため一時キーを自動生成しました。"
                "再起動すると全トークンが失効します。"
                "本番では .env に SECRET_KEY を設定してください"
            )
            return secrets.token_hex(32)
        return v

    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_top_k: int = 5

    # ファイルコピー途中の部分インデックスを防ぐための猶予秒数。
    # mtime からこの秒数未満しか経過していないファイルは処理を先送りする。
    index_settle_seconds: int = 10

    use_polling_watcher: bool = False
    smb_poll_interval: int = 300

    # 既定はローカル開発用フロントエンドのオリジン。
    # 本番ではカンマ区切りまたは JSON 配列文字列で CORS_ORIGINS を上書きする。
    cors_origins: list[str] = ["http://localhost:3000"]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, v: Any) -> Any:
        if isinstance(v, str):
            s = v.strip()
            if s.startswith("["):
                return json.loads(s)
            return [item.strip() for item in s.split(",") if item.strip()]
        return v

    class Config:
        env_file = ".env"


settings = Settings()
