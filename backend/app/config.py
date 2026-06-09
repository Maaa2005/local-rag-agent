import json
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"
    vector_size: int = 1024  # multilingual-e5-large

    vllm_base_url: str = "http://localhost:8001/v1"
    llm_model: str = "local-llm"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.1

    embed_model: str = "intfloat/multilingual-e5-large"

    watched_path: str = "/watched"
    data_dir: str = "/app/data"

    secret_key: str = "change-me-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 480

    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_top_k: int = 5

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
