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

    class Config:
        env_file = ".env"


settings = Settings()
