import json
import logging
import re
import secrets
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"
    # Qdrant への要求の時間切れ（秒・整数）。0 なら qdrant-client の既定のまま
    # （qdrant-client 1.12.1 は timeout を渡さないと httpx の既定 5 秒になる）。
    # 学校 PC で本体のメモリが尽きた回に検索が 5 秒で時間切れになった（Issue #1・#14）。
    # 既定は今までと同じ。測るときだけ環境変数 QDRANT_TIMEOUT_S で変える。
    qdrant_timeout_s: int = 0
    # 読むだけの要求（検索の query_points・scroll）を、時間切れのときだけ送り直す回数。0 なら送り直さない
    # （今までと同じ）。書き込み（upsert・delete・set_payload）は送り直さない。環境変数 QDRANT_SEARCH_RETRIES
    qdrant_search_retries: int = 0
    vector_size: int = 1024  # multilingual-e5-large

    # 推論プロバイダ: "vllm" | "openai" (将来 "transformers" 等を追加可能)
    llm_provider: str = "vllm"
    vllm_base_url: str = "http://localhost:8001/v1"
    llm_model: str = "local-llm"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.1

    embed_model: str = "intfloat/multilingual-e5-large"
    # e5 系は query/passage で異なる prefix が必要。Ruri 系など方式が異なる
    # モデルへ切り替える際はここを変更する (embed_model 変更時は全文書の再インデックスが必要)。
    embed_query_prefix: str = "query: "
    embed_passage_prefix: str = "passage: "

    # Cross-Encoder によるリランキング (検索品質向上)。
    rerank_enabled: bool = True
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    # dense+sparse+RRF で取得する候補数。retrieval_top_k より多めに取り、
    # リランク後に retrieval_top_k 件へ絞り込む。
    rerank_candidates: int = 30

    # 埋め込み・リランカーを動かす場所。既定は CPU で VRAM を LLM 専用にする。
    # GPU 8〜12GB で LLM と同居させる場合は "cuda" (または "cuda:N") にする。
    # CPU のリランクは候補 30 件で検索上限 30 秒を超えた (Issue #7 の B)。
    embed_device: str = "cpu"
    rerank_device: str = "cpu"

    @field_validator("qdrant_timeout_s")
    @classmethod
    def _check_qdrant_timeout(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"QDRANT_TIMEOUT_S は 0 以上（0 は qdrant-client の既定）: {v!r}")
        return v

    @field_validator("qdrant_search_retries")
    @classmethod
    def _check_qdrant_retries(cls, v: int) -> int:
        # 送り直しを増やすと、時間切れのたびに待ちが timeout 分ずつ延びる。15 秒の枠を考えて 2 回までにする
        if not 0 <= v <= 2:
            raise ValueError(f"QDRANT_SEARCH_RETRIES は 0〜2: {v!r}")
        return v

    @field_validator("embed_device", "rerank_device")
    @classmethod
    def _check_device(cls, v: str) -> str:
        # reranker はロード失敗時に元の順序へ黙って戻るため、書き間違いは起動時に止める
        if not re.fullmatch(r"cpu|mps|cuda(:\d+)?", v):
            raise ValueError(f"device は cpu / mps / cuda / cuda:N のいずれか: {v!r}")
        return v

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

    # 質問の値と規程の条件の比べをコードでやり、【計算の結果】として LLM へ渡す。
    # 既定は OFF (従来と同じ出力)。環境変数 NUMERIC_CHECK_ENABLED=true で ON。
    numeric_check_enabled: bool = False

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

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
