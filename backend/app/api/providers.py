"""LLM プロバイダの一覧取得と資格情報管理 API。

- 認証済みユーザー: 利用可能プロバイダ一覧を取得
- 管理者のみ: 資格情報の登録・更新・削除
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core.security import get_current_user, require_admin
from app.services import credentials
from app.services.providers import list_providers, get_meta

router = APIRouter(prefix="/api/providers", tags=["providers"])


class ProviderInfo(BaseModel):
    name: str
    display_name: str
    kind: str
    is_external: bool
    requires_credentials: bool
    default_model: str
    extra_fields: list[str]
    has_credentials: bool
    available: bool  # 認証情報込みで実際に使えるか


@router.get("", response_model=list[ProviderInfo])
async def list_all(_user: dict = Depends(get_current_user)) -> list[ProviderInfo]:
    out: list[ProviderInfo] = []
    for meta in list_providers():
        has = await credentials.has_credentials(meta.name)
        available = (not meta.requires_credentials) or has
        out.append(
            ProviderInfo(
                name=meta.name,
                display_name=meta.display_name,
                kind=meta.kind,
                is_external=meta.is_external,
                requires_credentials=meta.requires_credentials,
                default_model=meta.default_model,
                extra_fields=list(meta.extra_fields),
                has_credentials=has,
                available=available,
            )
        )
    return out


class CredentialUpsert(BaseModel):
    api_key: str
    extra: dict | None = None


@router.put("/{name}/credentials", dependencies=[Depends(require_admin)])
async def upsert_credentials(name: str, body: CredentialUpsert) -> dict:
    try:
        get_meta(name)
    except KeyError:
        raise HTTPException(404, f"Unknown provider: {name}")
    if not body.api_key.strip():
        raise HTTPException(400, "api_key is required")
    await credentials.save_credentials(name, body.api_key, body.extra)
    return {"message": f"{name} の資格情報を保存しました"}


@router.delete("/{name}/credentials", dependencies=[Depends(require_admin)])
async def remove_credentials(name: str) -> dict:
    try:
        get_meta(name)
    except KeyError:
        raise HTTPException(404, f"Unknown provider: {name}")
    await credentials.delete_credentials(name)
    return {"message": f"{name} の資格情報を削除しました"}
