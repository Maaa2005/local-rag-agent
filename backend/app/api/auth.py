from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from app.core.security import (
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
    require_admin,
)
from app.database import db

router = APIRouter(prefix="/api/auth", tags=["auth"])


class Token(BaseModel):
    access_token: str
    token_type: str


class UserCreate(BaseModel):
    username: str
    password: str
    access_level: int = 1


@router.post("/token", response_model=Token)
async def login(form: OAuth2PasswordRequestForm = Depends()):
    row = await db.fetchone(
        "SELECT password_hash, is_active FROM users WHERE username=?",
        (form.username,),
    )
    if row is None or not row["is_active"] or not verify_password(form.password, row["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="ユーザー名またはパスワードが正しくありません",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token({"sub": form.username})
    return Token(access_token=token, token_type="bearer")


@router.get("/me")
async def me(user: dict = Depends(get_current_user)):
    return {"username": user["username"], "access_level": user["access_level"]}


@router.post("/users", dependencies=[Depends(require_admin)])
async def create_user(body: UserCreate):
    if body.access_level not in (1, 2, 3):
        raise HTTPException(400, "access_level は 1, 2, 3 のいずれかです")
    existing = await db.fetchone("SELECT id FROM users WHERE username=?", (body.username,))
    if existing:
        raise HTTPException(400, "そのユーザー名は既に使われています")
    await db.execute(
        "INSERT INTO users (username, password_hash, access_level) VALUES (?, ?, ?)",
        (body.username, hash_password(body.password), body.access_level),
    )
    return {"message": f"ユーザー {body.username} を作成しました"}
