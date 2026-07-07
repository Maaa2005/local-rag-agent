import time

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

# ログイン失敗のユーザー名単位レート制限（インメモリ）。
# username → (連続失敗回数, 最終失敗時刻 time.monotonic())
# 単一プロセス・単一 asyncio イベントループ前提のため await をまたがず同期的に
# 読み書きしており、ロックは不要。
_failed_logins: dict[str, tuple[int, float]] = {}
_LOCKOUT_THRESHOLD = 5
_LOCKOUT_SECONDS = 60.0


class Token(BaseModel):
    access_token: str
    token_type: str
    must_change_password: bool = False


class UserCreate(BaseModel):
    username: str
    password: str
    access_level: int = 1


class PasswordChange(BaseModel):
    current_password: str
    new_password: str


def _validate_password(username: str, password: str) -> None:
    if len(password) < 8:
        raise HTTPException(400, "パスワードは8文字以上にしてください")
    if password == username:
        raise HTTPException(400, "パスワードはユーザー名と異なるものにしてください")


@router.post("/token", response_model=Token)
async def login(form: OAuth2PasswordRequestForm = Depends()):
    failures, last_failed_at = _failed_logins.get(form.username, (0, 0.0))
    if failures >= _LOCKOUT_THRESHOLD:
        if time.monotonic() - last_failed_at < _LOCKOUT_SECONDS:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="試行回数が多すぎます。しばらくしてから再度お試しください",
            )
        # ロックアウト期間が経過したのでカウンタをリセットして続行
        _failed_logins.pop(form.username, None)

    row = await db.fetchone(
        "SELECT password_hash, is_active FROM users WHERE username=?",
        (form.username,),
    )
    if row is None or not row["is_active"] or not verify_password(form.password, row["password_hash"]):
        failures, _ = _failed_logins.get(form.username, (0, 0.0))
        _failed_logins[form.username] = (failures + 1, time.monotonic())
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="ユーザー名またはパスワードが正しくありません",
            headers={"WWW-Authenticate": "Bearer"},
        )
    _failed_logins.pop(form.username, None)
    token = create_access_token({"sub": form.username})
    must_change_password = form.password in (form.username, "admin")
    return Token(access_token=token, token_type="bearer", must_change_password=must_change_password)


@router.get("/me")
async def me(user: dict = Depends(get_current_user)):
    return {"username": user["username"], "access_level": user["access_level"]}


@router.post("/users", dependencies=[Depends(require_admin)])
async def create_user(body: UserCreate):
    if body.access_level not in (1, 2, 3):
        raise HTTPException(400, "access_level は 1, 2, 3 のいずれかです")
    _validate_password(body.username, body.password)
    existing = await db.fetchone("SELECT id FROM users WHERE username=?", (body.username,))
    if existing:
        raise HTTPException(400, "そのユーザー名は既に使われています")
    await db.execute(
        "INSERT INTO users (username, password_hash, access_level) VALUES (?, ?, ?)",
        (body.username, hash_password(body.password), body.access_level),
    )
    return {"message": f"ユーザー {body.username} を作成しました"}


@router.post("/password")
async def change_password(body: PasswordChange, user: dict = Depends(get_current_user)):
    row = await db.fetchone(
        "SELECT password_hash FROM users WHERE username=?",
        (user["username"],),
    )
    if row is None or not verify_password(body.current_password, row["password_hash"]):
        raise HTTPException(400, "現在のパスワードが正しくありません")
    _validate_password(user["username"], body.new_password)
    await db.execute(
        "UPDATE users SET password_hash=? WHERE username=?",
        (hash_password(body.new_password), user["username"]),
    )
    return {"message": "パスワードを変更しました"}
