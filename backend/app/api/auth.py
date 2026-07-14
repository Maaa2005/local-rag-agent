import time
from datetime import datetime, timezone

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
from app.services.admin_events import record_admin_event

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
    # システム管理者権限 (項目7: 文書アクセスレベルとは独立)。作成できるのは
    # 既存の管理者 (require_admin) のみ。
    is_admin: bool = False


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
        "SELECT password_hash, is_active, must_change_password FROM users WHERE username=?",
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
    # DB に永続化されたフラグを単一の真実源とする (項目6)。ヒューリスティックな
    # 平文パスワード比較ではなく、実際にパスワードが変更されたかを追跡する。
    must_change_password = bool(row["must_change_password"])
    return Token(access_token=token, token_type="bearer", must_change_password=must_change_password)


@router.get("/me")
async def me(user: dict = Depends(get_current_user)):
    return {
        "username": user["username"],
        "access_level": user["access_level"],
        "is_admin": bool(user.get("is_admin")),
        "must_change_password": bool(user.get("must_change_password")),
    }


@router.post("/users")
async def create_user(body: UserCreate, actor: dict = Depends(require_admin)):
    if body.access_level not in (1, 2, 3):
        raise HTTPException(400, "access_level は 1, 2, 3 のいずれかです")
    _validate_password(body.username, body.password)
    existing = await db.fetchone("SELECT id FROM users WHERE username=?", (body.username,))
    if existing:
        raise HTTPException(400, "そのユーザー名は既に使われています")
    # 項目中6: 新規ユーザーは初回ログイン後に必ずパスワード変更を求める。
    await db.execute(
        "INSERT INTO users (username, password_hash, access_level, is_admin, must_change_password) "
        "VALUES (?, ?, ?, ?, 1)",
        (body.username, hash_password(body.password), body.access_level, int(body.is_admin)),
    )
    # 項目高4: admin による自己昇格経路自体は仕様上防がないが、事後検知できるよう記録する。
    await record_admin_event(
        actor,
        "create_user",
        {
            "username": body.username,
            "access_level": body.access_level,
            "is_admin": bool(body.is_admin),
        },
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
    now = datetime.now(timezone.utc).isoformat()
    # 項目高3: password_changed_at を更新し、これより古い iat を持つ既存 JWT を失効させる。
    await db.execute(
        "UPDATE users SET password_hash=?, must_change_password=0, password_changed_at=? "
        "WHERE username=?",
        (hash_password(body.new_password), now, user["username"]),
    )
    # 項目高4: パスワード自体は絶対に記録せず、誰が変更したか (本人) のみ記録する。
    await record_admin_event(user, "change_password", {"username": user["username"]})
    return {"message": "パスワードを変更しました"}
