from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import jwt
from passlib.context import CryptContext

from app.config import settings
from app.database import db

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(minutes=settings.access_token_expire_minutes))
    to_encode["exp"] = expire
    # 項目高3: パスワード変更で既存 JWT を失効させるための発行時刻。
    # get_current_user 側で users.password_changed_at と突き合わせる。
    to_encode["iat"] = int(now.timestamp())
    return jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)


def _password_changed_at_unix(value: str) -> int:
    """password_changed_at (UTC ISO8601) を unix 秒 (int, 秒単位切り捨て) に変換する。

    iat も秒単位の int で発行しているため、同一秒内の再ログインは許可し
    (秒未満の誤差で不当に弾かない)、それより厳密に古いトークンのみ拒否する。
    """
    return int(datetime.fromisoformat(value).timestamp())


async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="認証情報が無効です",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        username: str = payload.get("sub", "")
        if not username:
            raise credentials_exception
        iat = payload.get("iat")
    except jwt.PyJWTError:
        raise credentials_exception

    row = await db.fetchone(
        "SELECT id, username, access_level, is_admin, must_change_password, is_active, "
        "password_changed_at FROM users WHERE username = ?",
        (username,),
    )
    if row is None or not row["is_active"]:
        raise credentials_exception

    password_changed_at = row["password_changed_at"]
    if password_changed_at:
        # パスワード変更以降に発行されたトークンのみ有効。iat が無い旧トークンは
        # fail-closed で拒否する。
        if iat is None:
            raise credentials_exception
        changed_at_unix = _password_changed_at_unix(password_changed_at)
        if int(iat) < changed_at_unix:
            raise credentials_exception

    return dict(row)


async def require_password_changed(user: dict = Depends(get_current_user)) -> dict:
    """デフォルト資格情報 (admin/admin) 等、パスワード変更が未完了の間は
    /api/auth/password と /api/auth/me 以外の API を 403 でブロックする (項目6)。
    """
    if user.get("must_change_password"):
        raise HTTPException(
            status_code=403,
            detail="初回ログインです。先にパスワードを変更してください",
        )
    return user


async def require_admin(user: dict = Depends(require_password_changed)) -> dict:
    """システム管理者権限 (ユーザー管理・監視フォルダ設定・監査ログ閲覧等)。

    文書アクセスレベル (access_level) とは独立したロール。is_admin=1 でも
    access_level は別途適用されるため、管理者だからといって全文書を読める
    わけではない (項目7)。
    """
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="管理者権限が必要です")
    return user
