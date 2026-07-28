"""bcrypt と JWT 周りの検証。"""
from datetime import timedelta

import pytest
import jwt

from app.config import settings
from app.core.security import (
    create_access_token,
    hash_password,
    verify_password,
)


def test_hash_password_is_not_plain():
    h = hash_password("secret123")
    assert h != "secret123"
    assert h.startswith("$2b$") or h.startswith("$2a$")


def test_verify_password_roundtrip():
    h = hash_password("p@ssw0rd")
    assert verify_password("p@ssw0rd", h) is True
    assert verify_password("wrong", h) is False


def test_init_sql_admin_hash_matches_admin_password():
    """migrations/init.sql に焼かれた admin の bcrypt hash が 'admin' で通ること。

    init.sql からハッシュを抽出して検証する（ファイルに焼かれた値が壊れていないか実測）。
    """
    import re
    from pathlib import Path

    sql_path = Path(__file__).resolve().parent.parent / "migrations" / "init.sql"
    sql = sql_path.read_text(encoding="utf-8")
    m = re.search(r"'admin',\s*'(\$2[aby]\$[^']+)'", sql)
    assert m is not None, "init.sql から admin のハッシュを抽出できません"
    seeded_hash = m.group(1)
    assert verify_password("admin", seeded_hash) is True, (
        f"init.sql の admin ハッシュ {seeded_hash} が 'admin' で検証できません"
    )


def test_jwt_token_contains_sub_and_exp():
    token = create_access_token({"sub": "alice"})
    payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
    assert payload["sub"] == "alice"
    assert "exp" in payload


def test_jwt_token_respects_custom_expiry():
    token = create_access_token({"sub": "bob"}, expires_delta=timedelta(minutes=1))
    payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
    assert payload["sub"] == "bob"


def test_jwt_invalid_signature_rejected():
    token = create_access_token({"sub": "carol"})
    with pytest.raises(Exception):
        jwt.decode(token, "wrong-secret", algorithms=[settings.algorithm])
