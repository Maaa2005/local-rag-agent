"""FastAPI TestClient で /api/auth と /health の挙動を検証する。

DB を一時パスへ差し替え、Qdrant/Watcher/Task processor のライフサイクル副作用を回避する。
"""
import asyncio
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.database as db_module
from app.api import auth as auth_module
from app.config import Settings
from app.core.security import hash_password


@pytest.fixture(autouse=True)
def _reset_failed_logins():
    """テスト間でログイン失敗カウンタを持ち越さない。"""
    auth_module._failed_logins.clear()
    yield
    auth_module._failed_logins.clear()


# ─── DB を一時パスへ差し替え ──────────────────────────────
@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    db_path = tmp_path / "app.db"

    # DB_PATH を一時ファイルへ差し替え、シングルトン接続もリセット
    monkeypatch.setattr(db_module, "DB_PATH", db_path)
    db_module.db._conn = None

    # lifespan の init_db でもスキーマ初期化される (DEFAULT admin 投入)。
    yield db_path

    # 後始末: テスト間で接続を持ち越さない
    if db_module.db._conn is not None:
        asyncio.run(db_module.db.close())


@pytest.fixture()
def client(isolated_db, monkeypatch):
    """lifespan の重い依存をスタブして TestClient を起動。"""
    async def _noop(*args, **kwargs):
        return None

    def _noop_sync(*args, **kwargs):
        return None

    monkeypatch.setattr("app.main.ensure_collection", _noop)
    monkeypatch.setattr("app.main.start_watcher", _noop_sync)
    monkeypatch.setattr("app.main.stop_watcher", _noop_sync)

    async def _run_task_processor_stub():
        try:
            while True:
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            raise

    monkeypatch.setattr("app.main.run_task_processor", _run_task_processor_stub)

    from app.main import app

    with TestClient(app) as c:
        yield c


# ─── /api/auth/token ──────────────────────────────────
def test_login_success(client):
    r = client.post(
        "/api/auth/token",
        data={"username": "admin", "password": "admin"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]


def test_login_wrong_password(client):
    r = client.post(
        "/api/auth/token",
        data={"username": "admin", "password": "wrong"},
    )
    assert r.status_code == 401


def test_login_unknown_user(client):
    r = client.post(
        "/api/auth/token",
        data={"username": "ghost", "password": "x"},
    )
    assert r.status_code == 401


# ─── ログイン失敗レート制限 ─────────────────────────────
def test_login_lockout_after_repeated_failures(client):
    for _ in range(5):
        r = client.post(
            "/api/auth/token",
            data={"username": "admin", "password": "wrong"},
        )
        assert r.status_code == 401

    # 6回目は正しいパスワードでもロックアウトされる
    r = client.post(
        "/api/auth/token",
        data={"username": "admin", "password": "admin"},
    )
    assert r.status_code == 429


def test_login_lockout_clears_after_cooldown(client, monkeypatch):
    for _ in range(5):
        r = client.post(
            "/api/auth/token",
            data={"username": "admin", "password": "wrong"},
        )
        assert r.status_code == 401

    r = client.post(
        "/api/auth/token",
        data={"username": "admin", "password": "admin"},
    )
    assert r.status_code == 429

    # 60秒経過したものとして扱う (実時間 sleep はしない)
    failures, last_failed_at = auth_module._failed_logins["admin"]
    auth_module._failed_logins["admin"] = (failures, last_failed_at - 61.0)

    r = client.post(
        "/api/auth/token",
        data={"username": "admin", "password": "admin"},
    )
    assert r.status_code == 200


def test_login_success_resets_failure_counter(client):
    for _ in range(3):
        r = client.post(
            "/api/auth/token",
            data={"username": "admin", "password": "wrong"},
        )
        assert r.status_code == 401

    r = client.post(
        "/api/auth/token",
        data={"username": "admin", "password": "admin"},
    )
    assert r.status_code == 200
    assert "admin" not in auth_module._failed_logins


# ─── /api/auth/me ──────────────────────────────────────
def _login(client, username: str = "admin", password: str = "admin") -> str:
    r = client.post(
        "/api/auth/token",
        data={"username": username, "password": password},
    )
    token = r.json()["access_token"]
    if username == "admin" and password == "admin":
        # デフォルト資格情報 (admin/admin) のままだと must_change_password=1 のため
        # 他の API が 403 になる (項目6)。テストで admin 権限のその他 API を叩く
        # 前提として、ここで一度だけパスワードを変更しておく。
        client.post(
            "/api/auth/password",
            headers={"Authorization": f"Bearer {token}"},
            json={"current_password": "admin", "new_password": "adminpw12345"},
        )
        # 項目高3: パスワード変更で変更前に発行したトークンの iat が失効しうるため、
        # 変更後の状態で確実に使えるトークンを取り直す。
        r2 = client.post(
            "/api/auth/token", data={"username": username, "password": "adminpw12345"}
        )
        token = r2.json()["access_token"]
    return token


def test_me_returns_user_info(client):
    token = _login(client)
    r = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    body = r.json()
    assert body["username"] == "admin"
    assert body["access_level"] == 3


def test_me_rejects_missing_token(client):
    r = client.get("/api/auth/me")
    assert r.status_code == 401


def test_me_rejects_bogus_token(client):
    r = client.get("/api/auth/me", headers={"Authorization": "Bearer not-a-jwt"})
    assert r.status_code == 401


def test_logout_invalidates_token_and_allows_fresh_login(client):
    token = client.post(
        '/api/auth/token', data={'username': 'admin', 'password': 'admin'}
    ).json()['access_token']
    headers = {'Authorization': f'Bearer {token}'}
    assert client.get('/api/auth/me', headers=headers).status_code == 200

    logged_out = client.post('/api/auth/logout', headers=headers)
    assert logged_out.status_code == 204
    assert client.get('/api/auth/me', headers=headers).status_code == 401

    fresh = client.post(
        '/api/auth/token', data={'username': 'admin', 'password': 'admin'}
    )
    assert fresh.status_code == 200
    fresh_headers = {'Authorization': 'Bearer ' + fresh.json()['access_token']}
    assert client.get('/api/auth/me', headers=fresh_headers).status_code == 200


# ─── /api/auth/users (admin only) ────────────────────────
def test_create_user_requires_admin(client, isolated_db):
    with sqlite3.connect(isolated_db) as conn:
        conn.execute(
            "INSERT INTO users (username, password_hash, access_level) VALUES (?, ?, ?)",
            ("bob", hash_password("bobpw"), 1),
        )
        conn.commit()

    token = _login(client, "bob", "bobpw")
    r = client.post(
        "/api/auth/users",
        headers={"Authorization": f"Bearer {token}"},
        json={"username": "x", "password": "y", "access_level": 1},
    )
    assert r.status_code == 403


def test_admin_can_create_user(client):
    token = _login(client)
    r = client.post(
        "/api/auth/users",
        headers={"Authorization": f"Bearer {token}"},
        json={"username": "alice", "password": "alicepw1", "access_level": 2},
    )
    assert r.status_code == 200

    token2 = _login(client, "alice", "alicepw1")
    r2 = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token2}"})
    assert r2.status_code == 200
    assert r2.json()["access_level"] == 2


def test_admin_rejects_invalid_access_level(client):
    token = _login(client)
    r = client.post(
        "/api/auth/users",
        headers={"Authorization": f"Bearer {token}"},
        json={"username": "x", "password": "y", "access_level": 99},
    )
    assert r.status_code == 400


def test_admin_rejects_short_password(client):
    token = _login(client)
    r = client.post(
        "/api/auth/users",
        headers={"Authorization": f"Bearer {token}"},
        json={"username": "shorty", "password": "short", "access_level": 1},
    )
    assert r.status_code == 400


def test_admin_rejects_password_same_as_username(client):
    token = _login(client)
    r = client.post(
        "/api/auth/users",
        headers={"Authorization": f"Bearer {token}"},
        json={"username": "sameuser", "password": "sameuser", "access_level": 1},
    )
    assert r.status_code == 400


# ─── /api/auth/password ──────────────────────────────────
def test_change_password_success(client):
    token = _login(client)
    r = client.post(
        "/api/auth/users",
        headers={"Authorization": f"Bearer {token}"},
        json={"username": "carol", "password": "carolpw1", "access_level": 1},
    )
    assert r.status_code == 200

    token2 = _login(client, "carol", "carolpw1")
    r2 = client.post(
        "/api/auth/password",
        headers={"Authorization": f"Bearer {token2}"},
        json={"current_password": "carolpw1", "new_password": "carolpw2"},
    )
    assert r2.status_code == 200

    r3 = client.post(
        "/api/auth/token",
        data={"username": "carol", "password": "carolpw2"},
    )
    assert r3.status_code == 200


def test_change_password_wrong_current(client):
    token = _login(client)
    r = client.post(
        "/api/auth/password",
        headers={"Authorization": f"Bearer {token}"},
        json={"current_password": "wrong", "new_password": "newpassword1"},
    )
    assert r.status_code == 400


def test_change_password_rejects_weak_new_password(client):
    # _login はデフォルト資格情報だと admin のパスワードを adminpw12345 に変更する
    # ため、正しい現在パスワードで新パスワードの強度チェックに到達させる。
    token = _login(client)
    r = client.post(
        "/api/auth/password",
        headers={"Authorization": f"Bearer {token}"},
        json={"current_password": "adminpw12345", "new_password": "short"},
    )
    assert r.status_code == 400


# ─── must_change_password フラグ ──────────────────────────
def test_login_admin_default_password_flags_must_change(client):
    r = client.post(
        "/api/auth/token",
        data={"username": "admin", "password": "admin"},
    )
    assert r.status_code == 200
    assert r.json()["must_change_password"] is True


def test_login_after_password_change_clears_flag(client):
    # ここでは _login ヘルパー (デフォルト資格情報時に自動でパスワードを変える
    # 副作用を持つ) を使わず、素の admin/admin でログインしてからパスワードを
    # 変更する過程そのものを検証する。
    r0 = client.post("/api/auth/token", data={"username": "admin", "password": "admin"})
    token = r0.json()["access_token"]
    r = client.post(
        "/api/auth/password",
        headers={"Authorization": f"Bearer {token}"},
        json={"current_password": "admin", "new_password": "newadminpw1"},
    )
    assert r.status_code == 200
    assert r.json()['access_token']
    assert client.get(
        '/api/auth/me', headers={'Authorization': 'Bearer ' + token}
    ).status_code == 401
    assert client.get(
        '/api/auth/me',
        headers={'Authorization': 'Bearer ' + r.json()['access_token']},
    ).status_code == 200

    r2 = client.post(
        "/api/auth/token",
        data={"username": "admin", "password": "newadminpw1"},
    )
    assert r2.status_code == 200
    assert r2.json()["must_change_password"] is False


# ─── config: SECRET_KEY 自動生成 ──────────────────────────
def test_settings_generates_random_secret_key_when_empty():
    s1 = Settings(secret_key="")
    s2 = Settings(secret_key="")
    assert s1.secret_key != ""
    assert s1.secret_key != "change-me-in-production"
    assert s1.secret_key != s2.secret_key


# ─── /health ────────────────────────────────────────────
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
