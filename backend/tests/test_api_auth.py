"""FastAPI TestClient で /api/auth と /health の挙動を検証する。

DB を一時パスへ差し替え、Qdrant/Watcher/Task processor のライフサイクル副作用を回避する。
"""
import asyncio
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.database as db_module
from app.core.security import hash_password


# ─── DB を一時パスへ差し替え ──────────────────────────────
@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    db_path = tmp_path / "app.db"

    monkeypatch.setattr(db_module, "DB_PATH", db_path)
    monkeypatch.setattr(db_module.db, "_path", str(db_path))

    sql = (Path(__file__).resolve().parent.parent / "migrations" / "init.sql").read_text(
        encoding="utf-8"
    )
    with sqlite3.connect(db_path) as conn:
        conn.executescript(sql)

    yield db_path


@pytest.fixture()
def client(isolated_db, monkeypatch):
    """lifespan の重い依存をスタブして TestClient を起動。"""
    async def _noop(*args, **kwargs):
        return None

    def _noop_sync(*args, **kwargs):
        return None

    monkeypatch.setattr("app.main.init_db", _noop)
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


# ─── /api/auth/me ──────────────────────────────────────
def _login(client, username: str = "admin", password: str = "admin") -> str:
    r = client.post(
        "/api/auth/token",
        data={"username": username, "password": password},
    )
    return r.json()["access_token"]


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
        json={"username": "alice", "password": "alicepw", "access_level": 2},
    )
    assert r.status_code == 200

    token2 = _login(client, "alice", "alicepw")
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


# ─── /health ────────────────────────────────────────────
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
