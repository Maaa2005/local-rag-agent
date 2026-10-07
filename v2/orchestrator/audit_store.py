"""RunResult.audit を SQLite に追記専用で残す監査ストア。

設計書 F9: 判断モデル・revision・質問の版・結果・確率、承認者と対象、実際の送信内容、
要求 ID、送信結果を対応付けて残す。ここでは「実際の送信内容」を本文ではなく digest で
対応付ける（本文は Gateway が保持する。設計書 N7 の保持期限は本文 30 日・メタデータ 90 日で、
本文をここに複製すると保持の管理対象が増えるため）。承認済み説明文・依頼文・出力テキストは保存しない。

- 追記専用: UPDATE / DELETE の API を持たず、DB にもトリガーで禁止する。
  そのため N7 の保持期限による削除は未実装（「実装済み」と書かない）。改ざん検知（HMAC 等）も未実装。
- 1 実行 = 1 run_id。runs 1 行 + events n 行を 1 トランザクションで書く。
- 書き込み失敗は送信判断に影響させない。persist_run は例外を上げず結果に印を付けるだけ。
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Protocol, Sequence

if TYPE_CHECKING:
    from orchestrator.pipeline import RunResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    recorded_at TEXT NOT NULL,
    run_at TEXT,
    user_id TEXT,
    contract TEXT,
    destination TEXT,
    request_id TEXT,
    digest TEXT,
    outcome TEXT NOT NULL,
    stopped_at TEXT NOT NULL,
    reason TEXT NOT NULL,
    gateway_received INTEGER NOT NULL,
    attempted INTEGER NOT NULL,
    gateway_state TEXT,
    c1_route TEXT,
    c1_model TEXT,
    c1_revision TEXT,
    c1_probs TEXT,
    c2_decision TEXT,
    c2_model TEXT,
    c2_revision TEXT,
    c2_prob_block REAL,
    c2_truncated INTEGER,
    response_visibility TEXT,
    output_sha256 TEXT,
    output_chars INTEGER
);
CREATE TABLE IF NOT EXISTS events (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    seq INTEGER NOT NULL,
    stage TEXT NOT NULL,
    result TEXT NOT NULL,
    detail TEXT NOT NULL,
    PRIMARY KEY (run_id, seq)
);
CREATE INDEX IF NOT EXISTS runs_request_id ON runs(request_id);
CREATE TRIGGER IF NOT EXISTS runs_no_update BEFORE UPDATE ON runs BEGIN SELECT RAISE(ABORT, 'audit is append-only'); END;
CREATE TRIGGER IF NOT EXISTS runs_no_delete BEFORE DELETE ON runs BEGIN SELECT RAISE(ABORT, 'audit is append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT, 'audit is append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT, 'audit is append-only'); END;
"""

_RUN_COLUMNS = {
    line.split()[0]
    for line in SCHEMA.split("CREATE TABLE IF NOT EXISTS runs (")[1].split(");")[0].strip().splitlines()
}

REDACTED = "[本文除去]"
_MIN_FRAGMENT = 8  # 本文の行単位の断片もこの字数以上なら除去する


@dataclass(frozen=True)
class AuditRecord:
    run: dict[str, Any]
    events: tuple[tuple[str, str, str], ...]  # (stage, result, detail)
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)


class AuditSink(Protocol):
    def append(self, record: AuditRecord) -> None: ...


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def scrub(text: str, bodies: Iterable[str]) -> str:
    """本文（全体と 8 字以上の各行）が混ざっていれば除去する。長いものから置換する。"""
    frags: set[str] = set()
    for b in bodies:
        if not b:
            continue
        frags.add(b)
        frags.update(line.strip() for line in b.splitlines() if len(line.strip()) >= _MIN_FRAGMENT)
    for f in sorted(frags, key=len, reverse=True):
        if f and f in text:
            text = text.replace(f, REDACTED)
    return text


def build_record(
    result: RunResult,
    *,
    user: str | None,
    contract: str | None,
    destination: str | None,
    run_at: datetime | None,
    bodies: Sequence[str] = (),
    c1: Any = None,
) -> AuditRecord:
    """RunResult から監査対象の項目だけを取り出す。本文は digest・字数だけにする。"""
    out = result.output_text
    all_bodies = [*bodies, *([out] if out else [])]
    v = result.c2_verdict
    run = dict(
        recorded_at=datetime.now(timezone.utc).isoformat(),
        run_at=run_at.isoformat() if run_at else None,
        user_id=user,
        contract=contract,
        destination=destination,
        request_id=result.request_id,
        digest=result.digest,
        outcome=result.outcome,
        stopped_at=result.stopped_at,
        reason=scrub(result.reason, all_bodies),
        gateway_received=int(result.gateway_received),
        attempted=int(result.attempted),
        gateway_state=result.gateway_state.value if result.gateway_state else None,
        c1_route=getattr(c1, "route", None),
        c1_model=getattr(c1, "model", None),
        c1_revision=getattr(c1, "revision", None),
        c1_probs=json.dumps(c1.probs, sort_keys=True) if c1 is not None and getattr(c1, "probs", None) else None,
        c2_decision=v.decision if v else None,
        c2_model=v.model if v else None,
        c2_revision=v.revision if v else None,
        c2_prob_block=v.prob_block if v else None,
        c2_truncated=int(v.truncated) if v else None,
        response_visibility=result.response_visibility,
        output_sha256=_sha256(out) if out else None,
        output_chars=len(out) if out else None,
    )
    events = tuple(
        (e.stage, scrub(e.result, all_bodies), scrub(e.detail, all_bodies)) for e in result.audit
    )
    return AuditRecord(run=run, events=events)


def persist_run(
    sink: AuditSink | None,
    result: RunResult,
    *,
    user: str | None,
    contract: str | None,
    destination: str | None,
    run_at: datetime | None,
    bodies: Sequence[str] = (),
    c1: Any = None,
) -> RunResult:
    """結果確定時に一括で書く。失敗しても例外を上げず audit_persisted=False にする（再送はしない）。"""
    if sink is None:
        return result
    try:
        rec = build_record(result, user=user, contract=contract, destination=destination,
                           run_at=run_at, bodies=bodies, c1=c1)
        sink.append(rec)
    except Exception as e:  # noqa: BLE001  監査の失敗で送信判断を変えない
        result.audit_persisted = False
        result.audit_error = f"{type(e).__name__}"
        return result
    result.audit_persisted = True
    result.audit_run_id = rec.run_id
    return result


class AuditStore:
    """SQLite（WAL, synchronous=FULL）の追記専用ストア。append と読み出しだけを持つ。"""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def append(self, record: AuditRecord) -> None:
        cols = ["run_id", *record.run.keys()]
        unknown = set(cols) - _RUN_COLUMNS
        if unknown:
            raise ValueError(f"未知の監査項目: {sorted(unknown)}")
        with self._lock:
            c = self._conn
            c.execute("BEGIN IMMEDIATE")
            try:
                c.execute(
                    f"INSERT INTO runs ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                    [record.run_id, *record.run.values()],
                )
                c.executemany(
                    "INSERT INTO events (run_id, seq, stage, result, detail) VALUES (?,?,?,?,?)",
                    [(record.run_id, i, *ev) for i, ev in enumerate(record.events)],
                )
                c.execute("COMMIT")
            except BaseException:
                c.execute("ROLLBACK")
                raise

    def runs(self, request_id: str | None = None) -> list[dict[str, Any]]:
        q, args = "SELECT * FROM runs", ()
        if request_id is not None:
            q, args = q + " WHERE request_id = ?", (request_id,)
        return self._rows(q + " ORDER BY recorded_at, rowid", args)

    def events(self, run_id: str) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM events WHERE run_id = ? ORDER BY seq", (run_id,))

    def _rows(self, q: str, args: Sequence[Any]) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(q, args)
            names = [d[0] for d in cur.description]
            return [dict(zip(names, r)) for r in cur.fetchall()]

