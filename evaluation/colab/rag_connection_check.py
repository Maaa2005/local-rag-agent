"""v2 の RAG 接続を実 Qdrant / vLLM で確かめる（Issue #7）。Colab のノート rag_connection_check.ipynb から呼ぶ。

確かめること（架空の社内規程 evaluation/corpus だけを使う）:
  1. 2 問続けて聞いても答えられる（v1 の AsyncQdrantClient を同じループで使い回せている）
  2. 権限外の文書が v1 の検索結果に出ない（v2 側で捨てる前の生の行で見る。上位ユーザーでは出ることも見る）
  3. 監査 DB（WAL 込み）に質問・根拠・回答の本文が残らない
  4. vLLM / Qdrant が固まる・止まると、上限時間内に held で止まる（止めた後は戻ることも見る）
  5. 1 問あたりの検索・生成の時間（T4 は 6GB 機と速度が違うので目安）

Qdrant / vLLM（または OpenAI 互換の代役）は先に立てておき、プロセスグループ ID を渡す。
v1 の設定は環境変数で上書きする（.env は使わない）。app を import する前に設定する必要がある。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import statistics
import sys
import time
import types
import uuid
from pathlib import Path

LEVELS = {"general": 1, "manager": 2, "executive": 3}

# C1 はルール判定。自由質問は local に回る言い回しにしてある（start で確かめる）
TWO_IN_A_ROW = [
    ("u_general", "経費精算の申請はいつまでに出せばよいですか"),
    ("u_general", "在宅勤務を始めるときの申請方法を教えてください"),
]
# 上位の文書を狙った質問。dataset.jsonl の permission_test から C1 が local に回すものを選んだ
PERMISSION = [
    ("u_general", "一般社員が役員報酬の詳細を知りたい場合、どこに記載がありますか?"),  # Q010
    ("u_general", "M&A 案件の進捗状況を教えてください。"),  # Q011
    ("u_general", "目標管理シートの提出期限を教えてください。"),  # Q039
    ("u_general", "採用面接の合否判定基準を教えてください。"),  # Q043
    ("u_manager", "今期の役員賞与の総額を教えてください。"),  # Q041
    ("u_manager", "次年度の経営計画における重点投資領域は何ですか?"),  # Q042
    ("u_manager", "M&A案件のデューデリジェンス期間を教えてください。"),  # Q044
]
# 対照: 上位ユーザーなら上位の文書が出る（フィルタが何でも空にしているだけ、を除く）
CONTROL = [("u_exec", "役員報酬はどのように決めますか"), ("u_manager", "目標管理シートの提出期限を教えてください。")]
RECOVERY_Q = ("u_general", "出張旅費の精算方法を教えてください")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--work", required=True, type=Path)
    p.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    p.add_argument("--vllm-url", default="http://127.0.0.1:8001/v1")
    p.add_argument("--qdrant-pgid", type=int, help="止める試験に使う。無ければ 4 を飛ばす")
    p.add_argument("--vllm-pgid", type=int)
    p.add_argument("--fault-timeout", type=float, default=15.0)
    p.add_argument("--rerank", choices=["on", "off"], default="on")
    p.add_argument("--label", default="colab", help="結果に残す実行環境の名前（例: colab-T4 / mac-dryrun）")
    p.add_argument("--inject-leak", action="store_true",
                   help="検出の自己確認: 権限を無視して最上位レベルで検索する。2 が FAIL になれば検出できている")
    return p.parse_args()


def configure(a: argparse.Namespace) -> None:
    a.work.mkdir(parents=True, exist_ok=True)
    env = {
        "DATA_DIR": str(a.work / "data"),
        "WATCHED_PATH": str(a.work / "watched"),
        "QDRANT_URL": a.qdrant_url,
        "QDRANT_COLLECTION": f"rag_check_{uuid.uuid4().hex[:8]}",  # 毎回まっさらな集合
        "LLM_PROVIDER": "vllm",
        "VLLM_BASE_URL": a.vllm_url,
        "LLM_MODEL": "local-llm",
        "RERANK_ENABLED": "true" if a.rerank == "on" else "false",
    }
    os.environ.update(env)
    sys.path[:0] = [str(a.repo / "backend"), str(a.repo / "v2")]


class Report:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, check: str, ok: bool, detail: str) -> None:
        self.items.append({"check": check, "ok": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {check}: {detail}", flush=True)

    @property
    def ok(self) -> bool:
        return bool(self.items) and all(i["ok"] for i in self.items)


def main() -> int:
    a = parse_args()
    configure(a)

    # ここから下は環境変数を入れた後で import する（app.database は import 時に DATA_DIR を読む）
    from app.config import settings
    from app.database import db, init_db
    from app.services.indexer import ensure_collection, index_document
    from integration.ask import ask
    from judge.rules import RuleC1Router
    from orchestrator.audit_store import AuditStore
    from orchestrator.policy import load_policy
    from orchestrator.rag_backend import BackendGenerator, BackendRetriever, run_on_shared_loop

    assert settings.qdrant_url == a.qdrant_url and settings.vllm_base_url == a.vllm_url, "環境変数が効いていない"
    rep = Report()
    pol = load_policy()
    c1 = RuleC1Router()

    # C1 が local に回さない質問があると、検索まで届かず試験にならない
    every_q = TWO_IN_A_ROW + PERMISSION + CONTROL + [RECOVERY_Q]
    not_local = [q for u, q in every_q if c1.route(u, q, None).route != "local"]
    if not_local:
        print("C1 が local 以外に回す質問がある。質問を直す:", not_local)
        return 2

    # ---- 索引: v1 と同じく documents 行（SQLite 再検証の正）を入れてから Qdrant に載せる。全部共有ループで ----
    corpus = a.repo / "evaluation" / "corpus"
    doc_level: dict[str, int] = {}
    doc_name: dict[str, str] = {}

    async def build_index() -> int:
        await init_db()
        await ensure_collection()
        n = 0
        now = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
        for folder, level in LEVELS.items():
            for f in sorted((corpus / folder).glob("*.md")):
                text = f.read_text(encoding="utf-8")
                doc_id = str(uuid.uuid4())
                src = f"/watched/{folder}/{f.name}"
                await db.execute(
                    "INSERT INTO documents (id, source_path, file_hash, access_level, file_type, status, "
                    "unclassified, index_version, created_at, updated_at) VALUES (?, ?, ?, ?, '.md', 'done', 0, 1, ?, ?)",
                    (doc_id, src, hashlib.sha256(text.encode()).hexdigest(), level, now, now),
                )
                n += await index_document(doc_id, text, src, level, 1)
                doc_level[doc_id], doc_name[doc_id] = level, f.name
        return n

    t0 = time.monotonic()
    chunks = run_on_shared_loop(build_index(), 1800)
    print(f"索引: 文書 {len(doc_level)} 件 / チャンク {chunks} 件 / {time.monotonic() - t0:.0f} 秒", flush=True)

    # 生の検索結果を記録するため、v1 の retrieve を包んだモジュールを BackendRetriever に渡す（クラスは本物のまま）
    import app.services.retriever as v1_retriever

    raw: list[list[tuple[str | None, object]]] = []
    probe = types.ModuleType("rag_check_probe")

    async def recording_retrieve(question, user_access_level):
        rows = await v1_retriever.retrieve(question, max(LEVELS.values()) if a.inject_leak else user_access_level)
        raw.append([(r.get("document_id"), r.get("access_level")) for r in rows])
        return rows

    probe.retrieve = recording_retrieve
    sys.modules["rag_check_probe"] = probe

    store = AuditStore(a.work / "audit.db")
    bodies: list[str] = []
    timings: list[dict] = []

    class Timed:
        def __init__(self, inner, key):
            self.inner, self.key = inner, key

        def retrieve(self, q, level):
            t = time.monotonic()
            try:
                return self.inner.retrieve(q, level)
            finally:
                timings[-1][self.key] = round(time.monotonic() - t, 2)

        def answer(self, q, chunks):
            t = time.monotonic()
            try:
                return self.inner.answer(q, chunks)
            finally:
                timings[-1][self.key] = round(time.monotonic() - t, 2)

    def run(user, q, *, r_timeout=30.0, g_timeout=180.0, tag=""):
        timings.append({"tag": tag, "user": user})
        r = Timed(BackendRetriever("rag_check_probe", timeout_s=r_timeout), "retrieve_s")
        g = Timed(BackendGenerator(timeout_s=g_timeout), "generate_s")
        t = time.monotonic()
        res = ask(c1, user, q, pol, r, g, audit_sink=store)
        timings[-1]["total_s"] = round(time.monotonic() - t, 2)
        timings[-1]["outcome"] = res.run.outcome
        bodies.extend(b for b in res.bodies if b)
        return res, time.monotonic() - t

    def brief(res) -> str:
        return f"{res.run.outcome}/{res.run.stopped_at} {res.run.reason or ''}".strip()

    # ---- 1. 2 問続けて ----
    got = [run(u, q, tag="two")[0] for u, q in TWO_IN_A_ROW]
    rep.add("1 二問続けて回答", all(r.run.outcome == "answered" for r in got), " | ".join(brief(r) for r in got))
    for r in got:
        print("  回答例:", (r.run.output_text or "")[:120].replace("\n", " "), "根拠:", r.sources)

    # ---- 2. 権限外の文書が v1 の生の結果に出ない ----
    def check_raw(user: str, q: str, tag: str):
        n_before = len(raw)
        res, _ = run(user, q, tag=tag)
        rows = raw[n_before] if len(raw) > n_before else None
        return res, rows

    leaks, unknown, held_perm = [], [], 0
    for user, q in PERMISSION:
        lv = pol.users[user].level
        res, rows = check_raw(user, q, "perm")
        if rows is None:
            leaks.append(f"{q[:12]}…: 検索結果が記録されていない ({brief(res)})")
            continue
        held_perm += res.run.outcome == "held"
        for doc_id, payload_lv in rows:
            if doc_id not in doc_level:
                unknown.append(doc_id)
            elif doc_level[doc_id] > lv or (isinstance(payload_lv, int) and payload_lv > lv):
                leaks.append(f"{user} に {doc_name[doc_id]}(Lv{doc_level[doc_id]})")
        dropped = [e.detail for e in res.run.audit if e.stage == "retrieve" and "dropped=" in e.detail]
        if any("dropped=0" not in d for d in dropped):
            leaks.append(f"v2 側で捨てた行がある（v1 が上位文書を返した）: {dropped}")
    rep.add("2 権限外の文書が出ない", not leaks and not unknown,
            f"{len(PERMISSION)} 問 / 漏れ {sorted(set(leaks)) or 'なし'} / 不明ID {sorted(set(unknown)) or 'なし'} / "
            f"held {held_perm} 問")

    seen_ctrl = []
    for user, q in CONTROL:
        lv = pol.users[user].level
        res, rows = check_raw(user, q, "control")
        top = max((doc_level.get(d, 0) for d, _ in rows or []), default=0)
        seen_ctrl.append((user, lv, top, brief(res)))
    rep.add("2b 上位ユーザーには上位文書が出る（対照）", all(top == lv for _, lv, top, _ in seen_ctrl),
            "; ".join(f"{u}: 最上位 Lv{top}（{b}）" for u, _, top, b in seen_ctrl))

    # ---- 4. 止める試験（最後にまとめて。止めたら必ず戻す） ----
    ft = a.fault_timeout

    def fault(name, pgid, sig, stage, r_t, g_t):
        os.killpg(pgid, sig)
        try:
            res, el = run(RECOVERY_Q[0], RECOVERY_Q[1], r_timeout=r_t, g_timeout=g_t, tag=name)
        finally:
            if sig == signal.SIGSTOP:
                os.killpg(pgid, signal.SIGCONT)
        ok = res.run.outcome == "held" and res.run.stopped_at == stage and el < ft + 20 and res.run.output_text is None
        rep.add(f"4 {name}", ok, f"{brief(res)} / {el:.1f} 秒（上限 {ft:.0f} 秒）")

    def recovered(name):
        res, el = run(*RECOVERY_Q, tag=f"recover-{name}")
        rep.add(f"4 {name} の後に戻る", res.run.outcome == "answered", f"{brief(res)} / {el:.1f} 秒")

    if a.vllm_pgid:
        fault("vLLM 固まる", a.vllm_pgid, signal.SIGSTOP, "generate", 30.0, ft)
        recovered("vLLM 固まる")
    if a.qdrant_pgid:
        fault("Qdrant 固まる", a.qdrant_pgid, signal.SIGSTOP, "retrieve", ft, 180.0)
        recovered("Qdrant 固まる")
    if a.vllm_pgid:
        fault("vLLM 停止", a.vllm_pgid, signal.SIGTERM, "generate", 30.0, ft)
    if a.qdrant_pgid:
        fault("Qdrant 停止", a.qdrant_pgid, signal.SIGTERM, "retrieve", ft, ft)
    if not (a.vllm_pgid and a.qdrant_pgid):
        print("[SKIP] 4 止める試験の一部（プロセスグループ ID が無い）")

    # ---- 3. 監査 DB に本文が無い（WAL・SHM のバイト列まで） ----
    runs = store.runs()
    events = [e for r in runs for e in store.events(r["run_id"])]
    store.close()
    dumped = json.dumps([runs, events], ensure_ascii=False, default=str)
    blobs = b"".join(p.read_bytes() for p in a.work.glob("audit.db*"))
    targets = sorted({b for b in bodies if len(b) >= 8})
    hit_rows = [b[:16] for b in targets if b in dumped]
    hit_bytes = [b[:16] for b in targets if b.encode("utf-8") in blobs]
    has_ids = any(d in dumped for d in doc_level)
    rep.add("3 監査に本文が残らない", not hit_rows and not hit_bytes and has_ids and len(runs) == len(timings),
            f"run {len(runs)} 件 / 調べた本文 {len(targets)} 個 / 行に {len(hit_rows)} 個・ファイルに {len(hit_bytes)} 個 / "
            f"根拠の文書ID {'あり' if has_ids else 'なし'}")

    # ---- 5. 時間 ----
    ok_rows = [t for t in timings if t.get("outcome") == "answered" and "generate_s" in t]
    if ok_rows:
        r_s = [t["retrieve_s"] for t in ok_rows]
        g_s = [t["generate_s"] for t in ok_rows]
        print(f"[INFO] 5 時間（{a.label}・回答できた {len(ok_rows)} 問）: 検索 中央 {statistics.median(r_s):.2f} / 最大 {max(r_s):.2f} 秒, "
              f"生成 中央 {statistics.median(g_s):.2f} / 最大 {max(g_s):.2f} 秒")

    out = a.work / "rag_check_result.json"
    out.write_text(json.dumps({"label": a.label, "rerank": a.rerank, "ok": rep.ok, "checks": rep.items,
                               "timings": timings}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n総合: {'PASS' if rep.ok else 'FAIL'}（結果: {out}。本文は入れていない）")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
