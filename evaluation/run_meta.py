"""評価 1 回ぶんの実行条件を run_meta.json に残す (docs/v2-light-model-plan.md の Step 0.5)。

run_eval 本体は重い依存を読み込むので、テストできるよう問の読み込みと記録をここに分ける。
run_meta.json は最初の問を流す前に書き (status: running)、終わったら書き直す (done / failed)。
途中で落ちても、どの条件で何問まで流したかが残る。
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

META_NAME = "run_meta.json"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ─── 問の読み込み ──────────────────────────────────────────────
def parse_ids(text: str | None) -> list[str] | None:
    """--ids の "Q001,N03" を ["Q001", "N03"] にする。未指定なら None。"""
    if text is None:
        return None
    ids = [s.strip() for s in text.split(",") if s.strip()]
    if not ids:
        raise ValueError("--ids が空")
    return ids


def load_questions(paths: list[Path], ids: list[str] | None = None) -> tuple[list[dict], list[dict]]:
    """複数の jsonl を順につなげて読み、ids があればその問だけ残す (並びはファイルの順)。

    ファイルをまたいで同じ ID があれば ValueError (どちらを流したか分からなくなるため)。
    ids に無い ID が入っていれば ValueError (打ち間違いで問が黙って減るのを防ぐ)。
    戻り値の 2 つ目は各ファイルの {path, sha256, rows}。データセットの版を記録に残すため。
    """
    rows: list[dict] = []
    infos: list[dict] = []
    seen: dict[str, str] = {}
    for p in paths:
        p = Path(p)
        n = 0
        with p.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                qid = row["id"]
                if qid in seen:
                    raise ValueError(f"ID {qid} が {seen[qid]} と {p} の両方にある")
                seen[qid] = str(p)
                rows.append(row)
                n += 1
        infos.append({"path": str(p), "sha256": file_sha256(p), "rows": n})

    if ids is not None:
        unknown = [i for i in ids if i not in seen]
        if unknown:
            raise ValueError(f"--ids に無い ID がある: {', '.join(unknown)}")
        wanted = set(ids)
        rows = [r for r in rows if r["id"] in wanted]
    return rows, infos


def parse_meta_args(items: list[str] | None) -> dict[str, str]:
    """--meta KEY=VALUE の並びを dict にする。= が無い・キーが空・キーの重複は ValueError。"""
    out: dict[str, str] = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ValueError(f"--meta は KEY=VALUE の形で渡す: {item!r}")
        if key in out:
            raise ValueError(f"--meta のキーが重複: {key}")
        out[key] = value
    return out


# ─── 実行環境 ──────────────────────────────────────────────────
def git_info(root: Path, run=subprocess.run) -> dict:
    """評価のコミットと、コミットしていない変更の有無。git が使えなければ error だけ返す。"""
    try:
        head = run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        status = run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"head": None, "error": str(exc)}
    changes = [line for line in status.splitlines() if line.strip()]
    return {"head": head, "dirty": bool(changes), "changes": changes}


def fetch_vllm_models(base_url: str, timeout: float = 5.0, opener=urllib.request.urlopen) -> dict:
    """vLLM の /models から、載っているモデルの id・元の名前・max_model_len を取る。

    取れなくても評価は止めない (error を記録に残す)。
    """
    url = base_url.rstrip("/") + "/models"
    try:
        with opener(url, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - 記録用なので理由だけ残す
        return {"url": url, "error": f"{type(exc).__name__}: {exc}"}
    models = [
        {"id": m.get("id"), "root": m.get("root"), "max_model_len": m.get("max_model_len")}
        for m in body.get("data", [])
    ]
    return {"url": url, "models": models}


# ─── 記録の中身 ────────────────────────────────────────────────
_SETTINGS_KEYS = (
    "llm_provider", "llm_model", "vllm_base_url", "llm_max_tokens", "llm_temperature",
    "embed_model", "embed_device", "rerank_enabled", "rerank_model", "rerank_device",
    "rerank_candidates", "qdrant_collection",
    # 検索の結果を変える欄。索引と評価で DATA_DIR / Qdrant がずれると全問の根拠が消えるので場所も残す
    "retrieval_top_k", "chunk_size", "chunk_overlap", "vector_size", "data_dir", "qdrant_url",
)


def build_run_meta(
    *,
    argv: list[str],
    datasets: list[dict],
    ids: list[str] | None,
    n_questions: int,
    providers: list[str],
    rerank_modes: list[bool | None],
    judge: str | None,
    settings,
    extras: dict[str, str],
    git: dict,
    vllm: dict | None,
) -> dict:
    return {
        "status": "running",
        "started_at": _now(),
        "argv": list(argv),
        "python": sys.version.split()[0],
        "git": git,
        "datasets": datasets,
        "ids": ids,
        "n_questions": n_questions,
        "providers": providers,
        "rerank_modes": rerank_modes,
        "judge": judge,
        "settings": {k: getattr(settings, k, None) for k in _SETTINGS_KEYS},
        "vllm": vllm,
        "extras": extras,
    }


def write_run_meta(out_dir: Path, meta: dict) -> Path:
    """run_meta.json を書く。途中で落ちても壊れた JSON を残さないよう、別名に書いてから置き換える。"""
    path = Path(out_dir) / META_NAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def finish_run_meta(meta: dict, counts: dict[str, int], error: str | None = None) -> dict:
    """終わった時刻と、run_label ごとに保存した行数を足す。error があれば status は failed。"""
    meta["status"] = "failed" if error else "done"
    meta["finished_at"] = _now()
    meta["results"] = counts
    if error:
        meta["error"] = error
    return meta


def judged_meta(
    source_raw: Path, judge: str, judge_model: str, argv: list[str], extras: dict[str, str]
) -> dict:
    """--judge-from の記録。採点した raw.jsonl の sha256 と、回答を作った実行の run_meta.json を写す。"""
    source_raw = Path(source_raw)
    src_meta_path = source_raw.parent / META_NAME
    src_meta = (
        json.loads(src_meta_path.read_text(encoding="utf-8")) if src_meta_path.exists() else None
    )
    return {
        "status": "running",
        "started_at": _now(),
        "argv": list(argv),
        "judged_from": {"path": str(source_raw), "sha256": file_sha256(source_raw)},
        "judge": {"provider": judge, "model": judge_model},
        "extras": extras,
        "source_run_meta": src_meta,
    }
