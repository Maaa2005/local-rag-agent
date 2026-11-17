"""nuggets.jsonl の読み込み・検査・freeze、答えファイルの読み込み。"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path
from typing import Mapping

IMPORTANCES = ("vital", "okay")
FREEZE_NAME = "freeze.json"


class SchemaError(ValueError):
    """nuggets.jsonl / 答えファイルの形が不正。"""


class FreezeError(RuntimeError):
    """freeze と nuggets.jsonl が合わない（採点を止める）。"""


# ─── 逐語の照合 ─────────────────────────────────────────────────
def norm_text(s: str) -> str:
    """改行・空白（全角含む）をすべて除いた形。quote の照合専用。"""
    return "".join(ch for ch in s if not ch.isspace())


def quote_in_text(quote: str, body: str) -> bool:
    q = norm_text(quote or "")
    return bool(q) and q in norm_text(body or "")


# ─── 読み込み ───────────────────────────────────────────────────
def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SchemaError(f"{path}:{ln}: JSON が読めない ({e})") from e
            if not isinstance(obj, dict):
                raise SchemaError(f"{path}:{ln}: 行がオブジェクトでない")
            rows.append(obj)
    return rows


def load_nuggets(path: str | Path) -> list[dict]:
    return _read_jsonl(Path(path))


def load_corpus(corpus_dir: str | Path) -> dict[str, str]:
    """`<corpus_dir>/<rule_id>.md` を {rule_id: 本文} にする。"""
    d = Path(corpus_dir)
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(d.glob("*.md"))}


def default_corpus_dir(nuggets_path: str | Path) -> Path:
    """nuggets.jsonl は `<set>/` に置き、規程は `<set>/corpus/` にある前提。"""
    return Path(nuggets_path).resolve().parent / "corpus"


def vital_nuggets(row: Mapping) -> list[dict]:
    return [n for n in row.get("nuggets", []) if n.get("importance") == "vital"]


# ─── 検査 ───────────────────────────────────────────────────────
def validate_nuggets(rows: list[dict], corpus: Mapping[str, str]) -> list[str]:
    """問題の一覧を返す。空なら合格。"""
    errors: list[str] = []
    seen_q: set[str] = set()
    seen_n: set[str] = set()
    for idx, row in enumerate(rows, start=1):
        qid = row.get("qid")
        where = f"行{idx}({qid})"
        if not isinstance(qid, str) or not qid:
            errors.append(f"行{idx}: qid が無い")
            continue
        if qid in seen_q:
            errors.append(f"{where}: qid が重複")
        seen_q.add(qid)
        rule_id = row.get("rule_id")
        if not isinstance(rule_id, str) or not rule_id:
            errors.append(f"{where}: rule_id が無い")
            body = None
        elif rule_id not in corpus:
            errors.append(f"{where}: 規程 {rule_id} が corpus に無い")
            body = None
        else:
            body = corpus[rule_id]
        nuggets = row.get("nuggets")
        if not isinstance(nuggets, list) or not nuggets:
            errors.append(f"{where}: nuggets が空")
            continue
        n_vital = 0
        for jdx, n in enumerate(nuggets, start=1):
            if not isinstance(n, dict):
                errors.append(f"{where}: nuggets[{jdx}] がオブジェクトでない")
                continue
            nid = n.get("nid")
            nw = f"{where}/{nid}"
            if not isinstance(nid, str) or not nid:
                errors.append(f"{where}: nuggets[{jdx}] の nid が無い")
            elif nid in seen_n:
                errors.append(f"{nw}: nid が重複")
            else:
                seen_n.add(nid)
            if not isinstance(n.get("text"), str) or not n.get("text", "").strip():
                errors.append(f"{nw}: text が空")
            imp = n.get("importance")
            if imp not in IMPORTANCES:
                errors.append(f"{nw}: importance は vital/okay のどちらか (値: {imp!r})")
            elif imp == "vital":
                n_vital += 1
            kws = n.get("keywords", [])
            if not isinstance(kws, list) or not all(isinstance(k, str) for k in kws):
                errors.append(f"{nw}: keywords は文字列の配列")
            quote = n.get("quote")
            if not isinstance(quote, str) or not quote.strip():
                errors.append(f"{nw}: quote が空")
            elif body is not None and not quote_in_text(quote, body):
                errors.append(f"{nw}: quote が規程 {rule_id} の本文に逐語で無い")
        if n_vital < 1:
            errors.append(f"{where}: vital が 1 つも無い")
    return errors


def validate_file(
    nuggets_path: str | Path, corpus: Mapping[str, str] | None = None
) -> tuple[list[dict], list[str]]:
    rows = load_nuggets(nuggets_path)
    if corpus is None:
        corpus = load_corpus(default_corpus_dir(nuggets_path))
    return rows, validate_nuggets(rows, corpus)


def unreviewed_qids(rows: list[dict]) -> list[str]:
    return [r.get("qid", "?") for r in rows if not str(r.get("reviewed_by") or "").strip()]


# ─── freeze ─────────────────────────────────────────────────────
def sha256_lf(path: str | Path) -> str:
    """改行を LF にそろえた sha256。"""
    data = Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def freeze_path_for(nuggets_path: str | Path) -> Path:
    return Path(nuggets_path).resolve().parent / FREEZE_NAME


def _row_version(rows: list[dict]) -> str:
    versions = {str(r.get("version") or "") for r in rows}
    if len(versions) != 1 or "" in versions:
        raise FreezeError(f"nuggets の version が行ごとに違う、または空: {sorted(versions)}")
    return versions.pop()


def write_freeze(
    nuggets_path: str | Path,
    version: str | None = None,
    freeze_path: str | Path | None = None,
    now: _dt.datetime | None = None,
) -> dict:
    """sha256・時刻・版を freeze.json に書く。同じ版で中身が変わっていたら拒否する。"""
    rows = load_nuggets(nuggets_path)
    version = version or _row_version(rows)
    fp = Path(freeze_path) if freeze_path else freeze_path_for(nuggets_path)
    sha = sha256_lf(nuggets_path)
    if fp.exists():
        old = json.loads(fp.read_text(encoding="utf-8"))
        if old.get("sha256") != sha and old.get("version") == version:
            raise FreezeError(
                f"版 {version} は固定済みで中身が変わっている。直したなら版を上げる（前の版の点と混ぜない）"
            )
    now = now or _dt.datetime.now(_dt.timezone.utc)
    info = {
        "sha256": sha,
        "frozen_at": now.astimezone(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "version": version,
        "n_questions": len(rows),
    }
    fp.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return info


def check_freeze(nuggets_path: str | Path, freeze_path: str | Path | None = None) -> dict:
    """freeze と合っていれば freeze の中身を返す。合わなければ FreezeError。"""
    fp = Path(freeze_path) if freeze_path else freeze_path_for(nuggets_path)
    if not fp.exists():
        raise FreezeError(f"freeze が無い: {fp}（回答を見る前に freeze する）")
    info = json.loads(fp.read_text(encoding="utf-8"))
    sha = sha256_lf(nuggets_path)
    if info.get("sha256") != sha:
        raise FreezeError(
            f"nuggets.jsonl が freeze と食い違う（freeze {str(info.get('sha256'))[:12]}… / 現在 {sha[:12]}…）。"
            "直したなら版を上げて freeze し直す"
        )
    return info


# ─── 答えの読み込み ─────────────────────────────────────────────
def load_answers(path: str | Path) -> list[dict]:
    """{"id"|"qid", "text"|"answer", 任意 "run"} → [{"qid","text","run"}]。

    ファイル 1 つ = 1 回分。run が無ければファイル名（拡張子なし）。
    PR #32 の runs/<名前>.jsonl ({"id","text",...}) をそのまま読める。
    """
    p = Path(path)
    out: list[dict] = []
    seen: set[str] = set()
    for idx, row in enumerate(_read_jsonl(p), start=1):
        qid = row.get("qid", row.get("id"))
        text = row.get("text", row.get("answer"))
        if qid is None or text is None:
            raise SchemaError(f"{p}:{idx}: id(qid) と text(answer) が要る")
        qid = str(qid)
        if qid in seen:
            raise SchemaError(f"{p}: qid {qid} が同じファイルに 2 回ある（1 ファイル = 1 回分）")
        seen.add(qid)
        out.append({"qid": qid, "text": str(text), "run": str(row.get("run") or p.stem)})
    return out
