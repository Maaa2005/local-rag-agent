"""監視フォルダ ⇔ パスのアクセスレベル判定ロジック (純関数)。

watcher (ファイル監視時) と access_sync (フォルダ変更の波及) の両方から
参照される単一の真実源。判定ロジックがずれると「無効化したのに検索できる」
「昇格したのにチャンクは旧レベルのまま」といった穴が生まれるため、
ここに一本化する。
"""
from __future__ import annotations

from pathlib import Path


def _normalized_path(value: str) -> str:
    """Return an absolute, case-normalized path with platform-neutral separators."""
    return Path(value).resolve(strict=False).as_posix().rstrip("/").casefold()


def match_folder(path: str, folders: list[dict]) -> dict | None:
    """path に最も長く一致する監視フォルダを返す (有効・無効を問わない)。

    一致はパス境界で判定する。`/watched/exec` の設定が `/watched/exec-public/...`
    のような**兄弟ディレクトリ**へ誤って適用されるのを防ぐため、完全一致または
    `フォルダパス.rstrip("/") + "/"` で始まる場合のみ一致とみなす。
    複数一致する場合はパスが最長 (= 最も深い) フォルダを優先する。
    どの監視フォルダにも一致しなければ None を返す。

    folders: {"path", "access_level", "is_active", ...} を持つ dict のリスト。
    is_active の値はこの関数では判定に使わず、呼び出し側が結果の
    is_active を見て「無効フォルダ配下は対象外」等の扱いを決める。
    """
    best: dict | None = None
    best_len = -1
    normalized_path = _normalized_path(path)
    for folder in folders:
        fp = _normalized_path(folder["path"])
        if (
            normalized_path == fp or normalized_path.startswith(fp + "/")
        ) and len(fp) > best_len:
            best_len = len(fp)
            best = folder
    return best
