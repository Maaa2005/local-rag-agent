"""検査の入口の正規化（Issue #17・docs/research/2026-11-02/judge.md の B「正規化 + 辞書」）。

全角・互換文字とゼロ幅文字での言い換え（「Ｍ＆Ａ」「買‍収」など）で手がかり語の検査を
すり抜けられないように、検査の前に NFKC をかけ、目に見えない文字を除く。
除くのは Unicode の書式文字（Cf: ゼロ幅スペース・ゼロ幅接合子・語結合子・BOM・双方向の制御・
ソフトハイフン・タグ文字など）と異体字セレクタ。改行やタブ（Cc）は残す。

正規化するのは検査に使う写しだけで、送信する本文は変えない（送る内容は組み立て時に確定している）。
"""
from __future__ import annotations

import unicodedata


def _invisible(ch: str) -> bool:
    o = ord(ch)
    return unicodedata.category(ch) == "Cf" or 0xFE00 <= o <= 0xFE0F or 0xE0100 <= o <= 0xE01EF


def normalize_text(text: str) -> str:
    # NFKC の前に除く（ゼロ幅文字が合成を妨げる場合があるため）。NFKC の後にもう一度除く
    text = "".join(ch for ch in text if not _invisible(ch))
    text = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in text if not _invisible(ch))
