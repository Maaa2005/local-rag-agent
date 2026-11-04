"""検査の入口の正規化（Issue #17・docs/research/2026-11-02/judge.md の B「正規化 + 辞書」）。

全角・互換文字と見えない文字での言い換え（「Ｍ＆Ａ」「買‍収」「買́収」など）で手がかり語の
検査をすり抜けられないように、検査の前に次をする:
1. 書式文字（Cf: ゼロ幅スペース・ゼロ幅接合子・語結合子・BOM・双方向の制御・ソフトハイフン・タグ文字など）、
   制御文字（Cc。改行は残し、タブは空白にする）、異体字セレクタ、見えない字（ハングル埋め字・点字の空白など）を除く
2. NFKC（濁点などの結合はここで合成される）
3. 合成されずに残った結合文字（Mn・Mc・Me）と、1 で除く文字をもう一度除く（Mc はハングルの傍点 U+302E など。
   「買〮収」のように語を割る。Codex レビュー 11/2）

compact_text は、さらに空白をすべて除いた写し（「買 収」「M & A」のように語を空白で割る言い換え用）。
正規化するのは検査に使う写しだけで、送信する本文は変えない（送る内容は組み立て時に確定している）。
"""
from __future__ import annotations

import unicodedata

from judge.base import question_version

# 見た目が空白か何も無いのに、空白（Zs）にも書式文字にも入らない字
_INVISIBLE_EXTRA = frozenset([0x115F, 0x1160, 0x3164, 0xFFA0, 0x2800, 0x17B4, 0x17B5])
_DROP_BEFORE = ("Cf", "Cc")
_DROP_AFTER = ("Cf", "Cc", "Mn", "Mc", "Me")

NORMALIZE_SPEC = {
    "steps": ["tab->space", "drop Cf/Cc/VS/extra (keep LF)", "NFKC", "drop Cf/Cc/Mn/Mc/Me/VS/extra (keep LF)"],
    "extra": sorted(_INVISIBLE_EXTRA), "variation_selectors": ["FE00-FE0F", "E0100-E01EF"],
    "compact": "normalized minus str.isspace",
}
NORMALIZE_VERSION = question_version("normalize@3", NORMALIZE_SPEC)


def _drop(ch: str, cats: tuple[str, ...]) -> bool:
    if ch == "\n":
        return False
    o = ord(ch)
    return (unicodedata.category(ch) in cats or o in _INVISIBLE_EXTRA
            or 0xFE00 <= o <= 0xFE0F or 0xE0100 <= o <= 0xE01EF)


def normalize_text(text: str) -> str:
    text = text.replace("\t", " ")
    text = "".join(ch for ch in text if not _drop(ch, _DROP_BEFORE))
    text = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in text if not _drop(ch, _DROP_AFTER))


def compact_text(normalized: str) -> str:
    return "".join(ch for ch in normalized if not ch.isspace())
