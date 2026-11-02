"""手がかり語と文字種の検査（Issue #17）。送信時は依頼文だけ、登録時は承認済み本文に当てる。

手がかり語は契約 B の「禁止する情報」（docs/v2-contract-examples.md: 個人名・社員番号、取引先名、
金額・日付の実データ、社内のテーブル名、社内コード）と、未公開の M&A、指示の注入、送信審査への働きかけ、
区分の見出しの偽装、契約の制約を外す依頼、承認済み本文の ID・ハッシュの書き込みから作る。
金額・日付は「〜円を超える」「〜以降」のような処理条件（契約 B が送ってよいとする閾値）には当てない。

各語は正規化した写し（judge.normalize.normalize_text）に当て、compact=True の語は空白を除いた写しにも当てる
（「買 収」「M & A」）。行頭や区切りに意味がある語（見出し・テーブル名・メール・ID）は空白を除くと
誤って当たるので、正規化した写しだけに当てる。

文字種: 正規化した後に、かな・漢字・ASCII のラテン文字以外の文字（キリル文字の「М」、結合文字で作った「Ḿ」
など、見た目が同じ・近い別の字）があれば unexpected_letters が返す。送信時は保留、登録時は当たりとして扱う。
"""
from __future__ import annotations

import re
import unicodedata

from judge.base import question_version
from judge.normalize import NORMALIZE_VERSION, compact_text, normalize_text

_NOT_COND_AMOUNT = r"(?!\s*(を?超|以上|以下|未満|より(大き|小さ|多|少|高|安)))"
_NOT_COND_DATE = r"(?!\s*日?\s*(以降|以前|から|まで|より(前|後)))"
# ASCII の語の境目（日本語の直後でも当たるように、\w ではなく ASCII だけで見る）
_A = r"(?<![A-Za-z0-9_])"
_Z = r"(?![A-Za-z0-9_])"
_MONTHS = _A + r"(?i:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"

# (名前, 正規表現, 空白を除いた写しにも当てるか)
REQUEST_CUES: list[tuple[str, re.Pattern[str], bool]] = [
    ("employee_id", re.compile(_A + r"([Ee]-?\d{3,}|(?i:emp)-?\d{2,})|社員(番号|ID|コード|No\.?)\s*[:=は]?\s*([A-Za-z]{0,3}-?\d+|[〇一二三四五六七八九]+)"), True),
    ("person_name", re.compile(r"[一-龥]{1,4}(さん|くん|氏|(?<![仕同模多各異一皆])様)"), True),
    ("counterparty", re.compile(r"([A-Z]社|株式会社|有限会社|合同会社|\(株\)|\(有\)|[一-龥ァ-ヶー]{1,8}(商事|物産|製作所|銀行|証券)|"
                                + _A + r"(Inc\.|Corp\.|Co\.,?\s*Ltd\.?|LLC)" + _Z + ")"), True),
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), True),
    ("phone", re.compile(r"(内線\s*\d{3,}|(?<!\d)\d{2,4}-\d{2,4}-\d{3,4}(?!\d)|(?<![\d.,])0\d{9,10}(?![\d.,])|"
                       r"(?<!\d)0\d{0,4}\(\d{1,4}\)-?\d{3,4}(?!\d)|\(0\d{0,4}\)-?\d{1,4}-?\d{3,4}(?!\d))"), True),
    ("amount", re.compile(r"((?<![\d,.])\d[\d,.]*\s*(億|万|千)?\s*円|[億万]\s*円|[〇一二三四五六七八九十百千]+\s*(万|億)?\s*円|"
                          r"[¥$€£₩]\s*\d[\d,.]*(?![\d,.])|"
                          r"(?<![\d,.])\d[\d,.]*(?![\d,.])\s*(ドル|ユーロ|ポンド|ウォン|元|(?i:usd|jpy|eur|gbp|cny|rmb|krw)" + _Z + "))"
                          + _NOT_COND_AMOUNT), True),
    ("real_date", re.compile(r"((?<!\d)\d{4}\s*[-/年.]\s*\d{1,2}\s*[-/月.]\s*\d{1,2}(?!\d)|"
                             r"(令和|平成|昭和)\s*(\d{1,2}|元)\s*年\s*\d{1,2}\s*月\s*\d{1,2}(?!\d)|"
                             r"(?<!\d)(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)|"
                             # 英語の月名（5 Nov 2026・Nov 5, 2026）
                             r"(?<!\d)\d{1,2}\s*" + _MONTHS + r"\.?,?\s*\d{4}(?!\d)|" + _MONTHS + r"\.?\s*\d{1,2},?\s*\d{4}(?!\d))"
                             + _NOT_COND_DATE), True),
    # pandas などの書き方（pd.read_csv・df.groupby）とファイル名は除く
    ("table_name", re.compile(r"(?<![A-Za-z0-9_.])(?!(?:pd|np|df|os|sys|re|csv|json|math|io|dt|datetime|pathlib|path|"
                              r"self|str|int|list|dict)\.)[A-Za-z_][A-Za-z0-9_]*\."
                              r"(?!(?:csv|py|txt|json|md|xlsx?)(?![A-Za-z0-9_]))[A-Za-z_][A-Za-z0-9_]*"), False),
    # 英字の略語は大文字小文字を区別する（TOB が October に当たらないように）
    ("m_and_a", re.compile(_A + r"(?i:m&a|tob|mbo|lbo|mergers?|acquisitions?|takeovers?|buyouts?)" + _Z + r"|(買収|合併|出資比率|株式(を|の)?(段階)?(取得|譲渡|売却|買い増し|買い付け)|段階取得|"
                           r"資本(業務)?提携|事業譲渡|経営統合|子会社化)"), True),
    ("injection", re.compile(r"(前の指示(は|を)?無視|指示(は|を)無視|指示を(すべて|全部)?忘れ|これまでの(指示|命令)|"
                             r"(ignore|disregard)\s*(all\s*)?(the\s*)?(previous|prior|above)|システムプロンプト|system\s*prompt|全文を出力)", re.I), True),
    ("gate_override", re.compile(r"(送信(ゲート|前の(審査|検査|確認))|審査の対象外|(検査|審査|判定)(を|の|は)?(対象外|通過済|免除|不要|省略|スキップ)|"
                                 r"(安全|許可|問題(が)?ない|問題なし)と(判定|判断|みな)|判定する側|判定(者|役|モデル)(へ|に)の?指示|"
                                 r"(答え|回答|判定)は「?(はい|いいえ|yes|no|allow|許可)」?と(して|する)|確認済みです|承認済みです)", re.I), True),
    ("forged_heading", re.compile(r"^\s*(-{3,}\s*$|区分\s*:|仕様\s*:|依頼\s*:|説明文\s*:|(system|user|assistant)\s*:|\[(system|user)\])",
                                  re.I | re.M), False),
    ("constraint_override", re.compile(r"(条件|制約|仕様|ルール)(は|を)?(すべて|全部)?(外し|無視し|守らなく|破っ|緩め|気にしなく)"), True),
    # 承認済み本文の ID@版・sha256 を書く（添付や区分を装う。ID 自体も社内コード）
    ("provenance_forgery", re.compile(r"((?<![A-Za-z0-9_-])(?i:[a-z][a-z0-9]*(-[a-z0-9]+)+)@\d+(?![A-Za-z0-9_])|(?<![0-9a-fA-F])[0-9a-fA-F]{64}(?![0-9a-fA-F]))"),
     False),
]

# かな・漢字・ラテン文字と、かなの繰り返し・長音などの記号
_SCRIPT_PREFIXES = ("LATIN ", "HIRAGANA ", "KATAKANA ", "CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH")
_SCRIPT_EXTRA = frozenset("々〆ーゝゞヽヾ")

CUES_VERSION = question_version("request-cues@3", {
    "normalize": NORMALIZE_VERSION,
    "cues": [(n, p.pattern, p.flags, c) for n, p, c in REQUEST_CUES],
    "scripts": {"letters": list(_SCRIPT_PREFIXES), "extra": sorted(_SCRIPT_EXTRA)},
})


def cue_hits(text: str) -> list[str]:
    """text は生の文字列。正規化した写しと（語によっては）空白を除いた写しに当てる。"""
    n = normalize_text(text)
    c = compact_text(n)
    return [name for name, pat, compact in REQUEST_CUES if pat.search(n) or (compact and pat.search(c))]


def cue_spans(text: str) -> tuple[list[tuple[str, int, int, str]], list[str]]:
    """登録時の承認（本文の箇所）に使う。正規化した写しの上の当たり (名前, 始まり, 終わり, 文字列) と、
    空白を除いた写しだけで当たった語の名前（位置を本文に戻せないので、承認の対象にしない）。"""
    n = normalize_text(text)
    c = compact_text(n)
    spans = [(name, m.start(), m.end(), m.group(0)) for name, pat, _ in REQUEST_CUES for m in pat.finditer(n)]
    found = {s[0] for s in spans}
    compact_only = [name for name, pat, compact in REQUEST_CUES if compact and name not in found and pat.search(c)]
    return spans, compact_only


def unexpected_letters(text: str) -> list[str]:
    """正規化した後に残る、かな・漢字・ラテン文字以外の文字（U+XXXX の形）。"""
    n = normalize_text(text)
    bad = {ch for ch in n if unicodedata.category(ch).startswith("L") and ch not in _SCRIPT_EXTRA
           and (not unicodedata.name(ch, "").startswith(_SCRIPT_PREFIXES) or (ch > "\x7f" and "LATIN" in unicodedata.name(ch, "")))}
    return sorted(f"U+{ord(ch):04X}" for ch in bad)
