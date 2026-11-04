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

# 閉じ引用符をはさむ形（「10万円」を超える）も処理条件とみなす
_NOT_COND_AMOUNT = r"(?![」』\"”’')\]]?\s*(を?超|を?上回|を?下回|以上|以下|以内|未満|に満た|より(大き|小さ|多|少|高|安)))"
_NOT_COND_DATE = r"(?![」』\"”’')\]]?\s*日?\s*(以降|以前|から|まで|より(前|後)))"
# ASCII の語の境目（日本語の直後でも当たるように、\w ではなく ASCII だけで見る）
_A = r"(?<![A-Za-z0-9_])"
_Z = r"(?![A-Za-z0-9_])"
_MONTHS = _A + r"(?i:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"

# よくある姓（Issue #17・11/3）。敬称の無い人名（「佐藤の経費」「山田太郎の行」）用。開発用の集合から
# 選んだのではなく、全国でよくある姓の上位から 2 字以上の物を取り、都道府県名と同じ物（山口・石川・千葉・
# 宮崎・福島など）は地域の値と紛れるので除いた。このリポの日本語の文書 2,654 行で当たったのは本物の人名の
# 2 行だけ（11/3 測定）。一覧に無い姓・名だけ・ひらがなの人名は拾えない（残る危険・#17）
_SURNAMES = """
    佐藤 鈴木 高橋 田中 伊藤 渡辺 渡邊 渡邉 山本 中村 小林 加藤 吉田 山田 佐々木 松本 井上 木村 斎藤 斉藤 齋藤 清水 山崎 池田 橋本 阿部 山下 中島 石井 小川 前田 岡田 長谷川
    藤田 後藤 近藤 村上 遠藤 青木 坂本 福田 太田 西村 藤井 金子 岡本 藤原 中野 三浦 原田 中川 松田 竹内 小野 田村 中山 和田 石田 森田 上田 内田 柴田 酒井 横山 高木 安藤
    宮本 大野 小島 谷口 今井 工藤 高田 増田 丸山 杉山 村田 大塚 新井 小山 平野 藤本 河野 上野 野口 武田 松井 岩崎 菅原 木下 久保 佐野 野村 松尾 市川 菊地 杉本 古川 大西
    島田 水野 桜井 高野 渡部 吉川 西田 飯田 菊池 西川 小松 北村 安田 五十嵐 川口 平田 中田 久保田 服部 岩田 土屋 川崎 本田 樋口 秋山 田口 永井 中西 吉村 石原 大橋 松岡 馬場
    浅野 荒木 大久保 野田 小池 松下 浅井 菅野 熊谷 大石 川村 望月 星野 内藤
    髙橋 髙田 髙木 髙野 山﨑 齊藤 嶋田
""".split()

# (名前, 正規表現, 空白を除いた写しにも当てるか)
REQUEST_CUES: list[tuple[str, re.Pattern[str], bool]] = [
    ("employee_id", re.compile(_A + r"([Ee]-?\d{3,}|(?i:emp)-?\d{2,})|社員(番号|ID|コード|No\.?)\s*[:=は]?\s*([A-Za-z]{0,3}-?\d+|[〇一二三四五六七八九]+)"), True),
    ("person_name", re.compile(r"[一-龥]{1,4}((?<![客皆])(さん|くん)|(?<![摂華])氏(?!名)|(?<![仕同模多各異一皆客])様(?![式子々]))"), True),
    ("person_surname", re.compile("(" + "|".join(sorted(_SURNAMES, key=len, reverse=True)) + ")(?!部(?!長)|店(?!長)|洋|馬場|の舞台|[市区町村郡県府都道駅線])"), True),
    ("counterparty", re.compile(r"([A-Z]社|株式会社|有限会社|合同会社|\(株\)|\(有\)|[一-龥ァ-ヶー]{1,8}(商事|物産|製作所|銀行|証券)|"
                                + _A + r"(Inc\.|Corp\.|Co\.,?\s*Ltd\.?|LLC)" + _Z + ")"), True),
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), True),
    ("phone", re.compile(r"(内線\s*\d{3,}|(?<!\d)\d{2,4}-\d{2,4}-\d{3,4}(?!\d)|(?<![\d.,])0\d{9,10}(?![\d.,])|"
                       r"(?<!\d)0\d{0,4}\(\d{1,4}\)-?\d{3,4}(?!\d)|\(0\d{0,4}\)-?\d{1,4}-?\d{3,4}(?!\d))"), True),
    ("amount", re.compile(r"((?<![\d,.])\d[\d,.]*\s*(億|万|千)?\s*円|[億万]\s*円|[〇一二三四五六七八九十百千壱弐参拾]+\s*(万|億)?\s*円|(まん|おく|せん|ひゃく)えん|"
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
    ("table_name", re.compile(r"(?<![A-Za-z0-9_.])(?!(?i:pd|np|df|os|sys|re|csv|json|math|io|dt|datetime|pathlib|path|pandas|numpy|statistics|collections|"
                              r"logging|itertools|functools|e|i|self|str|int|list|dict)\.)[A-Za-z_][A-Za-z0-9_]*\."
                              r"(?!(?i:csv|tsv|py|txt|json|md|xlsx?|g|e)(?![A-Za-z0-9_]))[A-Za-z_][A-Za-z0-9_]*"), False),
    # 「社内の」に英字の名前が続く（社内のテーブル名・列名・システム名。契約 B が禁じる。Codex 11/3）
    ("internal_name", re.compile(r"社内(の|で使う|向けの|システムの)?\s*(?!(?i:python|pandas|csv|tsv|excel|vpn|pc|wi-?fi|windows|mac)(?![A-Za-z0-9_]))[A-Za-z_][A-Za-z0-9_]*"), True),
    # 英字の略語は大文字小文字を問わず、ASCII の境目で見る（TOB が October に当たらないように）
    ("m_and_a", re.compile(_A + r"(?i:m&a|tob|mbo|lbo|mergers?|acquisitions?|takeovers?|buyouts?)" + _Z + r"|(買収|合併|出資比率|株式(を|の)?(段階)?(取得|譲渡|売却|買い増し|買い付け)|段階取得|"
                           r"資本(業務)?提携|事業譲渡|経営統合|子会社化)"), True),
    ("injection", re.compile(r"(前の指示(は|を)?無視|指示(は|を)無視|指示を(すべて|全部)?忘れ|これまでの(指示|命令)|"
                             r"(ignore|disregard)\s*(all\s*)?(the\s*)?(previous|prior|above)|システムプロンプト|system\s*prompt|全文を出力|"
                             r"(従来|以前|これまで|上記|先)の(命令|指示)|(従来|以前|これまで|上記|先)の(ルール|設定)(を|は)?(無視|忘れ|破棄|撤回|取り消|リセット|上書き|解除|無効)|(命令|指示|ルール|制約)(を|は)?(破棄|撤回|取り消|リセット|上書き|解除|無効)|"
                             r"(デバッグ|開発者|管理者|メンテナンス)モード(に|へ)|内部の(設定|指示|プロンプト)(を|も)?(すべて|全部)?(出力|表示|教え|開示))", re.I), True),
    ("gate_override", re.compile(r"(送信(ゲート|前の(審査|検査|確認))|審査の対象外|(検査|審査|判定)(を|の|は)?(対象外|通過済|免除|不要|省略|スキップ)|"
                                 r"(安全|許可|問題(が)?ない|問題なし)と(判定|判断|みな)|判定する側|判定(者|役|モデル)(へ|に)の?指示|"
                                 r"(答え|回答|判定)は「?(はい|いいえ|yes|no|allow|許可)」?と(して|する)|確認済みです|承認済みです|"
                                 r"(判定|審査|検査|承認)(結果)?(は|を)?(必ず|常に|すべて|全部)?\s*(許可|通過|合格|allow|ok)(に|と|で)(して|する)|"
                                 r"(安全|セキュリティ|送信|審査|検査|フィルタ)(チェック|確認|機能|審査|検査)?(を|は)?(無効|停止|解除|オフ|切っ|切る|切った)|"
                                 r"(上長|法務|部長|課長|管理者|担当者?|監査)(が|の|は|に)?\S{0,8}(確認|承認)(済|ずみ|した|して(いる|います|もらっ)|を(得て(いる|います)|得た|もらっ)|もらっ)|問題(は|が)?(ありません|ない)(です)?(?=[。!?\n]|$)|"
                                 r"(審査|判定)(役|担当)|管理者権限|(ルール|規則|決まり|制約)に従わなく|社外秘|極秘|"
                                 r"添付(?!(は|を)?(参考|参照|もと))(は|を)?\S{0,12}として)", re.I), True),
    ("forged_heading", re.compile(r"^\s*(-{3,}\s*$|区分\s*:|仕様\s*:|依頼\s*:|説明文\s*:|(system|user|assistant)\s*:|\[(system|user)\])",
                                  re.I | re.M), False),
    ("constraint_override", re.compile(r"(条件|制約|仕様|ルール)(は|を)?(すべて|全部)?(外し|無視し|守らなく|破っ|緩め|気にしなく)"), True),
    # 承認済み本文の ID@版・sha256 を書く（添付や区分を装う。ID 自体も社内コード）
    ("provenance_forgery", re.compile(r"((?<![A-Za-z0-9_.-])(?i:[a-z][a-z0-9_]*(-[a-z0-9_]+)*)@\d+(?![A-Za-z0-9_.])|(?<![0-9a-fA-F])[0-9a-fA-F]{64}(?![0-9a-fA-F]))"),
     False),
]

# かな・漢字・ラテン文字と、かなの繰り返し・長音などの記号
_SCRIPT_PREFIXES = ("LATIN ", "HIRAGANA ", "KATAKANA ", "CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH")
_SCRIPT_EXTRA = frozenset("々〆ーゝゞヽヾ")

CUES_VERSION = question_version("request-cues@7", {
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
    compact_only = []
    for name, pat, compact in REQUEST_CUES:
        if not compact:
            continue
        # 正規化した写しで同じ種類が当たっていても、空白を除いた写しにしか無い当たりがあれば別の箇所
        seen = {compact_text(m.group(0)) for m in pat.finditer(n)}
        if any(compact_text(m.group(0)) not in seen for m in pat.finditer(c)):
            compact_only.append(name)
    return spans, compact_only


def unexpected_letters(text: str) -> list[str]:
    """正規化した後に残る、かな・漢字・ラテン文字以外の文字（U+XXXX の形）。"""
    n = normalize_text(text)
    bad = {ch for ch in n if unicodedata.category(ch).startswith("L") and ch not in _SCRIPT_EXTRA
           and (not unicodedata.name(ch, "").startswith(_SCRIPT_PREFIXES) or (ch > "\x7f" and "LATIN" in unicodedata.name(ch, "")))}
    return sorted(f"U+{ord(ch):04X}" for ch in bad)
