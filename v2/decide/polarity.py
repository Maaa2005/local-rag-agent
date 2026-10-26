"""質問の中の「手がかりの語」が、満たされている／満たされていない／分からない のどれに読めるか（計画 §3・§7.9 (c)）。

語を一覧に足して「含まれていれば否定」と読むのではなく、次の順で読む:
 1. 数の比べ方の言い回し（「14日未満」「3万円を超えない」）は parse の表で先に除く（否定ではなく範囲の言葉）
 2. 否定を含む決まり文句（やむを得ない・少なくとも…）と、手がかりの語そのものは見ない
 3. 分からない・不明・確認中・二重否定・「〜しなければならない」は「分からない」の印にする
 4. 「問題ない」「支障ない」「構わない」のように否定が良い意味になる言い回しは、否定として数えない
 5. 述語の否定（ない・なく・なかった・ません・ず・ぬ・無い・なし…）と、否定の意味を持つ語（未提出・不備・紛失・忘れ…）を拾い、
    手がかりの語に付く物だけで決める。同じ節に別の語があって否定がそちらに付く時は、手がかりの語の側は否定と読まない
 6. 迷う形は「分からない」に倒す（読めない書き方を黙って「満たしている」にしない）。否定の印が何も無い節は「満たしている」

kind: REQUIREMENT（前提。「領収書は添付済み」＝満たしている）と EVENT（例外の事情。「災害による遅延」＝当たる）。
EVENT では「紛失・忘れ・未・不…」のような否定の意味を持つ語は、事情を述べる文の中身であって事情の否定ではないので数えない。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Sequence

from . import parse

MET = "met"          # 満たしている（EVENT では「当たる」）
UNMET = "unmet"      # 満たしていない（EVENT では「当たらない」）
UNKNOWN = "unknown"  # 分からない

REQUIREMENT = "requirement"
EVENT = "event"

LABEL = {
    REQUIREMENT: {MET: "満たしている", UNMET: "満たしていない", UNKNOWN: "分からない"},
    EVENT: {MET: "当たる", UNMET: "当たらない", UNKNOWN: "当たるか分からない"},
}

_NUM_COMMA = re.compile(r"(?<=\d)[,，](?=\d{3})")
_MASK = "□"  # 読み取りの対象から外した範囲
_CUE = "◆"   # 手がかりの語そのもの


def clause_bounds(text: str, pos: int) -> tuple[int, int]:
    """pos を含む節の [始まり, 終わり)。区切りは parse.CLAUSE_BREAKS（98,000 の桁区切りは区切りではない）。"""
    text = _NUM_COMMA.sub("0", text)
    a = max((text.rfind(c, 0, pos) for c in parse.CLAUSE_BREAKS), default=-1) + 1
    ends = [i for i in (text.find(c, pos) for c in parse.CLAUSE_BREAKS) if i != -1]
    return a, (min(ends) if ends else len(text))


@dataclass(frozen=True)
class Reading:
    status: str   # MET / UNMET / UNKNOWN
    cue: str      # 読んだ手がかりの語
    clause: str   # 読んだ節
    why: str      # どう読んだか（短い説明）


# ---------------------------------------------------------------- 否定の形

_NEG_SRC = "|".join([
    r"ませんでした", r"ません",
    r"なかっ", r"なかろ", r"なければ", r"なけりゃ", r"なきゃ", r"なくて", r"なく",
    r"(?<!あん)(?<!しゃ)(?<![少危切])ない(?!しょ|ぶ|かく|てい|せん|ぞう|がしろ)",
    r"無かっ", r"無けれ", r"無い", r"無く", r"無し", r"なし(?!とげ)",
    r"(?<![ま必きみすいあ])ず(?![っれつらいぶさるんばかこ])",
    r"(?<=[あかがさざたなばまらわせ])ぬ(?![いけ])",
])
_NEG = re.compile(_NEG_SRC)
_NEG_Q = re.compile(r"(?:ん?です|でしょう|だろう)?かな?(?![らどもぎ])")  # 「…ませんか」「…ないでしょうか」は問い。述べた事ではない

# 否定を含む決まり文句。否定ではない。
_IDIOM = re.compile("|".join([
    r"[やヤ止已]む(?:を|お)(?:得|え)(?:な(?:い|かっ|く|けれ)|ず)",
    r"余儀な(?:い|く|かっ)", r"(?:仕方|しかた)(?:が)?な(?:い|く|かっ)", r"致し方(?:が)?な(?:い|く|かっ)",
    r"やむな(?:い|く|し)", r"申し訳な(?:い|く)", r"無理もな(?:い|く)",
    r"欠か(?:せ|さ)(?:な(?:い|く|かっ)|ず)", r"少な(?:い|く|かっ)", r"危な(?:い|く|かっ)", r"切な(?:い|く)",
    r"に(?:違い|相違|ちがい)な(?:い|かっ)", r"に(?:過|す)ぎな(?:い|かっ)", r"にほかな(?:ら|り)(?:な(?:い|かっ|く)|ず)",
    r"ざるを(?:得|え)", r"思わず", r"構わず", r"かまわず",
]))

_UNKNOWN_MARK = re.compile("|".join([
    r"[分わ判]か(?:らな(?:い|かっ|く|けれ)|らず|らぬ|らん|りません(?:でした)?|りかね|りづら"
    r"|って(?:い)?(?:な(?:い|かっ|く)|ません(?:でした)?)|んな(?:い|かっ|く))",
    r"知らな(?:い|かっ|く)", r"知りません(?:でした)?", r"存じません", r"存じ上げません",
    r"覚えて(?:い)?(?:な(?:い|かっ|く)|ません)", r"覚えがな(?:い|く)", r"覚えがありません",
    r"不明", r"不詳", r"不確か", r"不確定", r"不透明", r"未定", r"未詳", r"未確認", r"未確定", r"未判明",
    r"確認中", r"確認待ち", r"調査中", r"検討中", r"審査中", r"申請中", r"処理中", r"手続き中",
    r"確認でき(?:て(?:い)?)?(?:な(?:い|かっ|く)|ません)", r"確認が取れて(?:い)?(?:な(?:い|かっ|く)|ません)",
    r"有無", r"かどうか", r"か否か", r"かもし(?:れ|知れ)(?:な(?:い|かっ)|ません)",
    r"定かでな(?:い|かっ|く)", r"はっきりしな(?:い|かっ|く)", r"はっきりせず", r"判然とし(?:な(?:い|かっ|く)|ず)",
]))

# 二重否定。「ないわけではない」は「ある」とも「ない」とも言い切っていない
_NEGW = r"(?:ない|なく|ません|無い|無く)"
_DOUBLE_NEG = re.compile("|".join([
    _NEGW + r"(?:わけ|訳|こと|はず|もの|の)(?:では|じゃ|も|は)(?:ない|なく|なかっ|ありません|無い)",
    _NEGW + r"(?:は|も)(?:ない|なく|なかっ|ありません|無い)",
    _NEGW + r"(?:と|とは|とも)(?:言え|いえ|言い切れ|言いきれ|限ら|思え|断言でき|断定でき)(?:ない|ません|なかっ)",
    r"否定(?:でき|しき|し切れ|しきれ)(?:な(?:い|かっ)|ません|ず)",
]))

# 「〜しなければならない」「〜しなくてもよい」: 質問が前提の中身を問うているだけで、満たしているかどうかは述べていない
_OBLIGATION = re.compile("|".join([
    r"な(?:ければ|けりゃ|くては|くちゃ|きゃ)(?:ならな(?:い|かっ|く)|なりません(?:でした)?|いけな(?:い|かっ|く)|いけません|だめ|ダメ)",
    r"ねばならな(?:い|かっ)", r"ねばなりません",
    r"なくても(?:いい|よい|良い|大丈夫|(?:かまわ|構わ|かまい|構い|問題|差し支え|差支え)(?:な(?:い|く)|ません|ありません))",
]))

# 否定が「良い」を意味する言い回し
_GOOD_NOUN = (r"(?:問題|支障|差し?支え|心配|間違い|間違え|誤り|懸念|疑い|異常|ミス|漏れ|不備|不足|欠け|欠落|落ち度|違反|不正|遅れ"
              r"|不明(?:な)?点|不安|障害|トラブル)")
_NEG_END = r"(?:な(?:い|く|かっ)|なし|ありません(?:でした)?|ございません|無い|無く|無し|無かっ)"
_GOOD = re.compile("|".join([
    _GOOD_NOUN + r"(?:は|も|が)?(?:特に|全く|まったく|一切|何も)?" + _NEG_END,
    r"(?:構わな(?:い|く|かっ)|かまわな(?:い|く|かっ)|構いません|かまいません|差し支えありません|差支えありません)",
]))

# 述語の否定ではないが、当たらない・無い事を言う語
_NEG_WORD = re.compile(r"非該当|非対象|対象外|適用外|範囲外|該当外")

# 否定の意味を持つ語（REQUIREMENT でだけ数える）
_BAD = r"(?:紛失|遺失|失念|忘れ|(?:失|無|な)くし|失っ|失い|欠(?:く|け|い|如|落|損)|漏れ|漏らし|抜け(?!る)|抜かし|落とし)"
_BAD_MERGE = re.compile(_BAD + r"[ぁ-んー]{0,6}?(?:" + _NEG_SRC + r")")  # 「紛失していない」「忘れていません」は良い側
_BAD_ONLY = re.compile(_BAD)

_PREFIX = re.compile(r"[未不無][一-龥ァ-ヶー]{1,3}")
_PREFIX_NEUTRAL = ("未来", "未成年", "未曾有", "未婚", "未開", "不動産", "不思議", "不意", "不況", "不特定", "不定期", "不景気", "不可欠",
                   "不断", "無料", "無償", "無事", "無数", "無限", "無制限", "無地", "無人", "無休")
_PREFIX_UNKNOWN = ("未決", "未知", "不要", "不必要")
_PREFIX_BAD_MU = ("無記入", "無記載", "無署名", "無効", "無添付", "無申請", "無承認", "無届", "無提出", "無押印", "無捺印", "無資格",
                  "無許可", "無連絡", "無通知", "無報告", "無入力", "無回答")

_STILL = re.compile(r"まだ|未だ|いまだ|これから|追って|のちほど|後日|予定(?:です|でして|でした|だ|ですが)")

# 節の中の切れ目（読点が無くても、ここで別の述語になる）
_CONNECTOR = re.compile(r"(?<=[だたるいすんうくつぐぬむぶ])が|けれども|けれど|けど|ものの|のに|ので|一方|しかし|ただし")
_RENYO = ("ず", "ずに", "なく", "なくて", "ないで")  # 「…せず、〇〇は…」の連用形。ここでも切る

_HIRA = re.compile(r"[ぁ-ん]")


# ---------------------------------------------------------------- 1 つの手がかりの読み取り

@dataclass(frozen=True)
class _Ev:
    a: int
    b: int
    kind: str   # N 否定 / B 否定の意味の語 / U 分からない / M 否定+否定の意味の語（良い側）/ S 「まだ」など
    token: str


def _take(w: list[str], evs: list[_Ev], a: int, b: int, kind: str | None, mode: str) -> None:
    tok = "".join(w[a:b])
    for i in range(a, b):
        w[i] = _MASK
    if kind is None:
        return
    if mode == EVENT and kind in ("B", "M"):
        return  # 事情の文の中身。事情の否定ではない
    evs.append(_Ev(a, b, kind, tok))


def _scan(w: list[str], rx: re.Pattern[str], kind: str | None, evs: list[_Ev], mode: str) -> None:
    for m in rx.finditer("".join(w)):
        if m.end() > m.start():
            _take(w, evs, m.start(), m.end(), kind, mode)


def _scan_negations(w: list[str], evs: list[_Ev], mode: str) -> None:
    t = "".join(w)
    for m in _NEG.finditer(t):
        if m.end() == m.start():
            continue
        q = _NEG_Q.match(t, m.end())
        if q is not None and q.end() > q.start():
            _take(w, evs, m.start(), q.end(), "U", mode)
        else:
            _take(w, evs, m.start(), m.end(), "N", mode)


def _scan_prefix(w: list[str], evs: list[_Ev], mode: str) -> None:
    t = "".join(w)
    for m in _PREFIX.finditer(t):
        word = m.group(0)
        end = m.end()
        kind: str | None
        for k in _PREFIX_NEUTRAL:
            if word.startswith(k):
                _take(w, evs, m.start(), m.start() + len(k), None, mode)
                break
        else:
            for k in _PREFIX_UNKNOWN:
                if word.startswith(k):
                    _take(w, evs, m.start(), m.start() + len(k), "U", mode)
                    break
            else:
                if word[0] == "無" and not any(word.startswith(k) for k in _PREFIX_BAD_MU):
                    _take(w, evs, m.start(), end, "U", mode)  # 読み方が決まらない「無」の語は分からない側へ
                    continue
                for k in _PREFIX_BAD_MU:
                    if word.startswith(k):
                        end = m.start() + len(k)
                        break
                merge = re.compile(r"[ぁ-んー]{0,6}?(?:" + _NEG_SRC + r")").match("".join(w), end)
                if merge is not None:
                    _take(w, evs, m.start(), merge.end(), "M", mode)
                else:
                    _take(w, evs, m.start(), end, "B", mode)


def _topic_particles(w: list[str], lo: int, hi: int) -> list[int]:
    out = []
    for p in range(max(lo, 1), hi):
        if w[p] in "はがも" and (not _HIRA.match(w[p - 1]) or w[p - 1] in "でにと"):
            out.append(p)
    return out


def _compose(evs: list[_Ev]) -> tuple[str, str]:
    kinds = [e.kind for e in evs]
    n, b = kinds.count("N"), kinds.count("B")
    toks = "・".join(f"「{e.token}」" for e in evs)
    if "U" in kinds:
        return UNKNOWN, f"分からないと読める言い方（{toks}）"
    if n > 1:
        return UNKNOWN, f"否定が 2 つ以上ある（{toks}）"
    if n == 1 and b:
        return UNKNOWN, f"否定と否定の意味を持つ語が重なる（{toks}）"
    if n == 1 or b:
        return UNMET, f"否定（{toks}）が付く"
    if "S" in kinds:
        return UNKNOWN, f"まだ・これから等の言い方（{toks}）"
    return MET, "否定の印が付かない" if not evs else f"良い意味の否定・否定の相手が別の語（{toks}）"


# 述語が無いまま終わる区切り。述語は次の区切りにある（「領収書は、ありません」「領収書については、残念ながら、ありません」）
_OPEN_END = re.compile(
    r"(?:について|に関して|に関し|につき|としては|においては|に対しては|ながら|ては|には|では|とは|からは|までは|でも|とも)$"
    r"|(?:(?<![ぁ-ん])|(?<=[でにと]))[はがもをにでとのへ]$")
_ADVERB_ONLY = re.compile(r"(?:残念ながら|あいにく|実は|正直|特に|もちろん|ちなみに|一応|念のため|なお|また|そして|それと|ただ|現在|今のところ|いまのところ)+")

# 事情（EVENT）の否定が事情そのものに付く時の、事情の語と否定の間。ここに別の語（原因の「で」・別の述語）が挟まる否定は、事情の否定ではない
_EVENT_GAP = re.compile(
    r"□*(?:という|との|の|による|によるもの|に当たる|にあたる|に該当する)?"
    r"(?:事情|事由|理由|原因|状況|遅延|もの|こと|事|ケース|場合|件|事故|遅れ)?"
    r"(?:は|が|も|を|では|には|とは|とも|じゃ|で)?"
    r"(?:特に|全く|まったく|一切|何も|別に|あまり|特別)?"
    r"(?:関係|該当|当てはま|あてはま|影響)?(?:あり|ござい)?"
    r"(?:(?:理由|原因)(?:に|と)(?:は|も)?)?(?:し(?:て(?:い)?)?)?")


def _segment_cuts(raw: str) -> list[int]:
    cuts = {0, len(raw)}
    for m in _CONNECTOR.finditer(raw):
        cuts.add(m.end())
    for i, ch in enumerate(raw):
        if ch in "、，," and not (0 < i and raw[i - 1].isdigit() and raw[i + 1:i + 4].isdigit()):
            cuts.add(i + 1)
    return sorted(cuts)


def read_at(text: str, cs: int, ce: int, mode: str = REQUIREMENT,
            bounds: Callable[[str, int], tuple[int, int]] = clause_bounds,
            ranges: Sequence[tuple[int, int]] | None = None) -> Reading:
    """text[cs:ce] の手がかりの語を含む節を読む。text は parse.clean_question 済み（正規化済み）の質問。"""
    if ranges is None:
        ranges = parse.range_spans(text)
    a, b = bounds(text, cs)
    b = max(b, ce)
    # 述語が読点の先にある書き方（「領収書を、紛失しました」）の為に、続く節（最大 3 つ）まで読み取りの範囲に入れる。
    # 手がかりに付く否定かどうかは、下で区切り（読点・接続）ごとに決める
    for _ in range(3):
        if b < len(text) and text[b] in "、，,":
            b = max(bounds(text, b + 1)[1], b + 1)
        else:
            break
    raw = text[a:b]
    w = list(raw)
    evs: list[_Ev] = []
    for ra, rb in list(ranges) + [(m.start() + a, m.end() + a) for m in re.finditer("未満", raw)]:
        for i in range(max(ra, a), min(rb, b)):
            w[i - a] = _MASK
    _scan(w, _IDIOM, None, evs, mode)
    for i in range(cs - a, ce - a):
        w[i] = _CUE
    _scan(w, _UNKNOWN_MARK, "U", evs, mode)
    _scan(w, _DOUBLE_NEG, "U", evs, mode)
    _scan(w, _OBLIGATION, "U", evs, mode)
    _scan(w, _GOOD, None, evs, mode)
    _scan(w, _NEG_WORD, "N", evs, mode)
    _scan(w, _BAD_MERGE, "M", evs, mode)
    _scan(w, _BAD_ONLY, "B", evs, mode)
    _scan(w, _STILL, "S", evs, mode)
    _scan_prefix(w, evs, mode)
    _scan_negations(w, evs, mode)

    # 節の中の切れ目（述語が変わる所）
    cuts = _segment_cuts(raw)
    for e in evs:
        if e.kind == "N" and e.token in _RENYO and e.b not in cuts:
            cuts.append(e.b)
    cuts.sort()
    lo = max(c for c in cuts if c <= cs - a)
    hi = min((c for c in cuts if c > cs - a), default=len(raw))
    # 手がかりの区切りが述語を持たずに終わる時は、次の区切りまで手がかりの述語として読む
    while hi < len(raw):
        last = re.split("[、，,]", raw[lo:hi].rstrip("、，, "))[-1]
        if not (_OPEN_END.search(last) or _ADVERB_ONLY.fullmatch(last)):
            break
        hi = min((c for c in cuts if c > hi), default=len(raw))
    pre = _topic_particles(w, lo, cs - a)
    post = _topic_particles(w, ce - a, hi)
    first = post[0] if post else None

    near: list[_Ev] = []
    for e in evs:
        if not (lo <= e.a < hi):
            continue
        if e.b <= cs - a:  # 手がかりより前の否定: 間に別の「Xは」があればそちらの物
            if any(e.b <= p < cs - a for p in pre):
                continue
        elif e.a >= ce - a:  # 後ろの否定: 手がかり自身の「は」の後に別の「Xは」があればそちらの物
            if first is not None and first < e.a:
                others = [p for p in post if first < p < e.a and not (w[first] == "も" and w[p] == "も")]
                if others:
                    continue
        near.append(e)
    if mode == EVENT:
        # 事情の否定は、事情そのものに付く物（「災害ではありません」）だけ。原因の「で」を挟んだ別の述語の否定
        # （「災害の影響で申請できなかった」）は、事情が述べられているのであって否定ではない（保留側）
        near = [e for e in near
                if e.kind != "N" or (e.a >= ce - a and _EVENT_GAP.fullmatch("".join(w[ce - a:e.a])))]
    elif ce - a < len(w) and w[ce - a] in "はも" and not any(e.kind in "NBU" for e in near):
        # 主題の省略: 「領収書は14日未満ですが、ありません」の次の区切りは、自分の主題（Xは・Xが）を持たない否定なら手がかりの述語
        nxt = min((c for c in cuts if c > hi and raw[hi:c].strip("、，, ")), default=len(raw))
        for e in evs:
            if hi <= e.a < nxt and e.kind in ("N", "B") and not _topic_particles(w, hi, e.a):
                near.append(e)
    status, why = _compose(near)
    return Reading(status, raw[cs - a:ce - a], raw[lo:hi].strip("、，, ") or raw, why)


def read_cues(text: str, cues: Sequence[str], mode: str = REQUIREMENT,
              bounds: Callable[[str, int], tuple[int, int]] = clause_bounds) -> Reading | None:
    """text に出てくる cues の全部の出現を読み、まとめる。1 つも出てこなければ None。

    まとめ方: REQUIREMENT は 満たしていない > 分からない > 満たしている（悪い読みを採る）。
    EVENT は 当たる > 分からない > 当たらない（保留に倒れる読みを採る）。
    """
    ranges = parse.range_spans(text)
    found: list[Reading] = []
    for cue in cues:
        k = parse._normalize(cue)
        if not k:
            continue
        j = text.find(k)
        while j != -1:
            found.append(read_at(text, j, j + len(k), mode, bounds, ranges))
            j = text.find(k, j + 1)
    if not found:
        return None
    order = (UNMET, UNKNOWN, MET) if mode == REQUIREMENT else (MET, UNKNOWN, UNMET)
    for st in order:
        for r in found:
            if r.status == st:
                return r
    return found[0]
