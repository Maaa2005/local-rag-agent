"""学習用・開発用の問を作る（life #45・docs/verified-distill-plan.md §2）。

1 問は「本番と同じ形の入力（指示文＋【参考情報】＋【質問】）」と「facts からコードが出した正解」の組。
正解は facts から 3 値で出す（evaluation/teacher_check/rules.py とは別の実装。境目の定義は spec.holds）。
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from fractions import Fraction

from evaluation.distill.gen.families import DEV_FAMILIES, FAMILIES, TRAIN_FAMILIES
from evaluation.distill.gen.spec import WORDS, holds

UNDECIDED = "判断できない"
# backend/app/services/llm.py の SYSTEM_PROMPT と同じ（試験で一致を確かめる）
SYSTEM_PROMPT = """あなたは社内情報専門のアシスタントです。
以下の【参考情報】のみを根拠として質問に回答してください。
参考情報に含まれていない情報については「その情報は現在の資料には含まれていません」と回答してください。
回答は必ず日本語で行ってください。
回答中で根拠にした参考情報には、該当箇所の文末に [1] のようにその番号を付けて引用してください。"""

PREDICATE = {"以上": "{t}以上である", "以下": "{t}以下である", "未満": "{t}未満である", "超える": "{t}を超える",
             "以内": "{t}以内である", "を超えない": "{t}を超えない", "を下回る": "{t}を下回る", "に満たない": "{t}に満たない"}
# 同じ値で結論が変わる対の語（最小の差の対）
COUNTERPART = {"以上": "超える", "超える": "以上", "以下": "未満", "未満": "以下", "以内": "未満",
               "を超えない": "未満", "を下回る": "以下", "に満たない": "以下"}

# 問の書き方（型によらない包み）。0〜5 は学習、6〜7 は開発
WRAPS = ("{facts}{ask}", "社員です。{facts}この場合、{ask}", "確認させてください。{facts}{ask}",
         "{facts}規程に照らすと、{ask}", "今度の件で質問です。{facts}{ask}", "{facts}念のため教えてください。{ask}",
         "部下から相談を受けました。{facts}{ask}", "{facts}私の場合、{ask}")
SPLIT_WRAPS = {"train": (0, 1, 2, 3, 4, 5), "dev": (6, 7)}
SPLIT_FAMILIES = {"train": TRAIN_FAMILIES, "dev": DEV_FAMILIES}

REFUSAL_ASKS = ("この制度を昨年使った社員は何人でしたか。", "この規程は何年に作られましたか。",
                "この制度の担当者の内線番号を教えてください。", "この規程はいつ改正される予定ですか。")
DESC = {
    "F01": ("制服はいつ返さなければなりませんか。", ["退職"]), "F02": ("駐車場の利用料はいくらですか。", ["3,500円"]),
    "F03": ("名刺の文字の大きさや配置はどう決めますか。", ["会社"]), "F04": ("ポイントの有効期限はいつまでですか。", ["2年"]),
    "F05": ("再検査の費用は誰が負担しますか。", ["会社"]), "F06": ("印刷はどのような形を基本としますか。", ["両面"]),
    "F07": ("補助を受けたサークルは年度末に何を出しますか。", ["報告書"]), "F08": ("夜勤の後はどれだけ休みを取りますか。", ["11時間"]),
    "F09": ("補助は1日に何食までですか。", ["1食"]), "F10": ("単身赴任手当の額はいくらですか。", ["4万5千円"]),
    "F11": ("貸した携帯電話を私用に使ってよいですか。", ["私用"]), "F12": ("出願した時の出願報奨はいくらですか。", ["1万5千円"]),
    "F13": ("駐車券はどこで受け取りますか。", ["受付"]), "F14": ("献血の休暇はいつまでに取りますか。", ["1か月"]),
    "F15": ("備蓄品はいつ点検しますか。", ["9月"]), "F16": ("提案はいつ締め切りますか。", ["月末"]),
    "F17": ("出向の期間は原則としてどれくらいまでですか。", ["4年"]), "F18": ("謝礼はいつ支払われますか。", ["3か月"]),
    "F19": ("1回の利用は何泊までですか。", ["2泊"]), "F20": ("出展の場所はどう決めますか。", ["抽選"]),
    "F21": ("速達はどう頼みますか。", ["1通ごと"]), "F22": ("休暇はいつまでに取りますか。", ["1年"]),
    "F23": ("再雇用の期間はどのように更新しますか。", ["1年ごと"]), "F24": ("支援の期間はどれくらいですか。", ["1年間"]),
    "F25": ("表彰はいつ行いますか。", ["4月"]), "F26": ("補助はどのように支払われますか。", ["領収書"]),
    "F27": ("安全靴はどこで買いますか。", ["会社の決めた店"]), "F28": ("休暇の前に何をしますか。", ["所属長"]),
    "F29": ("受講を終えたら何を出しますか。", ["修了"]), "F30": ("休暇の届には何を付けますか。", ["診断書"]),
}
# 割合（docs/verified-distill-plan.md §2）
MIX = {"boundary": 0.40, "conversion": 0.20, "exception": 0.15, "undecidable": 0.10, "decidable_missing": 0.05,
       "refusal": 0.05, "describe": 0.05}

FAM = {f["id"]: f for f in FAMILIES}


# ─── 値と語 ────────────────────────────────────────────────────
def num(v) -> str:
    v = Fraction(v)
    return f"{int(v):,}" if v.denominator == 1 else f"{float(v):g}"


def unit_of(c: dict) -> str:
    return c.get("unit_text") or c["unit"]


def plain(v, c: dict) -> str:
    return f"{num(v)}{unit_of(c)}"


def converted(v, c: dict) -> str | None:
    """別の単位で書く。換算の書き方が無い単位は None。"""
    u, v = c["unit"], Fraction(v)
    if u == "万円":
        return f"{int(v * 10_000):,}円"
    if u == "円" and v.denominator == 1 and int(v) % 100 == 0 and v >= 1000:
        return f"{int(v) // 1000}千{int(v) % 1000}円".replace("千0円", "千円")
    if u == "時間":
        whole, mins = int(v), int((v - int(v)) * 60)
        return f"{whole}時間{mins}分" if mins else f"{int(v * 60)}分"
    if u == "km":
        return f"{int(v * 1000):,}m"
    if u == "か月" and v >= 12:
        y, m = divmod(int(v), 12)
        return f"{y}年{m}か月" if m else f"{y}年"
    return None


def op(word: str) -> str:
    return WORDS[word][0]


def attr(word: str, t_text: str) -> str:
    return WORDS[word][1].format(t=t_text)


def sat_value(c: dict, want: bool, rng: random.Random):
    """条件を満たす（want=True）か満たさない値。境目から 1〜4 歩離す（満たさない側は境目ちょうどもあり）。"""
    t, s, o = Fraction(c["t"]), Fraction(str(c["step"])), op(c["word"])
    k = rng.randint(1, 4)
    up = o in (">=", ">")
    if want:
        v = t + (k - (1 if o == ">=" else 0)) * s if up else t - (k - (1 if o == "<=" else 0)) * s
    else:
        v = t - k * s if up else t + k * s
        if rng.random() < 0.3 and o in (">", "<"):
            v = t
    assert holds(v, o, t) == want
    return v


# ─── 規程の文 ──────────────────────────────────────────────────
def purpose_sentence(f: dict) -> str:
    p = f["purpose"]
    return f"この規程は、{p}を定める。" if p.endswith("決まり") else f"この規程は、{p}。"


def rule_text(f: dict, style: int, word: str | None = None) -> str:
    t = f["type"]
    if t in ("single", "exception"):
        w = word or f["word"]
        tt = plain(f["t"], f)
        if w == "上限とする":
            body = f"{f['subj']}は、{tt}を上限とする。"
        elif style == 2:
            body = f"次に当てはまる場合は、{f['then']}。\n・{f['subj']}が{PREDICATE[w].format(t=tt)}こと"
        else:
            body = f"{f['subj']}が{attr(w, tt)}{'場合' if style == 0 else 'とき'}は、{f['then']}。"
        if t == "exception":
            body += f"\nただし、{f['exc']}の場合は、この限りでない。"
        return body
    if t == "range":
        (lo, lw), (hi, hw) = f["lo"], f["hi"]
        a, b = plain(lo, f), plain(hi, f)
        if style == 2:
            return (f"次のすべてに当てはまる場合は、{f['then']}。\n・{f['subj']}が{PREDICATE[lw].format(t=a)}こと\n"
                    f"・{f['subj']}が{PREDICATE[hw].format(t=b)}こと")
        return f"{f['subj']}が{PREDICATE[lw].format(t=a)}、かつ{attr(hw, b)}{'場合' if style == 0 else 'とき'}は、{f['then']}。"
    if t in ("and", "or"):
        kind = "すべて" if t == "and" else "いずれか"
        lines = [f"{c['subj']}が{PREDICATE[c['word']].format(t=plain(c['t'], c))}こと" for c in (f["c1"], f["c2"])]
        if style == 1:
            return f"{f['then']}のは、次の{kind}に当てはまる場合とする。\n一 {lines[0]}\n二 {lines[1]}"
        return f"次の{kind}に当てはまる場合は、{f['then']}。\n一 {lines[0]}\n二 {lines[1]}"
    if t == "ratio":
        tt = f"{f['t'][0]}{f['t'][1]}"
        if style == 2:
            return f"次に当てはまる場合は、{f['then']}。\n・{f['part']}が、{f['whole']}の{PREDICATE[f['word']].format(t=tt)}こと"
        return f"{f['part']}が、{f['whole']}の{attr(f['word'], tt)}{'場合' if style == 0 else 'とき'}は、{f['then']}。"
    if t == "deadline":
        return (f"{f['event']}の{f['n']}日前の日までに、{f['action']}。" if style == 1 else
                f"{f['event']}の{f['n']}日前までに、{f['action']}。")
    if t == "age":
        what = "勤続年数" if f["basis"] == "entry" else "年齢"
        tt = f"{f['t']}{'年' if f['basis'] == 'entry' else '歳'}"
        rule = (f"基準日（{f['base_name']}）に{what}が{attr(f['word'], tt)}社員は、{f['then']}。" if style != 2 else
                f"{f['then']}のは、基準日（{f['base_name']}）に{what}が{attr(f['word'], tt)}社員とする。")
        return rule + ("勤続年数は、入社した日を起点に、基準日の時点で満了している年の数とする。" if f["basis"] == "entry" else "年齢は、基準日の時点の満年齢とする。")
    if t == "tier":
        a, b = plain(f["a"], f), plain(f["b"], f)
        l0, l1, l2 = f["labels"]
        if style == 2:
            return f"{f['subj']}に応じて、次のとおりとする。\n・{a}未満: {l0}\n・{a}以上{b}未満: {l1}\n・{b}以上: {l2}"
        return f"{f['subj']}が{a}未満の場合は{l0}、{a}以上{b}未満の場合は{l1}、{b}以上の場合は{l2}とする。"
    raise ValueError(t)


TITLES = {"single": "基準", "range": "対象", "and": "条件", "or": "条件", "ratio": "基準", "deadline": "期限",
          "age": "対象", "tier": "区分", "exception": "基準"}


def render_doc(f: dict, style: int, word: str | None = None) -> tuple[str, int]:
    """規程の全文と、主な条の番号。style 1 は主な条を第3条に置く（条の番号を覚えさせないため）。"""
    rule = rule_text(f, style, word)
    if style == 1:
        arts = [("目的", purpose_sentence(f)), ("手続き", f["other"]), (TITLES[f["type"]], rule)]
    else:
        arts = [("目的", purpose_sentence(f)), (TITLES[f["type"]], rule), ("その他", f["other"])]
    body = "\n\n".join(f"第{i}条（{h}）\n{t}" for i, (h, t) in enumerate(arts, 1))
    return f"# {f['title']}\n\n{body}\n", (3 if style == 1 else 2)


def chunks_of(doc: str) -> list[str]:
    head, *arts = doc.split("\n\n第")
    return [p.strip() for p in [head + "\n\n第" + arts[0]] + ["第" + a for a in arts[1:]]]


# ─── 問の部品 ──────────────────────────────────────────────────
@dataclass
class Item:
    family: str
    category: str
    facts_text: str
    expected: str
    style: int
    word: str | None = None
    three: bool = True
    check: dict = field(default_factory=dict)
    no_footer: bool = False
    keywords: list = field(default_factory=list)


def conclude(f: dict, v) -> str:
    if f["type"] == "tier":
        return UNDECIDED if v is None else f["labels"][v]
    return UNDECIDED if v is None else (f["yes"] if v else f["no"])


def pos_values(t, step):
    s = Fraction(str(step))
    return (("below", Fraction(t) - s), ("at", Fraction(t)), ("above", Fraction(t) + s))


def chk(value_txt: str, threshold_txt: str, at: bool, word: str | None) -> dict:
    return {"value_text": value_txt, "threshold_text": threshold_txt, "at": at, "word": word}


def items_single(f: dict, rng: random.Random) -> list[Item]:
    out = []
    for style in range(3):
        words = [f["word"]] + ([COUNTERPART[f["word"]]] if f["word"] in COUNTERPART else [])
        for w in words:
            for pos, v in pos_values(f["t"], f["step"]):
                hit = holds(v, op(w), Fraction(f["t"]))
                exc_note = f"これは{f['exc']}ではありません。" if f["type"] == "exception" else ""
                res = hit  # 例外の型でも「当たらない」と書くので、境目どおり
                for conv in (False, True):
                    vt = converted(v, f) if conv else plain(v, f)
                    if vt is None:
                        continue
                    out.append(Item(f["id"], "conversion" if conv else "boundary",
                                    f"{f['subj']}は{vt}です。{exc_note}", conclude(f, res), style, w, rng.random() < 0.5,
                                    chk(vt, plain(f["t"], f), pos == "at", w)))
        if f["type"] == "single":
            out.append(Item(f["id"], "undecidable", f"{f['subj']}は、まだ調べていません。", UNDECIDED, style, f["word"], True))
    return out


def items_exception(f: dict, rng: random.Random) -> list[Item]:
    out = items_single(f, rng)
    o, t, s = op(f["word"]), Fraction(f["t"]), Fraction(str(f["step"]))
    for style in range(3):
        for k in range(1, 7):
            hit_v = t + k * s if o in (">=", ">") else t - k * s       # 境目に当てはまる値
            miss_v = t - k * s if o in (">=", ">") else t + k * s      # 当てはまらない値
            assert holds(hit_v, o, t) and not holds(miss_v, o, t)
            # 例外が効く向き: exc_effect=no は「当てはまる」を打ち消す、yes は「当てはまらない」を救う
            v_matter = hit_v if f["exc_effect"] == "no" else miss_v
            v_free = miss_v if f["exc_effect"] == "no" else hit_v
            base_hit = holds(v_matter, o, t)
            flipped = (not base_hit)
            vt, ft = plain(v_matter, f), plain(v_free, f)
            out.append(Item(f["id"], "exception", f"{f['subj']}は{vt}です。これは{f['exc']}です。",
                            conclude(f, flipped), style, f["word"], rng.random() < 0.5, chk(vt, plain(t, f), False, f["word"])))
            out.append(Item(f["id"], "exception", f"{f['subj']}は{vt}です。これは{f['exc_near']}です。",
                            conclude(f, base_hit), style, f["word"], rng.random() < 0.5, chk(vt, plain(t, f), False, f["word"])))
            out.append(Item(f["id"], "undecidable", f"{f['subj']}は{vt}です。理由がどれに当たるかは、まだ聞いていません。",
                            UNDECIDED, style, f["word"], True))
            out.append(Item(f["id"], "decidable_missing", f"{f['subj']}は{ft}です。理由がどれに当たるかは、まだ聞いていません。",
                            conclude(f, holds(v_free, o, t)), style, f["word"], True))
    return out


def items_range(f: dict, rng: random.Random) -> list[Item]:
    out = []
    (lo, lw), (hi, hw) = f["lo"], f["hi"]
    for style in range(3):
        for t in (lo, hi):
            for pos, v in pos_values(t, f["step"]):
                res = holds(v, op(lw), Fraction(lo)) and holds(v, op(hw), Fraction(hi))
                for conv in (False, True):
                    vt = converted(v, f) if conv else plain(v, f)
                    if vt is None:
                        continue
                    word = lw if t == lo else hw
                    out.append(Item(f["id"], "conversion" if conv else "boundary", f"{f['subj']}は{vt}です。",
                                    conclude(f, res), style, word, rng.random() < 0.5, chk(vt, plain(t, f), pos == "at", word)))
        out.append(Item(f["id"], "undecidable", f"{f['subj']}はまだ決まっていません。", UNDECIDED, style, None, True))
    return out


def items_two(f: dict, rng: random.Random) -> list[Item]:
    out, is_and = [], f["type"] == "and"
    cs = (f["c1"], f["c2"])
    for style in range(3):
        for i, c in enumerate(cs):
            other = cs[1 - i]
            for pos, v in pos_values(c["t"], c["step"]):
                hit = holds(v, op(c["word"]), Fraction(c["t"]))
                ov = sat_value(other, is_and, rng)   # and: もう一方は満たす / or: もう一方は満たさない
                res = hit  # どちらも、結論は変える方の条件だけで決まる
                for conv in (False, True):
                    vt = converted(v, c) if conv else plain(v, c)
                    if vt is None:
                        continue
                    facts = [f"{c['subj']}は{vt}です。", f"{other['subj']}は{plain(ov, other)}です。"]
                    if i == 1:
                        facts.reverse()
                    out.append(Item(f["id"], "conversion" if conv else "boundary", "".join(facts), conclude(f, res),
                                    style, c["word"], rng.random() < 0.5, chk(vt, plain(c["t"], c), pos == "at", c["word"])))
            # 片方だけ分からない
            v_dec = sat_value(c, not is_and, rng)     # and: 満たさない→決まる / or: 満たす→決まる
            v_und = sat_value(c, is_and, rng)         # and: 満たす→決まらない / or: 満たさない→決まらない
            miss = f"{other['subj']}は、まだ調べていません。"
            out.append(Item(f["id"], "decidable_missing", f"{c['subj']}は{plain(v_dec, c)}です。{miss}",
                            conclude(f, not is_and), style, c["word"], True))
            out.append(Item(f["id"], "undecidable", f"{c['subj']}は{plain(v_und, c)}です。{miss}", UNDECIDED, style, c["word"], True))
    return out


def items_ratio(f: dict, rng: random.Random) -> list[Item]:
    out = []
    tv, tu = f["t"]
    r = Fraction(tv, 100) if tu == "%" else Fraction(tv, 10)
    for style in range(3):
        for whole in (r.denominator * k for k in (4, 7, 12, 20)):
            at = r * whole
            for pos, part in (("below", at - 1), ("at", at), ("above", at + 1)):
                res = holds(Fraction(part, whole), op(f["word"]), r)
                out.append(Item(f["id"], "conversion", f"{f['part']}は{num(part)}{f['unit']}で、{f['whole']}は{num(whole)}{f['unit']}です。",
                                conclude(f, res), style, f["word"], rng.random() < 0.5,
                                chk(f"{num(part)}{f['unit']}", f"{tv}{tu}", pos == "at", f["word"])))
        out.append(Item(f["id"], "undecidable", f"{f['part']}は{num(r.denominator * 3)}{f['unit']}ですが、{f['whole']}は、まだ調べていません。",
                        UNDECIDED, style, f["word"], True))
    return out


def jdate(d: date) -> str:
    return f"{d.year}年{d.month}月{d.day}日"


def items_deadline(f: dict, rng: random.Random) -> list[Item]:
    out = []
    for style in range(3):
        for _ in range(4):
            event = date(2027, 1, 1) + timedelta(days=rng.randint(60, 600))
            limit = event - timedelta(days=f["n"])
            for pos, d in (("below", limit - timedelta(days=1)), ("at", limit), ("above", limit + timedelta(days=1))):
                res = d <= limit
                out.append(Item(f["id"], "conversion", f"{f['event']}は{jdate(event)}で、手続きは{jdate(d)}に行いました。",
                                conclude(f, res), style, None, rng.random() < 0.5, chk(jdate(d), f"{f['n']}日前", pos == "at", None)))
        out.append(Item(f["id"], "undecidable", f"{f['event']}は{jdate(event)}です。手続きをした日は記録が残っていません。",
                        UNDECIDED, style, None, True))
    return out


def items_age(f: dict, rng: random.Random) -> list[Item]:
    out = []
    unit = "年" if f["basis"] == "entry" else "歳"
    for style in range(3):
        for _ in range(4):
            year = rng.choice((2027, 2028))
            base = date(year, f["base_month"], f["base_day"])
            for pos, years in (("below", f["t"] - 1), ("at", f["t"]), ("above", f["t"] + 1)):
                # 基準日から 20〜200 日ずらした日（応当日の近くを避ける）
                off = rng.randint(20, 200)
                d = date(base.year - years, base.month, base.day) - timedelta(days=off)
                full = base.year - d.year - ((base.month, base.day) < (d.month, d.day))
                assert full == years
                res = holds(full, op(f["word"]), f["t"])
                lead = f"入社したのは{jdate(d)}です。" if f["basis"] == "entry" else f"生年月日は{jdate(d)}です。"
                out.append(Item(f["id"], "conversion", f"{lead}{year}年の基準日（{jdate(base)}）で判断します。",
                                conclude(f, res), style, f["word"], rng.random() < 0.5,
                                chk(jdate(d), f"{f['t']}{unit}", pos == "at", f["word"])))
    return out


def items_tier(f: dict, rng: random.Random) -> list[Item]:
    out = []
    a, b = Fraction(f["a"]), Fraction(f["b"])
    for style in range(3):
        for t in (a, b):
            for pos, v in pos_values(t, 1):
                idx = 0 if v < a else (1 if v < b else 2)
                for conv in (False, True):
                    vt = converted(v, f) if conv else plain(v, f)
                    if vt is None:
                        continue
                    out.append(Item(f["id"], "conversion" if conv else "boundary", f"{f['subj']}は{vt}です。",
                                    conclude(f, idx), style, "以上", rng.random() < 0.5, chk(vt, plain(t, f), pos == "at", "以上")))
        out.append(Item(f["id"], "undecidable", f"{f['subj']}はまだ数えていません。", UNDECIDED, style, None, True))
    return out


BUILDERS = {"single": items_single, "exception": items_exception, "range": items_range, "and": items_two,
            "or": items_two, "ratio": items_ratio, "deadline": items_deadline, "age": items_age, "tier": items_tier}


def items_extra(f: dict) -> list[Item]:
    """断り（資料に無い事を聞く）と、資料の文を答える問。"""
    out = []
    for style in range(3):
        for a in REFUSAL_ASKS:
            out.append(Item(f["id"], "refusal", f"{f['title']}について質問です。{a}", "refusal", style, no_footer=True))
        q, kws = DESC[f["id"]]
        out.append(Item(f["id"], "describe", f"{f['title']}について質問です。{q}", "describe", style, no_footer=True, keywords=kws))
    return out


# ─── 組み立て ──────────────────────────────────────────────────
def options_for(f: dict, three: bool) -> list[str]:
    base = list(f["labels"]) if f["type"] == "tier" else [f["yes"], f["no"]]
    return base + [UNDECIDED] if three else base


def footer(options: list[str]) -> str:
    q = [f"「結論: {o}」" for o in options]
    if len(q) == 2:
        return f"\n最後の行には{q[0]}または{q[1]}のどちらかだけを書いてください。"
    return f"\n最後の行には{'、'.join(q)}のどれかだけを書いてください。"


def ask_of(f: dict) -> str:
    a = f["ask"]
    return a if a.endswith(("？", "。")) else a + "？"


def assemble(item: Item, split: str, rng: random.Random) -> dict:
    f = FAM[item.family]
    doc, main_art = render_doc(f, item.style, item.word if item.word != f.get("word") else None)
    target = [(f["title"], c) for c in chunks_of(doc)]
    others = [x for x in SPLIT_FAMILIES[split] if x != f["id"]]
    distract = []
    for oid in rng.sample(others, rng.randint(1, 3)):
        of = FAM[oid]
        odoc, _ = render_doc(of, rng.randrange(3))
        distract.append((of["title"], rng.choice(chunks_of(odoc))))
    ctx = target + distract
    rng.shuffle(ctx)
    rule_marker = f"第{(2 if item.style == 1 else 3) if item.category == 'describe' else main_art}条"
    rule_ids = [i for i, (title, c) in enumerate(ctx, 1) if title == f["title"] and c.startswith(rule_marker) or
                (title == f["title"] and f"\n\n{rule_marker}" in c)]
    context = "\n\n".join(f"[{i}] {title}.md\n{c}" for i, (title, c) in enumerate(ctx, 1))
    wrap_idx = rng.choice(SPLIT_WRAPS[split])
    wrap = WRAPS[wrap_idx]
    if item.no_footer:
        question = wrap.format(facts="", ask=item.facts_text)
        options = []
    else:
        options = options_for(f, item.three)
        if item.expected == UNDECIDED and not item.three:
            options = options_for(f, True)
        question = wrap.format(facts=item.facts_text, ask=ask_of(f)) + footer(options)
    user = f"【参考情報】\n{context}\n\n【質問】\n{question}"
    return {"family": f["id"], "type": f["type"], "category": item.category, "split": split,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            "question": question, "options": options, "expected": item.expected, "keywords": item.keywords,
            "rule_chunk_ids": rule_ids, "doc_title": f["title"], "style": item.style, "word": item.word, "wrap": wrap_idx,
            "check": item.check}


def pool(split: str, seed: int) -> dict[str, list[Item]]:
    rng = random.Random(seed)
    by_cat: dict[str, list[Item]] = {c: [] for c in MIX}
    for fid in SPLIT_FAMILIES[split]:
        f = FAM[fid]
        for it in BUILDERS[f["type"]](f, rng) + items_extra(f):
            by_cat[it.category].append(it)
    return by_cat


def generate(split: str, n: int, seed: int) -> list[dict]:
    """割合 MIX どおりに n 問を作る。同じ問の重複は、包みと混ぜ物が違えば許す（入力の文が同じ物は除く）。"""
    rng = random.Random(seed)
    by_cat = pool(split, seed)
    rows, seen = [], set()
    # 作れない区分（開発用の家族に例外の型が無い等）の分は、作れる区分に割合で配り直す
    live = {c: w for c, w in MIX.items() if by_cat[c]}
    total = sum(live.values())
    for cat, share in live.items():
        want = round(n * share / total)
        cands = by_cat[cat]
        tries = 0
        got = 0
        while got < want and tries < want * 20:
            tries += 1
            row = assemble(rng.choice(cands), split, rng)
            key = row["messages"][1]["content"]
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
            got += 1
    rng.shuffle(rows)
    for i, r in enumerate(rows):
        r["id"] = f"{split[0].upper()}{seed % 1000:03d}-{i:05d}"
    return rows


def item_dict(it: Item) -> dict:
    return asdict(it)
