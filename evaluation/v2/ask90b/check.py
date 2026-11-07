# ask90b の検算。式は規程の文だけから、規程ごとに 2 通り（f1: 条の順に判定 / f2: 妨げの列挙や別の計算）書いた。
# make_questions.py は読まず、questions.jsonl と facts.jsonl だけを見る。
import itertools
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).parent
HOLD = "保留"
KINDS = {"complete": 4, "exc_hit": 1, "exc_miss": 1, "missing_one": 2, "missing_many": 3,
         "irrelevant_gap": 1, "dont_know": 3}
ERRORS = []


def err(msg):
    ERRORS.append(msg)


def d(s):
    return date.fromisoformat(s)


def years(day, n):
    return date(day.year + n, day.month, day.day)


def months_back(day, n):
    m = day.month - 1 - n
    return date(day.year + m // 12, m % 12 + 1, min(day.day, 28))


def mins(s):
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def hhmm(n):
    return f"{n // 60:02d}:{n % 60:02d}"


def iso(day, delta=0):
    return (day + timedelta(days=delta)).isoformat()


# ------------------------------------------------ 独身寮入寮規程
def dorm_f1(f):
    if not f["正社員"] or f["配偶者と同居"]:
        return "見送り"
    first = d(f["入寮希望月初日"])
    if f["半壊以上認定日"] is not None and first <= years(d(f["半壊以上認定日"]), 1):
        return "入寮"
    if f["障害者手帳"] and f["産業医の軽減認定"]:
        return "入寮"
    b = d(f["生年月日"])
    age = first.year - b.year - ((first.month, first.day) < (b.month, b.day))
    return "入寮" if f["通勤分"] >= 90 and age < 35 else "見送り"


def dorm_f2(f):
    premise = f["正社員"] and f["配偶者と同居"] is False
    first = d(f["入寮希望月初日"])
    tokurei = False
    if f["半壊以上認定日"]:
        tokurei = (d(f["半壊以上認定日"]).year + 1, d(f["半壊以上認定日"]).month, d(f["半壊以上認定日"]).day) >= (first.year, first.month, first.day)
    tokurei = tokurei or (f["障害者手帳"] and f["産業医の軽減認定"])
    missing = []
    if f["通勤分"] / 60 < 1.5:
        missing.append("通勤")
    if first >= years(d(f["生年月日"]), 35):  # 35 歳の誕生日以後は欠ける
        missing.append("年齢")
    return "入寮" if premise and (tokurei or not missing) else "見送り"


# ------------------------------------------------ 予防接種費用補助規程
def vac_f1(f):
    if not f["本人の接種"]:
        return "自己負担"
    day = d(f["接種日"])
    if (d(f["領収書提出日"]) - day).days > 30:
        return "自己負担"
    if f["妊娠中"] or f["基礎疾患で勧奨"]:
        return "補助"
    in_season = day.month in (10, 11, 12, 1)
    cost = f["費用額"] * 11 / 10 if f["税区分"] == "税抜" else f["費用額"]
    return "補助" if in_season and cost <= 5000 else "自己負担"


def vac_f2(f):
    day = d(f["接種日"])
    y = day.year if day.month >= 10 else day.year - 1
    season = date(y, 10, 1) <= day <= date(y + 1, 1, 31)
    tenths = f["費用額"] * (11 if f["税区分"] == "税抜" else 10)
    x, count = day, 0  # 翌日を 1 日目として歩く
    while x < d(f["領収書提出日"]):
        x += timedelta(days=1)
        count += 1
    bad = []
    if f["本人の接種"] is not True:
        bad.append("対象外")
    if count > 30:
        bad.append("提出")
    if not (f["妊娠中"] or f["基礎疾患で勧奨"]):
        if not season:
            bad.append("期間")
        if tenths > 50000:
            bad.append("費用")
    return "自己負担" if bad else "補助"


# ------------------------------------------------ 語学研修選抜規程
def lang_score(f):
    dl = d(f["応募締切日"])
    win = years(dl, -2)
    return max([s for t, s in f["検定履歴"] if win <= d(t) <= dl], default=0)


def lang_f1(f):
    if f["過去に修了"]:
        return "落選"
    if f["海外赴任内示"]:
        return "選抜"
    dl = d(f["応募締切日"])
    if not f["正社員"] or years(d(f["入社日"]), 3) > dl:
        return "落選"
    line = 700
    if f["育休復職日"] and d(f["育休復職日"]) <= dl <= years(d(f["育休復職日"]), 1):
        line = 600
    return "選抜" if lang_score(f) >= line and f["直前評価"] in ("S", "A", "B") else "落選"


def lang_f2(f):
    dl = d(f["応募締切日"])
    hire = d(f["入社日"])
    tenure_ok = (dl.year - 3, dl.month, dl.day) >= (hire.year, hire.month, hire.day)
    back = f["育休復職日"]
    relief = bool(back) and 0 <= (dl - d(back)).days and dl <= date(d(back).year + 1, d(back).month, d(back).day)
    scores = sorted(s for t, s in f["検定履歴"] if date(dl.year - 2, dl.month, dl.day) <= d(t) <= dl)
    best = scores[-1] if scores else -1
    rank = "SABCD".index(f["直前評価"])
    normal = f["正社員"] and tenure_ok and rank <= 2 and best >= (600 if relief else 700)
    return "選抜" if (normal or f["海外赴任内示"]) and not f["過去に修了"] else "落選"


# ------------------------------------------------ 休日出勤振替規程
def work_f1(f):
    if not f["所属長の事前命令"]:
        return "精算"
    if f["時間外制限"]:
        return "振替"
    worked = mins(f["終了時刻"]) - mins(f["開始時刻"]) - f["休憩分"]
    need = 180 if f["育児介護の短時間勤務"] else 240
    days = (d(f["振替予定日"]) - d(f["出勤日"])).days
    return "振替" if worked >= need and days <= 56 else "精算"


def work_f2(f):
    slack = (mins(f["終了時刻"]) - f["休憩分"]) - (mins(f["開始時刻"]) + 60 * (3 if f["育児介護の短時間勤務"] else 4))
    deadline = d(f["出勤日"]) + timedelta(weeks=8)
    ok = slack >= 0 and d(f["振替予定日"]) <= deadline
    return "振替" if f["所属長の事前命令"] and (ok or f["時間外制限"]) else "精算"


# ------------------------------------------------ 海外出張渡航規程
def trip_f1(f):
    if not f["部長の出張命令"]:
        return "延期"
    limit = 2
    if f["渡航先に配偶者か子が居住"]:
        limit = 3
    if f["妊娠中"]:
        limit = 1
    if f["渡航先危険度"] > limit:
        return "延期"
    if f["救援要員"]:
        return "渡航"
    return "渡航" if (d(f["出発日"]) - d(f["渡航届提出日"])).days >= 14 else "延期"


def trip_f2(f):
    limit = 1 if f["妊娠中"] else (3 if f["渡航先に配偶者か子が居住"] else 2)
    last = d(f["出発日"]) - timedelta(weeks=2)
    reasons = [not f["部長の出張命令"], limit < f["渡航先危険度"],
               d(f["渡航届提出日"]) > last and not f["救援要員"]]
    return "延期" if any(reasons) else "渡航"


# ------------------------------------------------ 業務端末持出規程
def pc_f1(f):
    if not f["資産台帳登録"]:
        return "禁止"
    if f["顧客情報件数"] >= 1000 or not f["暗号化"]:
        return "禁止"
    if f["両立支援認定"]:
        return "持出"
    limit = 30 if f["事業継続要員"] else 7
    return "持出" if f["持出日数"] <= limit else "禁止"


def pc_f2(f):
    days_ok = f["両立支援認定"] or f["持出日数"] < (31 if f["事業継続要員"] else 8)
    conds = [f["資産台帳登録"], days_ok, f["顧客情報件数"] <= 999, f["暗号化"]]
    return "持出" if all(conds) else "禁止"


# ------------------------------------------------ あり得る値（基準日は場面の値から作る）
def dom_dorm(f, name):
    first = d(f["入寮希望月初日"])
    table = {
        "正社員": [True, False], "配偶者と同居": [True, False], "通勤分": [60, 89, 90, 120],
        "入寮希望月初日": [f["入寮希望月初日"]],
        "生年月日": [iso(years(first, -30)), iso(years(first, -35)), iso(years(first, -35), 1), iso(years(first, -40))],
        "半壊以上認定日": [None, iso(months_back(first, 6)), iso(years(first, -1)), iso(years(first, -1), -1), iso(years(first, -3))],
        "障害者手帳": [True, False], "産業医の軽減認定": [True, False]}
    return table[name]


def dom_vac(f, name):
    if name == "接種日":
        base = f["領収書提出日"]
        if base == "不明":
            return [f["接種日"]]
        return [iso(d(base), -k) for k in (2, 14, 28, 45, 100)]
    if name == "領収書提出日":
        return [iso(d(f["接種日"]), k) for k in (5, 30, 31, 60)]
    return {"本人の接種": [True, False], "費用額": [3000, 4545, 4546, 5000, 5500, 8000],
            "税区分": ["税込", "税抜"], "妊娠中": [True, False], "基礎疾患で勧奨": [True, False]}[name]


def dom_lang(f, name):
    dl = d(f["応募締切日"])
    recent = iso(months_back(dl, 6))
    return {"正社員": [True, False], "応募締切日": [f["応募締切日"]],
            "入社日": [iso(years(dl, -3)), iso(years(dl, -3), 1), iso(years(dl, -1)), iso(years(dl, -8))],
            "検定履歴": [[[recent, s]] for s in (550, 600, 650, 699, 700, 800)],
            "直前評価": ["S", "B", "C", "D"], "海外赴任内示": [True, False],
            "育休復職日": [None, recent, iso(years(dl, -3))], "過去に修了": [True, False]}[name]


def dom_work(f, name):
    start = mins(f["開始時刻"])
    work = d(f["出勤日"])
    return {"所属長の事前命令": [True, False], "開始時刻": [f["開始時刻"]],
            "終了時刻": [hhmm(start + h * 60) for h in (2, 3, 4, 5, 9)], "休憩分": [0, 30, 60, 90],
            "出勤日": [f["出勤日"]], "振替予定日": [iso(work, k) for k in (1, 56, 57, 100)],
            "育児介護の短時間勤務": [True, False], "時間外制限": [True, False]}[name]


def dom_trip(f, name):
    out = d(f["出発日"])
    return {"部長の出張命令": [True, False], "渡航先危険度": [1, 2, 3, 4], "出発日": [f["出発日"]],
            "渡航届提出日": [iso(out, -k) for k in (30, 14, 13, 3)],
            "渡航先に配偶者か子が居住": [True, False], "妊娠中": [True, False], "救援要員": [True, False]}[name]


def dom_pc(f, name):
    return {"資産台帳登録": [True, False], "持出日数": [3, 7, 8, 30, 31, 60], "顧客情報件数": [100, 999, 1000, 5000],
            "暗号化": [True, False], "事業継続要員": [True, False], "両立支援認定": [True, False]}[name]


RULES = {
    "独身寮入寮規程": dict(f1=dorm_f1, f2=dorm_f2, dom=dom_dorm,
                    core=["正社員", "配偶者と同居", "通勤分", "入寮希望月初日", "生年月日"],
                    exc={"半壊以上認定日": None, "障害者手帳": False, "産業医の軽減認定": False},
                    words=r"災害|被災|半壊|全壊|罹災|手帳|障害|産業医"),
    "予防接種費用補助規程": dict(f1=vac_f1, f2=vac_f2, dom=dom_vac,
                       core=["本人の接種", "接種日", "費用額", "税区分", "領収書提出日"],
                       exc={"妊娠中": False, "基礎疾患で勧奨": False},
                       words=r"妊娠|基礎疾患|産業医"),
    "語学研修選抜規程": dict(f1=lang_f1, f2=lang_f2, dom=dom_lang,
                     core=["正社員", "入社日", "応募締切日", "検定履歴", "直前評価"],
                     exc={"海外赴任内示": False, "育休復職日": None, "過去に修了": False},
                     words=r"内示|赴任|育休|育児休業|修了|この研修[をにはも]|終えた"),
    "休日出勤振替規程": dict(f1=work_f1, f2=work_f2, dom=dom_work,
                     core=["所属長の事前命令", "開始時刻", "終了時刻", "休憩分", "出勤日", "振替予定日"],
                     exc={"育児介護の短時間勤務": False, "時間外制限": False},
                     words=r"短時間|時短|育児|介護|産業医|時間外"),
    "海外出張渡航規程": dict(f1=trip_f1, f2=trip_f2, dom=dom_trip,
                     core=["部長の出張命令", "渡航先危険度", "出発日", "渡航届提出日"],
                     exc={"渡航先に配偶者か子が居住": False, "妊娠中": False, "救援要員": False},
                     words=r"配偶者|家族|子ども|妊娠|救援"),
    "業務端末持出規程": dict(f1=pc_f1, f2=pc_f2, dom=dom_pc,
                     core=["資産台帳登録", "持出日数", "顧客情報件数", "暗号化"],
                     exc={"事業継続要員": False, "両立支援認定": False},
                     words=r"事業継続|両立|治療"),
}
# 事実の語が文に出るか。書いた事実は出て、書いていない事実は出ない事を確かめる。
FIELD_RE = {
    "配偶者と同居": r"同居|独身|未婚|結婚|妻|夫|別居|配偶者", "通勤分": r"通勤|片道|経路検索",
    "生年月日": r"生まれ|生年月日|誕生日|\d歳", "半壊以上認定日": r"半壊|全壊|災害|被災|罹災|損壊",
    "障害者手帳": r"手帳", "産業医の軽減認定": r"産業医",
    "本人の接種": r"本人|自分|私|娘|息子|家族|妻|夫|子ども", "費用額": r"\d円|\d千円|費用|金額",
    "税区分": r"税", "領収書提出日": r"提出|へ出|領収書を出", "妊娠中": r"妊娠", "基礎疾患で勧奨": r"基礎疾患|産業医",
    "入社日": r"入社|勤続|年目", "検定履歴": r"検定", "直前評価": r"評価", "海外赴任内示": r"内示|赴任",
    "育休復職日": r"育休|育児休業|復職", "過去に修了": r"修了|この研修[をにはも]|終えた|行った事|受けるのは初めて|参加して",
    "所属長の事前命令": r"命令|所属長|上司|言われ|命じ", "休憩分": r"休憩", "終了時刻": r"\d時.*まで|〜\d|終わ",
    "振替予定日": r"予定|休みを取|休みたい|日を代わり|振替の休み|代わりの休み", "育児介護の短時間勤務": r"短時間|時短",
    "時間外制限": r"産業医|時間外", "部長の出張命令": r"命令|命じ", "渡航先危険度": r"危険度",
    "渡航届提出日": r"渡航届|届は", "渡航先に配偶者か子が居住": r"配偶者|家族|子ども", "救援要員": r"救援",
    "資産台帳登録": r"台帳", "持出日数": r"\d日|週間|返します", "顧客情報件数": r"顧客|\d件",
    "暗号化": r"暗号", "事業継続要員": r"事業継続", "両立支援認定": r"両立",
    "正社員": r"正社員|契約社員|パート|派遣|嘱託",
}


def value_forms(value):
    """事実の値の、文に出そうな書き方。見つからない物は一覧に出して目で確かめる。"""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return [str(value), f"{value:,}"]
    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        y, m, dd = value.split("-")
        return [f"{int(m)}月{int(dd)}日", f"{int(m)}/{int(dd)}", f"{int(dd)}日", f"{int(m)}月"]
    if isinstance(value, str) and re.fullmatch(r"\d{2}:\d{2}", value):
        h, m = value.split(":")
        return [f"{int(h)}時{int(m)}分" if int(m) else f"{int(h)}時", f"{int(h)}:{m}"]
    return None


def stated_value_gaps(qs, fs):
    gaps = []
    for q in qs:
        body = q["question"].split("\n")[0]
        rec = fs[q["id"]]
        for name in rec["stated"]:
            if name not in rec["facts"]:
                continue
            val = rec["facts"][name]
            items = [(name, val)]
            if name == "検定履歴":
                items = [(name, s) for _, s in val]
            for n, v in items:
                forms = value_forms(v)
                if forms and not any(x in body for x in forms):
                    gaps.append(f"{q['id']} {n}={v}: {body[:60]}")
    return gaps


def conclude(rule, f):
    return RULES[rule]["f1"](f)


def assignments(rule, facts, free):
    """free の事実にあり得る値（場面の真の値も含む）を全部入れる"""
    dom = RULES[rule]["dom"]
    lists = []
    for name in free:
        vals = list(dom(facts, name))
        if facts[name] != "不明" and facts[name] not in vals:
            vals.append(facts[name])
        lists.append(vals)
    for combo in itertools.product(*lists):
        yield dict(zip(free, combo))


def analyse(rule, facts, stated):
    """stated だけを固定し、残りを動かす。結論の集合と、結論を割る事実の名前を返す。"""
    free = [k for k in facts if k not in stated]
    table = {}
    for a in assignments(rule, facts, free):
        g = {**facts, **a}
        table[tuple(json.dumps(a[k], ensure_ascii=False) for k in free)] = conclude(rule, g)
    outcomes = set(table.values())
    splitters = []
    for i, name in enumerate(free):
        groups = defaultdict(set)
        for key, val in table.items():
            groups[key[:i] + key[i + 1:]].add(val)
        if any(len(v) > 1 for v in groups.values()):
            splitters.append(name)
    return outcomes, splitters


def main():
    qs = [json.loads(x) for x in open(HERE / "questions.jsonl", encoding="utf-8")]
    fs = {r["id"]: r for r in map(json.loads, open(HERE / "facts.jsonl", encoding="utf-8"))}

    # 5. 件数・種類・id・書式
    if len(qs) != 90 or len({q["id"] for q in qs}) != 90:
        err("件数か id の重複")
    if sorted(q["id"] for q in qs) != [f"B{i:03d}" for i in range(1, 91)]:
        err("id の並び")
    combos = 0
    for rule in RULES:
        rows = [q for q in qs if q["rule_id"] == rule]
        if len(rows) != 15:
            err(f"{rule}: {len(rows)} 件")
        if Counter(q["kind"] for q in rows) != Counter(KINDS):
            err(f"{rule}: 種類の内訳 {Counter(q['kind'] for q in rows)}")
        dirs = Counter(q["expected_conclusion"] for q in rows if q["kind"] != "dont_know")
        if sorted(dirs.values()) != [6, 6]:
            err(f"{rule}: 結論の向き {dirs}")
        print(f"{rule}: 向き {dict(dirs)}")
        if sum(1 for q in rows if q["pair_id"]) != 2:
            err(f"{rule}: pair の数")

    pairs = defaultdict(list)
    for q in qs:
        qid, rule, kind = q["id"], q["rule_id"], q["kind"]
        rec = fs[qid]
        facts, stated, unknowns = rec["facts"], rec["stated"], rec["unknowns"]
        R = RULES[rule]
        expected_keys = set(R["core"]) | set(R["exc"])
        if set(facts) != expected_keys:
            err(f"{qid}: 事実の名前 {set(facts) ^ expected_keys}")
            continue
        if not set(stated) <= set(facts):
            err(f"{qid}: stated の名前")
        lines = q["question"].split("\n")
        a, b = q["conclusion_options"]
        if len(lines) != 2 or lines[1] != f"最後の行には「結論: {a}」または「結論: {b}」のどちらかだけを書いてください。":
            err(f"{qid}: 最後の行")
        if set(q["conclusion_options"]) != set({"独身寮入寮規程": ["入寮", "見送り"], "予防接種費用補助規程": ["補助", "自己負担"],
                                               "語学研修選抜規程": ["選抜", "落選"], "休日出勤振替規程": ["振替", "精算"],
                                               "海外出張渡航規程": ["渡航", "延期"], "業務端末持出規程": ["持出", "禁止"]}[rule]):
            err(f"{qid}: 選択肢")
        if q["group"] != kind:
            err(f"{qid}: group")
        if q["should_hold"] != (kind == "dont_know"):
            err(f"{qid}: should_hold")
        body = lines[0]
        if re.search(r"分かりません|わかりません|不明|覚えていません|確認できていません", body):
            err(f"{qid}: 文で知らない事が分かる")
        # 語の確認: 書いた事実は語が出て、書いていない事実は語が出ない
        for name, pat in FIELD_RE.items():
            if name not in facts:
                continue
            found = re.search(pat, body) is not None
            if name in stated and not found:
                err(f"{qid}: 書いたはずの {name} の語が文に無い")
            if name not in stated and found:
                err(f"{qid}: 書いていない {name} の語が文に出る")
        if q["pair_id"]:
            pairs[q["pair_id"]].append((q, rec))

        # 1. 全場面で f1 と f2 が一致し期待と同じ
        if kind == "dont_know":
            if q["expected_conclusion"] != HOLD:
                err(f"{qid}: 保留でない")
            if not unknowns:
                err(f"{qid}: unknowns が空")
            for name in unknowns:
                if facts[name] != "不明":
                    err(f"{qid}: {name} の値が不明でない")
                if name in stated:
                    err(f"{qid}: {name} が stated にある")
            # 2. unknowns の値で割れる
            seen = set()
            for combo in itertools.product(*unknowns.values()):
                g = {**facts, **dict(zip(unknowns, combo))}
                c1, c2 = R["f1"](g), R["f2"](g)
                if c1 != c2:
                    err(f"{qid}: f1 {c1} と f2 {c2} が違う {combo}")
                seen.add(c1)
            if len(seen) != 2:
                err(f"{qid}: unknowns で割れない {seen}")
        else:
            if unknowns or "不明" in facts.values():
                err(f"{qid}: 不明があってはいけない")
            c1, c2 = R["f1"](facts), R["f2"](facts)
            if c1 != c2:
                err(f"{qid}: f1 {c1} と f2 {c2} が違う")
            if c1 != q["expected_conclusion"]:
                err(f"{qid}: 期待 {q['expected_conclusion']} に対し {c1}")

        # 3 と 4. stated だけで読む
        probe = {k: (v if v != "不明" else unknowns[k][0]) for k, v in facts.items()}
        outcomes, splitters = analyse(rule, probe, stated)
        missing_core = [k for k in R["core"] if k not in stated]
        if kind in ("missing_one", "missing_many", "exc_hit", "exc_miss", "dont_know"):
            if len(outcomes) != 2:
                err(f"{qid}: {kind} なのに stated で割れない")
        if kind == "missing_one":
            if len(splitters) != 1 or splitters[0] not in R["core"]:
                err(f"{qid}: missing_one の割る事実が {splitters}")
        if kind == "missing_many":
            if len([s for s in splitters if s in R["core"]]) < 2:
                err(f"{qid}: missing_many の割る事実が {splitters}")
        if kind in ("exc_hit", "exc_miss"):
            if missing_core:
                err(f"{qid}: exc なのに基本の事実が欠ける {missing_core}")
            if not splitters or not set(splitters) <= set(R["exc"]):
                err(f"{qid}: exc の割る事実が {splitters}")
            # 例外を当たらない側に戻して読む
            plain = {**facts, **{k: v for k, v in R["exc"].items() if k not in stated}}
            read = R["f1"](plain)
            if kind == "exc_hit" and read == q["expected_conclusion"]:
                err(f"{qid}: exc_hit なのに書かれた読みと結論が同じ")
            if kind == "exc_miss" and read != q["expected_conclusion"]:
                err(f"{qid}: exc_miss なのに書かれた読みと結論が違う")
        if kind in ("complete", "irrelevant_gap"):
            if outcomes != {q["expected_conclusion"]}:
                err(f"{qid}: {kind} なのに {outcomes}")
            if kind == "complete" and missing_core:
                err(f"{qid}: complete なのに基本の事実が欠ける {missing_core}")
            if kind == "irrelevant_gap" and not missing_core:
                err(f"{qid}: irrelevant_gap なのに基本の事実が全部ある")
        if kind == "dont_know":
            if not (set(unknowns) & set(splitters)):
                err(f"{qid}: unknowns が割る事実でない {splitters}")

        # 全組合せで f1 と f2 が一致
        for a in assignments(rule, probe, list(probe)):
            g = {**probe, **a}
            combos += 1
            if R["f1"](g) != R["f2"](g):
                err(f"{qid}: 全組合せで f1 と f2 が違う {a}")
                break

    # 5. pair
    if len(pairs) != 6:
        err(f"pair の組数 {len(pairs)}")
    for pid, rows in pairs.items():
        (q1, r1), (q2, r2) = rows
        if q1["rule_id"] != q2["rule_id"] or r1["facts"] != r2["facts"] or sorted(r1["stated"]) != sorted(r2["stated"]):
            err(f"{pid}: 事実が同じでない")
        if q1["expected_conclusion"] != q2["expected_conclusion"] or q1["kind"] != "complete" or q2["kind"] != "complete":
            err(f"{pid}: 結論か種類が違う")
        if q1["question"] == q2["question"]:
            err(f"{pid}: 文が同じ")

    # 6. 例外の語の割合と文の長さ
    print("\n種類ごとの例外の語が出る割合 / 文の平均字数")
    per = defaultdict(list)
    table = defaultdict(lambda: defaultdict(list))
    lens = defaultdict(list)
    for q in qs:
        body = q["question"].split("\n")[0]
        hit = re.search(RULES[q["rule_id"]]["words"], body) is not None
        per[q["kind"]].append(hit)
        table[q["rule_id"]][q["kind"]].append(hit)
        lens[q["kind"]].append(len(body))
    for kind in KINDS:
        v = per[kind]
        print(f"  {kind:15s} {sum(v):2d}/{len(v):2d} = {sum(v) / len(v):.0%}  平均 {sum(lens[kind]) / len(lens[kind]):.0f} 字")
    print("規程ごと（complete, exc, missing_one, missing_many, irrelevant_gap, dont_know）")
    for rule in RULES:
        t = table[rule]
        exc = t["exc_hit"] + t["exc_miss"]
        cells = [t["complete"], exc, t["missing_one"], t["missing_many"], t["irrelevant_gap"], t["dont_know"]]
        print(f"  {rule}: " + " ".join(f"{sum(c)}/{len(c)}" for c in cells))

    gaps = stated_value_gaps(qs, fs)
    print(f"\n書いた値の字面が文に無い物（書き方の違い。目で確かめる）: {len(gaps)} 件")
    for g in gaps:
        print(" ", g)
    print(f"\n全組合せの f1/f2 一致を確かめた数: {combos}")
    if ERRORS:
        print(f"\nNG {len(ERRORS)} 件")
        for e in ERRORS:
            print(" ", e)
        sys.exit(1)
    print("全部 OK")


if __name__ == "__main__":
    main()
