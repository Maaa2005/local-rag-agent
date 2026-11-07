# ask90 の検算。facts.jsonl は全部の事実を持ち、stated が問の文に書いた事実の名前。
# 文に無い事実にあり得る値を全部入れて、規程ごとの式 2 通り（f1 / f2）で結論を出す。
import itertools
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

HERE = Path(__file__).parent
HOLD = "保留"
UNK = "不明"
KINDS = {"complete": 4, "exc_hit": 1, "exc_miss": 1, "missing_one": 2, "missing_many": 3, "irrelevant_gap": 1, "dont_know": 3}


def d(s):
    return date.fromisoformat(s)


def anniversary(s, years):
    x = d(s)
    return x.replace(year=x.year + years)


def minutes(s):
    t = datetime.strptime(s, "%H:%M")
    return t.hour * 60 + t.minute


# ---------------------------- 独身寮入寮規程
def dorm_f1(f):
    if f["雇用形態"] != "正社員" or f["配偶者と同居"]:
        return "見送り"
    hope = d(f["入寮希望月初日"])
    if f["被災認定日"] and hope <= anniversary(f["被災認定日"], 1):
        return "入寮"
    if f["障害者手帳"] and f["産業医の軽減認定"]:
        return "入寮"
    if f["通勤分"] >= 90 and hope < anniversary(f["生年月日"], 35):
        return "入寮"
    return "見送り"


def dorm_f2(f):
    b, h = d(f["生年月日"]), d(f["入寮希望月初日"])
    age = h.year - b.year - ((h.month, h.day) < (b.month, b.day))
    base = f["通勤分"] > 89 and age <= 34
    c = f["被災認定日"]
    special = False
    if c:
        c = d(c)
        last = date(c.year + 1, c.month, c.day)
        special = (last - h).days >= 0
    special = special or (f["障害者手帳"] is True and f["産業医の軽減認定"] is True)
    ok = f["雇用形態"] == "正社員" and not f["配偶者と同居"]
    return "入寮" if ok and (base or special) else "見送り"


# ---------------------------- 予防接種費用補助規程
def vac_f1(f):
    if not f["在籍"] or f["接種者"] != "本人":
        return "自己負担"
    x = d(f["接種日"])
    if (d(f["提出日"]) - x).days > 30:
        return "自己負担"
    if f["妊娠中"] or f["基礎疾患の勧奨"]:
        return "補助"
    in_season = x.month >= 10 or x.month == 1
    cost10 = f["領収書額"] * (11 if f["税区分"] == "税抜" else 10)
    return "補助" if in_season and cost10 <= 50000 else "自己負担"


def vac_f2(f):
    x = d(f["接種日"])
    start = date(x.year if x.month >= 2 else x.year - 1, 10, 1)
    end = date(start.year + 1, 1, 31)
    amount = f["領収書額"] * 1.1 if f["税区分"] == "税抜" else f["領収書額"]
    limit_ok = round(amount, 6) <= 5000
    waived = bool(f["妊娠中"]) or bool(f["基礎疾患の勧奨"])
    blockers = []
    if not f["在籍"]:
        blockers.append("在籍")
    if f["接種者"] == "家族":
        blockers.append("本人でない")
    if d(f["提出日"]) > x + timedelta(days=30):
        blockers.append("提出")
    if not waived and not (start <= x <= end):
        blockers.append("期間")
    if not waived and not limit_ok:
        blockers.append("額")
    return "自己負担" if blockers else "補助"


# ---------------------------- 語学研修選抜規程
RANK = ["D", "C", "B", "A", "S"]


def lang_f1(f):
    if f["研修修了歴"]:
        return "落選"
    if f["海外赴任内示"]:
        return "選抜"
    close = date(f["応募年"], 4, 30)
    if f["雇用形態"] != "正社員" or anniversary(f["入社日"], 3) > close:
        return "落選"
    start = close.replace(year=close.year - 2)
    best = max([p for day, p in f["検定履歴"] if start <= d(day) <= close], default=-1)
    need = 700
    r = f["育休復職日"]
    if r and d(r) <= close <= anniversary(r, 1):
        need = 600
    return "選抜" if best >= need and RANK.index(f["人事評価"]) >= RANK.index("B") else "落選"


def lang_f2(f):
    y = f["応募年"]
    ymd = lambda s: tuple(int(p) for p in s.split("-"))
    hired = ymd(f["入社日"])
    tenure_ok = (hired[0] + 3, hired[1], hired[2]) <= (y, 4, 30)
    scores = [p for day, p in f["検定履歴"] if (y - 2, 4, 30) <= ymd(day) <= (y, 4, 30)]
    reduced = False
    if f["育休復職日"]:
        r = ymd(f["育休復職日"])
        reduced = r <= (y, 4, 30) <= (r[0] + 1, r[1], r[2])
    passes = any(p >= (600 if reduced else 700) for p in scores) and f["人事評価"] in ("S", "A", "B")
    normal = f["雇用形態"] == "正社員" and tenure_ok and passes
    return "選抜" if not f["研修修了歴"] and (f["海外赴任内示"] or normal) else "落選"


# ---------------------------- 休日出勤振替規程
def hol_f1(f):
    if not f["事前命令"]:
        return "精算"
    if f["時間外制限"]:
        return "振替"
    worked = minutes(f["終了時刻"]) - minutes(f["開始時刻"]) - f["休憩分"]
    need = 180 if f["短時間勤務"] else 240
    within = (d(f["取得予定日"]) - d(f["出勤日"])).days <= 56
    return "振替" if worked >= need and within else "精算"


def hol_f2(f):
    # 時間を小数の時間で比べ、期限の日を週で出す
    worked_h = (minutes(f["終了時刻"]) - f["休憩分"] - minutes(f["開始時刻"])) / 60
    last = d(f["出勤日"]) + timedelta(weeks=8)
    conds = [worked_h >= (3 if f["短時間勤務"] else 4), d(f["取得予定日"]) <= last]
    result = "振替" if all(conds) or f["時間外制限"] else "精算"
    return result if f["事前命令"] else "精算"


# ---------------------------- 海外出張渡航規程
def trip_f1(f):
    if not f["出張命令"]:
        return "延期"
    limit = 2
    if f["家族在住"]:
        limit = 3
    if f["妊娠中"]:
        limit = 1
    if f["危険度"] > limit:
        return "延期"
    if f["救援要員"]:
        return "渡航"
    return "渡航" if d(f["届出日"]) <= d(f["出発日"]) - timedelta(days=14) else "延期"


def trip_f2(f):
    limit = 1 if f["妊娠中"] else (3 if f["家族在住"] else 2)
    days_before = (d(f["出発日"]) - d(f["届出日"])).days
    blockers = [not f["出張命令"], f["危険度"] >= limit + 1, days_before < 14 and not f["救援要員"]]
    return "延期" if any(blockers) else "渡航"


# ---------------------------- 業務端末持出規程
def dev_f1(f):
    if not f["台帳登録"]:
        return "禁止"
    if f["顧客情報件数"] >= 1000 or not f["暗号化"]:
        return "禁止"
    if f["両立支援認定"]:
        return "持出"
    days = (d(f["返却日"]) - d(f["持出日"])).days + 1
    return "持出" if days <= (30 if f["事業継続要員"] else 7) else "禁止"


def dev_f2(f):
    # 返却が許される最後の日を出して比べる
    if f["両立支援認定"]:
        period = True
    else:
        last = d(f["持出日"]) + timedelta(days=(29 if f["事業継続要員"] else 6))
        period = d(f["返却日"]) <= last
    all_ok = f["台帳登録"] and f["顧客情報件数"] <= 999 and f["暗号化"] and period
    return "持出" if all_ok else "禁止"


def rel(key, delta):
    def g(t):
        return [] if t[key] == UNK else [str(d(t[key]) + timedelta(days=delta))]
    return g


def rels(key, *deltas):
    def g(t):
        return sum((rel(key, x)(t) for x in deltas), [])
    return g


def add_min(t, m):
    if t["開始時刻"] == UNK:
        return []
    x = minutes(t["開始時刻"]) + m
    return [f"{x // 60:02d}:{x % 60:02d}"]


BOOL = lambda t: [True, False]
RULES = {
    "独身寮入寮規程": {
        "f": (dorm_f1, dorm_f2),
        "positive": "入寮", "negative": "見送り",
        "off": {"被災認定日": None, "障害者手帳": False, "産業医の軽減認定": False},
        "domain": {
            "雇用形態": lambda t: ["正社員", "契約社員"], "配偶者と同居": BOOL,
            "通勤分": lambda t: [30, 89, 90, 150],
            "生年月日": lambda t: [str(anniversary(t["入寮希望月初日"], -25)), str(anniversary(t["入寮希望月初日"], -35)),
                               str(anniversary(t["入寮希望月初日"], -35) + timedelta(days=1)), str(anniversary(t["入寮希望月初日"], -45))],
            "入寮希望月初日": lambda t: [], "被災認定日": rels("入寮希望月初日", -60, -500),
            "障害者手帳": BOOL, "産業医の軽減認定": BOOL,
        },
        "word": {
            "雇用形態": "正社員|契約社員|雇用", "配偶者と同居": "独身|配偶者|妻|夫|結婚|未婚",
            "通勤分": "片道|通勤|経路検索", "生年月日": "生まれ|生年月日|歳", "入寮希望月初日": r"\d月",
            "被災認定日": "半壊|被災|災害|認定", "障害者手帳": "手帳", "産業医の軽減認定": "産業医",
        },
    },
    "予防接種費用補助規程": {
        "f": (vac_f1, vac_f2),
        "positive": "補助", "negative": "自己負担",
        "off": {"妊娠中": False, "基礎疾患の勧奨": False},
        "domain": {
            "在籍": BOOL, "接種者": lambda t: ["本人", "家族"],
            "接種日": lambda t: ["2026-08-20", "2026-11-01", "2026-12-20"],
            "領収書額": lambda t: [3000, 4700, 5000, 8000], "税区分": lambda t: ["税込", "税抜"],
            "提出日": rels("接種日", 3, 30, 31, 60), "妊娠中": BOOL, "基礎疾患の勧奨": BOOL,
        },
        "word": {
            "在籍": "入社|在籍|勤続", "接種者": "本人|自分|私|妻|夫|息子|娘|母|父|家族",
            "接種日": r"\d月\d+日[^。]*(接種|受け)|接種日", "領収書額": r"\d[\d,千]*円", "税区分": "税",
            "提出日": "提出|へ出|出しま", "妊娠中": "妊娠", "基礎疾患の勧奨": "産業医|基礎疾患",
        },
    },
    "語学研修選抜規程": {
        "f": (lang_f1, lang_f2),
        "positive": "選抜", "negative": "落選",
        "off": {"海外赴任内示": False, "育休復職日": None, "研修修了歴": False},
        "domain": {
            "雇用形態": lambda t: ["正社員", "契約社員"],
            "入社日": lambda t: ["2010-04-01", "2024-04-30", "2024-05-01", "2026-04-01"],
            "応募年": lambda t: [],
            "検定履歴": lambda t: [[["2026-06-01", 550]], [["2026-06-01", 650]], [["2026-06-01", 800]], [["2024-06-01", 800]], []],
            "人事評価": lambda t: ["S", "B", "C", "D"], "海外赴任内示": BOOL,
            "育休復職日": lambda t: [None, "2026-09-01", "2024-09-01"], "研修修了歴": BOOL,
        },
        "word": {
            "雇用形態": "正社員|契約社員", "入社日": "入社|勤続", "応募年": "募集|応募",
            "検定履歴": "検定", "人事評価": "評価", "海外赴任内示": "内示|赴任",
            "育休復職日": "育児休業|育休|復職", "研修修了歴": "修了|受けた事",
        },
    },
    "休日出勤振替規程": {
        "f": (hol_f1, hol_f2),
        "positive": "振替", "negative": "精算",
        "off": {"短時間勤務": False, "時間外制限": False},
        "domain": {
            "事前命令": BOOL, "出勤日": lambda t: [],
            "開始時刻": lambda t: [],
            "終了時刻": lambda t: add_min(t, 60) + add_min(t, 200) + add_min(t, 480),
            "休憩分": lambda t: [0, 30, 60, 90],
            "取得予定日": rels("出勤日", 10, 56, 57, 90), "短時間勤務": BOOL, "時間外制限": BOOL,
        },
        "word": {
            "事前命令": "命令", "出勤日": "出勤", "開始時刻": r"\d時",
            "終了時刻": "まで", "休憩分": "休憩", "取得予定日": "代わりの休み|いつ休",
            "短時間勤務": "短時間", "時間外制限": "制限",
        },
    },
    "海外出張渡航規程": {
        "f": (trip_f1, trip_f2),
        "positive": "渡航", "negative": "延期",
        "off": {"家族在住": False, "妊娠中": False, "救援要員": False},
        "domain": {
            "出張命令": BOOL, "危険度": lambda t: [1, 2, 3, 4], "出発日": lambda t: [],
            "届出日": rels("出発日", -30, -14, -13, -3), "家族在住": BOOL, "妊娠中": BOOL, "救援要員": BOOL,
        },
        "word": {
            "出張命令": "命令|部長", "危険度": "危険度", "出発日": r"\d日[^。]*(出発|発)",
            "届出日": "届", "家族在住": "配偶者|妻|夫|子", "妊娠中": "妊娠", "救援要員": "救援",
        },
    },
    "業務端末持出規程": {
        "f": (dev_f1, dev_f2),
        "positive": "持出", "negative": "禁止",
        "off": {"事業継続要員": False, "両立支援認定": False},
        "domain": {
            "台帳登録": BOOL, "持出日": lambda t: [],
            "返却日": rels("持出日", 2, 6, 7, 29, 30, 60),
            "顧客情報件数": lambda t: [50, 999, 1000, 5000], "暗号化": BOOL, "事業継続要員": BOOL, "両立支援認定": BOOL,
        },
        "word": {
            "台帳登録": "台帳", "持出日": r"\d日に(社外へ)?持ち出", "返却日": "返却|戻",
            "顧客情報件数": "顧客情報|件", "暗号化": "暗号", "事業継続要員": "事業継続|要員", "両立支援認定": "両立",
        },
    },
}

errors = []


def err(qid, msg):
    errors.append(f"{qid}: {msg}")


def keyed(v):
    return json.dumps(v, ensure_ascii=False, sort_keys=True)


def values_of(rule, name, true_facts, row):
    vals = []
    if name in row["unknowns"]:
        vals += row["unknowns"][name]
    vals += RULES[rule]["domain"][name](true_facts)
    if true_facts[name] != UNK:
        vals.append(true_facts[name])
    out, seen = [], set()
    for v in vals:
        if keyed(v) not in seen:
            seen.add(keyed(v))
            out.append(v)
    return out


def evaluate(rule, qid, f):
    a, b = (fn(f) for fn in RULES[rule]["f"])
    if a != b:
        err(qid, f"f1 と f2 が違う: {a} / {b} {keyed(f)}")
    return a


def sweep(rule, qid, base, names, row, true_facts):
    """names の全組合せの結論。戻りは {組合せ: 結論}"""
    pools = [values_of(rule, n, true_facts, row) for n in names]
    for n, p in zip(names, pools):
        if not p:
            err(qid, f"{n} に入れる値が無い")
    res = {}
    for combo in itertools.product(*pools):
        f = dict(base)
        f.update(zip(names, combo))
        res[tuple(keyed(c) for c in combo)] = evaluate(rule, qid, f)
    return res


def splitters(names, res):
    out = []
    for i, n in enumerate(names):
        groups = defaultdict(set)
        for combo, c in res.items():
            groups[combo[:i] + combo[i + 1:]].add(c)
        if any(len(g) > 1 for g in groups.values()):
            out.append(n)
    return out


def main():
    qs = [json.loads(x) for x in open(HERE / "questions.jsonl", encoding="utf-8")]
    fs = {r["id"]: r for r in map(json.loads, open(HERE / "facts.jsonl", encoding="utf-8"))}
    combos = 0

    # 5. 件数と形
    if len(qs) != 90 or len(fs) != 90:
        err("all", f"件数 {len(qs)} / {len(fs)}")
    if sorted(q["id"] for q in qs) != [f"A{i:03d}" for i in range(1, 91)]:
        err("all", "id が A001〜A090 でない")
    by_rule = defaultdict(list)
    for q in qs:
        by_rule[q["rule_id"]].append(q)
    if set(by_rule) != set(RULES):
        err("all", f"規程の名前 {sorted(by_rule)}")
    for rule, rows in by_rule.items():
        c = Counter(q["kind"] for q in rows)
        if c != Counter(KINDS):
            err(rule, f"種類の内訳 {dict(c)}")
        spec = RULES[rule]
        v = Counter(q["expected_conclusion"] for q in rows if q["kind"] != "dont_know")
        if v[spec["positive"]] != 6 or v[spec["negative"]] != 6:
            err(rule, f"結論の向き {dict(v)}")
        hit = [q for q in rows if q["kind"] == "exc_hit"]
        miss = [q for q in rows if q["kind"] == "exc_miss"]
        if hit and miss and hit[0]["expected_conclusion"] == miss[0]["expected_conclusion"]:
            err(rule, "exc_hit と exc_miss の結論が同じ")
        pairs = defaultdict(list)
        for q in rows:
            if q["pair_id"]:
                pairs[q["pair_id"]].append(q)
        if len(pairs) != 1:
            err(rule, f"pair_id が {len(pairs)} 組")
        for pid, ps in pairs.items():
            if len(ps) != 2 or any(p["kind"] != "complete" for p in ps):
                err(rule, f"{pid} の組が complete の 2 問でない")
                continue
            a, b = ps
            if keyed(fs[a["id"]]["facts"]) != keyed(fs[b["id"]]["facts"]) or fs[a["id"]]["stated"] != fs[b["id"]]["stated"]:
                err(pid, "組の事実が違う")
            if a["expected_conclusion"] != b["expected_conclusion"]:
                err(pid, "組の結論が違う")
            if a["question"].split("\n")[0] == b["question"].split("\n")[0]:
                err(pid, "組の文が同じ")

    # 1〜4. 場面ごと
    for q in qs:
        qid, rule, kind, row = q["id"], q["rule_id"], q["kind"], fs.get(q["id"])
        if row is None or rule not in RULES:
            err(qid, "事実か規程が無い")
            continue
        spec = RULES[rule]
        facts, stated, unknowns = row["facts"], row["stated"], row["unknowns"]
        expected = q["expected_conclusion"]
        opts = q["conclusion_options"]
        if set(opts) != {spec["positive"], spec["negative"]} or len(opts) != 2:
            err(qid, f"選択肢 {opts}")
        last = q["question"].split("\n")[-1]
        if last != f"最後の行には「結論: {opts[0]}」または「結論: {opts[1]}」のどちらかだけを書いてください。":
            err(qid, "最後の行の書式")
        if q["group"] != kind or q["should_hold"] != (kind == "dont_know"):
            err(qid, "group か should_hold が違う")
        if (kind == "dont_know") != (expected == HOLD):
            err(qid, "保留の印が違う")
        if set(facts) != set(spec["domain"]):
            err(qid, f"事実の名前が欠ける・余る: {set(facts) ^ set(spec['domain'])}")
            continue
        if not set(stated) <= set(facts) or len(set(stated)) != len(stated):
            err(qid, f"stated が事実にない: {stated}")
            continue
        bad_stated = [s for s in stated if facts[s] == UNK]
        if bad_stated:
            err(qid, f"不明の事実が stated にある: {bad_stated}")
        unk_names = [k for k, v in facts.items() if v == UNK]
        if set(unk_names) != set(unknowns):
            err(qid, f"不明の事実と unknowns が合わない: {unk_names} / {list(unknowns)}")
        if kind != "dont_know" and (unk_names or unknowns):
            err(qid, "dont_know 以外に不明がある")

        # 文に書いた事実の語は文にあり、書いていない事実の語は文に無い（推測の手がかりも防ぐ）
        body = q["question"].split("\n")[0]
        for name, pat in spec["word"].items():
            found = re.search(pat, body) is not None
            if name in stated and not found:
                err(qid, f"stated の {name} が文に見つからない")
            if name not in stated and found:
                err(qid, f"書いていない {name} の語が文にある: {re.search(pat, body).group(0)}")
            if name not in stated and isinstance(facts[name], int) and not isinstance(facts[name], bool) and facts[name] >= 10:
                if re.search(rf"(?<!\d){facts[name]:,}|(?<!\d){facts[name]}(?!\d)", body):
                    err(qid, f"書いていない {name} の値が文にある")

        # 全部の事実で結論を出す
        if kind == "dont_know":
            res = sweep(rule, qid, facts, unk_names, row, facts) if unk_names else {}
            combos += len(res)
            if len(set(unknowns)) < 1 or {c for c in res.values()} != {spec["positive"], spec["negative"]}:
                err(qid, f"不明の値で結論が割れない: {set(res.values())}")
            # unknowns に書いた値だけでも割れる
            only = [evaluate(rule, qid, {**facts, unk_names[0]: v}) for v in unknowns.get(unk_names[0], [])] if unk_names else []
            if len(set(only)) != 2:
                err(qid, f"unknowns の値だけで割れない: {only}")
        else:
            got = evaluate(rule, qid, facts)
            if got != expected:
                err(qid, f"正解 {expected} / 式 {got}")

        # stated だけのとき
        missing = [k for k in facts if k not in stated]
        res = sweep(rule, qid, facts, missing, row, facts)
        combos += len(res)
        concl = set(res.values())
        split = splitters(missing, res)
        exc_names = set(spec["off"])
        if kind in ("missing_one", "missing_many", "exc_hit", "exc_miss", "dont_know") and len(concl) != 2:
            err(qid, f"stated だけで結論が割れない: {concl}")
        if kind == "missing_one" and len(split) != 1:
            err(qid, f"結論を割る欠けが 1 つでない: {split}")
        if kind == "missing_many" and len(split) < 2:
            err(qid, f"結論を割る欠けが 2 つ未満: {split}")
        if kind in ("complete", "irrelevant_gap") and (len(concl) != 1 or expected not in concl):
            err(qid, f"stated だけで決まらない: {concl} 割る事実 {split}")
        if kind == "complete" and not set(missing) <= exc_names:
            err(qid, f"complete で基準の事実が欠ける: {set(missing) - exc_names}")
        if kind == "irrelevant_gap" and not set(missing) - exc_names:
            err(qid, "irrelevant_gap で基準の事実が欠けていない")
        if kind in ("exc_hit", "exc_miss"):
            if not set(missing) <= exc_names or set(stated) & exc_names:
                err(qid, f"例外の形が違う: 欠け {missing}")
            reading = evaluate(rule, qid, {**facts, **spec["off"]})
            if kind == "exc_hit" and reading == expected:
                err(qid, "exc_hit なのに、書かれた事だけの読みが正解と同じ")
            if kind == "exc_miss" and reading != expected:
                err(qid, "exc_miss なのに、書かれた事だけの読みが正解と違う")

    if errors:
        print("\n".join(errors))
        print(f"NG: {len(errors)} 件")
        sys.exit(1)
    print(f"OK: 90 場面、確かめた組合せ {combos}")
    for rule, rows in by_rule.items():
        print(rule, dict(Counter(q["kind"] for q in rows)), dict(Counter(q["expected_conclusion"] for q in rows)))


if __name__ == "__main__":
    main()
