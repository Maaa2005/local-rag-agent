# numeric_v8b の検算。facts.jsonl は「問の文に書いた事実」だけを持つ。
# 書いていない事実は DOMAIN（または問ごとの unknowns）のあり得る値を全部入れる。
#   回答できる問: どの値でも結論が 1 つで、expected_conclusion と同じ
#   保留の問: unknowns の値で結論が割れ、それ以外の未記載の事実では割れない
# 式は規程ごとに 2 通り（f1: 上から順に判定 / f2: 別の考え方）で、全組合せで一致を確かめる。
# 最後に、わざと誤りを入れて check が止まる事（自己テスト 7 種）も確かめる。
import copy
import itertools
import json
import math
import re
import sys
from collections import Counter
from datetime import date, datetime, time, timedelta
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).parent
HOLD = "保留"
GROUPS = {"below": 1, "at": 1, "above": 1, "paraphrase": 2, "exception_hit": 1, "exception_near_miss": 1,
          "premise_unmet": 1, "second_condition": 1, "irrelevant_number": 1, "compute": 1,
          "determined_unstated": 1, "hold_missing_value": 1, "hold_unknown_condition": 1, "hold_range_straddle": 1}
BANNED = ["株式会社", "有限会社", "労働基準法", "育児・介護休業法", "労働安全衛生法", "民法", "会社法", "所得税法"]


def d(s):
    return date.fromisoformat(s)


def fr(x):
    return Fraction(str(x))


# ------------------------------------------------ 海外出張日当規程
KOU = {"米国", "英国", "ドイツ", "フランス", "スイス"}


def trip_f1(f):
    cs = f["countries"]
    if all(c == "日本" for c in cs):
        return "部長決裁"
    rate = 8000 if any(c in KOU for c in cs) else 5000
    n = (d(f["arr"]) - d(f["dep"])).days + 1
    units = Fraction(1, 2) if n == 1 else n - 1
    total = 0 if f["host"] == "full" else rate * units
    if f["alert"]:
        return "役員決裁"
    return "役員決裁" if total >= 100000 else "部長決裁"


def trip_f2(f):
    # 日ごとに重みを足していく。日付を 1 日ずつ歩く
    cs = f["countries"]
    domestic = all(c == "日本" for c in cs)
    r = max(8000 if c in KOU else 5000 for c in cs)
    days = [d(f["dep"]) + timedelta(days=i) for i in range((d(f["arr"]) - d(f["dep"])).days + 1)]
    total = Fraction(0)
    for i in range(len(days)):
        w = Fraction(1, 2) if i in (0, len(days) - 1) else Fraction(1)
        total += r * w
    paid = f["host"] != "full"
    need = (not domestic) and (f["alert"] or (paid and total >= 100000))
    return "役員決裁" if need else "部長決裁"


# ------------------------------------------------ 介護休業規程
ELIG = {"配偶者", "事実婚の相手", "父", "母", "子", "配偶者の父", "配偶者の母", "祖父", "祖母", "兄", "姉", "弟", "妹",
        "孫"}


def care_f1(f):
    if f["rel"] not in ELIG or f["care_days"] < 14:
        return "却下"
    ud, ut = (0, 0) if f["renewed"] else (f["used_days"], f["used_times"])
    if ud + f["req_days"] > 93 or ut + 1 > 3:
        return "却下"
    if (d(f["start"]) - d(f["submit"])).days < 14:
        return "却下"
    return "承認"


def care_f2(f):
    left_days = 93 - (0 if f["renewed"] else f["used_days"])
    left_times = 3 - (0 if f["renewed"] else f["used_times"])
    deadline = d(f["start"]) - timedelta(days=14)
    ok = [f["rel"] in ELIG, f["care_days"] >= 14, f["req_days"] <= left_days, left_times >= 1,
          d(f["submit"]) <= deadline]
    return "承認" if all(ok) else "却下"


# ------------------------------------------------ 単身赴任帰省旅費規程
CAP = [12000, 36000, 70000]
TRIPS = [2, 2, 1]


def trav_f1(f):
    if f["dest"] != "family":
        return "差戻し"
    km = math.ceil(fr(f["dist"]))
    b = 0 if km < 100 else (1 if km < 400 else 2)
    lim = TRIPS[b] + (1 if f["hosp"] == "admitted" else 0)
    if f["n"] > lim:
        return "差戻し"
    return "支給" if f["claim"] <= CAP[b] * f["n"] else "差戻し"


def trav_f2(f):
    # 切り上げを「引いて切り捨て」で行い、1 往復あたりの実費で比べる
    x = fr(f["dist"])
    km = -((-x.numerator) // x.denominator)
    table = [(400, 1, 70000), (100, 2, 36000), (0, 2, 12000)]
    for lo, cnt, cap in table:
        if km >= lo:
            break
    cnt += 1 if f["hosp"] == "admitted" else 0
    blockers = [f["dest"] != "family", f["n"] > cnt, Fraction(f["claim"], f["n"]) > cap]
    return "差戻し" if any(blockers) else "支給"


# ------------------------------------------------ 会議室予約規程
RANGE = {"小": (1, 6), "中": (1, 12), "大": (15, 30)}


def is_bd(x):
    return x.weekday() < 5 and not (x.month == 12 and x.day >= 29) and not (x.month == 1 and x.day <= 3)


def room_f1(f):
    if f["kind"] == "cancel":
        x, k = d(f["usage"]), 0
        while k < 2:
            x -= timedelta(days=1)
            if is_bd(x):
                k += 1
        cut = datetime(x.year, x.month, x.day, 17, 0)
        return "受付" if datetime.strptime(f["cancel_at"], "%Y-%m-%d %H:%M") <= cut else "断り"
    if f["purpose"] != "work":
        return "断り"
    lo, hi = RANGE[f["room"]]
    if not lo <= f["people"] <= hi:
        return "断り"
    if f["exec"]:
        return "受付"
    total = f["minutes"] + (f["prev"] if f["prev"] > 0 and f["gap"] < 30 else 0)
    return "受付" if total <= 180 else "断り"


def room_f2(f):
    if f["kind"] == "cancel":
        at = datetime.strptime(f["cancel_at"], "%Y-%m-%d %H:%M")
        c, u = at.date(), d(f["usage"])
        # 取消しの日の翌日から利用日までに、営業日がいくつあるか
        cnt = sum(is_bd(c + timedelta(days=i)) for i in range(1, (u - c).days + 1))
        ok = cnt > 2 or (cnt == 2 and is_bd(c) and at.time() <= time(17, 0))
        return "受付" if ok else "断り"
    lo, hi = RANGE[f["room"]]
    merged = f["minutes"] + (f["prev"] if (f["prev"] > 0 and f["gap"] < 30) else 0)
    blockers = [f["purpose"] != "work", f["people"] < lo, f["people"] > hi, (not f["exec"]) and merged > 180]
    return "断り" if any(blockers) else "受付"


# ------------------------------------------------ 試用期間規程
def add_months(x, m):
    return date(x.year + (x.month - 1 + m) // 12, (x.month - 1 + m) % 12 + 1, x.day)


def prob_f1(f):
    if f["grp"]:
        return "本採用"
    if f["hire"] is not None and d(f["eff"]) <= add_months(d(f["hire"]), 3) - timedelta(days=1):
        return "見送り"
    denom = f["sched"] - f["inj"]
    th = Fraction(9, 10) if f["stage"] == 1 else Fraction(8, 10)
    if Fraction(f["present"], denom) < th:
        return "見送り"
    return "本採用" if f["grade"] in ("A", "B") else "見送り"


def prob_f2(f):
    # 日付は（年*12+月, 日）の組で比べ、出勤率は百分率の整数式で比べる
    if f["grp"]:
        return "本採用"
    early = False
    if f["hire"] is not None:
        h, e = d(f["hire"]), d(f["eff"])
        early = (e.year * 12 + e.month, e.day) < (h.year * 12 + h.month + 3, h.day)
    need = 90 if f["stage"] == 1 else 80
    short = f["present"] * 100 < need * (f["sched"] - f["inj"])
    rank = {"A": 4, "B": 3, "C": 2, "D": 1}[f["grade"]]
    return "見送り" if (early or short or rank < 3) else "本採用"


# ------------------------------------------------ 高所作業許可規程
def wind_ms(w):
    if isinstance(w, dict):
        return fr(w["knot"]) / 2
    return fr(w)


def hs_f1(f):
    h = fr(f["height"])
    if h < 2:
        return "着手"
    w = wind_ms(f["wind"])
    if h < 5:
        need_c, need_w, cap = 1, 0, 10
    elif h < 10:
        need_c, need_w, cap = 2, 1, 10
    else:
        need_c, need_w, cap = 2, 2, 8
    if f["fixed"] and f["mins"] <= 15:
        need_c = 0
    have = {"none": 0, "special": 1, "skill": 2}[f["cert"]]
    if have < need_c or f["wit"] < need_w or w >= cap:
        return "待機"
    return "着手"


def hs_f2(f):
    # 風速をノットに直して比べる。高さは段の番号にして表で引く
    h = fr(f["height"])
    step = (h >= 2) + (h >= 5) + (h >= 10)
    if step == 0:
        return "着手"
    knots = wind_ms(f["wind"]) * 2
    cert_need = [0, 1, 2, 2][step]
    wit_need = [0, 0, 1, 2][step]
    cap_knots = [0, 20, 20, 16][step]
    if f["fixed"] and f["mins"] <= 15:
        cert_need = 0
    cert_have = {"none": 0, "special": 1, "skill": 2}[f["cert"]]
    blockers = [cert_have < cert_need, f["wit"] < wit_need, not knots < cap_knots]
    return "待機" if any(blockers) else "着手"


# 破った式（自己テスト用）。事実の 1 つを 1 だけずらしてから本来の式を呼ぶ
def _shift(f1, key, fn):
    return lambda f: f1({**f, key: fn(f[key])})


RULES = {
    "海外出張日当規程": dict(
        f1=trip_f1, f2=trip_f2,
        domain={"countries": [["米国"], ["タイ"]], "dep": ["2026-03-02"], "arr": ["2026-03-05", "2026-03-25"],
                "host": ["none", "part", "full"], "alert": [True, False]},
        absent={}, brk=("arr", lambda v: (d(v) - timedelta(days=1)).isoformat())),
    "介護休業規程": dict(
        f1=care_f1, f2=care_f2,
        domain={"rel": ["母", "叔母"], "care_days": [10, 60], "used_days": [0, 60, 90], "used_times": [0, 2, 3],
                "renewed": [True, False], "req_days": [10, 40, 98], "start": ["2026-07-01"],
                "submit": ["2026-06-10", "2026-06-25"]},
        absent={}, brk=("req_days", lambda v: v + 1)),
    "単身赴任帰省旅費規程": dict(
        f1=trav_f1, f2=trav_f2,
        domain={"dest": ["family", "parents"], "dist": [50, 99.5, 250, 399.5, 450], "n": [1, 2, 3],
                "claim": [10000, 36000, 52000, 80000], "hosp": ["none", "admitted", "outpatient"]},
        absent={}, brk=("dist", lambda v: v - 1)),
    "会議室予約規程": dict(
        f1=room_f1, f2=room_f2,
        domain={"kind": ["book", "cancel"], "room": ["小", "中", "大"], "people": [3, 12, 13, 20],
                "minutes": [60, 180, 240], "prev": [0, 120], "gap": [0, 20, 30, 120], "purpose": ["work", "private"],
                "exec": [True, False], "usage": ["2026-06-15", "2026-06-17"],
                "cancel_at": ["2026-06-12 10:00", "2026-06-13 09:00", "2026-06-15 16:50"]},
        absent={"purpose": "work", "exec": False}, brk=("minutes", lambda v: v + 1)),
    "試用期間規程": dict(
        f1=prob_f1, f2=prob_f2,
        domain={"hire": [None], "eff": [None], "stage": [1, 2], "sched": [60], "inj": [0, 6],
                "present": [40, 54, 60], "grade": ["A", "B", "C", "D"], "grp": [True, False]},
        absent={"hire": None, "eff": None, "stage": 1, "inj": 0, "grp": False},
        brk=("present", lambda v: v - 1)),
    "高所作業許可規程": dict(
        f1=hs_f1, f2=hs_f2,
        domain={"height": [1.8, 3, 7, 12], "cert": ["none", "special", "skill"], "wit": [0, 1, 2],
                "wind": [3, 9.9, 12], "fixed": [True, False], "mins": [10, 15, 16, 60]},
        absent={"fixed": False, "mins": 999}, brk=("wind", lambda v: v - 1)),
}
for _r in RULES.values():
    _r["f1_ok"] = _r["f1"]


def enumerate_cases(rules, rule, facts, unknowns):
    """facts を固定し、書いていない事実に DOMAIN / unknowns の値を入れる。(割り当て, f1結論, f2結論) を返す。"""
    spec = rules[rule]
    domain, absent = spec["domain"], spec["absent"]
    for k in facts:
        assert k in domain, f"{rule}: facts のキー {k} は domain に無い"
    for k in unknowns:
        assert k in domain, f"unknowns のキー {k} は domain に無い"
        assert k not in facts, f"unknowns のキー {k} は facts と重なる"
    keys = [k for k in domain if k not in facts]
    choices = []
    for k in keys:
        if k in unknowns:
            choices.append(unknowns[k])
        elif k in absent:
            choices.append([absent[k]])
        else:
            choices.append(domain[k])
    for combo in itertools.product(*choices):
        full = dict(facts)
        full.update(zip(keys, combo))
        yield dict(zip(keys, combo)), spec["f1"](full), spec["f2"](full)


def pair_norm(rule, facts):
    out = dict(facts)
    if "wind" in out:
        out["wind"] = str(wind_ms(out["wind"]))
    return out


def num_forms(v):
    """本文に現れてほしい数の書き方の候補（報告用）。"""
    out = set()
    if isinstance(v, bool) or v is None:
        return out
    if isinstance(v, (int, float)):
        if float(v).is_integer():
            n = int(v)
            out |= {str(n), f"{n:,}"}
            if n % 10000 == 0:
                out.add(f"{n // 10000}万")
            if n % 1000 == 0:
                out.add(f"{n // 1000}千")
        else:
            out.add(str(v))
    elif isinstance(v, str):
        m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", v)
        if m:
            y, mo, da = map(int, m.groups())
            out |= {f"{y}年{mo}月{da}日", f"{mo}月{da}日"}
        m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2})", v)
        if m:
            y, mo, da, hh, mm = map(int, m.groups())
            out |= {f"{mo}月{da}日"}
    elif isinstance(v, dict) and "knot" in v:
        out.add(str(v["knot"]))
    elif isinstance(v, list):
        for x in v:
            out.add(x)
    return out


# ------------------------------------------------ 構造の確かめ
def check_structure(qs, fs, rules, root=HERE):
    errors = []

    def err(*a):
        errors.append(" ".join(str(x) for x in a))

    if len(qs) != 90:
        err("件数", len(qs))
    ids = [q["id"] for q in qs]
    if len(set(ids)) != len(ids):
        err("id 重複")
    if set(ids) != {f"VB{i:03d}" for i in range(1, 91)}:
        err("id の範囲")
    if ids == sorted(ids):
        err("id が順に並んでいる")
    if set(fs) != set(ids):
        err("facts の id が questions と合わない")
    need = {"id", "question", "rule_id", "should_hold", "expected_conclusion", "conclusion_options", "group",
            "pair_id", "basis"}
    per_rule = Counter(q["rule_id"] for q in qs)
    if len(per_rule) != 6 or set(per_rule.values()) != {15}:
        err("規程ごとの数", per_rule)
    if set(per_rule) != set(rules):
        err("規程名", per_rule)
    for rule in rules:
        cp = root / "corpus" / f"{rule}.md"
        if not cp.exists():
            err("corpus が無い", rule)
        else:
            text = cp.read_text(encoding="utf-8")
            n = len(text.splitlines())
            if not 15 <= n <= 30:
                err("corpus の行数", rule, n)
            if any(w in text for w in BANNED):
                err("実在の会社・法令を思わせる語", rule)
            if not any(w in text for w in ("以上", "以下", "未満", "超")):
                err("数値の境目の語が無い", rule)
        groups = Counter(q["group"] for q in qs if q["rule_id"] == rule)
        if groups != Counter(GROUPS):
            err("内訳", rule, groups)
    corpus_files = {p.stem for p in (root / "corpus").glob("*.md")}
    if corpus_files != set(rules):
        err("corpus のファイル名と rule_id が合わない", corpus_files ^ set(rules))

    optsets = {}
    for q in qs:
        if set(q) != need:
            err("項目", q["id"], set(q) ^ need)
        a, b = q["conclusion_options"]
        if a in b or b in a or a == b or HOLD in a or HOLD in b:
            err("部分文字列または保留を含む語", q["id"], a, b)
        optsets.setdefault(q["rule_id"], set()).add(frozenset((a, b)))
        last = q["question"].split("\n")[-1]
        if last != f"最後の行には「結論: {a}」または「結論: {b}」のどちらかだけを書いてください。":
            err("最後の行", q["id"])
        if q["question"].count("\n") != 1:
            err("改行", q["id"])
        if q["should_hold"]:
            if q["expected_conclusion"] != HOLD or not q["group"].startswith("hold_"):
                err("保留の項目", q["id"])
        else:
            if q["expected_conclusion"] not in q["conclusion_options"] or q["group"].startswith("hold_"):
                err("回答の項目", q["id"])
    if any(len(v) != 1 for v in optsets.values()):
        err("規程内で語の組が揃っていない", optsets)
    else:
        words = [w for v in optsets.values() for w in next(iter(v))]
        if len(set(words)) != 12:
            err("語の組が規程間で重なる", words)
        for x, y in itertools.permutations(words, 2):
            if x in y:
                err("語が別の語の部分文字列", x, y)

    if sum(q["should_hold"] for q in qs) != 18:
        err("保留の数")
    for rule in rules:
        if Counter(q["should_hold"] for q in qs if q["rule_id"] == rule) != Counter({False: 12, True: 3}):
            err("保留の数（規程ごと）", rule)
        c = Counter(q["expected_conclusion"] for q in qs if q["rule_id"] == rule and not q["should_hold"])
        if sorted(c.values()) != [6, 6]:
            err("向き", rule, c)

    pairs = {}
    for q in qs:
        if q["pair_id"]:
            pairs.setdefault(q["pair_id"], []).append(q)
        elif q["group"] == "paraphrase":
            err("paraphrase に pair_id が無い", q["id"])
    if set(pairs) != {f"PB{i:02d}" for i in range(1, 13)}:
        err("pair の名前", sorted(pairs))
    for p, m in pairs.items():
        gs = sorted(x["group"] for x in m)
        if len(m) != 2 or "paraphrase" not in gs or gs[0] not in ("above", "at", "below"):
            err("pair の形", p, gs)
            continue
        if len({x["rule_id"] for x in m}) != 1 or len({x["expected_conclusion"] for x in m}) != 1:
            err("pair の結論", p)
        if m[0]["question"].split("\n")[0] == m[1]["question"].split("\n")[0]:
            err("pair の文が同じ", p)
        fa, fb = fs[m[0]["id"]], fs[m[1]["id"]]
        rule = m[0]["rule_id"]
        if pair_norm(rule, fa["facts"]) != pair_norm(rule, fb["facts"]) or fa["unknowns"] != fb["unknowns"]:
            err("pair の事実", p, fa, fb)
    for rule in rules:
        c = Counter(next(x["group"] for x in pairs[p] if x["group"] != "paraphrase")
                    for p in pairs if pairs[p][0]["rule_id"] == rule)
        if sum(c.values()) != 2:
            err("規程ごとの pair は 2 組", rule, c)
    return errors


# ------------------------------------------------ 正解の再計算
def check_semantics(qs, fs, rules, only=None):
    errors, n_cases = [], 0

    def err(*a):
        errors.append(" ".join(str(x) for x in a))

    for q in qs:
        if only is not None and q["id"] not in only:
            continue
        rule = q["rule_id"]
        rec = fs[q["id"]]
        facts, unknowns = rec["facts"], rec["unknowns"]
        try:
            results = list(enumerate_cases(rules, rule, facts, unknowns))
        except (AssertionError, KeyError, TypeError, ValueError, ZeroDivisionError) as e:
            err("計算できない", q["id"], repr(e))
            continue
        for asg, r1, r2 in results:
            n_cases += 1
            if r1 != r2:
                err("式の不一致", q["id"], asg, r1, r2)
        outs = {r1 for _, r1, _ in results}
        if q["should_hold"]:
            if len(outs) < 2:
                err("保留が割れない", q["id"], outs)
            if not unknowns:
                err("保留に unknowns が無い", q["id"])
            byu = {}
            for asg, r1, _ in results:
                key = tuple(json.dumps(asg[k], sort_keys=True) for k in unknowns)
                byu.setdefault(key, set()).add(r1)
            if any(len(v) != 1 for v in byu.values()):
                err("保留が unknowns 以外でも割れる", q["id"], byu)
            if len({next(iter(v)) for v in byu.values() if len(v) == 1}) < 2:
                err("unknowns で割れない", q["id"], byu)
        else:
            if outs != {q["expected_conclusion"]}:
                err("正解違い", q["id"], q["group"], outs, q["expected_conclusion"], facts)
    return errors, n_cases


# ------------------------------------------------ 自己テスト（わざと誤りを入れて止まるか）
def self_test(qs, fs, rules):
    by_rule = {}
    for q in qs:
        by_rule.setdefault(q["rule_id"], {})[q["group"]] = q
    fails, table = [], {}

    def rec(kind, rule, detected):
        table.setdefault(kind, []).append(detected)
        if not detected:
            fails.append(f"{kind}: {rule} で止まらなかった")

    for rule in rules:
        at = by_rule[rule]["at"]
        hold = by_rule[rule]["hold_missing_value"]
        # 1 結論の反転
        qs2 = copy.deepcopy(qs)
        t = next(x for x in qs2 if x["id"] == at["id"])
        t["expected_conclusion"] = next(w for w in t["conclusion_options"] if w != t["expected_conclusion"])
        e, _ = check_semantics(qs2, fs, rules, only={at["id"]})
        rec("結論の反転", rule, bool(e))
        # 2 保留の候補の削り
        fs2 = copy.deepcopy(fs)
        k = next(iter(fs2[hold["id"]]["unknowns"]))
        fs2[hold["id"]]["unknowns"][k] = fs2[hold["id"]]["unknowns"][k][:1]
        e, _ = check_semantics(qs, fs2, rules, only={hold["id"]})
        rec("保留の候補の削り", rule, bool(e))
        # 3 事実の値の変更（数・日付・真偽を 1 つずつ動かして、1 つでも止まれば良い）
        hit = False
        for key, v in fs[at["id"]]["facts"].items():
            if isinstance(v, bool):
                alts = [not v]
            elif isinstance(v, (int, float)):
                alts = [v + 1, v - 1]
            elif isinstance(v, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
                alts = [(d(v) + timedelta(days=1)).isoformat(), (d(v) - timedelta(days=1)).isoformat()]
            else:
                continue
            for a in alts:
                fs2 = copy.deepcopy(fs)
                fs2[at["id"]]["facts"][key] = a
                e, _ = check_semantics(qs, fs2, rules, only={at["id"]})
                hit = hit or bool(e)
        rec("事実の値の変更", rule, hit)
        # 4 必要な事実の削除
        hit = False
        for key in fs[at["id"]]["facts"]:
            fs2 = copy.deepcopy(fs)
            del fs2[at["id"]]["facts"][key]
            e, _ = check_semantics(qs, fs2, rules, only={at["id"]})
            hit = hit or bool(e)
        rec("必要な事実の削除", rule, hit)
        # 7 式を 1 つ壊す（f1 の入力を 1 だけずらす）
        key, fn = rules[rule]["brk"]
        rules2 = dict(rules)
        rules2[rule] = dict(rules[rule], f1=_shift(rules[rule]["f1"], key, fn))
        e, _ = check_semantics([x for x in qs if x["rule_id"] == rule], fs, rules2)
        rec("式を1つ壊す", rule, bool(e))
    # 5 id の重複
    qs2 = copy.deepcopy(qs)
    qs2[1]["id"] = qs2[0]["id"]
    rec("idの重複", "全体", bool([x for x in check_structure(qs2, fs, rules) if "id" in x]))
    # 6 部分文字列の語
    qs2 = copy.deepcopy(qs)
    a = qs2[0]["conclusion_options"][0]
    qs2[0]["conclusion_options"] = [a, a + "ず"]
    rec("部分文字列の語", "全体", bool([x for x in check_structure(qs2, fs, rules) if "部分文字列" in x]))
    return fails, table


def main():
    root = HERE
    qs = [json.loads(x) for x in open(root / "questions.jsonl", encoding="utf-8")]
    fs = {r["id"]: r for r in (json.loads(x) for x in open(root / "facts.jsonl", encoding="utf-8"))}
    errors = check_structure(qs, fs, RULES, root)
    sem_errors, n_cases = check_semantics(qs, fs, RULES)
    errors += sem_errors

    unmatched = []
    for q in qs:
        text = q["question"].split("\n")[0]
        for k, v in fs[q["id"]]["facts"].items():
            forms = num_forms(v)
            if forms and not any(x in text for x in forms):
                unmatched.append((q["id"], q["group"], k, v))

    print(f"件数 {len(qs)} / 回答 {sum(not q['should_hold'] for q in qs)} / 保留 {sum(q['should_hold'] for q in qs)}")
    print(f"pair {len({q['pair_id'] for q in qs if q['pair_id']})} 組 / 検算した問 {len(qs)} / 入れた組合せ {n_cases}")
    print(f"本文に書き方が見つからない数 {len(unmatched)} 件")
    if "-v" in sys.argv:
        for u in unmatched:
            print("  ", u)

    fails, table = self_test(qs, fs, RULES)
    for kind, res in table.items():
        print(f"自己テスト {kind}: {sum(res)}/{len(res)} 検出")
    errors += fails
    if errors:
        print("NG", len(errors))
        for e in errors:
            print(" ", e)
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
