# numeric_v8d の検算。facts.jsonl は「問の文に書いた事実」だけを持つ。
# 書いていない事実は DOMAIN（または問ごとの unknowns）のあり得る値を全部入れる。
#   回答できる問: どの値でも結論が 1 つで、expected_conclusion と同じ
#   保留の問: unknowns の値で結論が割れ、それ以外の未記載の事実では割れない
# 式は規程ごとに 2 通り（f1: 上から順に判定 / f2: 妨げの列挙や別の計算）で、全組合せで一致を確かめる。
# 最後に、わざと誤りを入れて check が止まる事を 7 種で確かめる（selftest）。
import copy
import itertools
import json
import re
import sys
from collections import Counter
from datetime import date, timedelta
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).parent
HOLD = "保留"
GROUPS = {"below": 1, "at": 1, "above": 1, "paraphrase": 2, "exception_hit": 1, "exception_near_miss": 1,
          "premise_unmet": 1, "second_condition": 1, "irrelevant_number": 1, "compute": 1,
          "determined_unstated": 1, "hold_missing_value": 1, "hold_unknown_condition": 1, "hold_range_straddle": 1}
BANNED = ["株式会社", "有限会社", "労働基準法", "労働安全衛生法", "民法", "会社法", "高年齢者雇用安定法",
          "育児・介護休業法", "労働契約法"]


def d(s):
    return date.fromisoformat(s)


def add_months(dt, n):
    y, m0 = divmod(dt.year * 12 + dt.month - 1 + n, 12)
    return date(y, m0 + 1, dt.day)


# ------------------------------------------------ 休日出勤振替休日規程
def vac_f1(f):
    if not f["ordered"]:
        return "認めない"
    w = d(f["work_date"])
    if (w - d(f["notify_date"])).days < 3:
        return "認めない"
    m = f["worked_min"]
    cap = 0 if m < 240 else 4 if m < 480 else 8
    if f["take_h"] > cap:
        return "認めない"
    lim = 90 if (f["disaster"] or f["overseas"]) else 60
    n = (d(f["take_date"]) - w).days
    if n < 1 or n > lim:
        return "認めない"
    return "認める"


def vac_f2(f):
    work = d(f["work_date"])
    cap_min = [0, 240, 480][(f["worked_min"] >= 240) + (f["worked_min"] >= 480)]
    last_day = work + timedelta(days=60 + (30 if (f["disaster"] or f["overseas"]) else 0))
    take = d(f["take_date"])
    latest_notice = work - timedelta(days=3)
    blockers = [
        f["ordered"] is not True,
        d(f["notify_date"]) > latest_notice,
        f["take_h"] * 60 > cap_min,
        take <= work,
        take > last_day,
    ]
    return "認めない" if any(blockers) else "認める"


# ------------------------------------------------ 定年後再雇用規程
def age_on(b, ref):
    return ref.year - b.year - ((ref.month, ref.day) < (b.month, b.day))


def re_f1(f):
    ry = f["renew_year"]
    if f["submit"] is None or d(f["submit"]) > date(ry - 1, 12, 31):
        return "終了"
    sk = f["skill"]
    if age_on(d(f["birth"]), date(ry, 4, 1)) >= (67 if sk else 65):
        return "終了"
    if ry - f["first_year"] + 1 > (7 if sk else 5):
        return "終了"
    sc = [60 if s is None else s for s in (f["s1"], f["s2"])]
    if Fraction(sum(sc), 2) < 60:
        return "終了"
    return "更新"


def re_f2(f):
    ry = f["renew_year"]
    renew = date(ry, 4, 1)
    b = d(f["birth"])
    lim_age = 67 if f["skill"] else 65
    turn = date(b.year + lim_age, b.month, b.day)  # 上限の年齢に達する日（誕生日の当日）
    contracts = [y for y in range(f["first_year"], ry + 1)]  # 4月1日に始まる契約の年
    scores = [60 if s is None else s for s in (f["s1"], f["s2"])]
    deadline = date(ry - 1, 12, 31)
    blockers = [
        f["submit"] is None or d(f["submit"]) > deadline,
        turn <= renew,
        len(contracts) > (7 if f["skill"] else 5),
        sum(scores) < 120,
    ]
    return "終了" if any(blockers) else "更新"


# ------------------------------------------------ 制服貸与交換規程
UNI_CAP = {("製造", "上衣"): 3, ("製造", "下衣"): 3, ("製造", "作業帽"): 2,
           ("営業", "上衣"): 2, ("営業", "下衣"): 2, ("営業", "作業帽"): 1}
UNI_CYC = {"上衣": 2, "下衣": 3, "作業帽": 1}


def uni_f1(f):
    if not f["applied"] or not f["returned"]:
        return "自己負担"
    if f["n_req"] > UNI_CAP[(f["dept"], f["item"])]:
        return "自己負担"
    last = d(f["last_lend"])
    due = date(last.year + UNI_CYC[f["item"]], last.month, last.day)
    if d(f["req_date"]) >= due or f["accident"]:
        return "無償"
    return "自己負担"


def uni_f2(f):
    last, req = d(f["last_lend"]), d(f["req_date"])
    months = 12 * {"上衣": 2, "下衣": 3, "作業帽": 1}[f["item"]]
    elapsed = (req.year * 12 + req.month, req.day) >= (last.year * 12 + last.month + months, last.day)
    cap = {"製造": {"上衣": 3, "下衣": 3, "作業帽": 2}, "営業": {"上衣": 2, "下衣": 2, "作業帽": 1}}[f["dept"]][f["item"]]
    blockers = [not f["applied"], not f["returned"], f["n_req"] > cap, not (elapsed or f["accident"])]
    return "自己負担" if any(blockers) else "無償"


# ------------------------------------------------ 夏季休暇規程
def sum_f1(f):
    if f["dpw"] < 4:
        return "取得不可"
    j = d(f["join"])
    if j <= date(2026, 4, 1):
        grant = 5
    else:
        first = j.month if j.day == 1 else j.month + 1
        grant = 5 * (16 - first) // 12
    if f["nights"] >= 8:
        grant += 2
    if f["used"] + f["req_days"] > grant:
        return "取得不可"
    last_ok = date(2026, 10, 31) if f["ext"] else date(2026, 9, 30)
    if d(f["take_from"]) < date(2026, 7, 1) or d(f["take_to"]) > last_ok:
        return "取得不可"
    return "取得可"


def sum_f2(f):
    j = d(f["join"])
    if j <= date(2026, 4, 1):
        grant = 5
    else:
        y, m = j.year, j.month + (0 if j.day == 1 else 1)
        if m > 12:
            y, m = y + 1, 1
        count = 0
        while (y, m) <= (2027, 3):  # 翌年3月まで月を数える
            count += 1
            m += 1
            if m > 12:
                y, m = y + 1, 1
        grant = int(Fraction(5 * count, 12))
    if f["nights"] >= 8:
        grant += 2
    end_limit = date(2026, 9, 30).toordinal() + (31 if f["ext"] else 0)
    blockers = [
        f["dpw"] <= 3,
        f["used"] + f["req_days"] - grant > 0,
        d(f["take_from"]).toordinal() < date(2026, 7, 1).toordinal(),
        d(f["take_to"]).toordinal() > end_limit,
    ]
    return "取得不可" if any(blockers) else "取得可"


# ------------------------------------------------ 契約締結決裁規程
def months_inclusive(s, e):
    return (e.year - s.year) * 12 + e.month - s.month + 1


def con_total_a(f):
    a = f["amount"]
    if isinstance(a, dict):
        return a["usd_month"] * months_inclusive(d(f["start"]), d(f["end"])) * 150
    return a


def con_total_b(f):
    a = f["amount"]
    if isinstance(a, dict):
        s, e = d(f["start"]), d(f["end"])
        n, y, m = 0, s.year, s.month
        while (y, m) <= (e.year, e.month):  # 月を 1 つずつ数える
            n += 1
            m += 1
            if m > 12:
                y, m = y + 1, 1
        return Fraction(a["usd_month"]) * n * 150
    return Fraction(a)


def con_f1(f):
    if not f["pay"]:
        return "部長決裁"
    if f["related"]:
        return "役員決裁"
    if f["base"]:
        return "部長決裁"
    s, e = d(f["start"]), d(f["end"])
    long_term = f["auto"] or e >= date(s.year + 3, s.month, s.day)
    big = con_total_a(f) >= 10000000
    return "役員決裁" if (big or long_term) else "部長決裁"


def con_f2(f):
    s, e = d(f["start"]), d(f["end"])
    days = (e - s).days + 1
    three_years = (date(s.year + 3, s.month, s.day) - s).days
    total = con_total_b(f)
    if f["pay"] is False:
        return "部長決裁"
    needs_exec = [
        f["related"],
        (not f["base"]) and total >= 10000000,
        (not f["base"]) and (f["auto"] or days > three_years),
    ]
    return "役員決裁" if any(needs_exec) else "部長決裁"


# ------------------------------------------------ 騒音作業場保護具規程
BANDS = [(85, 88, 8), (88, 91, 4), (91, 94, 2), (94, 100, 1)]


def eff(f):
    ex = [(Fraction(str(db)), m) for db, m in f["exposures"]]
    if f["booth"] is not None:
        b = Fraction(str(f["booth"]))
        ex = [(b, m) for _, m in ex]
    return ex


def noi_f1(f):
    if not f["designated"]:
        return "不要"
    ex = eff(f)
    mx = max(db for db, _ in ex)
    if mx >= 100:
        return "着用"
    if mx >= 85:
        cyc = 6 if mx >= 91 else 12
        if d(f["today"]) > add_months(d(f["last_check"]), cyc):
            return "着用"
    idx = Fraction(0)
    for db, m in ex:
        for lo, hi, h in BANDS:
            if lo <= db < hi:
                idx += Fraction(m, 60 * h)
    return "着用" if idx >= 1 else "不要"


def noi_f2(f):
    if not f["designated"]:
        return "不要"
    tenths = [(int(db * 10), m) for db, m in eff(f)]  # 0.1 dB 単位の整数
    top = max(t for t, _ in tenths)
    weighted = 0  # 分 × (8 ÷ 許容時間)。480 以上で指数が 1 以上
    for t, m in tenths:
        if 850 <= t < 880:
            weighted += m * 1
        elif 880 <= t < 910:
            weighted += m * 2
        elif 910 <= t < 940:
            weighted += m * 4
        elif 940 <= t < 1000:
            weighted += m * 8
    last, today = d(f["last_check"]), d(f["today"])
    months = 6 if top >= 910 else 12
    passed = (today.year * 12 + today.month, today.day) > (last.year * 12 + last.month + months, last.day)
    blockers = [top >= 1000, top >= 850 and passed, weighted >= 480]
    return "着用" if any(blockers) else "不要"


RULES = {
    "休日出勤振替休日規程": (vac_f1, vac_f2, {
        "ordered": [True, False], "work_date": ["2026-03-08"], "notify_date": ["2026-03-01", "2026-03-05", "2026-03-06"],
        "take_date": ["2026-03-20", "2026-05-08", "2026-06-01"], "worked_min": [200, 239, 240, 479, 480, 600],
        "take_h": [4, 8], "disaster": [True, False], "overseas": [True, False]}),
    "定年後再雇用規程": (re_f1, re_f2, {
        "submit": [None, "2025-12-20", "2026-01-05"], "birth": ["1961-04-01", "1961-04-02", "1963-08-10"],
        "renew_year": [2026], "first_year": [2020, 2022, 2023], "s1": [None, 40, 60, 80], "s2": [None, 40, 60, 80],
        "skill": [True, False]}),
    "制服貸与交換規程": (uni_f1, uni_f2, {
        "item": ["上衣", "下衣", "作業帽"], "dept": ["製造", "営業"], "n_req": [1, 2, 3, 4],
        "last_lend": ["2022-01-10", "2025-05-01"], "req_date": ["2026-06-10"], "accident": [True, False],
        "applied": [True, False], "returned": [True, False]}),
    "夏季休暇規程": (sum_f1, sum_f2, {
        "dpw": [3, 4, 5], "join": ["2020-04-01", "2026-04-02", "2026-06-02", "2026-08-01"],
        "nights": [0, 7, 8, 12], "used": [0, 2], "req_days": [1, 3, 5],
        "take_from": ["2026-07-06", "2026-09-28"], "take_to": ["2026-08-10", "2026-09-30", "2026-10-02", "2026-10-16"],
        "ext": [True, False]}),
    "契約締結決裁規程": (con_f1, con_f2, {
        "pay": [True, False], "amount": [5000000, 9999999, 10000000, 30000000], "start": ["2026-04-01"],
        "end": ["2027-03-31", "2029-03-31", "2029-04-01"], "auto": [True, False], "base": [True, False],
        "related": [True, False]}),
    "騒音作業場保護具規程": (noi_f1, noi_f2, {
        "designated": [True, False],
        "exposures": [[[84.0, 480]], [[86.0, 120]], [[89.0, 240]], [[96.0, 60]], [[100.0, 10]]],
        "booth": [None, 79.0], "last_check": ["2025-01-10", "2026-07-06"], "today": ["2026-11-10"]}),
}

# 「at」の問で、わざと変える事実（selftest 用）。値を 1 単位動かすと結論が変わる事実を選ぶ。
def _plus_day(v):
    return (d(v) + timedelta(days=1)).isoformat()


MUTATE = {
    "休日出勤振替休日規程": ("worked_min", lambda v: v - 1),
    "定年後再雇用規程": ("birth", _plus_day),
    "制服貸与交換規程": ("last_lend", _plus_day),
    "夏季休暇規程": ("req_days", lambda v: v + 1),
    "契約締結決裁規程": ("amount", lambda v: v - 1),
    "騒音作業場保護具規程": ("exposures", lambda v: [[v[0][0], v[0][1] - 1]]),
}


def enumerate_cases(rules, rule, facts, unknowns):
    """facts を固定し、書いていない事実に DOMAIN / unknowns の値を入れる。(割り当て, f1結論, f2結論) を返す。"""
    f1, f2, domain = rules[rule]
    keys = [k for k in domain if k not in facts]
    choices = []
    for k in keys:
        choices.append(unknowns[k] if k in unknowns else domain[k])
    for k in unknowns:
        assert k in domain, f"unknowns のキー {k} は domain に無い"
        assert k not in facts, f"unknowns のキー {k} は facts と重なる"
    for k in facts:
        assert k in domain, f"facts のキー {k} は domain に無い"
    for combo in itertools.product(*choices):
        full = dict(facts)
        full.update(zip(keys, combo))
        yield dict(zip(keys, combo)), f1(full), f2(full)


def num_forms(v):
    """本文に現れてほしい数の書き方の候補（報告用）。"""
    out = set()
    if isinstance(v, bool) or v is None:
        return out
    if isinstance(v, (int, float)):
        if isinstance(v, float):
            out.add(str(v))
            out.add(f"{int(v)}" if v == int(v) else str(v))
            return out
        n = v
        out |= {str(n), f"{n:,}"}
        if n % 10000 == 0:
            man = n // 10000
            out |= {f"{man}万", f"{man:,}万"}
            if man % 1000 == 0:
                out.add(f"{man // 1000}千万")
        if n % 1000 == 0:
            out |= {f"{n // 1000:,}千"}
    elif isinstance(v, str):
        m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", v)
        if m:
            y, mo, da = map(int, m.groups())
            out |= {f"{y}年{mo}月{da}日", f"{y}/{mo}/{da}", f"{mo}月{da}日"}
    return out


def verify(qs, fs, rules, corpus=None, only=None):
    """検査の本体。誤りの一覧と、報告用の数を返す。only を渡すと、その id の問だけ再計算する。"""
    errors = []

    def err(kind, *a):
        errors.append(kind + " " + " ".join(str(x) for x in a))

    # --- 件数・内訳
    if len(qs) != 90:
        err("件数", len(qs))
    ids = [q["id"] for q in qs]
    if len(set(ids)) != len(ids):
        err("id 重複")
    if set(ids) != {f"VD{i:03d}" for i in range(1, 91)}:
        err("id の範囲")
    if ids == sorted(ids):
        err("id が順に並んでいる")
    need = {"id", "question", "rule_id", "should_hold", "expected_conclusion", "conclusion_options", "group",
            "pair_id", "basis"}
    per_rule = Counter(q["rule_id"] for q in qs)
    if len(per_rule) != 6 or set(per_rule.values()) != {15}:
        err("規程ごとの数", per_rule)
    if set(per_rule) != set(rules):
        err("規程名", per_rule)
    for rule in rules:
        if corpus is not None:
            cp = corpus / f"{rule}.md"
            if not cp.exists():
                err("corpus が無い", rule)
            else:
                text = cp.read_text(encoding="utf-8")
                n = len(text.splitlines())
                if not 15 <= n <= 30:
                    err("corpus の行数", rule, n)
                if any(w in text for w in BANNED):
                    err("実在の会社・法令の語", rule)
        groups = Counter(q["group"] for q in qs if q["rule_id"] == rule)
        if groups != Counter(GROUPS):
            err("内訳", rule, groups)
    if set(fs) != set(ids):
        err("facts の id", len(fs))

    # --- 項目・結論の語
    optsets = {}
    for q in qs:
        if set(q) != need:
            err("項目", q["id"], set(q) ^ need)
            continue
        a, b = q["conclusion_options"]
        if a in b or b in a or a == b:
            err("部分文字列", q["id"], a, b)
        if HOLD in a or HOLD in b:
            err("保留を含む語", q["id"])
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
    all_pairs = [next(iter(v)) for v in optsets.values() if len(v) == 1]
    words = [w for p in all_pairs for w in p]
    if len(all_pairs) != 6 or len(set(words)) != 12:
        err("語の組が規程間で重なる", all_pairs)
    for w1, w2 in itertools.permutations(set(words), 2):
        if w1 in w2:
            err("語が別の語の一部", w1, w2)

    # --- 結論の向き・保留の数
    if sum(q["should_hold"] for q in qs) != 18:
        err("保留の数")
    for rule in rules:
        c = Counter(q["expected_conclusion"] for q in qs if q["rule_id"] == rule and not q["should_hold"])
        if sorted(c.values()) != [6, 6]:
            err("向き", rule, c)
        if sum(q["should_hold"] for q in qs if q["rule_id"] == rule) != 3:
            err("規程ごとの保留の数", rule)

    # --- pair_id
    pairs = {}
    for q in qs:
        if q["pair_id"]:
            pairs.setdefault(q["pair_id"], []).append(q)
        elif q["group"] == "paraphrase":
            err("paraphrase に pair_id が無い", q["id"])
    if len(pairs) != 12 or set(pairs) != {f"PD{i:02d}" for i in range(1, 13)}:
        err("pair の数", len(pairs))
    pair_per_rule = Counter(m[0]["rule_id"] for m in pairs.values())
    if set(pair_per_rule.values()) != {2} or len(pair_per_rule) != 6:
        err("規程ごとの pair の数", pair_per_rule)
    for p, m in pairs.items():
        gs = sorted(x["group"] for x in m)
        if len(m) != 2 or "paraphrase" not in gs or gs[0] not in ("above", "at", "below"):
            err("pair の形", p, gs)
            continue
        if len({x["rule_id"] for x in m}) != 1 or len({x["expected_conclusion"] for x in m}) != 1:
            err("pair の結論", p)
        if m[0]["question"].split("\n")[0] == m[1]["question"].split("\n")[0]:
            err("pair の文が同じ", p)
        if m[0]["id"] in fs and m[1]["id"] in fs:
            fa, fb = fs[m[0]["id"]], fs[m[1]["id"]]
            if fa["facts"] != fb["facts"] or fa["unknowns"] != fb["unknowns"]:
                err("pair の事実", p)

    # --- 正解の再計算
    n_checked, n_cases = 0, 0
    for q in qs:
        if only is not None and q["id"] not in only:
            continue
        rule = q["rule_id"]
        if q["id"] not in fs or rule not in rules:
            continue
        rec = fs[q["id"]]
        facts, unknowns = rec["facts"], rec["unknowns"]
        try:
            results = list(enumerate_cases(rules, rule, facts, unknowns))
        except AssertionError as e:
            err("定義の誤り", q["id"], e)
            continue
        except Exception as e:  # 事実の欠けなど
            err("計算できない", q["id"], type(e).__name__, e)
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
                err("正解違い", q["id"], q["group"], outs, q["expected_conclusion"])
        n_checked += 1

    # --- 数の写し間違いの確かめ（書いた数が本文にあるか。目で確かめる）
    unmatched = []
    for q in qs:
        if q["id"] not in fs:
            continue
        text = q["question"].split("\n")[0]
        for k, v in fs[q["id"]]["facts"].items():
            forms = num_forms(v)
            if forms and not any(x in text for x in forms):
                unmatched.append((q["id"], q["group"], k, v))
    return {"errors": errors, "unmatched": unmatched, "n_checked": n_checked, "n_cases": n_cases,
            "pairs": len(pairs)}


def load():
    qs = [json.loads(x) for x in open(HERE / "questions.jsonl", encoding="utf-8")]
    fs = {r["id"]: r for r in (json.loads(x) for x in open(HERE / "facts.jsonl", encoding="utf-8"))}
    return qs, fs


def selftest(qs, fs):
    """わざと誤りを入れ、verify が止まる事を確かめる。検出できなかった物の一覧を返す。"""
    results, missed = [], []

    def run(label, qs2, fs2, rules2, want, only):
        res = verify(qs2, fs2, rules2, only=only)
        hit = [e for e in res["errors"] if e.startswith(want)]
        ok = bool(hit)
        results.append((label, ok, len(res["errors"])))
        if not ok:
            missed.append(label)

    ans = [q for q in qs if not q["should_hold"]]
    hold = [q for q in qs if q["should_hold"]]
    # 1. 結論の反転（回答できる 72 問を 1 問ずつ）
    for q in ans:
        qs2 = copy.deepcopy(qs)
        t = next(x for x in qs2 if x["id"] == q["id"])
        t["expected_conclusion"] = next(w for w in t["conclusion_options"] if w != t["expected_conclusion"])
        run(f"1 結論の反転 {q['id']}", qs2, fs, RULES, "正解違い", {q["id"]})
    # 2. 保留の候補の削り（保留 18 問を 1 問ずつ。unknowns の値を 1 つだけにする）
    for q in hold:
        fs2 = copy.deepcopy(fs)
        u = fs2[q["id"]]["unknowns"]
        k = next(iter(u))
        u[k] = u[k][:1]
        run(f"2 保留の候補の削り {q['id']}", qs, fs2, RULES, "保留が割れない", {q["id"]})
    # 3. 事実の値の変更 / 4. 必要な事実の削除 / 7. 式を 1 つ壊す（規程ごとに「at」の問で）
    for rule, (key, fn) in MUTATE.items():
        at = next(q for q in qs if q["rule_id"] == rule and q["group"] == "at")
        fs2 = copy.deepcopy(fs)
        fs2[at["id"]]["facts"][key] = fn(fs2[at["id"]]["facts"][key])
        run(f"3 事実の値の変更 {rule}", qs, fs2, RULES, "正解違い", {at["id"]})
        fs3 = copy.deepcopy(fs)
        del fs3[at["id"]]["facts"][key]
        run(f"4 必要な事実の削除 {rule}", qs, fs3, RULES, "正解違い", {at["id"]})
        f1, f2, dom = RULES[rule]

        def broken(f, f2=f2, key=key, fn=fn):
            g = dict(f)
            g[key] = fn(g[key])
            return f2(g)

        rules2 = dict(RULES)
        rules2[rule] = (f1, broken, dom)
        run(f"7 式を 1 つ壊す {rule}", qs, fs, rules2, "式の不一致", {at["id"]})
    # 5. id の重複
    qs2 = copy.deepcopy(qs)
    qs2[1]["id"] = qs2[0]["id"]
    run("5 id の重複", qs2, fs, RULES, "id 重複", set())
    # 6. 部分文字列の語
    qs2 = copy.deepcopy(qs)
    a = qs2[0]["conclusion_options"][0]
    qs2[0]["conclusion_options"][1] = a[:-1]
    run("6 部分文字列の語", qs2, fs, RULES, "部分文字列", set())
    kinds = Counter(r[0].split()[0] for r in results)
    return results, missed, kinds


def main():
    qs, fs = load()
    res = verify(qs, fs, RULES, corpus=HERE / "corpus")
    print(f"本文に書き方が見つからない数 {len(res['unmatched'])} 件（下は目で確かめる）")
    for u in res["unmatched"]:
        print("  ", u)
    if res["errors"]:
        print("NG", len(res["errors"]))
        for e in res["errors"]:
            print(" ", e)
        sys.exit(1)
    results, missed, kinds = selftest(qs, fs)
    print(f"件数 {len(qs)} / 回答 {sum(not q['should_hold'] for q in qs)} / 保留 {sum(q['should_hold'] for q in qs)}")
    print(f"pair {res['pairs']} 組 / 検算した問 {res['n_checked']} / 入れた組合せ {res['n_cases']}")
    print("わざと誤りを入れた確かめ: " + " / ".join(
        f"{k}={sum(1 for r in results if r[0].split()[0] == k and r[1])}/{v}" for k, v in sorted(kinds.items())))
    if missed:
        print("NG 見つけられなかった誤り", len(missed))
        for m in missed:
            print(" ", m)
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
