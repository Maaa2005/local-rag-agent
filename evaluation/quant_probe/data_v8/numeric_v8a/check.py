# numeric_v8a の検算。facts.jsonl は「問の文に書いた事実」だけを持つ。
# 書いていない事実は DOMAIN（または問ごとの unknowns）のあり得る値を全部入れる。
#   回答できる問: どの値でも結論が 1 つで、expected_conclusion と同じ
#   保留の問: unknowns の値で結論が割れ、それ以外の未記載の事実では割れない
# 式は規程ごとに 2 通り（f1: 上から順に判定 / f2: 妨げの列挙や別の計算）で、全組合せで一致を確かめる。
# 最後に、わざと誤りを入れた 7 種のデータを読み込んで、全部が止まる事を確かめる。
import calendar
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
          "determined_unstated": 1, "hold_missing_value": 1, "hold_unknown_condition": 1,
          "hold_range_straddle": 1}


def d(s):
    return date.fromisoformat(s)


def eom(y, m):
    return calendar.monthrange(y, m)[1]


# ------------------------------------------------ 通勤定期代支給規程
def km_total(f):
    x = f["dist"]
    if isinstance(x, dict):
        return sum(Fraction(v) for v in x["lines"])
    return Fraction(x)


def com_f1(f):
    if f["mode"] != "transit":
        return "却下"
    km = int(km_total(f))
    if km < 2:
        return "却下"
    if f["route"] == "other":
        return "却下"
    if f["route"] == "designated" and f["outage"] < 14:
        return "却下"
    lim = 8000 if km < 10 else (15000 if km < 25 else 25000)
    return "支給" if f["amount"] <= lim * f["months"] else "却下"


BANDS = [(2, 8000), (10, 15000), (25, 25000)]


def com_f2(f):
    x = km_total(f)
    whole = x.numerator // x.denominator
    limits = [lim for lo, lim in BANDS if whole >= lo]
    limit = max(limits) if limits else None
    bad_route = {"cert": False, "designated": f["outage"] < 14, "other": True}[f["route"]]
    blockers = [f["mode"] != "transit", limit is None, bad_route,
                limit is not None and Fraction(f["amount"], f["months"]) > limit]
    return "却下" if any(blockers) else "支給"


# ------------------------------------------------ 育児短時間勤務規程
def one_month_before(s):
    y, m = (s.year, s.month - 1) if s.month > 1 else (s.year - 1, 12)
    return date(y, m, min(s.day, eom(y, m)))


def add_month(x):
    y, m = (x.year, x.month + 1) if x.month < 12 else (x.year + 1, 1)
    return date(y, m, min(x.day, eom(y, m)))


def kid_f1(f):
    if not f["cohabit"]:
        return "認めない"
    b, s, fl = d(f["birth"]), d(f["start"]), d(f["filed"])
    if s >= date(b.year + 3, b.month, b.day):
        return "認めない"
    w = f["work_min"]
    if not (300 <= w <= 360 and w % 30 == 0):
        return "認めない"
    if fl <= one_month_before(s):
        return "認める"
    if f["unforeseen"] and fl < s:
        return "認める"
    return "認めない"


def kid_f2(f):
    b, s, fl = d(f["birth"]), d(f["start"]), d(f["filed"])
    x = s
    while add_month(x) > s:  # 1か月後が開始日以前になる最も遅い日を、暦を戻って探す
        x -= timedelta(days=1)
    yrs = s.year - b.year - ((s.month, s.day) < (b.month, b.day))
    excused = f["unforeseen"] and fl <= s - timedelta(days=1)
    blockers = [not f["cohabit"], yrs >= 3, f["work_min"] not in (300, 330, 360), fl > x and not excused]
    return "認めない" if any(blockers) else "認める"


# ------------------------------------------------ 社宅入居規程
def rent_of(r):
    if isinstance(r, dict):
        return r["base"] + r["fee"]
    return r


def home_f1(f):
    if f["housing"]:
        return "見送り"
    pct = 20 if f["deps"] else 30
    if rent_of(f["rent"]) * (100 - pct) > 7000000:
        return "見送り"
    m = f["moved_m"]
    if m is not None and m <= 3:
        return "入居"
    return "入居" if f["tenure_m"] // 12 >= 2 and f["commute"] > 60 else "見送り"


def home_f2(f):
    rent = Fraction(rent_of(f["rent"]))
    company = rent - rent * (Fraction(1, 5) if f["deps"] else Fraction(3, 10))
    waived = f["moved_m"] is not None and f["moved_m"] <= 3
    blockers = [f["housing"], company > 70000,
                (not waived) and (f["tenure_m"] < 24 or f["commute"] <= 60)]
    return "見送り" if any(blockers) else "入居"


# ------------------------------------------------ 交際費規程
def total_of(t):
    if isinstance(t, dict):
        return t["food"] + t["service"]
    return t


def ent_f1(f):
    if f["client"] < 1 or f["own"] > f["client"]:
        return "私費扱い"
    t = total_of(f["total"])
    per = t // (f["client"] + f["own"])
    mgr = f["appr"] in ("manager", "director") and f["appr_before"]
    dire = f["appr"] == "director" and f["appr_before"]
    if per > (20000 if dire else 10000):
        return "私費扱い"
    if t >= 50000 and not mgr:
        return "私費扱い"
    return "経費可"


def ent_f2(f):
    t = Fraction(total_of(f["total"]))
    n = f["client"] + f["own"]
    per = int(t / n)
    limit = 10000 + (10000 if (f["appr"] == "director" and f["appr_before"]) else 0)
    unapproved = f["appr"] is None or not f["appr_before"]
    blockers = [f["client"] == 0, f["own"] > f["client"], per > limit, t >= 50000 and unapproved]
    return "私費扱い" if any(blockers) else "経費可"


# ------------------------------------------------ 入館証再発行規程
def acc_f1(f):
    if f["holder"] == "visitor":
        return "受付不可"
    aware, rep = d(f["aware"]), d(f["report"])
    x, n = aware, 0
    while n < 3:
        x += timedelta(days=1)
        if x.weekday() < 5:
            n += 1
    if rep > x:
        return "受付不可"
    if f["stolen"] and f["police"]:
        return "再発行"
    anniv = date(rep.year - 1, rep.month, rep.day)
    cnt = sum(1 for p in f["prior"] if d(p) > anniv and d(p) >= date(2025, 4, 1))
    if cnt >= 3:
        return "受付不可"
    fee = [0, 1000, 3000][cnt]
    return "再発行" if f["cash"] >= fee else "受付不可"


def acc_f2(f):
    aware, rep = d(f["aware"]), d(f["report"])
    elapsed = sum(1 for k in range(1, (rep - aware).days + 1)
                  if (aware + timedelta(days=k)).weekday() < 5)
    anniv = date(rep.year - 1, rep.month, rep.day)
    span = (rep - anniv).days
    cnt = sum(1 for p in f["prior"] if (rep - d(p)).days < span and d(p) >= date(2025, 4, 1))
    fee = {0: 0, 1: 1000, 2: 3000}.get(cnt)
    special = f["stolen"] and f["police"]
    blockers = [f["holder"] == "visitor", elapsed > 3,
                (not special) and (fee is None or f["cash"] < fee)]
    return "受付不可" if any(blockers) else "再発行"


# ------------------------------------------------ 冷蔵倉庫作業規程
def eff_cont(f, rest_need):
    c = f["cont"]
    if isinstance(c, dict):
        return c["after"] if c["rest"] >= rest_need else c["before"] + c["rest"] + c["after"]
    return c


def cold_f1(f):
    t = Fraction(str(f["temp"]))
    if t > 5:
        return "作業継続"
    if t <= -20:
        lim, rest, freezer = 30, 15, True
    elif t <= -5:
        lim, rest, freezer = 60, 10, False
    else:
        lim, rest, freezer = 120, 10, False
    if freezer and f["workers"] < 2:
        return "退庫"
    if f["emerg"]:
        lim *= 2
    return "作業継続" if eff_cont(f, rest) <= lim else "退庫"


def cold_f2(f):
    tf = Fraction(str(f["temp"])) * 9 / 5 + 32  # 華氏に直して表の境目（-4, 23, 41）と比べる
    zone = "free" if tf > 41 else ("freeze" if tf <= -4 else ("low" if tf <= 23 else "cool"))
    table = {"freeze": (30, 15), "low": (60, 10), "cool": (120, 10)}
    if zone == "free":
        return "作業継続"
    lim, rest = table[zone]
    lim = lim * (2 if f["emerg"] else 1)
    blockers = [zone == "freeze" and f["workers"] <= 1, eff_cont(f, rest) - lim > 0]
    return "退庫" if any(blockers) else "作業継続"


RULES = {
    "通勤定期代支給規程": (com_f1, com_f2, {
        "mode": ["transit", "other"], "dist": ["1.5", "5.0", "12.0", "30.0"],
        "route": ["cert", "designated", "other"], "outage": [0, 13, 14, 30],
        "amount": [5000, 9000, 20000], "months": [1, 6]}),
    "育児短時間勤務規程": (kid_f1, kid_f2, {
        "cohabit": [True, False], "birth": ["2023-03-15", "2024-06-15"], "start": ["2026-08-01"],
        "work_min": [300, 330, 360, 390, 420], "filed": ["2026-05-01", "2026-07-20"],
        "unforeseen": [True, False]}),
    "社宅入居規程": (home_f1, home_f2, {
        "housing": [True, False], "tenure_m": [8, 18, 24, 60], "commute": [30, 60, 61, 90],
        "deps": [True, False], "rent": [60000, 100000], "moved_m": [None, 2, 3, 4]}),
    "交際費規程": (ent_f1, ent_f2, {
        "client": [0, 3, 4], "own": [1, 3, 5], "total": [30000, 49999, 50000, 90000],
        "appr": [None, "manager", "director"], "appr_before": [True, False]}),
    "入館証再発行規程": (acc_f1, acc_f2, {
        "holder": ["employee", "visitor"], "aware": ["2026-05-11"], "report": ["2026-05-13", "2026-05-20"],
        "prior": [[], ["2025-06-02", "2025-11-14", "2026-01-20"]], "cash": [0, 1000, 3000, 10000],
        "stolen": [True, False], "police": [True, False]}),
    "冷蔵倉庫作業規程": (cold_f1, cold_f2, {
        "temp": [-30, -25, -20, -12, 0, 8], "cont": [20, 45, 100, 200], "workers": [1, 2],
        "emerg": [True, False]}),
}
# 書いていなければ「無い」と読む事実は置かない（問の文に全て書いた）
ABSENT = {}


def enumerate_cases(rules, rule, facts, unknowns):
    """facts を固定し、書いていない事実に DOMAIN / unknowns の値を入れる。(割り当て, f1結論, f2結論) を返す。"""
    f1, f2, domain = rules[rule]
    keys = [k for k in domain if k not in facts]
    choices = []
    absent = ABSENT.get(rule, {})
    for k in keys:
        if k in unknowns:
            choices.append(unknowns[k])
        elif k in absent:
            choices.append([absent[k]])
        else:
            choices.append(domain[k])
    for k in unknowns:
        assert k in domain, f"unknowns のキー {k} は domain に無い"
        assert k not in facts, f"unknowns のキー {k} は facts と重なる"
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
        n = int(v)
        out |= {str(n), f"{n:,}"}
        if n % 10000 == 0:
            man = n // 10000
            out |= {f"{man}万", f"{man:,}万"}
        if n % 1000 == 0:
            out |= {f"{n // 1000:,}千", f"{n // 10000}万{(n % 10000) // 1000}千"}
    elif isinstance(v, str):
        m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", v)
        if m:
            y, mo, da = map(int, m.groups())
            out |= {f"{y}年{mo}月{da}日", f"{y}/{mo}/{da}"}
        elif re.fullmatch(r"[\d.]+", v):
            out.add(v)
    elif isinstance(v, list):
        for x in v:
            out |= num_forms(x)
    return out


def ms_key(rec):
    return json.dumps(rec, sort_keys=True, ensure_ascii=False)


def validate(qs, fs, rules, root, report=False):
    errors = []

    def err(*a):
        errors.append(" ".join(str(x) for x in a))

    # --- 件数・内訳
    if len(qs) != 90:
        err("件数", len(qs))
    ids = [q["id"] for q in qs]
    if len(set(ids)) != len(ids):
        err("id 重複")
    if set(ids) != {f"VA{i:03d}" for i in range(1, 91)}:
        err("id の範囲")
    if ids == sorted(ids):
        err("id が順に並んでいる")
    if set(fs) != set(ids):
        err("facts の id が問と合わない")
    need = {"id", "question", "rule_id", "should_hold", "expected_conclusion", "conclusion_options", "group",
            "pair_id", "basis"}
    per_rule = Counter(q["rule_id"] for q in qs)
    if len(per_rule) != 6 or set(per_rule.values()) != {15}:
        err("規程ごとの数", per_rule)
    if set(per_rule) != set(rules):
        err("規程名", per_rule)
    files = {p.stem for p in (root / "corpus").glob("*.md")}
    if files != set(rules):
        err("corpus のファイルと規程名が合わない", files ^ set(rules))
    for rule in rules:
        cp = root / "corpus" / f"{rule}.md"
        if not cp.exists():
            err("corpus が無い", rule)
        else:
            n = len(cp.read_text(encoding="utf-8").splitlines())
            if not 15 <= n <= 30:
                err("規程の行数", rule, n)
        groups = Counter(q["group"] for q in qs if q["rule_id"] == rule)
        if groups != Counter(GROUPS):
            err("内訳", rule, groups)
        holds = sum(1 for q in qs if q["rule_id"] == rule and q["should_hold"])
        if holds != 3:
            err("規程ごとの保留の数", rule, holds)

    # --- 項目・結論の語
    optsets = {}
    orders = {}
    for q in qs:
        if set(q) != need:
            err("項目", q["id"], set(q) ^ need)
            continue
        a, b = q["conclusion_options"]
        if a in b or b in a or a == b:
            err("部分文字列", q["id"], a, b)
        if HOLD in a or HOLD in b:
            err("結論の語に保留が入っている", q["id"])
        optsets.setdefault(q["rule_id"], set()).add(frozenset((a, b)))
        orders.setdefault(q["rule_id"], set()).add((a, b))
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
    elif optsets:
        all_pairs = [next(iter(v)) for v in optsets.values()]
        if len(set(all_pairs)) != 6 or len({w for p in all_pairs for w in p}) != 12:
            err("語の組が規程間で重なる", all_pairs)
    if any(len(v) < 2 for v in orders.values()):
        err("選択肢の並びが混ざっていない", {k: len(v) for k, v in orders.items()})

    # --- 結論の向き・保留の数
    if sum(q["should_hold"] for q in qs) != 18:
        err("保留の数")
    for rule in rules:
        c = Counter(q["expected_conclusion"] for q in qs if q["rule_id"] == rule and not q["should_hold"])
        if sorted(c.values()) != [6, 6]:
            err("向き", rule, c)

    # --- pair_id
    pairs = {}
    for q in qs:
        if q["pair_id"]:
            pairs.setdefault(q["pair_id"], []).append(q)
        elif q["group"] == "paraphrase":
            err("paraphrase に pair_id が無い", q["id"])
    if len(pairs) != 12 or set(pairs) != {f"PA{i:02d}" for i in range(1, 13)}:
        err("pair の数", len(pairs))
    pair_rules = Counter(m[0]["rule_id"] for m in pairs.values())
    if set(pair_rules.values()) != {2} or len(pair_rules) != 6:
        err("規程ごとの pair の数", pair_rules)
    for p, m in pairs.items():
        gs = sorted(x["group"] for x in m)
        if len(m) != 2 or "paraphrase" not in gs or gs[0] not in ("above", "at", "below"):
            err("pair の形", p, gs)
            continue
        if len({x["rule_id"] for x in m}) != 1 or len({x["expected_conclusion"] for x in m}) != 1:
            err("pair の結論", p)
        if m[0]["question"].split("\n")[0] == m[1]["question"].split("\n")[0]:
            err("pair の文が同じ", p)
        if all(x["id"] in fs for x in m) and ms_key(fs[m[0]["id"]]) != ms_key(
                {**fs[m[1]["id"]], "id": m[0]["id"]}):
            err("pair の事実", p, fs[m[0]["id"]], fs[m[1]["id"]])

    # --- 正解の再計算
    n_checked, n_cases = 0, 0
    for q in qs:
        if q["id"] not in fs or q["rule_id"] not in rules:
            continue
        rec = fs[q["id"]]
        facts, unknowns = rec["facts"], rec["unknowns"]
        try:
            results = list(enumerate_cases(rules, q["rule_id"], facts, unknowns))
        except (AssertionError, KeyError) as e:
            err("再計算できない", q["id"], repr(e))
            continue
        for asg, r1, r2 in results:
            n_cases += 1
            if r1 != r2:
                err("式の不一致", q["id"], asg, r1, r2)
        outs = {r1 for _, r1, _ in results}
        n_checked += 1
        if q["should_hold"]:
            if len(outs) < 2:
                err("保留が割れない", q["id"], outs)
            if not unknowns:
                err("保留に unknowns が無い", q["id"])
                continue
            byu = {}
            for asg, r1, _ in results:
                key = tuple(json.dumps(asg[k], sort_keys=True) for k in unknowns)
                byu.setdefault(key, set()).add(r1)
            if any(len(v) != 1 for v in byu.values()):
                err("保留が unknowns 以外でも割れる", q["id"], byu)
            if len({next(iter(v)) for v in byu.values() if v}) < 2:
                err("unknowns で割れない", q["id"], byu)
        else:
            if unknowns and q["group"] != "determined_unstated":
                err("回答できる問に unknowns がある", q["id"])
            if q["group"] == "determined_unstated" and not unknowns:
                err("determined_unstated に unknowns が無い", q["id"])
            if outs != {q["expected_conclusion"]}:
                err("正解違い", q["id"], q["group"], outs, q["expected_conclusion"], facts)

    # --- 数の写し間違いの確かめ（書いた数が本文にあるか）
    unmatched = []
    for q in qs:
        if q["id"] not in fs:
            continue
        text = q["question"].split("\n")[0]
        for k, v in fs[q["id"]]["facts"].items():
            forms = num_forms(v)
            if forms and not any(x in text for x in forms):
                unmatched.append((q["id"], q["group"], k, v))
    if report:
        return errors, dict(n_checked=n_checked, n_cases=n_cases, pairs=len(pairs), unmatched=unmatched)
    return errors, None


def load(root):
    qs = [json.loads(x) for x in open(root / "questions.jsonl", encoding="utf-8")]
    fs = {r["id"]: r for r in (json.loads(x) for x in open(root / "facts.jsonl", encoding="utf-8"))}
    return qs, fs


# ------------------------------------------------ わざと誤りを入れる（7 種）
def find(qs, rule, group):
    return next(q for q in qs if q["rule_id"] == rule and q["group"] == group)


def mut_flip(qs, fs, rules):
    q = find(qs, "通勤定期代支給規程", "below")
    a, b = q["conclusion_options"]
    q["expected_conclusion"] = b if q["expected_conclusion"] == a else a


def mut_cut_unknowns(qs, fs, rules):
    q = find(qs, "社宅入居規程", "hold_range_straddle")
    u = fs[q["id"]]["unknowns"]
    k = next(iter(u))
    u[k] = u[k][:1]


def mut_change_fact(qs, fs, rules):
    q = find(qs, "通勤定期代支給規程", "at")
    fs[q["id"]]["facts"]["amount"] += 1000


def mut_delete_fact(qs, fs, rules):
    q = find(qs, "通勤定期代支給規程", "at")
    del fs[q["id"]]["facts"]["amount"]


def mut_dup_id(qs, fs, rules):
    qs[1]["id"] = qs[0]["id"]


def mut_substring(qs, fs, rules):
    q = find(qs, "通勤定期代支給規程", "above")
    q["conclusion_options"] = ["支給", "支給外"]
    lines = q["question"].split("\n")
    lines[-1] = "最後の行には「結論: 支給」または「結論: 支給外」のどちらかだけを書いてください。"
    q["question"] = "\n".join(lines)


def mut_break_formula(qs, fs, rules):
    f1, f2, dom = rules["冷蔵倉庫作業規程"]
    rules["冷蔵倉庫作業規程"] = (lambda f: "作業継続", f2, dom)


MUTATIONS = [("結論の反転", mut_flip), ("保留の候補の削り", mut_cut_unknowns), ("事実の値の変更", mut_change_fact),
             ("必要な事実の削除", mut_delete_fact), ("id の重複", mut_dup_id), ("部分文字列の語", mut_substring),
             ("式を 1 つ壊す", mut_break_formula)]


def run_mutations(qs, fs, rules, root):
    bad = 0
    for name, fn in MUTATIONS:
        q2, f2_, r2 = copy.deepcopy(qs), copy.deepcopy(fs), dict(rules)
        fn(q2, f2_, r2)
        errs, _ = validate(q2, f2_, r2, root)
        status = "止まった" if errs else "止まらなかった"
        print(f"  誤り「{name}」: {status}（{len(errs)} 件）")
        if not errs:
            bad += 1
    return bad


def main():
    root = HERE
    qs, fs = load(root)
    errors, st = validate(qs, fs, RULES, root, report=True)
    print(f"件数 {len(qs)} / 回答 {sum(not q['should_hold'] for q in qs)} / 保留 {sum(q['should_hold'] for q in qs)}")
    print(f"pair {st['pairs']} 組 / 検算した問 {st['n_checked']} / 入れた組合せ {st['n_cases']}")
    print(f"本文に書き方が見つからない数 {len(st['unmatched'])} 件（下は目で確かめる）")
    for u in st["unmatched"]:
        print("  ", u)
    if errors:
        print("NG", len(errors))
        for e in errors:
            print(" ", e)
        sys.exit(1)
    print("誤りを入れた確かめ:")
    bad = run_mutations(qs, fs, RULES, root)
    if bad:
        print("NG 誤りを見逃した", bad)
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
