"""numeric_dev90c の検算。

別の書き方の式で全問を再計算し、questions.jsonl の正解・build.py の式と突き合わせる。
保留は結論が割れる事、それ以外は割れない事を確かめる。
件数・内訳・id・pair_id・語・事実の値が問の文にある事も確かめる。
"""
import calendar
import datetime as dt
import itertools
import json
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build  # noqa: E402

OLD_WORDS = {"もらえる", "もらえない", "出る", "出ない", "支給", "見送り", "受理", "却下", "必要", "不要", "補助可", "補助不可"}
OLD_TOPICS = ["慶弔", "食堂", "在宅", "副業", "社外講師", "再検査"]
SINGLE = ["below", "at", "above", "exception_hit", "exception_near_miss", "premise_unmet", "second_condition",
          "irrelevant_number", "compute", "determined_unstated", "hold_missing_value", "hold_unknown_condition",
          "hold_range_straddle"]
errors = []


def err(*a):
    errors.append(" ".join(str(x) for x in a))


def day_no(s):
    y, m, dd = map(int, s.split("-"))
    return dt.date(y, m, dd).toordinal()


# ---- 別の書き方の式 ----
def alt_taxi(f):
    """落ちる理由を順に探す。時刻は (日, 時, 分) で比べる。"""
    t = (f["leave"] // 1440, (f["leave"] % 1440) // 60, f["leave"] % 60)
    begin = (0, 21, 0) if f["pregnant"] is True else (0, 22, 30)
    reasons = []
    if f["approved"] is not True:
        reasons.append("no approval")
    if t < begin or t > (1, 5, 0):
        reasons.append("time")
    if f["fare"] > 15000:
        reasons.append("fare")
    return not reasons


def fy_label(s):
    y, m, _ = map(int, s.split("-"))
    return y - (m < 4)


def alt_exam(f):
    """年度を番号で表して足す。上限超えは必須資格だけ免れる。"""
    within_limit = sum(a for s, a in f["prior"] if fy_label(s) == fy_label(f["pass_date"])) + f["fee"] <= 6 * 10**4
    days = day_no(f["apply_date"]) - day_no(f["pass_date"])
    return bool(f["regular"]) and 1 <= days <= 90 and (within_limit or bool(f["required"]))


def alt_resort(f):
    """繁忙期の日を集合で作り、上限を表で引く。"""
    y = int(f["checkin"][:4])
    busy = set()
    for yy in (y - 1, y):
        a, b = dt.date(yy, 7, 20), dt.date(yy, 8, 31)
        busy |= {a.toordinal() + i for i in range((b - a).days + 1)}
        a, b = dt.date(yy, 12, 28), dt.date(yy + 1, 1, 3)
        busy |= {a.toordinal() + i for i in range((b - a).days + 1)}
    ci = day_no(f["checkin"])
    protected = False
    if f["award"]:
        a = f["award"]
        one_year = f"{int(a[:4]) + 1}{a[4:]}"
        protected = day_no(a) <= ci <= day_no(one_year)
    limit = {True: 2, False: 4}[ci in busy and not protected]
    return f["accompany"] is True and ci - day_no(f["apply"]) >= 7 and f["nights"] <= limit


def alt_mycar(f):
    """キロと円に直して比べる。"""
    km = Fraction(f["dist_m"], 1000)
    yen = f["coverage_man"] * 10000
    far = not (km < 2)
    return all([f["probation_done"], yen >= 30000000, far or f["handicap"]])


def alt_circle(f):
    """小さい方を取らず、A と B の両方が 3,000 円以上かで見る（同じ意味）。"""
    if f["month"] == 4 and f["welcome"] is True:
        enough = f["members"] >= 3
    else:
        enough = f["members"] > 4
    a_ok = int(Fraction(f["cost"], 2)) >= 3000
    b_ok = f["members"] * 1000 >= 3000
    return bool(f["registered"]) and f["minutes"] > 89 and enough and a_ok and b_ok


def alt_refresh(f):
    """入社日から 1 か月ずつ進めて数える。"""
    y, m, _ = map(int, f["start"].split("-"))
    base = dt.date(y if m >= 4 else y - 1, 4, 1)
    hy, hm, hd = map(int, f["hire"].split("-"))
    n = 0
    while True:
        k = hm - 1 + n + 1
        ny, nm = hy + k // 12, k % 12 + 1
        nd = min(hd, calendar.monthrange(ny, nm)[1])
        if dt.date(ny, nm, nd) > base:
            break
        n += 1
    return f["regular"] is True and f["days_before"] > 13 and n + f["transfer_months"] > 119


ALT = {
    "深夜帰宅タクシー利用規程": alt_taxi,
    "資格受験料補助規程": alt_exam,
    "保養所利用規程": alt_resort,
    "マイカー通勤規程": alt_mycar,
    "社内サークル助成規程": alt_circle,
    "リフレッシュ休暇規程": alt_refresh,
}


# ---- 事実の値が問の文にあるか ----
def money_forms(n):
    forms = {f"{n:,}", str(n)}
    man, rest = divmod(n, 10000)
    if man:
        if rest == 0:
            forms.add(f"{man}万")
        else:
            if rest % 1000 == 0:
                forms.add(f"{man}万{rest // 1000}千")
                forms.add(f"{man}.{rest // 1000}万")
            forms.add(f"{man}万{rest:,}")
    elif n % 1000 == 0:
        forms.add(f"{n // 1000}千")
    if not man and n % 1000:
        forms.add(f"{n // 1000}千{n % 1000}")
    return forms


def clock_forms(v):
    h, m = (v % 1440) // 60, v % 60
    forms = {f"{h}時{m}分", f"{h}時{m:02d}分"}
    if m == 0:
        forms.add(f"{h}時")
    if m == 30:
        forms.add(f"{h}時半")
    if h > 12:
        forms |= {f"午後{h - 12}時{m}分", f"午後{h - 12}時半" if m == 30 else f"午後{h - 12}時{m}分"}
    return forms


def date_forms(s):
    y, m, dd = map(int, s.split("-"))
    forms = {f"{m}月{dd}日"}
    if y >= 2019:
        forms.add(f"令和{y - 2018}年{m}月{dd}日")
    if 1989 <= y < 2019:
        forms.add(f"平成{y - 1988}年{m}月{dd}日")
    return forms


def dist_forms(n):
    forms = {f"{n:,}メートル", f"{n}メートル"}
    km = Fraction(n, 1000)
    txt = str(float(km)).rstrip("0").rstrip(".")
    forms |= {f"{txt}キロ"}
    return forms


def duration_forms(n):
    forms = {f"{n}分"}
    h, m = divmod(n, 60)
    if m == 0:
        forms.add(f"{h}時間")
    else:
        forms.add(f"{h}時間{m}分")
    if m == 30:
        forms.add(f"{h}時間半")
    return forms


def cov_forms(n):
    forms = {f"{n:,}万", f"{n}万"}
    if n % 10000 == 0:
        forms.add(f"{n // 10000}億")
    if n % 1000 == 0:
        forms.add(f"{n // 1000}千万")
    return forms


FORM = {
    "leave": clock_forms, "fare": money_forms, "fee": money_forms,
    "pass_date": date_forms, "apply_date": date_forms, "checkin": date_forms, "apply": date_forms, "start": date_forms,
    "hire": date_forms, "nights": lambda n: {f"{n}泊"}, "dist_m": dist_forms, "coverage_man": cov_forms,
    "minutes": duration_forms, "members": lambda n: {f"{n}人"}, "cost": money_forms,
    "days_before": lambda n: {f"{n}日前"},
}


def check_text(q, f, skip):
    text = q["question"]
    for k, v in f.items():
        if k in skip or (isinstance(v, dict) and "any" in v):
            continue
        if k in FORM:
            if not any(x in text for x in FORM[k](v)):
                err("文に値が無い", q["id"], q["rule_id"], k, v)
        if k == "month" and v != 6 and f"{v}月" not in text:
            err("文に値が無い", q["id"], "month", v)
        if k == "prior":
            for s, a in v:
                if not any(x in text for x in money_forms(a)):
                    err("文に値が無い", q["id"], "prior", a)


def main():
    rows = [json.loads(line) for line in (HERE / "questions.jsonl").read_text(encoding="utf-8").splitlines() if line]
    facts = json.loads((HERE / "facts.json").read_text(encoding="utf-8"))

    # 件数・内訳
    if len(rows) != 90:
        err("件数", len(rows))
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        err("id の重複")
    if set(ids) != set(facts):
        err("facts と id が合わない")
    if sum(r["should_hold"] for r in rows) != 18:
        err("保留の数")
    required_keys = ["id", "question", "rule_id", "should_hold", "expected_conclusion", "conclusion_options", "group",
                     "pair_id", "basis"]
    by_rule = {}
    for r in rows:
        if list(r) != required_keys:
            err("項目", r["id"], list(r))
        by_rule.setdefault(r["rule_id"], []).append(r)
    if set(by_rule) != set(build.RULES):
        err("規程の組", set(by_rule))
    for rule, qs in by_rule.items():
        if len(qs) != 15:
            err("規程の問の数", rule, len(qs))
        g = Counter(q["group"] for q in qs)
        if g["paraphrase"] != 2:
            err("言い換えの数", rule, g["paraphrase"])
        for name in SINGLE:
            if g[name] != 1:
                err("内訳", rule, name, g[name])
        fn, yes, no = build.RULES[rule]
        c = Counter(q["expected_conclusion"] for q in qs)
        if c[yes] != 6 or c[no] != 6 or c["保留"] != 3:
            err("向きの数", rule, dict(c))
        if not (HERE / "corpus" / f"{rule}.md").exists():
            err("規程の文が無い", rule)
        if any(t in rule for t in OLD_TOPICS):
            err("dev90b と同じ話題", rule)
    if len(list((HERE / "corpus").glob("*.md"))) != 6:
        err("corpus の本数")

    # 語
    words = [w for _, y, n in build.RULES.values() for w in (y, n)]
    if len(set(words)) != 12:
        err("語の重複", words)
    for a, b in itertools.permutations(words, 2):
        if a in b:
            err("部分文字列", a, b)
    if set(words) & OLD_WORDS:
        err("dev90b と同じ語", set(words) & OLD_WORDS)
    if "保留" in words:
        err("語に保留")

    # 再計算
    agree = 0
    for r in rows:
        rule = r["rule_id"]
        f, skip = facts[r["id"]]["facts"], facts[r["id"]]["skip_text_check"]
        fn, yes, no = build.RULES[rule]
        unknown = [k for k, v in f.items() if isinstance(v, dict) and "any" in v]
        res_main, res_alt = [], []
        for g in build.expand(f):
            m = fn(g)
            a = ALT[rule](g)
            if bool(m) != bool(a):
                err("2 つの式が不一致", r["id"], g)
            res_main.append(bool(m))
            res_alt.append(bool(a))
        hold = len(set(res_alt)) == 2
        expected = "保留" if hold else (yes if res_alt[0] else no)
        if expected != r["expected_conclusion"] or hold != r["should_hold"]:
            err("正解が不一致", r["id"], expected, r["expected_conclusion"])
        else:
            agree += 1
        is_hold_group = r["group"].startswith("hold_")
        if is_hold_group != hold:
            err("保留の群と結論が不一致", r["id"], r["group"])
        if (is_hold_group or r["group"] == "determined_unstated") != bool(unknown):
            err("書いていない事実の有無", r["id"], r["group"], unknown)
        check_text(r, f, skip)

        # 語と最後の行
        opts = r["conclusion_options"]
        if set(opts) != {yes, no}:
            err("選択肢", r["id"], opts)
        if not r["should_hold"] and r["expected_conclusion"] not in opts:
            err("結論が選択肢に無い", r["id"])
        last = r["question"].split("\n")[-1]
        want = f"最後の行には「結論: {opts[0]}」または「結論: {opts[1]}」のどちらかだけを書いてください。"
        if last != want:
            err("最後の行", r["id"], last)
        if "保留" in r["question"]:
            err("文に保留", r["id"])

    # pair_id
    pairs = {}
    for r in rows:
        if r["pair_id"]:
            pairs.setdefault(r["pair_id"], []).append(r)
        elif r["group"] == "paraphrase":
            err("言い換えに pair_id が無い", r["id"])
    if len(pairs) != 12:
        err("pair の組数", len(pairs))
    for pid, ps in pairs.items():
        groups = sorted(p["group"] for p in ps)
        if len(ps) != 2 or groups[1] != "paraphrase" or groups[0] not in ("above", "at", "below"):
            err("pair の形", pid, groups)
        if len({p["expected_conclusion"] for p in ps}) != 1 or len({p["rule_id"] for p in ps}) != 1:
            err("pair の結論", pid)
        fa = facts[ps[0]["id"]]["facts"]
        fb = facts[ps[1]["id"]]["facts"]
        if fa != fb:
            err("pair の事実", pid)
        if ps[0]["question"].split("\n")[0] == ps[1]["question"].split("\n")[0]:
            err("pair の文が同じ", pid)
    for rule, qs in by_rule.items():
        n = len({q["pair_id"] for q in qs if q["pair_id"]})
        if n != 2:
            err("規程の pair 数", rule, n)

    # 問の文の重複
    texts = [r["question"] for r in rows]
    if len(set(texts)) != len(texts):
        err("問の文の重複")

    if errors:
        print("NG", len(errors))
        for e in errors:
            print(" -", e)
        sys.exit(1)
    print(f"OK 90問 再計算一致 {agree}/90 回答できる {sum(not r['should_hold'] for r in rows)} 保留 "
          f"{sum(r['should_hold'] for r in rows)} pair {len(pairs)}組")
    for rule, (_, y, n) in build.RULES.items():
        print(f"  {rule}: {y} / {n}")


if __name__ == "__main__":
    main()
