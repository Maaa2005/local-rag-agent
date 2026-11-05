# numeric_dev90d の検算。facts.jsonl は「問の文に書いた事実」だけを持つ。
# 書いていない事実は DOMAIN（または問ごとの unknowns）のあり得る値を全部入れる。
#   回答できる問: どの値でも結論が 1 つで、expected_conclusion と同じ
#   保留の問: unknowns の値で結論が割れ、それ以外の未記載の事実では割れない
# 式は規程ごとに 2 通り（f1: 上から順に判定 / f2: 妨げの列挙や別の計算）で、全組合せで一致を確かめる。
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
BANNED = ["慶弔", "社員食堂", "在宅勤務", "副業", "社外講師", "再検査", "深夜タクシー", "資格受験", "保養所",
          "マイカー", "サークル", "リフレッシュ"]


def d(s):
    return date.fromisoformat(s)


# ------------------------------------------------ 文書廃棄規程
YEARS = {"議事録": 3, "経費精算書": 5, "稟議書": 7, "契約書": 10}


def doc_f1(f):
    if not f["registered"]:
        return "保管継続"
    if f["hold"]:
        return "保管継続"
    if f["copy"]:
        return "廃棄可"
    b = d(f["base"])
    fy = b.year if b.month >= 4 else b.year - 1
    end = date(fy + 1 + YEARS[f["kind"]], 3, 31)
    if d(f["today"]) > end and f["approved"]:
        return "廃棄可"
    return "保管継続"


def doc_f2(f):
    # 基準日の後の最初の 4/1 を歩いて探し、保存年数ぶん進めて前日を満了日とする
    x = d(f["base"]) + timedelta(days=1)
    while (x.month, x.day) != (4, 1):
        x += timedelta(days=1)
    start = x
    end = date(start.year + YEARS[f["kind"]], 4, 1) - timedelta(days=1)
    blockers = []
    if not f["registered"]:
        blockers.append("未登録")
    if f["hold"]:
        blockers.append("保全")
    if not f["copy"]:
        if (d(f["today"]) - end).days <= 0:
            blockers.append("満了前")
        if not f["approved"]:
            blockers.append("未承認")
    return "保管継続" if blockers else "廃棄可"


# ------------------------------------------------ 物品購入相見積規程
def amt_a(a):
    if isinstance(a, dict):
        return int(a["unit"] * a["qty"] * 11 // 10) + a["ship"]
    return a


def amt_b(a):
    if isinstance(a, dict):
        ex = Fraction(a["unit"]) * a["qty"]
        tax = int(ex * Fraction(10, 100))
        return int(ex) + tax + a["ship"]
    return a


def buy_f1(f):
    if f["kind"] == "lease":
        return "省略可"
    if f["genuine"]:
        return "省略可"
    if f["urgent_days"] <= 3 and f["approved"]:
        return "省略可"
    a = amt_a(f["amount"])
    if a >= 300000:
        return "必須"
    if a + f["prior"] >= 500000:
        return "必須"
    return "省略可"


def buy_f2(f):
    a = amt_b(f["amount"])
    triggers = [a - 300000 >= 0, f["prior"] + a - 500000 >= 0]
    waivers = [f["kind"] != "goods", f["genuine"], f["approved"] and not f["urgent_days"] > 3]
    return "必須" if any(triggers) and not any(waivers) else "省略可"


# ------------------------------------------------ 屋外作業中止規程
def ms(w):
    if isinstance(w, dict):
        return Fraction(str(w["kmh"])) / Fraction(36, 10)
    return Fraction(str(w))


def kmh(w):
    if isinstance(w, dict):
        return Fraction(str(w["kmh"]))
    return Fraction(str(w)) * Fraction(36, 10)


def out_f1(f):
    if f["indoor"]:
        return "続行"
    if f["emergency"] and f["approved"]:
        return "続行"
    if ms(f["wind"]) >= 15:
        return "中止"
    m, dd = map(int, f["md"].split("-"))
    summer = (6, 1) <= (m, dd) <= (9, 30)
    if summer:
        w = Fraction(str(f["wbgt"]))
        if w >= 31:
            return "中止"
        if w >= 28 and f["cont"] > 120:
            return "中止"
    return "続行"


def out_f2(f):
    doy = date(2025, *map(int, f["md"].split("-"))).timetuple().tm_yday
    in_period = date(2025, 6, 1).timetuple().tm_yday <= doy <= date(2025, 9, 30).timetuple().tm_yday
    w = Fraction(str(f["wbgt"]))
    hit = [kmh(f["wind"]) >= 54,
           in_period and not w < 31,
           in_period and not w < 28 and w < 31 and not f["cont"] <= 120]
    exempt = f["indoor"] or (f["emergency"] and f["approved"])
    return "続行" if exempt or not any(hit) else "中止"


# ------------------------------------------------ 時間単位年休規程
def lv_f1(f):
    if f["tenure_m"] < 6:
        return "取れない"
    cap = 40 if f["short_h"] >= 8 else f["short_h"] * 5
    if f["ytd"] + f["req_h"] > cap:
        return "取れない"
    if f["days_before"] >= 2:
        return "取れる"
    if f["days_before"] == 1:
        return "取れる" if f["submit_min"] <= 17 * 60 else "取れない"
    if f["reason"] == "family_live" and f["submit_min"] <= f["start_min"]:
        return "取れる"
    return "取れない"


def lv_f2(f):
    # 分で計算し直す。期限は「取得日の 0 時から見た提出時刻」で比べる
    remain = min(40, f["short_h"] * 5) * 60 - f["ytd"] * 60
    deadline = -(24 * 60) + 17 * 60
    submitted = -(f["days_before"] * 24 * 60) + f["submit_min"]
    in_time = submitted <= deadline
    special = f["days_before"] == 0 and f["reason"] != "other" and submitted <= f["start_min"]
    blockers = [not (f["tenure_m"] >= 6), f["req_h"] * 60 > remain, not (in_time or special)]
    return "取れない" if any(blockers) else "取れる"


# ------------------------------------------------ 永年勤続表彰規程
def months_a(h, ref):
    m = (ref.year - h.year) * 12 + ref.month - h.month
    return m - 1 if ref.day < h.day else m


def months_b(h, ref):
    n, y, m = 0, h.year, h.month
    while True:
        m += 1
        if m > 12:
            y, m = y + 1, 1
        if date(y, m, h.day) > ref:
            return n
        n += 1


def ser(f, months):
    ref = date(2026, 10, 1)
    return months(d(f["hire"]), ref), ref


def aw_f1(f):
    m, ref = ser(f, months_a)
    total = m + f.get("old_m", 0) // 2 - f["leave_private"]
    if not f["regular"] or f["prior_award"]:
        return "対象外"
    if f["discipline"] and d(f["discipline"]) >= date(ref.year - 3, ref.month, ref.day):
        return "対象外"
    return "贈呈" if total >= 120 else "対象外"


def aw_f2(f):
    m, ref = ser(f, months_b)
    old = f.get("old_m", 0)
    # 休職を全部引いてから差し引かない分を戻す
    gross = m - f["leave_private"] - f.get("leave_work", 0) + f.get("leave_work", 0)
    bonus = int(Fraction(old, 2))
    blockers = [f["regular"] is False, f["prior_award"] is True,
                f["discipline"] is not None and (ref - d(f["discipline"])).days <= (ref - date(2023, 10, 1)).days,
                gross + bonus < 120]
    return "対象外" if any(blockers) else "贈呈"


# ------------------------------------------------ 社員貸付金規程
def ln_f1(f):
    if not f["regular"] or f["tenure_m"] < 36:
        return "謝絶"
    if f["installments"] > 60:
        return "謝絶"
    limit = 5000000 if f["disaster"] else min(f["income"] * 20 // 100, 3000000)
    return "貸付" if f["loan"] <= limit - f["balance"] else "謝絶"


def ln_f2(f):
    caps = [Fraction(f["income"]) / 5, Fraction(3000000)]
    if f["disaster"]:
        caps = [Fraction(5000000)]
    room = sorted(caps)[0] - f["balance"] - f["loan"]
    blockers = [f["regular"] is False, f["tenure_m"] - 36 < 0, not f["installments"] <= 60, room < 0]
    return "謝絶" if any(blockers) else "貸付"


RULES = {
    "文書廃棄規程": (doc_f1, doc_f2, {
        "registered": [True, False], "hold": [True, False], "copy": [True, False], "approved": [True, False],
        "kind": list(YEARS), "base": ["2015-06-01", "2025-06-01"], "today": ["2026-08-01"]}),
    "物品購入相見積規程": (buy_f1, buy_f2, {
        "kind": ["goods", "lease"], "amount": [100000, 299999, 300000, 600000], "prior": [0, 150000, 450000],
        "genuine": [True, False], "urgent_days": [2, 3, 4, 20], "approved": [True, False]}),
    "屋外作業中止規程": (out_f1, out_f2, {
        "indoor": [True, False], "md": ["05-01", "08-01"], "wind": [3, 15, 16], "wbgt": [25, 28, 29, 31, 33],
        "cont": [60, 120, 121], "emergency": [True, False], "approved": [True, False]}),
    "時間単位年休規程": (lv_f1, lv_f2, {
        "tenure_m": [3, 6, 24], "short_h": [6, 8], "ytd": [0, 10, 28, 39], "req_h": [1, 2],
        "days_before": [0, 1, 3], "submit_min": [500, 540, 600, 1020, 1021], "start_min": [540],
        "reason": ["family_live", "other"]}),
    "永年勤続表彰規程": (aw_f1, aw_f2, {
        "hire": ["2010-04-01", "2021-04-01"], "regular": [True, False], "prior_award": [True, False],
        "leave_private": [0, 40], "discipline": [None, "2025-01-10", "2020-01-10"], "old_m": [0],
        "leave_work": [0]}),
    "社員貸付金規程": (ln_f1, ln_f2, {
        "regular": [True, False], "tenure_m": [24, 36, 60], "income": [5000000, 10000000, 20000000],
        "loan": [500000, 4000000], "balance": [0, 800000], "installments": [36, 60, 72],
        "disaster": [True, False]}),
}
# 書いていなければ「無い」と読む事実（合併の移籍と業務災害の休職は、書かれた時だけ効く。入社日と休職の記載で足りる）
ABSENT = {"永年勤続表彰規程": {"old_m": 0, "leave_work": 0}, "時間単位年休規程": {"start_min": 540}}


def enumerate_cases(rule, facts, unknowns, keys_only=None):
    """facts を固定し、書いていない事実に DOMAIN / unknowns の値を入れる。(割り当て, f1結論) を返す。"""
    f1, f2, domain = RULES[rule]
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
            if man % 1000 == 0:
                out.add(f"{man // 1000}千万")
        if n % 1000 == 0:
            out |= {f"{n // 1000:,}千", f"{n // 10000}万{(n % 10000) // 1000}千"}
    elif isinstance(v, str):
        m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", v)
        if m:
            y, mo, da = map(int, m.groups())
            out |= {f"{y}年{mo}月{da}日", f"{y}/{mo}/{da}"}
        elif re.fullmatch(r"\d{2}-\d{2}", v):
            mo, da = map(int, v.split("-"))
            out.add(f"{mo}月{da}日")
        elif re.fullmatch(r"[\d.]+", v):
            out.add(v)
    elif isinstance(v, dict) and "kmh" in v:
        out.add(str(v["kmh"]))
    return out


def main():
    root = HERE
    qs = [json.loads(x) for x in open(root / "questions.jsonl", encoding="utf-8")]
    fs = {r["id"]: r for r in (json.loads(x) for x in open(root / "facts.jsonl", encoding="utf-8"))}
    errors = []

    def err(*a):
        errors.append(" ".join(str(x) for x in a))

    # --- 件数・内訳
    if len(qs) != 90:
        err("件数", len(qs))
    ids = [q["id"] for q in qs]
    if len(set(ids)) != len(ids):
        err("id 重複")
    if set(ids) != {f"W{i:03d}" for i in range(1, 91)}:
        err("id の範囲")
    if ids == sorted(ids):
        err("id が順に並んでいる")
    need = {"id", "question", "rule_id", "should_hold", "expected_conclusion", "conclusion_options", "group",
            "pair_id", "basis"}
    per_rule = Counter(q["rule_id"] for q in qs)
    if len(per_rule) != 6 or set(per_rule.values()) != {15}:
        err("規程ごとの数", per_rule)
    if set(per_rule) != set(RULES):
        err("規程名", per_rule)
    for rule in RULES:
        cp = root / "corpus" / f"{rule}.md"
        if not cp.exists():
            err("corpus が無い", rule)
        elif any(w in cp.read_text(encoding="utf-8") for w in BANNED):
            err("使わない話題の語", rule)
        groups = Counter(q["group"] for q in qs if q["rule_id"] == rule)
        if groups != Counter(GROUPS):
            err("内訳", rule, groups)

    # --- 項目・結論の語
    optsets = {}
    for q in qs:
        if set(q) != need:
            err("項目", q["id"], set(q) ^ need)
        a, b = q["conclusion_options"]
        if a in b or b in a or a == b:
            err("部分文字列", q["id"], a, b)
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
        if a in q["question"].split("\n")[0] or b in q["question"].split("\n")[0]:
            pass  # 本文に語が出る事自体は許す（下で数を報告）
    if any(len(v) != 1 for v in optsets.values()):
        err("規程内で語の組が揃っていない", optsets)
    all_pairs = [next(iter(v)) for v in optsets.values()]
    if len(set(all_pairs)) != 6 or len({w for p in all_pairs for w in p}) != 12:
        err("語の組が規程間で重なる", all_pairs)

    # --- 結論の向き・保留の数
    if sum(q["should_hold"] for q in qs) != 18:
        err("保留の数")
    for rule in RULES:
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
    if len(pairs) != 12:
        err("pair の数", len(pairs))
    for p, m in pairs.items():
        gs = sorted(x["group"] for x in m)
        if len(m) != 2 or len(gs) != 2 or "paraphrase" not in gs or gs[0] not in ("above", "at", "below"):
            err("pair の形", p, gs)
        if len({x["rule_id"] for x in m}) != 1 or len({x["expected_conclusion"] for x in m}) != 1:
            err("pair の結論", p)
        if m[0]["question"].split("\n")[0] == m[1]["question"].split("\n")[0]:
            err("pair の文が同じ", p)
        fa, fb = fs[m[0]["id"]]["facts"], fs[m[1]["id"]]["facts"]
        if fa != fb:
            # 表し方が違うだけなら許す（amount の値や km/h）。計算して比べる
            if m[0]["rule_id"] == "屋外作業中止規程":
                ga, gb = dict(fa), dict(fb)
                wa, wb = ms(ga.pop("wind")), ms(gb.pop("wind"))
                if ga != gb or wa != wb:
                    err("pair の事実", p, fa, fb)
            else:
                err("pair の事実", p, fa, fb)

    # --- 正解の再計算
    n_two, n_cases = 0, 0
    for q in qs:
        rule = q["rule_id"]
        rec = fs[q["id"]]
        facts, unknowns = rec["facts"], rec["unknowns"]
        results = list(enumerate_cases(rule, facts, unknowns))
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
            # unknowns の値を固定すれば、それ以外の未記載の事実では割れない
            byu = {}
            for asg, r1, _ in results:
                key = tuple(json.dumps(asg[k], sort_keys=True) for k in unknowns)
                byu.setdefault(key, set()).add(r1)
            if any(len(v) != 1 for v in byu.values()):
                err("保留が unknowns 以外でも割れる", q["id"], byu)
            if len({next(iter(v)) for v in byu.values()}) < 2:
                err("unknowns で割れない", q["id"], byu)
            n_two += 1
        else:
            if outs != {q["expected_conclusion"]}:
                err("正解違い", q["id"], q["group"], outs, q["expected_conclusion"], facts)
            n_two += 1

    # --- 数の写し間違いの確かめ（書いた数が本文にあるか）
    unmatched = []
    for q in qs:
        text = q["question"].split("\n")[0]
        for k, v in fs[q["id"]]["facts"].items():
            forms = num_forms(v)
            if forms and not any(x in text for x in forms):
                unmatched.append((q["id"], q["group"], k, v))

    print(f"件数 {len(qs)} / 回答 {sum(not q['should_hold'] for q in qs)} / 保留 {sum(q['should_hold'] for q in qs)}")
    print(f"pair {len(pairs)} 組 / 検算した問 {n_two} / 入れた組合せ {n_cases}")
    print(f"本文に書き方が見つからない数 {len(unmatched)} 件（下は目で確かめる）")
    for u in unmatched:
        print("  ", u)
    if errors:
        print("NG", len(errors))
        for e in errors:
            print(" ", e)
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
