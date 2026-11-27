# numeric_v8c の検算。facts.jsonl は「問の文に書いた事実」だけを持つ。
# 書いていない事実は DOMAIN（または問ごとの unknowns）のあり得る値を全部入れる。
#   回答できる問: どの値でも結論が 1 つで、expected_conclusion と同じ
#   保留の問: unknowns の値で結論が割れ、それ以外の未記載の事実では割れない
# 式は規程ごとに 2 通り（f1: 上から順に判定 / f2: 妨げの列挙や別の単位での計算）で、全組合せで一致を確かめる。
# 最後に、わざと誤りを入れた 7 種で、この検算が止まる事を確かめる（自己点検）。
import copy
import itertools
import json
import math
import re
import sys
import zlib
from collections import Counter
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).parent
HOLD = "保留"
GROUPS = {"below": 1, "at": 1, "above": 1, "paraphrase": 2, "exception_hit": 1, "exception_near_miss": 1,
          "premise_unmet": 1, "second_condition": 1, "irrelevant_number": 1, "compute": 1,
          "determined_unstated": 1, "hold_missing_value": 1, "hold_unknown_condition": 1, "hold_range_straddle": 1}
BANNED = ["慶弔", "社員食堂", "在宅勤務", "副業", "社外講師", "再検査", "深夜タクシー", "資格受験", "保養所",
          "マイカー", "サークル", "リフレッシュ"]


def num(x):
    return Fraction(str(x))


# ------------------------------------------------ 時間外勤務事前申請規程
def c15(x):
    return -(-x // 15) * 15


def ot_f1(f):
    if not f["confirmed"] or f["lead"] < 24 * 60:
        return "差戻し"
    total = sum(c15(x) for x in f["prior"]) + c15(f["req"])
    if f["child_m"] is not None and f["child_m"] < 36 and f["requested"]:
        return "受理" if total <= 24 * 60 else "差戻し"
    if total <= 45 * 60:
        return "受理"
    if total <= 80 * 60 and f["sp"] <= 5:
        return "受理"
    return "差戻し"


def ot_f2(f):
    # 15 分を 1 単位に数え直し、妨げを並べる
    quarters = sum(-(-x // 15) for x in f["prior"]) + -(-f["req"] // 15)
    limited = bool(f["requested"]) and f["child_m"] is not None and f["child_m"] < 36
    cap_q = (24 if limited else 45) * 4
    special_ok = (not limited) and quarters <= 80 * 4 and f["sp"] + 1 <= 6
    blockers = [not f["confirmed"], f["lead"] < 1440, quarters > cap_q and not special_ok]
    return "差戻し" if any(blockers) else "受理"


# ------------------------------------------------ 転勤支度金規程
def tr_f1(f):
    d = num(f["dist"])
    if not f["ordered"] or d < 50:
        return "却下"
    if f["prev_m"] is not None and f["prev_m"] < 12:
        return "却下"
    base = 100000 if d < 200 else 150000 if d < 500 else 200000
    n = f["dependents"] // 2 if f["solo"] else f["accompany"]
    limit = base + 30000 * min(n, 4) + (50000 if f["notice_days"] < 14 else 0)
    return "支給" if f["amount"] <= limit else "却下"


def tr_f2(f):
    d = num(f["dist"])
    table = [(Fraction(500), 200000), (Fraction(200), 150000), (Fraction(50), 100000)]
    base = next((v for t, v in table if d >= t), None)
    heads = f["dependents"] // 2 if f["solo"] else f["accompany"]
    blockers = [not f["ordered"], base is None, f["prev_m"] is not None and f["prev_m"] < 12]
    if any(blockers):
        return "却下"
    extra = 30000 * (heads if heads < 4 else 4) + 50000 * (f["notice_days"] < 14)
    return "却下" if f["amount"] - (base + extra) > 0 else "支給"


# ------------------------------------------------ 改善提案報奨金規程
def rw_f1(f):
    if f["elapsed_m"] < 6 or f["room"]:
        return "見送り"
    annual = f["monthly"] * 12
    total = 60000 if annual >= 1000000 else 25000 if annual >= 400000 else 9000 if annual >= 120000 else 0
    if f["n"] == 1:
        mine = total
    elif f["rep"]:
        mine = total * 50 // 100
    else:
        mine = total * 50 // 100 // (f["n"] - 1)
    return "支払" if mine >= 3000 else "見送り"


def rw_f2(f):
    table = [(0, 0), (120000, 9000), (400000, 25000), (1000000, 60000)]
    annual = Fraction(f["monthly"]) * 12
    total = [v for t, v in table if annual >= t][-1]
    if f["n"] == 1:
        share = Fraction(total)
    elif f["rep"]:
        share = Fraction(total, 2)
    else:
        share = Fraction(total, 2) / (f["n"] - 1)
    blockers = [f["elapsed_m"] < 6, f["room"], math.floor(share) < 3000]
    return "見送り" if any(blockers) else "支払"


# ------------------------------------------------ 駐輪場利用規程
def pk_f1(f):
    cc = f["cc"]
    if cc is not None and cc > 125:
        return "不可"
    if f["other"]:
        return "不可"
    if f["kind"] == "renew" and f["days"] < 30:
        return "不可"
    if f["handbook"]:
        return "許可"
    need = Fraction(2) if cc is None else Fraction(3) if cc <= 50 else Fraction(5)
    d = Fraction(math.floor(num(f["dist"]) * 10), 10)
    return "許可" if d >= need else "不可"


def pk_f2(f):
    cc = f["cc"]
    metres = int(num(f["dist"]) * 1000) // 100 * 100
    need_m = 2000 if cc is None else 3000 if cc <= 50 else 5000 if cc <= 125 else None
    blockers = [need_m is None, bool(f["other"]), f["kind"] == "renew" and f["days"] < 30]
    if any(blockers):
        return "不可"
    return "許可" if (f["handbook"] or metres >= need_m) else "不可"


# ------------------------------------------------ 情報機器社外持出規程
LADDER = ["所属長", "部門長", "情報システム部長", "役員"]


def im_f1(f):
    if not f["reg"]:
        return "持出禁止"
    if f["usb"] and f["conf"]:
        return "持出禁止"
    if f["conf"] and not f["usb"] and not f["enc"]:
        return "持出禁止"
    if f["days"] > 30:
        return "持出禁止"
    need = 2 if f["abroad"] else (0 if f["days"] <= 3 else 1 if f["days"] <= 14 else 2)
    return "持出可" if LADDER.index(f["approver"]) >= need else "持出禁止"


def im_f2(f):
    tiers = [(3, "所属長"), (14, "部門長"), (30, "情報システム部長")]
    required = next((t for lim, t in tiers if f["days"] <= lim), None)
    if f["abroad"] and required:
        required = "情報システム部長"
    blockers = [not f["reg"], required is None, bool(f["conf"]) and (bool(f["usb"]) or not f["enc"])]
    if any(blockers):
        return "持出禁止"
    return "持出禁止" if LADDER.index(f["approver"]) < LADDER.index(required) else "持出可"


# ------------------------------------------------ フォークリフト運転規程
QR = {"none": 0, "special": 1, "skill": 2}


def fk_f1(f):
    if not f["inspected"]:
        return "運転禁止"
    cap = num(f["cap"])
    need = 1 if cap < 1 else 2
    if max(QR[f["qual"]], QR[f["escort"]]) < need:
        return "運転禁止"
    c = f["center"]
    if c > 600:
        return "運転禁止"
    limit = cap * 1000 if c <= 500 else cap * 1000 * Fraction(80, 100)
    if f["load"] > limit:
        return "運転禁止"
    if num(f["speed"]) > (6 if f["indoor"] else 10):
        return "運転禁止"
    return "運転可"


def fk_f2(f):
    cap_kg = num(f["cap"]) * 1000
    pct = 100 if f["center"] <= 500 else 80 if f["center"] <= 600 else 0
    speed_m_min = num(f["speed"]) * 1000 / 60
    limit_m_min = Fraction((6 if f["indoor"] else 10) * 1000, 60)
    skill_needed = 1 if cap_kg < 1000 else 2
    blockers = [not f["inspected"], max(QR[f["qual"]], QR[f["escort"]]) < skill_needed, pct == 0,
                f["load"] * 100 > cap_kg * pct, speed_m_min > limit_m_min]
    return "運転禁止" if any(blockers) else "運転可"


RULES = {
    "時間外勤務事前申請規程": (ot_f1, ot_f2, {
        "confirmed": [True, False], "lead": [1380, 1440, 2880], "prior": [[], [1500], [2640]],
        "req": [60, 600, 1200], "sp": [0, 5, 6], "child_m": [None, 24, 38], "requested": [True, False]}),
    "転勤支度金規程": (tr_f1, tr_f2, {
        "ordered": [True, False], "dist": [30, 49.9, 50, 120, 200, 500], "solo": [True, False],
        "accompany": [0, 2, 5], "dependents": [0, 3, 6], "notice_days": [7, 13, 14, 30],
        "prev_m": [None, 8, 12], "amount": [100000, 160000, 240000, 340000]}),
    "改善提案報奨金規程": (rw_f1, rw_f2, {
        "elapsed_m": [3, 6, 12], "monthly": [5000, 9999, 10000, 33333, 33334, 100000], "room": [True, False],
        "n": [1, 3, 7], "rep": [True, False]}),
    "駐輪場利用規程": (pk_f1, pk_f2, {
        "kind": ["new", "renew"], "cc": [None, 50, 125, 250], "dist": ["1.9", "2.0", "3.0", "4.99", "5.0", "6.0"],
        "days": [29, 30, 45], "handbook": [True, False], "other": [True, False]}),
    "情報機器社外持出規程": (im_f1, im_f2, {
        "reg": [True, False], "usb": [True, False], "conf": [True, False], "enc": [True, False],
        "days": [3, 4, 14, 15, 30, 31], "approver": LADDER, "abroad": [True, False]}),
    "フォークリフト運転規程": (fk_f1, fk_f2, {
        "qual": ["none", "special", "skill"], "escort": ["none", "special", "skill"], "cap": ["0.8", "1.0", "2.5"],
        "load": [500, 1500, 2490], "center": [450, 520, 620], "speed": ["4", "6", "10", "12"],
        "indoor": [True, False], "inspected": [True, False]}),
}
# 書いていなければ「無い」と読む事実（例外の発動条件。請求・手帳・別の許可・海外・同乗者・室員）
ABSENT = {
    "時間外勤務事前申請規程": {"child_m": None, "requested": False},
    "転勤支度金規程": {"prev_m": None},
    "改善提案報奨金規程": {"room": False},
    "駐輪場利用規程": {"handbook": False, "other": False},
    "情報機器社外持出規程": {"abroad": False},
    "フォークリフト運転規程": {"escort": "none"},
}
# 本文の数と直接は比べない事実（日付や時刻から計算して入れた値）
DERIVED = {
    "時間外勤務事前申請規程": {"lead", "prior", "req", "child_m", "requested", "sp", "confirmed"},
    "転勤支度金規程": {"notice_days", "prev_m", "accompany", "dependents"},
    "改善提案報奨金規程": {"elapsed_m", "n"},
    "駐輪場利用規程": {"days"},
    "情報機器社外持出規程": {"days"},
    "フォークリフト運転規程": set(),
}


def enumerate_cases(rule, facts, unknowns, rules=RULES):
    f1, f2, domain = rules[rule]
    keys = [k for k in domain if k not in facts]
    absent = ABSENT.get(rule, {})
    choices = []
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
    for k in facts:
        assert k in domain, f"facts のキー {k} は domain に無い"
    for combo in itertools.product(*choices):
        full = dict(facts)
        full.update(zip(keys, combo))
        yield dict(zip(keys, combo)), f1(full), f2(full)


def norm(v):
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, list):
        return [norm(x) for x in v]
    if isinstance(v, (int, float)):
        return ("n", Fraction(str(v)))
    if isinstance(v, str) and re.fullmatch(r"-?[\d.]+", v):
        return ("n", Fraction(v))
    return v


def num_forms(v):
    """本文に現れてほしい数の書き方の候補（報告用）。"""
    out = set()
    if isinstance(v, bool) or v is None or isinstance(v, list):
        return out
    if isinstance(v, (int, float)) or (isinstance(v, str) and re.fullmatch(r"[\d.]+", v)):
        x = num(v)
        if x.denominator == 1:
            n = int(x)
            out |= {str(n), f"{n:,}"}
            if n % 10000 == 0:
                out |= {f"{n // 10000}万", f"{n // 10000:,}万"}
            if n % 1000 == 0:
                out |= {f"{n // 1000}千", f"{n / 1000:g}", f"{n // 1000:,}"}
                out |= {f"{n // 10000}万{(n % 10000) // 1000}千"}
            out |= {f"{n}.0"}
            if n >= 1000 and n % 100 == 0:
                out |= {f"{n / 1000:g}"}
        else:
            out.add(str(float(x)))
            out.add(f"{float(x):g}")
            out.add(f"{float(x * 1000):g}")
    return out


# ------------------------------------------------ 構造の確かめ
def check_struct(qs, fs, corpus_dir):
    errors = []

    def err(*a):
        errors.append(" ".join(str(x) for x in a))

    if len(qs) != 90:
        err("件数", len(qs))
    ids = [q["id"] for q in qs]
    if len(set(ids)) != len(ids):
        err("id 重複")
    if set(ids) != {f"VC{i:03d}" for i in range(1, 91)}:
        err("id の範囲")
    if ids == sorted(ids):
        err("id が順に並んでいる")
    if set(fs) != set(ids):
        err("facts の id が合わない")
    need = {"id", "question", "rule_id", "should_hold", "expected_conclusion", "conclusion_options", "group",
            "pair_id", "basis"}
    per_rule = Counter(q["rule_id"] for q in qs)
    if len(per_rule) != 6 or set(per_rule.values()) != {15}:
        err("規程ごとの数", per_rule)
    files = {p.stem for p in Path(corpus_dir).glob("*.md")}
    if set(per_rule) != files or set(RULES) != files:
        err("規程名と corpus のファイル名が合わない", set(per_rule) ^ files)
    for rule in files:
        cp = Path(corpus_dir) / f"{rule}.md"
        text = cp.read_text(encoding="utf-8")
        nlines = len(text.rstrip("\n").split("\n"))
        if not 15 <= nlines <= 30:
            err("規程の行数", rule, nlines)
        if any(w in text for w in BANNED):
            err("使わない話題の語", rule)
        groups = Counter(q["group"] for q in qs if q["rule_id"] == rule)
        if groups != Counter(GROUPS):
            err("内訳", rule, groups)

    optsets = {}
    for q in qs:
        if set(q) != need:
            err("項目", q["id"], set(q) ^ need)
        a, b = q["conclusion_options"]
        if a in b or b in a or a == b:
            err("部分文字列", q["id"], a, b)
        if HOLD in a or HOLD in b:
            err("結論の語に保留を含む", q["id"])
        optsets.setdefault(q["rule_id"], set()).add(frozenset((a, b)))
        lines = q["question"].split("\n")
        if lines[-1] != f"最後の行には「結論: {a}」または「結論: {b}」のどちらかだけを書いてください。":
            err("最後の行", q["id"])
        if len(lines) != 2:
            err("改行", q["id"])
        if q["should_hold"]:
            if q["expected_conclusion"] != HOLD or not q["group"].startswith("hold_"):
                err("保留の項目", q["id"])
        else:
            if q["expected_conclusion"] not in q["conclusion_options"] or q["group"].startswith("hold_"):
                err("回答の項目", q["id"])
        if (q["pair_id"] is None) == (q["group"] == "paraphrase"):
            if q["group"] == "paraphrase":
                err("paraphrase に pair_id が無い", q["id"])
    if any(len(v) != 1 for v in optsets.values()):
        err("規程内で語の組が揃っていない", optsets)
    else:
        pairs_ = [tuple(next(iter(v))) for v in optsets.values()]
        words = [w for p in pairs_ for w in p]
        if len(set(words)) != 12:
            err("語が規程間で重なる", pairs_)
        for w1, w2 in itertools.permutations(range(len(words)), 2):
            if words[w1] in words[w2] and w1 // 2 != w2 // 2 and words[w1] != words[w2]:
                err("語が他の規程の語の一部", words[w1], words[w2])

    if sum(q["should_hold"] for q in qs) != 18:
        err("保留の数")
    for rule in files:
        c = Counter(q["expected_conclusion"] for q in qs if q["rule_id"] == rule and not q["should_hold"])
        if sorted(c.values()) != [6, 6]:
            err("向き", rule, c)
        if sum(q["should_hold"] for q in qs if q["rule_id"] == rule) != 3:
            err("規程ごとの保留の数", rule)

    pairs = {}
    for q in qs:
        if q["pair_id"]:
            pairs.setdefault(q["pair_id"], []).append(q)
    if set(pairs) != {f"PC{i:02d}" for i in range(1, 13)}:
        err("pair_id の範囲", sorted(pairs))
    per_rule_pairs = Counter()
    for p, m in pairs.items():
        gs = sorted(x["group"] for x in m)
        if len(m) != 2 or "paraphrase" not in gs or gs[0] not in ("above", "at", "below"):
            err("pair の形", p, gs)
            continue
        per_rule_pairs[m[0]["rule_id"]] += 1
        if len({x["rule_id"] for x in m}) != 1 or len({x["expected_conclusion"] for x in m}) != 1:
            err("pair の結論", p)
        if m[0]["question"].split("\n")[0] == m[1]["question"].split("\n")[0]:
            err("pair の文が同じ", p)
        fa, fb = fs[m[0]["id"]], fs[m[1]["id"]]
        if norm(fa["facts"]) != norm(fb["facts"]) or norm(fa["unknowns"]) != norm(fb["unknowns"]):
            err("pair の事実", p, fa, fb)
    if set(per_rule_pairs.values()) != {2} or len(per_rule_pairs) != 6:
        err("規程ごとの pair の数", per_rule_pairs)
    return errors


# ------------------------------------------------ 問ごとの再計算
def check_question(q, rec, rules=RULES):
    errors = []
    rule = q["rule_id"]
    facts, unknowns = rec["facts"], rec["unknowns"]
    try:
        results = list(enumerate_cases(rule, facts, unknowns, rules))
    except Exception as e:  # 事実が壊れて式が動かない場合も、検算が止まったものとして数える
        return [f"式が動かない {q['id']} {type(e).__name__} {e}"], 0
    for asg, r1, r2 in results:
        if r1 != r2:
            errors.append(f"式の不一致 {q['id']} {asg} {r1} {r2}")
            break
    outs = {r1 for _, r1, _ in results}
    if q["should_hold"]:
        if len(outs) < 2:
            errors.append(f"保留が割れない {q['id']} {outs}")
        if not unknowns:
            errors.append(f"保留に unknowns が無い {q['id']}")
        else:
            byu = {}
            for asg, r1, _ in results:
                key = tuple(json.dumps(asg[k], sort_keys=True) for k in unknowns)
                byu.setdefault(key, set()).add(r1)
            if any(len(v) != 1 for v in byu.values()):
                errors.append(f"保留が unknowns 以外でも割れる {q['id']} {byu}")
            if len({next(iter(v)) for v in byu.values() if len(v) == 1}) < 2:
                errors.append(f"unknowns で割れない {q['id']} {byu}")
            if any(len(v) < 2 for v in unknowns.values()):
                errors.append(f"unknowns の候補が 1 つ {q['id']}")
    else:
        if outs != {q["expected_conclusion"]}:
            errors.append(f"正解違い {q['id']} {q['group']} {outs} {q['expected_conclusion']} {facts}")
        if (q["group"] == "determined_unstated") != bool(unknowns):
            errors.append(f"unknowns の置き方 {q['id']} {q['group']}")
    return errors, len(results)


def run_all(qs, fs, rules=RULES, corpus_dir=HERE / "corpus"):
    errors = check_struct(qs, fs, corpus_dir)
    n_cases = 0
    for q in qs:
        e, n = check_question(q, fs[q["id"]], rules)
        errors += e
        n_cases += n
    return errors, n_cases


# ------------------------------------------------ 自己点検（わざと誤りを入れて止まるか）
def broken(fn):
    """式を 1 つ壊す。事実の中身で決まる条件のときだけ結論を入れ替える。"""
    def g(f):
        r = fn(f)
        key = json.dumps(f, sort_keys=True, default=str)
        if zlib.crc32(key.encode("utf-8")) % 5 == 0:
            alt = {"受理": "差戻し", "差戻し": "受理", "支給": "却下", "却下": "支給", "支払": "見送り",
                   "見送り": "支払", "許可": "不可", "不可": "許可", "持出可": "持出禁止", "持出禁止": "持出可",
                   "運転可": "運転禁止", "運転禁止": "運転可"}
            return alt[r]
        return r
    return g


def self_test(qs, fs):
    report = []
    bad = []
    answerable = [q for q in qs if not q["should_hold"]]
    holds = [q for q in qs if q["should_hold"]]

    # 1 結論の反転
    det = 0
    for q in answerable:
        q2 = dict(q)
        a, b = q["conclusion_options"]
        q2["expected_conclusion"] = b if q["expected_conclusion"] == a else a
        e, _ = check_question(q2, fs[q["id"]])
        det += bool(e)
    report.append(("結論の反転", det, len(answerable)))
    if det != len(answerable):
        bad.append("結論の反転")

    # 2 保留の候補の削り
    det = 0
    for q in holds:
        rec = copy.deepcopy(fs[q["id"]])
        rec["unknowns"] = {k: v[:1] for k, v in rec["unknowns"].items()}
        e, _ = check_question(q, rec)
        det += bool(e)
    report.append(("保留の候補の削り", det, len(holds)))
    if det != len(holds):
        bad.append("保留の候補の削り")

    # 3 事実の値の変更（問ごとに、変えて止まる値が 1 つ以上ある事）
    det, tot = 0, 0
    for q in answerable:
        rec = fs[q["id"]]
        dom = RULES[q["rule_id"]][2]
        tot += 1
        found = False
        for k in rec["facts"]:
            for alt in dom[k]:
                if norm(alt) == norm(rec["facts"][k]):
                    continue
                r2 = copy.deepcopy(rec)
                r2["facts"][k] = alt
                e, _ = check_question(q, r2)
                if e:
                    found = True
                    break
            if found:
                break
        det += found
    report.append(("事実の値の変更", det, tot))
    if det != tot:
        bad.append("事実の値の変更")

    # 4 必要な事実の削除（問ごとに、削って止まる事実が 1 つ以上ある事。保留でない問）
    det, tot = 0, 0
    for q in answerable:
        rec = fs[q["id"]]
        tot += 1
        found = False
        for k in rec["facts"]:
            r2 = copy.deepcopy(rec)
            del r2["facts"][k]
            e, _ = check_question(q, r2)
            if e:
                found = True
                break
        det += found
    report.append(("必要な事実の削除", det, tot))
    if det != tot:
        bad.append("必要な事実の削除")

    # 5 id の重複
    qs2 = copy.deepcopy(qs)
    qs2[1]["id"] = qs2[0]["id"]
    e = check_struct(qs2, fs, HERE / "corpus")
    ok = any(x.startswith("id 重複") for x in e)
    report.append(("id の重複", int(ok), 1))
    if not ok:
        bad.append("id の重複")

    # 6 部分文字列の語
    qs2 = copy.deepcopy(qs)
    a = qs2[0]["conclusion_options"][0]
    qs2[0]["conclusion_options"] = [a, a + "付"]
    e = check_struct(qs2, fs, HERE / "corpus")
    ok = any(x.startswith("部分文字列") for x in e)
    report.append(("部分文字列の語", int(ok), 1))
    if not ok:
        bad.append("部分文字列の語")

    # 7 式を 1 つ壊す（規程ごとに f1 と f2 のそれぞれ）
    det, tot = 0, 0
    for rule in RULES:
        for which in (0, 1):
            rules2 = dict(RULES)
            f1, f2, dom = RULES[rule]
            rules2[rule] = (broken(f1), f2, dom) if which == 0 else (f1, broken(f2), dom)
            found = False
            for q in qs:
                if q["rule_id"] != rule:
                    continue
                e, _ = check_question(q, fs[q["id"]], rules2)
                if e:
                    found = True
                    break
            tot += 1
            det += found
    report.append(("式を壊す", det, tot))
    if det != tot:
        bad.append("式を壊す")
    return report, bad


def main():
    qs = [json.loads(x) for x in open(HERE / "questions.jsonl", encoding="utf-8")]
    fs = {r["id"]: r for r in (json.loads(x) for x in open(HERE / "facts.jsonl", encoding="utf-8"))}
    errors, n_cases = run_all(qs, fs)

    pairs = {q["pair_id"] for q in qs if q["pair_id"]}
    unmatched = []
    for q in qs:
        text = q["question"].split("\n")[0]
        skip = DERIVED[q["rule_id"]]
        for k, v in fs[q["id"]]["facts"].items():
            if k in skip:
                continue
            forms = num_forms(v)
            if forms and not any(x in text for x in forms):
                unmatched.append((q["id"], q["group"], k, v))

    print(f"件数 {len(qs)} / 回答 {sum(not q['should_hold'] for q in qs)} / 保留 {sum(q['should_hold'] for q in qs)}")
    print(f"pair {len(pairs)} 組 / 検算した問 {len(qs)} / 入れた組合せ {n_cases}")
    print(f"本文に書き方が見つからない数 {len(unmatched)} 件（下は目で確かめる）")
    for u in unmatched:
        print("  ", u)
    if errors:
        print("NG", len(errors))
        for e in errors:
            print(" ", e)
        sys.exit(1)

    report, bad = self_test(qs, fs)
    print("自己点検（わざと誤りを入れて止まった数 / 試した数）")
    for name, det, tot in report:
        print(f"  {name}: {det} / {tot}")
    if bad:
        print("NG 自己点検で止まらなかった種類", bad)
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
