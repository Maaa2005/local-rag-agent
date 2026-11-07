"""facts.jsonl の隠した事実を、カードの枠の本当の値に機械的に写して truth.json を書く。

写しは facts だけから行い、結論（expected_conclusion）は使わない。

値の決まり:
- 元の事実が "不明" なら、その事実から作る枠は全部 null（計算される枠も含む）。
- 計算される枠は、元の枠に値があれば書かない（コードが計算する）。
- 実在しない事（認定の日・復職日が当たらない等）は ["なし"]。
- 同じ量の別の表し方で、facts がその表し方で示していない枠（税込み領収書の税抜き額、
  分で示された通勤の時間数）は作らず、省いて報告する。
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ASK90 = r"C:\Users\mtkzi\dev\.wt\lr-askback\evaluation\v2\ask90"
UNKNOWN = "不明"


class Skip:
    """計算される枠。元の枠に値があるので書かない。"""


class Omit:
    def __init__(self, reason):
        self.reason = reason


SKIP = Skip()


def yn(x):
    return ["あり" if x else "なし"]


def direct(fact):
    return ([fact], lambda f: [f[fact]])


def flag(fact):
    return ([fact], lambda f: yn(f[fact]))


def derived(*facts):
    return (list(facts), lambda f: SKIP)


def tax_included(f):
    if f["税区分"] == "税込":
        return [f["領収書額"]]
    return SKIP


def tax_excluded(f):
    if f["税区分"] == "税抜":
        return [f["領収書額"]]
    return Omit("領収書が税込みで、税抜きの額は示されていない（費用（税込み）に直接書いた）")


def deadline(f):
    return f'{f["応募年"]}-04-30'


def best_score(f):
    end = deadline(f)
    start = f'{f["応募年"] - 2}-04-30'
    scores = [s for d, s in f["検定履歴"] if start <= d <= end]
    if not scores:
        return Omit(f"{start}から{end}までに受けた検定が履歴に無い")
    return [max(scores)]


GRADE_B_OR_ABOVE = {"S": True, "A": True, "B": True, "C": False, "D": False}

TABLE = {
    "予防接種費用補助規程": ("yobousesshu-hiyou-hojo", {
        "社員本人の接種": (["在籍", "接種者"], lambda f: yn(f["在籍"] and f["接種者"] == "本人")),
        "接種日": direct("接種日"),
        "費用（税込み）": (["領収書額", "税区分"], tax_included),
        "費用（税抜き）": (["領収書額", "税区分"], tax_excluded),
        "提出までの日数": derived("提出日", "接種日"),
        "提出日": direct("提出日"),
        "妊娠中": flag("妊娠中"),
        "産業医から接種を勧められた": flag("基礎疾患の勧奨"),
    }),
    "休日出勤振替規程": ("kyujitsu-shukkin-furikae", {
        "事前の命令": flag("事前命令"),
        "実働": derived("開始時刻", "終了時刻", "休憩分"),
        "開始時刻": direct("開始時刻"),
        "終了時刻": direct("終了時刻"),
        "休憩の時間": direct("休憩分"),
        "出勤日からの日数": derived("出勤日", "取得予定日"),
        "出勤日": direct("出勤日"),
        "振替の取得予定日": direct("取得予定日"),
        "短時間勤務": flag("短時間勤務"),
        "産業医の制限": flag("時間外制限"),
    }),
    "業務端末持出規程": ("gyomu-tanmatsu-mochidashi", {
        "持出日数": derived("持出日", "返却日"),
        "持出日": direct("持出日"),
        "返却日": direct("返却日"),
        "顧客情報件数": direct("顧客情報件数"),
        "暗号化": flag("暗号化"),
        "台帳登録": flag("台帳登録"),
        "事業継続要員": flag("事業継続要員"),
        "両立支援認定": flag("両立支援認定"),
    }),
    "海外出張渡航規程": ("kaigai-shucchou-tokou", {
        "出張命令": flag("出張命令"),
        "社内危険度": direct("危険度"),
        "出発日": direct("出発日"),
        "渡航届を出した日": direct("届出日"),
        "出発日までの日数": derived("出発日", "届出日"),
        "配偶者又は子が住んでいる": flag("家族在住"),
        "妊娠中": flag("妊娠中"),
        "救援要員": flag("救援要員"),
    }),
    "独身寮入寮規程": ("dokushinryo-nyuryo", {
        "正社員": (["雇用形態"], lambda f: yn(f["雇用形態"] == "正社員")),
        "配偶者との同居": flag("配偶者と同居"),
        "通勤時間": direct("通勤分"),
        "通勤時間の時間数": (["通勤分"], lambda f: Omit(
            "facts は分で示され、時間の単位では示されていない（通勤時間に直接書いた）")),
        "年齢": derived("生年月日", "入寮希望月初日"),
        "生年月日": direct("生年月日"),
        "入寮希望月の初日": direct("入寮希望月初日"),
        "半壊以上の認定": (["被災認定日"], lambda f: yn(f["被災認定日"] is not None)),
        "認定の日": (["被災認定日"], lambda f: [f["被災認定日"] or "なし"]),
        "障害者手帳": flag("障害者手帳"),
        "産業医の認定": flag("産業医の軽減認定"),
    }),
    "語学研修選抜規程": ("gogaku-kenshu-senbatsu", {
        "正社員": (["雇用形態"], lambda f: yn(f["雇用形態"] == "正社員")),
        "入社日": direct("入社日"),
        "応募締切日": (["応募年"], lambda f: [deadline(f)]),
        "勤続月数": derived("入社日", "応募年"),
        "検定点数": (["検定履歴", "応募年"], best_score),
        "人事評価B以上": (["人事評価"], lambda f: yn(GRADE_B_OR_ABOVE[f["人事評価"]])),
        "復職日": (["育休復職日"], lambda f: [f["育休復職日"] or "なし"]),
        "復職後月数": (["育休復職日"], lambda f: SKIP if f["育休復職日"] else ["なし"]),
        "海外赴任の内示": flag("海外赴任内示"),
        "過去に修了": flag("研修修了歴"),
    }),
}


def read_jsonl(name):
    with open(os.path.join(ASK90, name), encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def check_value(where, slot_def, value):
    """枠の型と値の形が合うかを確かめる。"""
    if not (isinstance(value, list) and len(value) == 1):
        raise ValueError(f"{where}: 配列でない {value!r}")
    v = value[0]
    kind = slot_def["type"]
    if v == "なし":
        return
    if kind == "bool":
        ok = v == "あり"
    elif kind == "number":
        ok = isinstance(v, (int, float)) and not isinstance(v, bool)
    elif kind == "date":
        ok = isinstance(v, str) and len(v) == 10 and v[4] == "-" and v[7] == "-"
    elif kind == "time":
        ok = isinstance(v, str) and len(v) == 5 and v[2] == ":"
    else:
        ok = False
    if not ok:
        raise ValueError(f"{where}: 型 {kind} に合わない {v!r}")


def main():
    cards = {}
    for path in glob.glob(os.path.join(HERE, "cards", "*.json")):
        with open(path, encoding="utf-8") as fh:
            card = json.load(fh)
        cards[card["id"]] = card

    # 表とカードの枠、表と facts の語がずれていないかを先に確かめる
    rule_of = {q["id"]: q["rule_id"] for q in read_jsonl("questions.jsonl")}
    rows = read_jsonl("facts.jsonl")
    for rule, (card_id, table) in TABLE.items():
        want, have = set(cards[card_id]["slots"]), set(table)
        if want != have:
            raise ValueError(f"{rule}: 枠のずれ {sorted(want ^ have)}")
        names = {n for r in rows if rule_of[r["id"]] == rule for n in r["facts"]}
        used = {n for facts, _ in table.values() for n in facts}
        if names != used:
            raise ValueError(f"{rule}: 事実の語のずれ {sorted(names ^ used)}")

    truth, omitted = {}, []
    counts = {}
    for row in sorted(rows, key=lambda r: r["id"]):
        sid, facts = row["id"], row["facts"]
        rule = rule_of[sid]
        card_id, table = TABLE[rule]
        out = {}
        for slot, (sources, fn) in table.items():
            if any(facts[s] == UNKNOWN for s in sources):
                out[slot] = None
                continue
            value = fn(facts)
            if value is SKIP:
                continue
            if isinstance(value, Omit):
                omitted.append((sid, slot, value.reason))
                continue
            check_value(f"{sid}/{slot}", cards[card_id]["slots"][slot], value)
            out[slot] = value
        # null は unknowns の事実に対応する枠だけ
        nulls = {s for s, v in out.items() if v is None}
        expected = {s for s, (src, _) in table.items() if set(src) & set(row["unknowns"])}
        if nulls != expected:
            raise ValueError(f"{sid}: null のずれ {sorted(nulls ^ expected)}")
        truth[sid] = out
        counts[rule] = counts.get(rule, 0) + 1

    with open(os.path.join(HERE, "truth.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(truth, fh, ensure_ascii=False, indent=1, sort_keys=True)
        fh.write("\n")

    print("場面数", len(truth), counts)
    print("省いた枠", len(omitted))
    for item in omitted:
        print(" ", item)


if __name__ == "__main__":
    sys.exit(main())
