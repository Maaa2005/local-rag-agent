"""ask90b の隠した事実 (facts.jsonl) を、規程カードの枠の値に機械的に写す。

truth.json = {場面id: {枠の名前: 値の配列 か null}}
結論の答え (expected_conclusion) は読まない。questions.jsonl は id と rule_id だけ使う。
"""
import datetime as dt
import json
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
SRC = pathlib.Path(r"C:\Users\mtkzi\dev\.wt\lr-askback\evaluation\v2\ask90b")
UNKNOWN = "不明"
NONE = ["なし"]


class Omit(Exception):
    """facts から値を作れない枠。truth から省き、理由を出す。"""


def same(fact):
    """facts の値をそのまま 1 要素の配列にする。"""
    return fact, lambda v, f: [v]


def yesno(fact):
    """真偽をあり/なしにする。"""
    def conv(v, f):
        assert isinstance(v, bool), (fact, v)
        return ["あり" if v else "なし"]
    return fact, conv


def date_or_none(fact):
    """facts の null は「その日が無い人」なので なし にする。"""
    return fact, lambda v, f: NONE if v is None else [v]


def had_date(fact):
    """日の有無をあり/なしにする。"""
    return fact, lambda v, f: ["なし" if v is None else "あり"]


def omit(fact, reason):
    """facts に無い形の枠。fact が不明の時だけ null になる。"""
    def conv(v, f):
        raise Omit(reason)
    return fact, conv


def tax_included(v, f):
    if f["税区分"] == "税込":
        return [v]
    # 税抜きは元の枠に値があるのでコードが計算する。
    return "derive"


def tax_excluded(v, f):
    if f["税区分"] == "税抜":
        return [v]
    return NONE  # 領収書の額が税抜きではない人には当たらない量


def best_score(v, f):
    """応募締切日の 2 年前の同じ日から応募締切日までの最高点。"""
    end = dt.date.fromisoformat(f["応募締切日"])
    start = end.replace(year=end.year - 2)
    scores = [p for d, p in v if start <= dt.date.fromisoformat(d) <= end]
    if not scores:
        raise Omit("2年の範囲に検定が無い")
    return [max(scores)]


def grade_b_or_above(v, f):
    assert v in ("S", "A", "B", "C", "D"), v
    return ["あり" if v in ("S", "A", "B") else "なし"]


# 規程ごとの対応表: 枠 -> (事実の名前, 変換)。
# 表に無い枠は derive で計算される枠として後で扱う。
TABLE = {
    "独身寮入寮規程": {
        "正社員": yesno("正社員"),
        "配偶者との同居": yesno("配偶者と同居"),
        "通勤時間": same("通勤分"),
        "通勤時間の時間数": omit("通勤分", "事実は分で持ち、時間で示されていない（通勤時間に分を直接書いた）"),
        "生年月日": same("生年月日"),
        "入寮希望月の初日": same("入寮希望月初日"),
        "半壊以上の認定": had_date("半壊以上認定日"),
        "認定の日": date_or_none("半壊以上認定日"),
        "障害者手帳": yesno("障害者手帳"),
        "産業医の認定": yesno("産業医の軽減認定"),
    },
    "語学研修選抜規程": {
        "正社員": yesno("正社員"),
        "入社日": same("入社日"),
        "応募締切日": same("応募締切日"),
        "検定点数": ("検定履歴", best_score),
        "人事評価B以上": ("直前評価", grade_b_or_above),
        "復職日": date_or_none("育休復職日"),
        "海外赴任の内示": yesno("海外赴任内示"),
        "過去に修了": yesno("過去に修了"),
    },
    "業務端末持出規程": {
        "持出日数": same("持出日数"),
        "持出日": omit("持出日数", "事実に日付が無く日数だけ（持出日数に直接書いた）"),
        "返却日": omit("持出日数", "事実に日付が無く日数だけ（持出日数に直接書いた）"),
        "顧客情報件数": same("顧客情報件数"),
        "暗号化": yesno("暗号化"),
        "台帳登録": yesno("資産台帳登録"),
        "事業継続要員": yesno("事業継続要員"),
        "両立支援認定": yesno("両立支援認定"),
    },
    "海外出張渡航規程": {
        "出張命令": yesno("部長の出張命令"),
        "社内危険度": same("渡航先危険度"),
        "出発日": same("出発日"),
        "渡航届を出した日": same("渡航届提出日"),
        "配偶者又は子が住んでいる": yesno("渡航先に配偶者か子が居住"),
        "妊娠中": yesno("妊娠中"),
        "救援要員": yesno("救援要員"),
    },
    "休日出勤振替規程": {
        "事前の命令": yesno("所属長の事前命令"),
        "開始時刻": same("開始時刻"),
        "終了時刻": same("終了時刻"),
        "休憩の時間": same("休憩分"),
        "出勤日": same("出勤日"),
        "振替の取得予定日": same("振替予定日"),
        "短時間勤務": yesno("育児介護の短時間勤務"),
        "産業医の制限": yesno("時間外制限"),
    },
    "予防接種費用補助規程": {
        "社員本人の接種": yesno("本人の接種"),
        "接種日": same("接種日"),
        "費用（税込み）": ("費用額", tax_included),
        "費用（税抜き）": ("費用額", tax_excluded),
        "提出日": same("領収書提出日"),
        "妊娠中": yesno("妊娠中"),
        "産業医から接種を勧められた": yesno("基礎疾患で勧奨"),
    },
}
# 規程ごとに値を使う補助の事実（枠に直接写らないが値に効く）。
EXTRA_FACTS = {"予防接種費用補助規程": {"税区分"}}
# 費用は額と税区分の両方が分かって初めて値になる。
DEPENDS = {("予防接種費用補助規程", "費用額"): ["税区分"]}


def names_in(expr, slots):
    """derive の式に出る枠の名前を集める。"""
    if isinstance(expr, str):
        return [expr] if expr in slots else []
    if isinstance(expr, list):
        return [n for e in expr for n in names_in(e, slots)]
    if isinstance(expr, dict):
        return [n for e in expr.values() for n in names_in(e, slots)]
    return []


def read_jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def is_unknown(row, fact):
    value = row["facts"][fact]
    if value == UNKNOWN:
        assert fact in row["unknowns"], (row["id"], fact)
        return True
    assert fact not in row["unknowns"], (row["id"], fact)
    return False


def build(row, rule, card, omitted):
    sid = row["id"]
    slots = card["slots"]
    table = TABLE[rule]
    out = {}
    used = set(EXTRA_FACTS.get(rule, set()))
    for slot, (fact, conv) in table.items():
        assert slot in slots, (rule, slot)
        used.add(fact)
        deps = [fact] + DEPENDS.get((rule, fact), [])
        if any(is_unknown(row, d) for d in deps):
            out[slot] = None
            continue
        try:
            value = conv(row["facts"][fact], row["facts"])
        except Omit as e:
            omitted.append((sid, slot, str(e)))
            continue
        if value != "derive":
            out[slot] = value
    extra = set(row["facts"]) - used
    assert not extra, (sid, "写していない事実", extra)
    for slot, expr in card.get("derive", {}).items():
        if slot in out or any(o[:2] == (sid, slot) for o in omitted):
            continue
        sources = names_in(expr, slots)
        assert sources, slot
        if any(s in out and out[s] is None for s in sources):
            out[slot] = None
        elif any(out.get(s) == NONE for s in sources):
            out[slot] = NONE
        else:
            assert all(s in out for s in sources), (sid, slot, sources)
    return out


def validate(row, card, truth):
    """型と書き方を確かめる。"""
    sid = row["id"]
    derive = card.get("derive", {})
    for slot, value in truth.items():
        info = card["slots"][slot]
        if value is None:
            assert row["unknowns"], (sid, slot, "unknowns が無い場面の null")
            continue
        assert isinstance(value, list) and len(value) == 1, (sid, slot, value)
        v = value[0]
        if v == "なし":
            continue
        kind = info["type"]
        if kind == "bool":
            assert v == "あり", (sid, slot, v)
        elif kind == "number":
            assert isinstance(v, (int, float)) and not isinstance(v, bool), (sid, slot, v)
        elif kind == "date":
            dt.date.fromisoformat(v)
        elif kind == "time":
            assert re.fullmatch(r"\d\d:\d\d", v), (sid, slot, v)
        else:
            raise AssertionError((sid, slot, kind))
    # 元の枠が全部数や日付で揃っているのに、計算される枠まで書いていないか
    for slot, expr in derive.items():
        if truth.get(slot) in (None, NONE) or slot not in truth:
            continue
        sources = names_in(expr, card["slots"])
        real = [s for s in sources if truth.get(s) not in (None, NONE) and s in truth]
        assert len(real) < len(sources), (sid, slot, "元の枠があるのに書いた")


def main():
    cards = {}
    for path in (HERE / "cards").glob("*.json"):
        card = json.loads(path.read_text(encoding="utf-8"))
        cards[card["doc"]["source"]] = card
    assert set(cards) == set(TABLE), set(cards) ^ set(TABLE)
    rule_of = {q["id"]: q["rule_id"] for q in read_jsonl(SRC / "questions.jsonl")}
    rows = read_jsonl(SRC / "facts.jsonl")
    assert len(rows) == len(rule_of) == 90
    # 表が全部の枠を受け持つか（derive の枠は表に無くてよい）
    for rule, card in cards.items():
        left = set(card["slots"]) - set(TABLE[rule]) - set(card.get("derive", {}))
        assert not left, (rule, left)

    truth = {}
    omitted = []
    for row in sorted(rows, key=lambda r: r["id"]):
        rule = rule_of[row["id"]]
        truth[row["id"]] = build(row, rule, cards[rule], omitted)
        validate(row, cards[rule], truth[row["id"]])
    # unknowns の事実は必ずどこかの枠の null になる
    for row in rows:
        if row["unknowns"]:
            assert any(v is None for v in truth[row["id"]].values()), row["id"]

    (HERE / "truth.json").write_text(
        json.dumps(truth, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print("場面数", len(truth))
    print("省いた枠", len(omitted))
    for item in omitted:
        print(*item, sep="\t")
    print("null の枠")
    for sid, slots in truth.items():
        nulls = [s for s, v in slots.items() if v is None]
        if nulls:
            print(sid, rule_of[sid], ",".join(nulls), sep="\t")


if __name__ == "__main__":
    main()
