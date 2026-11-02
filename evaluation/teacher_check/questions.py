"""教師の確認（life #24）の 96 問。正解は rules.py が facts から出す（問の文から出さない）。

区分（2026-11-02 の協議で固定）:
- boundary（境目）48 問: 16 の条件 × 境目の下・ちょうど・上
- paraphrase（言い換え）24 問: 8 の条件 × 下・ちょうど・上。値を別の単位・日付・割合・計算で出す
- exception（例外・保留）24 問: 例外が当たる 6・例外に近いが当たらない 6・情報が足りず判断できない 6・情報が足りないが結論は決まる 6

文は人が書き、facts と食い違わないかは別の確認役が問だけを見て答えて照合する（seal.py の前の段）。
"""
from __future__ import annotations

import random
from datetime import date
from fractions import Fraction

from evaluation.teacher_check.rules import OUTCOMES, UNDECIDED, base_date, expected_conclusion

BASE_2027 = base_date(2027)
STAGE1_SEED = 20261102
STAGE1_PER_CATEGORY = 4


def footer(outcome: str) -> str:
    o = OUTCOMES[outcome]
    return ("答えの最後の2行は、次の形で書いてください。\n"
            "根拠: 判断に使った条（第○条の形で）\n"
            f"結論: 「{o.yes}」「{o.no}」「{UNDECIDED}」のどれか1つ")


def yen(n: int) -> str:
    return f"{n:,}円"


def _boundary() -> list[tuple]:
    """(outcome, facts, 文) を返す。境目の下・ちょうど・上の順。"""
    q = []
    for amt in (1_990_000, 2_000_000, 2_010_000):
        q.append(("loan_within_limit", {"amount": amt, "housing_purchase": False},
                  f"勤続3年の正社員です。子どもの教育費として{yen(amt)}を借りたいと考えています。この貸付額は、貸付限度額の範囲内ですか。"))
    for m in (59, 60, 61):
        q.append(("loan_repay_ok", {"repay_months": m},
                  f"社内貸付の返済期間を{m}か月にしたいと考えています。この返済期間は認められますか。"))
    for m in (11, 12, 13):
        q.append(("loan_eligible", {"tenure_months": m},
                  f"勤続{m}か月の正社員です。社内貸付の対象者に当たりますか。"))
    for h, s in ((Fraction(19, 2), "9.5"), (Fraction(10), "10"), (Fraction(21, 2), "10.5")):
        q.append(("side_permission_needed", {"side_hours": h},
                  f"副業を1週あたり{s}時間する予定です。所属長の許可が必要ですか。それとも事前の届出のみで足りますか。"))
    for x in (199_000, 200_000, 201_000):
        q.append(("side_report_needed", {"monthly_income": x},
                  f"副業による収入が月額{yen(x)}になる見込みです。人事部への収入額の報告は必要ですか。"))
    for y in (9, 10, 11):
        q.append(("long_service_leave", {"tenure_years": y, "leave_days": 0},
                  f"基準日に勤続満{y}年になる社員です。これまでに休職したことはありません。この基準日に永年勤続休暇は付与されますか。"))
    for d in (364, 365, 366):
        q.append(("long_service_leave", {"tenure_years": 15, "leave_days": d},
                  f"基準日に勤続満15年になる社員です。入社日から基準日までの休職期間は通算{d}日です。この基準日に永年勤続休暇は付与されますか。"))
    for x in (49_000, 50_000, 51_000):
        q.append(("housing_paid", {"monthly_rent": x, "age": 28, "dependent_child": False, "distance_m": 5_000},
                  f"自分で借りたアパートに住んでいて、家賃は月額{yen(x)}です。基準日の年齢は28歳で、扶養している子はいません。住宅から勤務地までは5kmです。住宅手当は支給されますか。"))
    for a in (34, 35, 36):
        q.append(("housing_paid", {"monthly_rent": 70_000, "age": a, "dependent_child": False, "distance_m": 8_000},
                  f"自分で借りたマンションに住んでいて、家賃は月額70,000円、住宅から勤務地までは8kmです。基準日の年齢は{a}歳で、扶養している子はいません。住宅手当は支給されますか。"))
    for dm, s in ((1_900, "1.9"), (2_000, "2"), (2_100, "2.1")):
        q.append(("housing_paid", {"monthly_rent": 60_000, "age": 30, "dependent_child": False, "distance_m": dm},
                  f"自分で借りたアパートに住んでいて、家賃は月額60,000円です。基準日の年齢は30歳で、扶養している子はいません。住宅から勤務地までの距離は{s}kmです。住宅手当は支給されますか。"))
    for x in (2_990_000, 3_000_000, 3_010_000):
        q.append(("trade_review_needed", {"deal_amount": x, "listed": False},
                  f"上場していない会社と新しく取引を始めます。1回あたりの取引額は{yen(x)}です。経理部の与信審査は必要ですか。"))
    for y in (2, 3, 4):
        q.append(("trade_prepay_needed", {"years_since_founding": y},
                  f"設立から満{y}年の会社と新しく取引を始めます。この取引で前払いを条件にする必要はありますか。"))
    for x in (19, 20, 21):
        q.append(("trade_forbidden", {"equity_ratio_pct": x},
                  f"直近の決算で自己資本比率が{x}%の会社があります。この会社と新たに取引を始めることは、規程で禁じられていますか。"))
    for d in (29, 30, 31):
        q.append(("trade_stop", {"overdue_days": d},
                  f"ある取引先から、支払期日を過ぎても入金がない日数が{d}日になりました。この取引先との取引を停止しますか。"))
    for x in (4_900, 5_000, 5_100):
        q.append(("party_full_subsidy", {"per_person": x},
                  f"懇親会を開き、参加人数・申請の期限・酒類の割合の決まりはすべて満たしています。1人あたりの費用は{yen(x)}でした。1人あたりの費用は全額補助されますか。"))
    for n in (9, 10, 11):
        q.append(("party_size_ok", {"participants": n, "cross_dept": False},
                  f"同じ部署の社員だけで懇親会を開きます。参加者は{n}名です。参加人数の点で、この懇親会は補助の対象になりますか。"))
    return q


def _paraphrase() -> list[tuple]:
    q = []
    for x, s in ((1_490_000, "149万円"), (1_500_000, "150万円"), (1_510_000, "151万円")):
        q.append(("loan_review_needed", {"amount": x, "annual_income": 4_500_000},
                  f"前年の年収が450万円の社員が、社内貸付を{s}申し込みます。人事部長の審査は必要ですか。"))
    for per_day, days in ((2, 7), (3, 5), (4, 4)):
        q.append(("side_allowed", {"side_hours": Fraction(per_day * days), "at_home": False, "kind": "接客", "self_scheduled": False},
                  f"副業として、飲食店での接客の仕事を1日{per_day}時間、週{days}日する予定です。勤務する時間はお店が決めます。所属長の許可はすでに得ています。この副業は認められますか。"))
    for entry in (date(2017, 6, 10), date(2017, 1, 10), date(2016, 1, 10)):
        q.append(("long_service_leave", {"entry_date": entry, "base_date": BASE_2027, "leave_days": 0},
                  f"{entry.year}年{entry.month}月{entry.day}日に入社した社員です。これまでに休職したことはありません。2027年4月1日の基準日に、永年勤続休暇は付与されますか。"))
    for birth in (date(1992, 10, 15), date(1991, 10, 15), date(1990, 10, 15)):
        q.append(("housing_paid", {"birth_date": birth, "base_date": BASE_2027, "dependent_child": False,
                                   "monthly_rent": 65_000, "distance_m": 4_000},
                  f"{birth.year}年{birth.month}月{birth.day}日生まれの社員です。扶養している子はいません。自分で借りた部屋に住み、家賃は月額6万5千円で、住宅から勤務地までは4kmです。2027年4月1日の基準日で判断すると、住宅手当は支給されますか。"))
    for day in (5, 6, 7):
        q.append(("party_deadline_ok", {"applied_on": date(2027, 6, day), "event_on": date(2027, 6, 20)},
                  f"懇親会を2027年6月20日に開きます。総務部への補助の申請は2027年6月{day}日に出しました。この申請は期限内のものとして受け付けられますか。"))
    for x in (59_000, 60_000, 61_000):
        q.append(("party_subsidy", {"participants": 12, "cross_dept": False, "applied_on": date(2027, 5, 31),
                                    "event_on": date(2027, 6, 20), "alcohol": x, "total": 120_000},
                  f"同じ部署の12名で、2027年6月20日に懇親会を開きました。補助の申請は2027年5月31日に総務部へ出しています。費用の総額は12万円で、そのうち酒類の代金は{yen(x)}でした。領収書は開催後10日以内に提出します。この懇親会は補助を受けられますか。"))
    for x in (790, 800, 810):
        q.append(("trade_credit_ok", {"credit_limit": x * 10_000, "net_assets": 80_000_000},
                  f"直近の決算の純資産が8,000万円の会社があります。この会社の与信限度額を{x}万円に設定できますか。"))
    for annual in (588_000, 600_000, 612_000):
        q.append(("housing_paid", {"monthly_rent": Fraction(annual, 12), "age": 29, "dependent_child": False, "distance_m": 3_000},
                  f"自分で借りたアパートに住んでいて、家賃は1年分で{yen(annual)}です（毎月同じ額を払います）。基準日の年齢は29歳で、扶養している子はいません。住宅から勤務地までは3kmです。住宅手当は支給されますか。"))
    return q


def _exception() -> list[tuple]:
    q = []
    # 例外が当たる 6
    q.append(("loan_within_limit", {"amount": 3_000_000, "housing_purchase": True},
              "勤続6年の正社員です。自分が住むマンションを買うために、300万円を借りたいと考えています。この貸付額は、貸付限度額の範囲内ですか。"))
    q.append(("housing_paid", {"age": 37, "dependent_child": True, "monthly_rent": 80_000, "distance_m": 6_000},
              "基準日の年齢は37歳で、扶養している6歳の子がいます。自分で借りた部屋に住み、家賃は月額80,000円、住宅から勤務地までは6kmです。住宅手当は支給されますか。"))
    q.append(("trade_review_needed", {"deal_amount": 5_000_000, "listed": True},
              "東京証券取引所に上場している会社と新しく取引を始めます。1回あたりの取引額は500万円です。経理部の与信審査は必要ですか。"))
    q.append(("side_allowed", {"side_hours": Fraction(20), "at_home": True, "kind": "執筆", "self_scheduled": True},
              "副業として、自宅で小説の原稿を書きます。書く時間は自分で自由に決められます。1週あたり20時間ほど書く予定です。所属長の許可はすでに得ています。この副業は認められますか。"))
    q.append(("long_service_leave", {"tenure_years": 12, "leave_days": 400},
              "基準日に勤続満12年になる社員です。入社日から基準日までの休職期間は通算400日です。この基準日に永年勤続休暇は付与されますか。"))
    q.append(("party_size_ok", {"participants": 7, "cross_dept": True},
              "営業部と開発部の社員が参加する交流会を開きます。参加者は7名です。参加人数の点で、この交流会は補助の対象になりますか。"))
    # 例外に近いが当たらない 6
    q.append(("loan_within_limit", {"amount": 5_200_000, "housing_purchase": True},
              "勤続8年の正社員です。自分が住む一戸建てを買うために、520万円を借りたいと考えています。この貸付額は、貸付限度額の範囲内ですか。"))
    q.append(("housing_paid", {"age": 37, "dependent_child": False, "monthly_rent": 80_000, "distance_m": 6_000},
              "基準日の年齢は37歳です。子が1人いますが、すでに就職して独立しており、扶養していません。自分で借りた部屋に住み、家賃は月額80,000円、住宅から勤務地までは6kmです。住宅手当は支給されますか。"))
    q.append(("trade_review_needed", {"deal_amount": 5_000_000, "listed": False},
              "来年の上場を目指しているが、まだ上場していない会社と新しく取引を始めます。1回あたりの取引額は500万円です。経理部の与信審査は必要ですか。"))
    q.append(("side_allowed", {"side_hours": Fraction(20), "at_home": True, "kind": "データ入力", "self_scheduled": True},
              "副業として、自宅でデータ入力の仕事をします。作業する時間は自分で自由に決められます。1週あたり20時間ほど作業する予定です。所属長の許可はすでに得ています。この副業は認められますか。"))
    q.append(("long_service_leave", {"tenure_years": 12, "leave_days": 365},
              "基準日に勤続満12年になる社員です。入社日から基準日までに休職が2回あり、1回目は200日、2回目は165日でした。この基準日に永年勤続休暇は付与されますか。"))
    q.append(("party_size_ok", {"participants": 4, "cross_dept": True},
              "営業部と開発部の社員が参加する交流会を開きます。参加者は4名です。参加人数の点で、この交流会は補助の対象になりますか。"))
    # 情報が足りず判断できない 6
    q.append(("loan_available", {"amount": 1_500_000, "housing_purchase": False},
              "正社員です。子どもの教育費として150万円を借りたいと考えています。社内貸付を受けられますか。"))
    q.append(("housing_paid", {"monthly_rent": 60_000, "age": 30, "dependent_child": False},
              "自分で借りたアパートに住んでいて、家賃は月額60,000円です。基準日の年齢は30歳で、扶養している子はいません。住宅手当は支給されますか。"))
    q.append(("trade_prepay_needed", {"equity_ratio_pct": 35},
              "直近の決算で自己資本比率が35%の会社と新しく取引を始めます。この取引で前払いを条件にする必要はありますか。"))
    q.append(("party_subsidy", {"participants": 12, "cross_dept": False, "alcohol": 30_000, "total": 100_000},
              "同じ部署の12名で懇親会を開きました。費用の総額は10万円で、そのうち酒類の代金は30,000円でした。領収書は開催後10日以内に提出します。この懇親会は補助を受けられますか。"))
    q.append(("side_permission_needed", {"monthly_income": 250_000},
              "副業による収入は月額250,000円になる見込みです。この副業をするのに、所属長の許可が必要ですか。それとも事前の届出のみで足りますか。"))
    q.append(("loan_review_needed", {"amount": 1_800_000},
              "社員が社内貸付を180万円申し込みます。人事部長の審査は必要ですか。"))
    # 情報が足りないが結論は決まる 6
    q.append(("loan_available", {"amount": 2_500_000, "housing_purchase": False},
              "正社員です。車を買うために250万円を借りたいと考えています。社内貸付を受けられますか。"))
    q.append(("housing_paid", {"monthly_rent": 40_000},
              "自分で借りたアパートに住んでいて、家賃は月額40,000円です。住宅手当は支給されますか。"))
    q.append(("trade_forbidden", {"equity_ratio_pct": 15},
              "直近の決算で自己資本比率が15%の会社があります。設立からの年数は分かりません。この会社と新たに取引を始めることは、規程で禁じられていますか。"))
    q.append(("party_size_ok", {"participants": 4},
              "懇親会の参加者は4名です。参加者がどの部署の社員かは、まだ確認していません。参加人数の点で、この懇親会は補助の対象になりますか。"))
    q.append(("long_service_leave", {"tenure_years": 8},
              "基準日に勤続満8年になる社員です。休職の記録はまだ確認できていません。この基準日に永年勤続休暇は付与されますか。"))
    q.append(("housing_paid", {"age": 40, "dependent_child": False},
              "基準日の年齢は40歳で、扶養している子はいません。家賃と、住宅から勤務地までの距離はまだ分かりません。住宅手当は支給されますか。"))
    return q


def build() -> list[dict]:
    """96 問を作る。id は区分の順に T001〜T096。stage1 は区分ごとに決まった種で 4 問ずつ選ぶ。"""
    rows = []
    for category, specs in (("boundary", _boundary()), ("paraphrase", _paraphrase()), ("exception", _exception())):
        for outcome, facts, text in specs:
            o = OUTCOMES[outcome]
            rows.append({
                "id": f"T{len(rows) + 1:03d}",
                "category": category,
                "doc": o.doc,
                "outcome": outcome,
                "question": text + "\n" + footer(outcome),
                "options": [o.yes, o.no, UNDECIDED],
                "expected_conclusion": expected_conclusion(outcome, facts),
                "expected_articles": list(o.articles),
                "facts": {k: (str(v) if isinstance(v, (date, Fraction)) else v) for k, v in facts.items()},
            })
    rng = random.Random(STAGE1_SEED)
    for category in ("boundary", "paraphrase", "exception"):
        ids = [r["id"] for r in rows if r["category"] == category]
        chosen = set(rng.sample(ids, STAGE1_PER_CATEGORY))
        for r in rows:
            if r["id"] in chosen:
                r["stage1"] = True
    for r in rows:
        r.setdefault("stage1", False)
    return rows
