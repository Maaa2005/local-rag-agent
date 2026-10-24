"""文ごと採点役 (evaluation/judges_sentence.py) のテスト。採点役は偽物 (決めた文字列を流す)。

確かめること:
- 厳格なパーサ: 要素数・文番号・型が期待と違えば ScoringFailure (phi-4 の [1,1,1]・ELYZA の JP4 [1,0,1] を含む)
- 空の回答は採点役を呼ばない / 忠実さと関連性は別の呼び出し / 期待キーワードを渡さない / 本文は全文
- judges.faithfulness_from_grounded_array は黙って切り詰めない
"""
from __future__ import annotations

import asyncio
import json

import pytest

from evaluation import judges, judges_sentence as js
from evaluation.metrics import split_sentences

SF = js.ScoringFailure


def _faith(items):
    """items: (文番号, 判定, 根拠) の並び → 採点役の出力 JSON。"""
    return json.dumps({"判定": [{"文番号": n, "判定": v, "根拠": r} for n, v, r in items]}, ensure_ascii=False)


def _faith_v(verdicts):
    return _faith([(i + 1, v, "根拠") for i, v in enumerate(verdicts)])


def _rel(v, reason="理由"):
    return json.dumps({"関連": v, "根拠": reason}, ensure_ascii=False)


class FakeProvider:
    def __init__(self, faith: str, rel: str):
        self.faith, self.rel = faith, rel
        self.calls: list[list[dict]] = []
        self.last_usage = None

    async def stream(self, messages):
        self.calls.append(messages)
        text = self.faith if "【回答の文】" in messages[-1]["content"] else self.rel
        for i in range(0, len(text), 5):
            yield text[i:i + 5]


def _run(provider, question="質問です。", chunks=None, answer="一文目です。二文目です。"):
    return asyncio.run(js.judge_sentences(provider, question, chunks or [{"source_file": "a.md", "content": "本文"}], answer))


# ─── パーサ ──────────────────────────────────────────────────────
def test_parse_ok_and_reorder():
    assert js.parse_sentence_verdicts(_faith_v([1, 0, 1]), 3) == ([1, 0, 1], ["根拠"] * 3)
    shuffled = _faith([(2, 0, "b"), (1, 1, "a")])
    assert js.parse_sentence_verdicts(shuffled, 2) == ([1, 0], ["a", "b"])  # 順が入れ替わっただけなら番号順に直す


def test_parse_accepts_text_around_json():
    assert js.parse_sentence_verdicts("結果:\n" + _faith_v([1]) + "\n以上", 1) == ([1], ["根拠"])


def test_one_sentence_answer_with_three_verdicts_fails():
    # phi-4 で実際に起きた形: 1 文の回答に [1,1,1]
    with pytest.raises(SF, match="要素数"):
        js.parse_sentence_verdicts(_faith_v([1, 1, 1]), 1)


def test_jp4_with_three_verdicts_fails():
    # ELYZA で実際に起きた形: JP4 (2 文) に [1,0,1]
    from evaluation import judge_precheck_s as jps

    jp4 = next(r for r in jps.load_fixture(jps.FIXTURE) if r["id"] == "JP4")
    n = len(split_sentences(jp4["answer"]))
    assert n == 2
    with pytest.raises(SF):
        js.parse_sentence_verdicts(_faith_v([1, 0, 1]), n)


@pytest.mark.parametrize("text", [
    "JSON ではありません",
    "{壊れた",
    "[1, 0]",                                               # オブジェクトでない
    json.dumps({"判定": "なし"}),                             # リストでない
    json.dumps({"verdicts": []}),                           # "判定" が無い
    _faith([(1, 1, "a")]),                                  # 要素が足りない
    _faith([(1, 1, "a"), (3, 1, "b")]),                     # 抜け (2 が無い)
    _faith([(1, 1, "a"), (1, 0, "b")]),                     # 重複
    _faith([(0, 1, "a"), (1, 1, "b")]),                     # 範囲外 (0)
    _faith([(1, 1, "a"), (3, 1, "b")]),                     # 範囲外 (3)
    _faith([(True, 1, "a"), (2, 1, "b")]),                  # 文番号が bool
    _faith([(1.0, 1, "a"), (2, 1, "b")]),                   # 文番号が float
    _faith([("1", 1, "a"), (2, 1, "b")]),                   # 文番号が文字列
    _faith([(1, True, "a"), (2, 1, "b")]),                  # 判定が bool
    _faith([(1, 1.0, "a"), (2, 1, "b")]),                   # 判定が float
    _faith([(1, "1", "a"), (2, 1, "b")]),                   # 判定が文字列
    _faith([(1, 2, "a"), (2, 1, "b")]),                     # 判定が 0/1 でない
    _faith([(1, None, "a"), (2, 1, "b")]),
    _faith([(1, 1, None), (2, 1, "b")]),                    # 根拠が文字列でない
    _faith([(1, 1, 5), (2, 1, "b")]),
    json.dumps({"判定": [{"文番号": 1, "判定": 1}, {"文番号": 2, "判定": 1, "根拠": "b"}]}),  # 根拠が無い
    json.dumps({"判定": [1, 0]}),                            # 要素がオブジェクトでない
])
def test_parse_sentence_verdicts_failures(text):
    with pytest.raises(SF):
        js.parse_sentence_verdicts(text, 2)


def test_parse_relevance():
    assert js.parse_relevance(_rel(1)) == 1
    assert js.parse_relevance("出力: " + _rel(0)) == 0


@pytest.mark.parametrize("text", [
    "なし", "{壊れた", "[1]", json.dumps({"根拠": "x"}),
    json.dumps({"関連": True}), json.dumps({"関連": 1.0}), json.dumps({"関連": "1"}),
    json.dumps({"関連": 2}), json.dumps({"関連": None}),
])
def test_parse_relevance_failures(text):
    with pytest.raises(SF):
        js.parse_relevance(text)


# ─── judge_sentences ─────────────────────────────────────────────
def test_judge_sentences_ok():
    p = FakeProvider(_faith_v([1, 0]), _rel(1))
    j = _run(p)
    assert j.sentences == ["一文目です。", "二文目です。"]
    assert j.verdicts == [1, 0] and j.reasons == ["根拠", "根拠"]
    assert j.faithfulness == 0.5 and j.relevance == 1
    assert set(j.raw) == {"faithfulness", "relevance"}
    assert j.raw["relevance"] == _rel(1)
    assert len(p.calls) == 2  # 忠実さと関連性は別々の呼び出し


@pytest.mark.parametrize("answer", ["", "   ", "\n"])
def test_empty_answer_does_not_call_judge(answer):
    p = FakeProvider(_faith_v([1]), _rel(1))
    j = _run(p, answer=answer)
    assert p.calls == []
    assert (j.sentences, j.verdicts, j.reasons, j.faithfulness, j.relevance) == ([], [], [], 0.0, 0)


def test_wrong_count_raises_with_raw():
    p = FakeProvider(_faith_v([1, 1, 1]), _rel(1))
    with pytest.raises(SF) as exc:
        _run(p)
    assert exc.value.raw["faithfulness"] == _faith_v([1, 1, 1])


def test_bad_relevance_raises():
    with pytest.raises(SF):
        _run(FakeProvider(_faith_v([1, 1]), _rel(True)))


def test_prompts_have_no_keywords_and_full_chunk():
    long_body = "あ" * 700 + "末尾の目印"
    p = FakeProvider(_faith_v([1, 1]), _rel(1))
    _run(p, chunks=[{"source_file": "/w/規程.md", "content": long_body}])
    faith_user = p.calls[0][-1]["content"]
    rel_user = p.calls[1][-1]["content"]
    assert long_body in faith_user                      # 600 字で切られない
    assert "/w/規程.md" in faith_user                    # 出典名
    assert "1. 一文目です。\n2. 二文目です。" in faith_user  # 1 から番号を付ける
    for text in (faith_user, rel_user):
        assert "キーワード" not in text
    assert "末尾の目印" not in rel_user and "本文" not in rel_user.split("判定の基準")[0]  # 関連性は質問と回答だけ
    import inspect
    assert "expected_keywords" not in inspect.signature(js.judge_sentences).parameters


def test_prompt_sha256_is_stable_hex():
    h = js.prompt_sha256()
    assert len(h) == 64 and h == js.prompt_sha256()
    int(h, 16)


# ─── faithfulness_from_grounded_array ────────────────────────────
def test_grounded_array_length_mismatch_raises():
    with pytest.raises(ValueError):
        judges.faithfulness_from_grounded_array("一文目です。二文目です。", [1])
    with pytest.raises(ValueError):
        judges.faithfulness_from_grounded_array("一文目です。", [1, 1, 1])
    with pytest.raises(ValueError):
        judges.faithfulness_from_grounded_array("一文目です。", [])


def test_grounded_array_mean_and_empty():
    assert judges.faithfulness_from_grounded_array("一文目です。二文目です。", [1, 0]) == 0.5
    assert judges.faithfulness_from_grounded_array("一文目です。", [1]) == 1.0
    assert judges.faithfulness_from_grounded_array("", []) == 0.0
    assert judges.faithfulness_from_grounded_array("", [1]) == 0.0


@pytest.mark.parametrize("bad", [[True], [1.0], ["1"], [2], [None]])
def test_grounded_array_bad_values_raise(bad):
    with pytest.raises(ValueError):
        judges.faithfulness_from_grounded_array("一文目です。", bad)
