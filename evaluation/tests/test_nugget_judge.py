"""evaluation.nugget.judge / draft のテスト。判定役は偽物（ネットに出ない）。"""
from __future__ import annotations

import io
import json
import re

import pytest

from evaluation.nugget import draft, judge
from evaluation.tests.test_nugget_schema import CORPUS, make_rows


class FakeJudge:
    """送る関数の偽物。

    - queue: 先頭から順に返す（文字列か例外）。空になったら自動応答。
    - always: {1|2: 文字列} 呼び出し種別ごとに常にその文字列を返す。
    - 自動応答: 文に「矛盾」→相反、「要点」→必須、それ以外→補足。
      事実の判定は回答中の <<nid:full|partial|none>> から読む（無ければ none）。
      full/partial の evidence は、その印そのもの（回答に一字一句ある部分）を写す。
    """

    def __init__(self, queue=None, always=None):
        self.calls: list[list] = []
        self.queue = list(queue or [])
        self.always = dict(always or {})

    def __call__(self, messages):
        self.calls.append(messages)
        if self.queue:
            item = self.queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        prompt = messages[1]["content"]
        kind = 1 if "呼び出し1" in prompt else 2
        if kind in self.always:
            return self.always[kind]
        return self._auto1(prompt) if kind == 1 else self._auto2(prompt)

    @staticmethod
    def _auto1(prompt):
        sec = prompt.split("## 回答の文")[1].split("## 出力")[0]
        sents = re.findall(r"^\d+\. (.*)$", sec, re.M)
        labels = ["相反" if "矛盾" in s else "必須" if "要点" in s else "補足" for s in sents]
        return "判定です。\n```json\n" + json.dumps({"labels": labels}, ensure_ascii=False) + "\n```\n以上。"

    @staticmethod
    def _auto2(prompt):
        if "## 回答の文" in prompt:  # NJ-3: 印のある文の番号を返す
            sec = prompt.split("## 回答の文")[1].split("## 必須の事実")[0]
            sents = re.findall(r"^(\d+)\. (.*)$", sec, re.M)
            facts_sec = prompt.split("## 必須の事実")[1].split("## 出力")[0]
            out = []
            for nid in re.findall(r"^- (\S+): ", facts_sec, re.M):
                score, nums = "none", []
                for num, text in sents:
                    m = re.search(r"<<" + re.escape(nid) + r":(full|partial|none)>>", text)
                    if m:
                        score = m.group(1)
                        nums = [int(num)] if score != "none" else []
                out.append({"nid": nid, "score": score, "sentences": nums})
            return json.dumps({"facts": out}, ensure_ascii=False)
        answer = prompt.split("## 回答\n")[1].split("## 必須の事実")[0]
        facts_sec = prompt.split("## 必須の事実")[1].split("## 出力")[0]
        nids = re.findall(r"^- (\S+): ", facts_sec, re.M)
        out = []
        for nid in nids:
            m = re.search(r"<<" + re.escape(nid) + r":(full|partial|none)>>", answer)
            score = m.group(1) if m else "none"
            out.append({"nid": nid, "score": score, "evidence": m.group(0) if m and score != "none" else ""})
        return json.dumps({"facts": out}, ensure_ascii=False)


ROW1 = make_rows()[0]  # Q1: vital 2 + okay 1


def run_judge(answer, send, row=ROW1, **kw):
    return judge.judge_answer(
        send,
        rule_text=CORPUS[row["rule_id"]],
        question=row["question"],
        answer=answer,
        nuggets=row["nuggets"],
        **kw,
    )


def run_nj2(answer, send):
    return run_judge(answer, send, version="NJ-2")


# ─── JSON を寛容に読む ──────────────────────────────────────────
def test_extract_json_variants():
    assert judge.extract_json('{"a": 1}') == {"a": 1}
    assert judge.extract_json('前置き\n```json\n{"a": [1, 2]}\n```\n後ろ') == {"a": [1, 2]}
    assert judge.extract_json('結果は {"labels": ["必須"]} です。') == {"labels": ["必須"]}
    assert judge.extract_json('["必須", "補足"]') == ["必須", "補足"]
    assert judge.extract_json("参考 [1] を見て {\"k\": 2}") == {"k": 2}  # 文中の [1] より辞書を取る


@pytest.mark.parametrize("bad", ["", "   ", "JSON ではない", '{"a": 1,', "```json\n{oops\n```"])
def test_extract_json_rejects_garbage(bad):
    with pytest.raises(ValueError):
        judge.extract_json(bad)


# ─── 文の分割 ───────────────────────────────────────────────────
def test_split_answer_splits_conclusion_line():
    text = "貸与期間は6か月です。延長は1回までです。\n結論: 貸与できる"
    assert judge.split_answer(text) == ["貸与期間は6か月です。", "延長は1回までです。", "結論: 貸与できる"]


def test_split_answer_blank_lines_and_empty():
    assert judge.split_answer("a。\n\n  \nb") == ["a。", "b"]
    assert judge.split_answer("") == []


# ─── 解釈 ───────────────────────────────────────────────────────
def test_parse_labels_accepts_japanese_and_internal_names():
    raw = json.dumps({"labels": ["必須", "okay", {"label": "相反"}, "記載なし", "定型", "不要"]}, ensure_ascii=False)
    assert judge.parse_labels(raw, 6) == ["vital", "okay", "contradiction", "not_stated", "boilerplate", "irrelevant"]


def test_parse_labels_rejects_unknown_and_length_mismatch():
    with pytest.raises(ValueError, match="知らないラベル"):
        judge.parse_labels('{"labels": ["必須", "たぶん正しい"]}', 2)
    with pytest.raises(ValueError, match="文の数"):
        judge.parse_labels('{"labels": ["必須"]}', 2)


def test_parse_facts_variants_and_errors():
    nids = ["a", "b"]
    assert judge.parse_facts('{"facts": [{"nid": "b", "score": "partial"}, {"nid": "a", "score": "full"}]}', nids) == {
        "b": "partial", "a": "full"}
    assert judge.parse_facts('{"facts": ["すべて", "一部"]}', nids) == {"a": "full", "b": "partial"}
    assert judge.parse_facts("[1, 0.5]", nids) == {"a": "full", "b": "partial"}
    with pytest.raises(ValueError, match="知らない判定"):
        judge.parse_facts('{"facts": ["full", "mostly"]}', nids)
    with pytest.raises(ValueError, match="vital の数"):
        judge.parse_facts('{"facts": ["full"]}', nids)
    with pytest.raises(ValueError, match="重複"):
        judge.parse_facts('{"facts": [{"nid": "a", "score": "full"}, {"nid": "a", "score": "none"}]}', nids)
    with pytest.raises(ValueError, match="合わない"):
        judge.parse_facts('{"facts": [{"nid": "a", "score": "full"}, {"nid": "zzz", "score": "none"}]}', nids)


# ─── 判定の流れ ─────────────────────────────────────────────────
def test_judge_answer_ok_two_calls():
    fake = FakeJudge()
    ans = "要点です<<Q1-n1:full>><<Q1-n2:partial>>。\n結論: 貸与できる"
    res = run_judge(ans, fake)
    assert res["status"] == "ok" and res["retries"] == 0 and res["error"] is None
    assert res["labels"] == ["vital", "okay"]  # 2 文: 「要点です…。」と「結論: …」
    assert res["facts"] == {"Q1-n1": "full", "Q1-n2": "partial"}  # okay の Q1-n3 は出さない
    assert len(fake.calls) == 2


def test_prompts_carry_version_rule_question_and_numbered_sentences():
    fake = FakeJudge()
    run_judge("第一文です。第二文です。", fake)
    p1 = fake.calls[0][1]["content"]
    p2 = fake.calls[1][1]["content"]
    assert judge.JUDGE_VERSION == "NJ-3" and judge.JUDGE_VERSIONS == ("NJ-2", "NJ-3")
    for p in (p1, p2):
        assert "NJ-3" in p and CORPUS["貸与規程"] in p and ROW1["question"] in p
    assert "1. 第一文です。" in p1 and "2. 第二文です。" in p1 and "全 2 文" in p1
    assert "1. 第一文です。" in p2 and "2. 第二文です。" in p2 and "全 2 文" in p2  # NJ-3 は呼び出し 2 も番号付き
    assert "Q1-n3" in p1 and "Q1-n3" not in p2  # 呼び出し 1 は全事実、呼び出し 2 は vital だけ
    assert "JSON だけ" in p1 and "JSON だけ" in p2


def test_nj2_prompts_carry_nj2_and_unnumbered_answer():
    fake = FakeJudge()
    run_nj2("第一文です。第二文です。", fake)
    p1 = fake.calls[0][1]["content"]
    p2 = fake.calls[1][1]["content"]
    assert "NJ-2" in p1 and "NJ-2" in p2 and "NJ-3" not in p1 + p2
    assert "1. 第一文です。" in p1 and "## 回答\n第一文です。第二文です。" in p2


def test_unknown_version_is_rejected():
    with pytest.raises(ValueError, match="判定の版"):
        run_judge("要点です。", FakeJudge(), version="NJ-9")


def test_prompt1_says_conclusion_line_is_not_boilerplate():
    fake = FakeJudge()
    run_judge("要点です。\n結論: 貸与できる", fake)
    p1 = fake.calls[0][1]["content"]
    assert "「結論: …」の行は定型ではない" in p1
    assert "規程に合う結論なら 必須" in p1 and "食い違う結論なら 相反" in p1


def test_prompt2_asks_for_verbatim_evidence():
    fake = FakeJudge()
    run_nj2("要点です。", fake)
    p2 = fake.calls[1][1]["content"]
    assert '"evidence"' in p2 and "一字一句そのまま写す" in p2
    assert "full にしない" in p2 and "partial か none" in p2


def test_prompt2_nj3_asks_for_sentence_numbers():
    fake = FakeJudge()
    run_judge("要点です。", fake)
    p2 = fake.calls[1][1]["content"]
    assert '"sentences"' in p2 and "evidence" not in p2
    assert "回答の文の番号" in p2 and "full にしない" in p2 and "none の時は空でよい" in p2


# ─── evidence の機械の検査 ──────────────────────────────────────
def facts_reply(*items):
    return json.dumps({"facts": [{"nid": n, "score": sc, "evidence": ev} for n, sc, ev in items]}, ensure_ascii=False)


def test_evidence_not_in_answer_is_downgraded_to_none():
    reply = facts_reply(("Q1-n1", "full", "勤続1年11か月"), ("Q1-n2", "partial", "貸与は6か月"))
    res = run_nj2("貸与は6か月です。", FakeJudge(always={2: reply}))
    assert res["status"] == "ok"
    assert res["facts"] == {"Q1-n1": "none", "Q1-n2": "partial"}
    assert res["downgraded"] == ["Q1-n1"]
    assert res["evidence"] == {"Q1-n1": "勤続1年11か月", "Q1-n2": "貸与は6か月"}  # 下げても記録に残す


@pytest.mark.parametrize("level", ["full", "partial"])
def test_empty_evidence_is_downgraded(level):
    reply = facts_reply(("Q1-n1", level, ""), ("Q1-n2", level, "   "))
    res = run_nj2("貸与は6か月です。", FakeJudge(always={2: reply}))
    assert res["facts"] == {"Q1-n1": "none", "Q1-n2": "none"}
    assert res["downgraded"] == ["Q1-n1", "Q1-n2"]


def test_missing_evidence_key_is_downgraded():
    reply = json.dumps({"facts": [{"nid": "Q1-n1", "score": "full"}, {"nid": "Q1-n2", "score": "none"}]})
    res = run_nj2("貸与は6か月です。", FakeJudge(always={2: reply}))
    assert res["facts"] == {"Q1-n1": "none", "Q1-n2": "none"} and res["downgraded"] == ["Q1-n1"]


def test_whitespace_only_difference_is_not_downgraded():
    ans = "貸与は 6か月\n延長は\u30001回\u3000までです。"
    reply = facts_reply(("Q1-n1", "full", "貸与は6か月延長は1回"), ("Q1-n2", "partial", "1回 まで"))
    res = run_nj2(ans, FakeJudge(always={2: reply}))
    assert res["facts"] == {"Q1-n1": "full", "Q1-n2": "partial"}
    assert res["downgraded"] == []


def test_none_may_have_no_evidence_and_is_not_counted_as_downgraded():
    reply = facts_reply(("Q1-n1", "none", ""), ("Q1-n2", "none", "回答に無い文"))
    res = run_nj2("貸与は6か月です。", FakeJudge(always={2: reply}))
    assert res["status"] == "ok"
    assert res["facts"] == {"Q1-n1": "none", "Q1-n2": "none"}
    assert res["downgraded"] == []


def test_downgraded_is_empty_list_when_all_evidence_ok_and_for_empty_answer():
    res = run_nj2("要点です<<Q1-n1:full>><<Q1-n2:full>>。", FakeJudge())
    assert res["downgraded"] == [] and res["facts"] == {"Q1-n1": "full", "Q1-n2": "full"}
    assert res["evidence"] == {"Q1-n1": "<<Q1-n1:full>>", "Q1-n2": "<<Q1-n2:full>>"}
    assert run_nj2("", FakeJudge())["downgraded"] == []
    assert run_judge("", FakeJudge())["downgraded"] == []


# ─── NJ-3: 文の番号の機械の検査 ─────────────────────────────────
ANS3 = "貸与は6か月です。延長は1回までです。\n結論: 貸与できる"  # 3 文


def sent_reply(*items):
    return json.dumps({"facts": [{"nid": n, "score": sc, "sentences": ns} for n, sc, ns in items]}, ensure_ascii=False)


def test_nj3_full_without_sentence_numbers_is_downgraded():
    reply = sent_reply(("Q1-n1", "full", []), ("Q1-n2", "partial", []))
    res = run_judge(ANS3, FakeJudge(always={2: reply}))
    assert res["status"] == "ok" and res["facts"] == {"Q1-n1": "none", "Q1-n2": "none"}
    assert res["downgraded"] == ["Q1-n1", "Q1-n2"]
    assert res["evidence"] == {"Q1-n1": "", "Q1-n2": ""}
    # sentences のキーが無い・null も同じ
    reply = json.dumps({"facts": [{"nid": "Q1-n1", "score": "full"}, {"nid": "Q1-n2", "score": "full", "sentences": None}]})
    res = run_judge(ANS3, FakeJudge(always={2: reply}))
    assert res["downgraded"] == ["Q1-n1", "Q1-n2"]


def test_nj3_out_of_range_numbers_only_is_downgraded():
    reply = sent_reply(("Q1-n1", "full", [0, 4, 99]), ("Q1-n2", "partial", [-1]))
    res = run_judge(ANS3, FakeJudge(always={2: reply}))
    assert res["facts"] == {"Q1-n1": "none", "Q1-n2": "none"}
    assert res["downgraded"] == ["Q1-n1", "Q1-n2"]


def test_nj3_valid_numbers_keep_score_and_record_sentence_text():
    reply = sent_reply(("Q1-n1", "full", [1]), ("Q1-n2", "partial", [2, 3]))
    res = run_judge(ANS3, FakeJudge(always={2: reply}))
    assert res["facts"] == {"Q1-n1": "full", "Q1-n2": "partial"}
    assert res["downgraded"] == []
    assert res["evidence"] == {"Q1-n1": "貸与は6か月です。", "Q1-n2": "延長は1回までです。結論: 貸与できる"}


def test_nj3_partly_valid_numbers_are_kept_and_only_valid_ones_are_quoted():
    reply = sent_reply(("Q1-n1", "full", [9, "2", 2.0, True]), ("Q1-n2", "none", [1]))
    res = run_judge(ANS3, FakeJudge(always={2: reply}))
    assert res["facts"] == {"Q1-n1": "full", "Q1-n2": "none"} and res["downgraded"] == []
    assert res["evidence"]["Q1-n1"] == "延長は1回までです。"


def test_nj3_none_needs_no_numbers_and_is_not_counted_as_downgraded():
    res = run_judge(ANS3, FakeJudge(always={2: sent_reply(("Q1-n1", "none", []), ("Q1-n2", "none", [7]))}))
    assert res["facts"] == {"Q1-n1": "none", "Q1-n2": "none"} and res["downgraded"] == []


def test_nj3_bad_reply_is_retried_then_undetermined():
    res = run_judge(ANS3, FakeJudge(always={2: '{"facts": ["full"]}'}))
    assert res["status"] == "undetermined" and "呼び出し2" in res["error"]


def test_nj3_end_to_end_with_fake_markers_and_version_is_default():
    ans = "要点です<<Q1-n1:full>>。\n延長は<<Q1-n2:partial>>です。"
    res = run_judge(ans, FakeJudge())
    assert res["facts"] == {"Q1-n1": "full", "Q1-n2": "partial"} and res["downgraded"] == []
    assert res["evidence"]["Q1-n2"] == "延長は<<Q1-n2:partial>>です。"


def test_broken_json_is_retried_once_then_ok():
    fake = FakeJudge(queue=["{labels: ここは壊れた"])
    res = run_judge("要点です<<Q1-n1:full>><<Q1-n2:full>>。", fake)
    assert res["status"] == "ok" and res["retries"] == 1
    assert len(fake.calls) == 3  # 呼び出し 1 を 2 回 + 呼び出し 2 を 1 回
    retry_msgs = fake.calls[1]
    assert retry_msgs[-2] == {"role": "assistant", "content": "{labels: ここは壊れた"}
    assert "もう一度" in retry_msgs[-1]["content"]


def test_unknown_label_twice_is_undetermined_not_zero():
    fake = FakeJudge(always={1: '{"labels": ["たぶん"]}'})
    res = run_judge("要点です。", fake)
    assert res["status"] == "undetermined"
    assert res["labels"] is None and res["facts"] is None
    assert "呼び出し1" in res["error"] and "知らないラベル" in res["error"]
    assert res["retries"] == 1
    assert len(fake.calls) == 2  # 頼み直しは 1 回だけ。呼び出し 2 へは進まない


def test_label_length_mismatch_retries_then_recovers():
    fake = FakeJudge(queue=['{"labels": ["必須", "必須", "必須"]}'])
    res = run_judge("一文だけ。", fake)
    assert res["status"] == "ok" and res["retries"] == 1


def test_call2_failure_is_undetermined():
    fake = FakeJudge(always={2: "これは JSON ではない"})
    res = run_judge("要点です。", fake)
    assert res["status"] == "undetermined" and "呼び出し2" in res["error"]
    assert res["retries"] == 1 and len(fake.calls) == 3  # 呼び出し 1 が 1 回 + 呼び出し 2 が 2 回


def test_send_exception_is_retried_and_then_undetermined():
    fake = FakeJudge(queue=[OSError("接続できない"), OSError("接続できない")])
    res = run_judge("要点です。", fake)
    assert res["status"] == "undetermined" and "OSError" in res["error"]
    assert len(fake.calls) == 2
    fake2 = FakeJudge(queue=[OSError("一瞬だけ失敗")])
    assert run_judge("要点です。", fake2)["status"] == "ok"


def test_empty_answer_is_ok_with_all_none_and_no_calls():
    fake = FakeJudge()
    res = run_judge("  \n ", fake)
    assert res["status"] == "ok" and res["labels"] == []
    assert res["facts"] == {"Q1-n1": "none", "Q1-n2": "none"}
    assert fake.calls == []


# ─── HTTP の送り方（urlopen を差し替える。ネットには出ない） ───────
def test_http_send_builds_openai_compatible_request(monkeypatch):
    seen = {}

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["payload"] = json.loads(req.data.decode("utf-8"))
        seen["timeout"] = timeout
        return Resp(json.dumps({"choices": [{"message": {"content": "返答"}}]}).encode("utf-8"))

    monkeypatch.setattr(judge.urllib.request, "urlopen", fake_urlopen)
    send = judge.make_http_send("http://127.0.0.1:8080/", model="m1", seed=7, max_tokens=99)
    assert send([{"role": "user", "content": "こんにちは"}]) == "返答"
    assert seen["url"] == "http://127.0.0.1:8080/v1/chat/completions"
    p = seen["payload"]
    assert p["temperature"] == 0 and p["seed"] == 7 and p["max_tokens"] == 99 and p["model"] == "m1"
    assert p["messages"][0]["content"] == "こんにちは"
    assert judge.DEFAULT_BASE_URL == "http://127.0.0.1:8080"


# ─── 下書き ─────────────────────────────────────────────────────
def test_draft_keeps_non_verbatim_quote_with_flag():
    reply = json.dumps({
        "nuggets": [
            {"text": "日当は3,000円", "importance": "vital", "quote": "国内出張の日当は3,000円とする。", "keywords": ["3,000円"]},
            {"text": "日当は4,000円", "importance": "必須", "quote": "国内出張の日当は4,000円とする。", "keywords": []},
            {"text": "補足", "importance": "謎", "quote": "宿泊費の上限は10,000円とする。"},
        ],
        "excluded": [{"text": "宿泊費", "why": "質問と無関係"}],
    }, ensure_ascii=False)
    row = draft.draft_question(
        FakeJudge(queue=["```json\n" + reply + "\n```"]),
        qid="Q9", question="日当は？", rule_id="出張規程", rule_text=CORPUS["出張規程"],
        set_name="toy", drafted_by="model-x",
    )
    assert row["qid"] == "Q9" and row["reviewed_by"] == "" and row["drafted_by"] == "model-x"
    assert row["version"] == "ng-1" and row["question"] == "日当は？"
    assert [n["nid"] for n in row["nuggets"]] == ["Q9-n1", "Q9-n2", "Q9-n3"]
    assert "flags" not in row["nuggets"][0]
    assert row["nuggets"][1]["flags"] == ["quote_not_verbatim"]  # 落とさずに印だけ
    assert row["nuggets"][1]["importance"] == "vital"
    assert row["nuggets"][2]["importance"] == "okay" and "importance_unknown" in row["nuggets"][2]["flags"]
    assert row["excluded"] == [{"text": "宿泊費", "why": "質問と無関係"}]


def test_draft_failure_is_kept_as_flagged_row():
    row = draft.draft_question(
        FakeJudge(queue=["壊れ", "壊れ"]),
        qid="Q9", question="q", rule_id="出張規程", rule_text=CORPUS["出張規程"],
    )
    assert row["nuggets"] == [] and row["flags"] == ["draft_failed"]


# ─── CLI: --judge-version ───────────────────────────────────────
def test_cli_judge_version_option(tmp_path):
    from evaluation.nugget import cli
    from evaluation.tests.test_nugget_schema import make_set, write_jsonl

    nug = make_set(tmp_path)
    assert cli.main(["freeze", "--nuggets", str(nug)]) == 0
    p = cli.build_parser()
    base = ["judge", "--nuggets", str(nug), "--answers", "x.jsonl", "--out-dir", "o"]
    assert p.parse_args(base).judge_version == "NJ-3"
    assert p.parse_args(base + ["--judge-version", "NJ-2"]).judge_version == "NJ-2"
    with pytest.raises(SystemExit):
        p.parse_args(base + ["--judge-version", "NJ-1"])

    ans = str(write_jsonl(tmp_path / "n.jsonl", [{"id": "Q1", "text": "要点です<<Q1-n1:full>><<Q1-n2:partial>>。"}]))
    results = {}
    for ver in ("NJ-2", "NJ-3"):
        out = tmp_path / f"judged-{ver}"
        send = FakeJudge()
        assert cli.main(["judge", "--nuggets", str(nug), "--answers", ans, "--out-dir", str(out),
                         "--model", "m", "--judge-version", ver], send=send) == 0
        (rec,) = [json.loads(l) for l in (out / "n.judged.jsonl").read_text(encoding="utf-8").splitlines()]
        assert rec["judge_version"] == ver and rec["status"] == "ok"
        assert all(ver in c[1]["content"] for c in send.calls)
        results[ver] = rec
    # どちらの版でも点は同じ。NJ-2 は写し（印そのもの）、NJ-3 は引いた文の本文
    assert results["NJ-2"]["facts"] == results["NJ-3"]["facts"] == {"Q1-n1": "full", "Q1-n2": "partial"}
    assert results["NJ-2"]["evidence"]["Q1-n1"] == "<<Q1-n1:full>>"
    assert results["NJ-3"]["evidence"]["Q1-n1"] == "要点です<<Q1-n1:full>><<Q1-n2:partial>>。"
