"""文ごと採点役の事前確認 (evaluation/judge_precheck_s.py) のテスト。採点役は偽物。

確かめること:
- fixture: JP1〜JP5 が旧 5 例の写し、JA・JB の本文がコーパスそのまま・質問が dataset と重ならない・文の数が揃う
- evaluate_run: 期待どおりで pass、同点・位置違い・relevance 違いで fail、error が 1 例でもあれば未判定
- 全体: 2 回目だけ未判定でも未判定。温度が 0 でない・--out が既にある・fixture 不正なら実行しない
"""
from __future__ import annotations

import asyncio
import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation import judge_precheck_s as jps
from evaluation import judges_sentence as js
from evaluation.metrics import split_sentences

ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = ROOT / "evaluation" / "corpus"
OLD_FIXTURE = ROOT / "evaluation" / "v2" / "judge_precheck.jsonl"


def _rows():
    return jps.load_fixture(jps.FIXTURE)


def _read(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


# ─── fixture ─────────────────────────────────────────────────────
def test_fixture_jp_rows_match_old_precheck():
    rows = {r["id"]: r for r in _rows()}
    old = _read(OLD_FIXTURE)
    assert [o["id"] for o in old] == ["JP1", "JP2", "JP3", "JP4", "JP5"]
    for o in old:
        r = rows[o["id"]]
        for key in ("id", "kind", "question", "answer", "chunks"):
            assert r[key] == o[key], (o["id"], key)
        assert r["expected_keywords"] == o["expected_keywords"]


def test_fixture_ids_and_sentence_counts():
    rows = _rows()
    assert sorted(r["id"] for r in rows) == sorted(jps.IDS) and len(rows) == 11
    for r in rows:
        assert len(split_sentences(r["answer"])) == len(r["expected_sentences"]), r["id"]
        assert all(v in (0, 1) and type(v) is int for v in r["expected_sentences"])


def test_fixture_ja_jb_chunks_are_verbatim_corpus_and_questions_are_new():
    rows = [r for r in _rows() if r["id"].startswith(("JA", "JB"))]
    assert len(rows) == 6
    dataset_questions = {q["question"] for q in _read(ROOT / "evaluation" / "dataset.jsonl")}
    for r in rows:
        assert r["question"] not in dataset_questions, r["id"]
        for c in r["chunks"]:
            name = Path(c["source_file"]).name
            corpus = next(CORPUS_DIR.rglob(name)).read_text(encoding="utf-8")
            assert c["content"] in corpus, (r["id"], name)


def test_fixture_expected_relevance_for_refusals():
    rows = {r["id"]: r for r in _rows()}
    assert rows["JA3"]["expected_relevance"] == 0 and rows["JB3"]["expected_relevance"] == 0


def _write(tmp_path, rows):
    path = tmp_path / "f.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def test_load_fixture_refuses_bad_fixtures(tmp_path):
    rows = _rows()
    with pytest.raises(ValueError, match="行数"):
        jps.load_fixture(_write(tmp_path, rows[:10]))
    renamed = copy.deepcopy(rows)
    renamed[0]["id"] = "JP9"
    with pytest.raises(ValueError, match="ID"):
        jps.load_fixture(_write(tmp_path, renamed))
    bad = copy.deepcopy(rows)
    bad[3]["expected_sentences"] = [1, 0, 1]
    with pytest.raises(ValueError, match="文の数"):
        jps.load_fixture(_write(tmp_path, bad))


# ─── evaluate_run ────────────────────────────────────────────────
GOOD_REL = {"JP1": 1, "JP2": 1, "JP3": 1, "JP4": 1, "JP5": 0, "JA1": 1, "JA2": 1, "JA3": 0, "JB1": 1, "JB2": 1, "JB3": 0}


def _entry(verdicts, relevance):
    return {
        "sentences": ["s"] * len(verdicts), "verdicts": list(verdicts), "reasons": ["r"] * len(verdicts),
        "faithfulness": sum(verdicts) / len(verdicts), "relevance": relevance, "raw": {}, "error": None,
    }


def _scored(**override):
    out = {r["id"]: _entry(r["expected_sentences"], GOOD_REL[r["id"]]) for r in _rows()}
    for k, v in override.items():
        out[k] = v
    return out


def test_expected_output_passes():
    v = jps.evaluate_run(_scored(), _rows())
    assert v["result"] == "pass" and v["broken"] == []
    assert len(v["checks"]) == 3 + 1 + 4 + 2
    assert all(c["ok"] is True for c in v["checks"])
    assert all(i["match"] is True for i in v["info"])


def test_tie_between_jp1_and_jp4_fails():
    v = jps.evaluate_run(_scored(JP4=_entry([1, 1], 1)), _rows())
    assert v["result"] == "fail"
    assert [c["name"] for c in v["checks"] if c["ok"] is False] == ["faithfulness: JP1 > JP4"]


def test_jp1_vs_jp5_relevance_tie_fails():
    v = jps.evaluate_run(_scored(JP5=_entry([0], 1)), _rows())
    assert v["result"] == "fail"
    assert [c["name"] for c in v["checks"] if c["ok"] is False] == ["relevance: JP1 > JP5"]


def test_ja2_wrong_position_fails():
    v = jps.evaluate_run(_scored(JA2=_entry([0, 1], 1)), _rows())
    assert v["result"] == "fail"
    assert [c["name"] for c in v["checks"] if c["ok"] is False] == ["verdicts: JA2 == expected_sentences"]


def test_ja3_relevance_one_fails():
    v = jps.evaluate_run(_scored(JA3=_entry([0], 1)), _rows())
    assert v["result"] == "fail"
    assert [c["name"] for c in v["checks"] if c["ok"] is False] == ["relevance: JA3 == expected_relevance"]


def test_reference_mismatch_is_info_only():
    # JP3 の文ごとの判定が expected とずれても、大小が保たれていれば合否は変わらない
    v = jps.evaluate_run(_scored(JP3=_entry([0, 1], 1)), _rows())
    assert v["result"] == "pass"
    assert next(i for i in v["info"] if i["id"] == "JP3")["match"] is False
    # JA3 の文ごとの判定 (relevance は合っている) も参考だけ
    v = jps.evaluate_run(_scored(JA3=_entry([1], 0)), _rows())
    assert v["result"] == "pass"
    assert next(i for i in v["info"] if i["id"] == "JA3")["match"] is False


@pytest.mark.parametrize("entry", [
    {"verdicts": None, "faithfulness": None, "relevance": None, "raw": {}, "error": "ScoringFailure: 要素数"},
    {"verdicts": None, "faithfulness": None, "relevance": None, "raw": {}, "error": "ConnectionError: refused"},
    None,
])
def test_one_error_is_undecided(entry):
    scored = _scored()
    scored["JB1"] = entry
    v = jps.evaluate_run(scored, _rows())
    assert v["result"] == "未判定" and v["broken"] == ["JB1"]
    # 他に fail 条件があっても未判定が優先
    scored["JP4"] = _entry([1, 1], 1)
    assert jps.evaluate_run(scored, _rows())["result"] == "未判定"


def test_overall():
    assert jps.overall(["pass", "pass"]) == "pass"
    assert jps.overall(["pass", "fail"]) == "fail"
    assert jps.overall(["fail", "未判定"]) == "未判定"
    assert jps.overall(["pass", "未判定"]) == "未判定"
    assert jps.REPEAT == 2


# ─── 偽プロバイダを通した全体 ────────────────────────────────────
def _faith_json(verdicts):
    return json.dumps({"判定": [{"文番号": i + 1, "判定": v, "根拠": "r"} for i, v in enumerate(verdicts)]}, ensure_ascii=False)


class FakeJudge:
    """回答の文から例を見分ける。例ごとに [1 回目, 2 回目...] の (忠実さ, 関連性) を返す。"""

    def __init__(self, script=None):
        rows = _rows()
        self.faith_by_numbered = {js._numbered(split_sentences(r["answer"])): r["id"] for r in rows}
        self.rel_by_answer = {r["answer"]: r["id"] for r in rows}
        self.script = {r["id"]: [(_faith_json(r["expected_sentences"]), json.dumps({"関連": GOOD_REL[r["id"]], "根拠": "r"}))] * 2
                       for r in rows}
        self.script.update(script or {})
        self.seen = {}
        self.calls = 0
        self.last_usage = None

    async def stream(self, messages):
        self.calls += 1
        prompt = messages[-1]["content"]
        if "【回答の文】" in prompt:
            rid = next(i for num, i in self.faith_by_numbered.items() if f"{num}\n\n判定の基準" in prompt)
            kind = 0
        else:
            rid = next(i for a, i in self.rel_by_answer.items() if f"【回答】\n{a}\n" in prompt)
            kind = 1
        n = self.seen.get((rid, kind), 0)
        self.seen[(rid, kind)] = n + 1
        text = self.script[rid][min(n, len(self.script[rid]) - 1)][kind]
        if isinstance(text, Exception):
            raise text
        for i in range(0, len(text), 9):
            yield text[i:i + 9]


def test_score_all_through_real_judge_sentences():
    j = FakeJudge()
    scored = asyncio.run(jps.score_all(j, _rows()))
    assert j.calls == 22  # 11 例 × (忠実さ 1 + 関連性 1)
    assert jps.evaluate_run(scored, _rows())["result"] == "pass"
    assert scored["JP4"]["verdicts"] == [1, 0] and scored["JP4"]["faithfulness"] == 0.5


def test_score_all_keeps_going_after_failures():
    j = FakeJudge({
        "JP4": [(_faith_json([1, 0, 1]), json.dumps({"関連": 1}))] * 2,   # ELYZA の形
        "JA1": [(ConnectionError("refused"), "")] * 2,
    })
    scored = asyncio.run(jps.score_all(j, _rows()))
    assert scored["JP4"]["error"].startswith("ScoringFailure")
    assert scored["JP4"]["raw"]["faithfulness"] == _faith_json([1, 0, 1])
    assert scored["JA1"]["error"] == "ConnectionError: refused"
    assert scored["JP1"]["error"] is None
    assert jps.evaluate_run(scored, _rows())["result"] == "未判定"


@pytest.fixture
def env(monkeypatch):
    def install(script=None, temperature=0.0):
        judge = FakeJudge(script)

        async def get_provider(name):
            assert name == "vllm"
            return judge

        monkeypatch.setattr(jps, "_get_provider", get_provider)
        monkeypatch.setattr(jps, "_get_settings", lambda: SimpleNamespace(
            llm_temperature=temperature, llm_model="m", vllm_base_url="http://x/v1", llm_max_tokens=2048))
        monkeypatch.setattr(jps.run_meta, "fetch_vllm_models", lambda url, **kw: {"url": url, "error": "offline"})
        return judge

    return install


def test_main_pass_writes_json(tmp_path, env):
    judge = env()
    out = tmp_path / "s" / "r.json"
    assert jps.main(["--out", str(out)]) == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert judge.calls == 11 * 2 * 2
    assert saved["result"] == "pass" and saved["judge"] == "vllm" and saved["llm_model"] == "m"
    assert saved["temperature"] == 0.0
    assert saved["max_tokens"] == 2048
    assert saved["vllm"]["error"] == "offline"
    assert saved["fixture"] == {"path": str(jps.FIXTURE), "sha256": jps.run_meta.file_sha256(jps.FIXTURE), "rows": 11}
    assert saved["prompt_sha256"] == js.prompt_sha256()
    assert len(saved["runs"]) == 2
    for r in saved["runs"]:
        assert {"checks", "info", "scores", "result"} <= set(r) and set(r["scores"]) == set(jps.IDS)


def test_main_fail_when_jp1_ties_jp4(tmp_path, env):
    env({"JP4": [(_faith_json([1, 1]), json.dumps({"関連": 1}))] * 2})
    out = tmp_path / "r.json"
    assert jps.main(["--out", str(out)]) == 1
    assert json.loads(out.read_text(encoding="utf-8"))["result"] == "fail"


def test_main_undecided_when_only_second_run_breaks(tmp_path, env):
    good = (_faith_json([1, 0]), json.dumps({"関連": 1}))
    bad = (_faith_json([1, 0, 1]), json.dumps({"関連": 1}))
    env({"JP4": [good, bad]})
    out = tmp_path / "r.json"
    assert jps.main(["--out", str(out)]) == 3
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["result"] == "未判定"
    assert [r["result"] for r in saved["runs"]] == ["pass", "未判定"]


def test_main_refuses_nonzero_temperature(tmp_path, env, capsys):
    judge = env(temperature=0.1)
    out = tmp_path / "r.json"
    assert jps.main(["--out", str(out)]) == 2
    assert "0.1" in capsys.readouterr().err
    assert judge.calls == 0 and not out.exists()


def test_main_refuses_existing_out(tmp_path, env):
    judge = env()
    out = tmp_path / "r.json"
    out.write_text("前の結果\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        jps.main(["--out", str(out)])
    assert exc.value.code == 2
    assert out.read_text(encoding="utf-8") == "前の結果\n" and judge.calls == 0


def test_main_refuses_bad_fixture(tmp_path, env):
    judge = env()
    path = _write(tmp_path, _rows()[:10])
    out = tmp_path / "r.json"
    assert jps.main(["--fixture", str(path), "--out", str(out)]) == 2
    assert judge.calls == 0 and not out.exists()


def test_module_form_help_runs_from_repo_root():
    p = subprocess.run(
        [sys.executable, "-m", "evaluation.judge_precheck_s", "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    assert p.returncode == 0, p.stderr
    assert "--out" in p.stdout
