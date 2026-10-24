"""文ごと採点役 (judges_sentence, M4-S) の事前確認。

正解の分かっている 11 例 (evaluation/v2/judge_precheck_s.jsonl: JP1〜JP5 は旧 5 例の写し、
JA1〜JA3・JB1〜JB3 が新規) を judges_sentence.judge_sentences で採点させ、同じ設定で
REPEAT 回 (2 回) 繰り返す。採点役を --judge-from に使ってよいかを決める。

1 回分の判定 (evaluate_run):
- 未判定: どれかの例に error (ScoringFailure・呼び出しの失敗を含む) がある。比べる値が欠けるので合否を出さない
- 旧 5 例の大小: faithfulness で JP1 > JP2・JP3・JP4、relevance で JP1 > JP5 (同点は不合格)
- 新 6 例: JA1・JB1・JA2・JB2 は verdicts が expected_sentences と完全一致、
  JA3・JB3 は relevance が expected_relevance と一致
- JP1〜JP5 と JA3・JB3 の文ごとの判定と expected_sentences のずれは info に残すだけ (合否に入れない)
全体: 2 回とも pass なら pass。どちらかが未判定なら未判定 (fail より優先)。それ以外は fail。

温度が 0 でない設定 (app.config.settings.llm_temperature) では実行しない。
温度は呼び出しごとでなく設定で決まるため、2 回の繰り返しが比較として成り立たなくなる。
LLM_TEMPERATURE=0 を付けて起動すること。

CLI (repo ルートから。採点役の vLLM を立てた後):
    LLM_TEMPERATURE=0 python -m evaluation.judge_precheck_s --out /content/step05/judge_precheck_s.json
終了コード: 0=pass / 1=fail / 3=未判定 / 2=引数の誤り・前提違反 (fixture 不正・温度が 0 でない・--out が既にある)。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))  # run_eval と同じ。app は設定・採点役を使うときだけ import する

from evaluation import judges_sentence, run_meta  # noqa: E402
from evaluation.metrics import split_sentences  # noqa: E402

FIXTURE = ROOT / "evaluation" / "v2" / "judge_precheck_s.jsonl"
IDS = ("JP1", "JP2", "JP3", "JP4", "JP5", "JA1", "JA2", "JA3", "JB1", "JB2", "JB3")
REPEAT = 2
EXIT = {"pass": 0, "fail": 1, "未判定": 3}

FAITH_ORDER = (("JP1", "JP2"), ("JP1", "JP3"), ("JP1", "JP4"))   # faithfulness: 左 > 右
REL_ORDER = (("JP1", "JP5"),)                                    # relevance: 左 > 右
EXACT_IDS = ("JA1", "JB1", "JA2", "JB2")                          # verdicts が expected_sentences と一致
REL_EXPECT_IDS = ("JA3", "JB3")                                   # relevance が expected_relevance と一致
INFO_IDS = ("JP1", "JP2", "JP3", "JP4", "JP5", "JA3", "JB3")      # 参考 (合否に入れない)


def load_fixture(path: Path) -> list[dict]:
    """読み込んで検証する。違えば ValueError。"""
    try:
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} を読めない: {exc}") from exc
    if len(rows) != len(IDS):
        raise ValueError(f"{path} の行数 {len(rows)} が {len(IDS)} と違う")
    ids = [r.get("id") if isinstance(r, dict) else None for r in rows]
    if sorted(map(str, ids)) != sorted(IDS):
        raise ValueError(f"{path} の ID が {list(IDS)} と合わない: {ids}")
    for r in rows:
        for key in ("question", "answer", "chunks", "expected_sentences"):
            if key not in r:
                raise ValueError(f"{r['id']}: {key} が無い")
        n = len(split_sentences(r["answer"]))
        if n != len(r["expected_sentences"]):
            raise ValueError(f"{r['id']}: 文の数 {n} が expected_sentences の長さ {len(r['expected_sentences'])} と違う")
    return rows


def _entry_ok(e: dict | None) -> bool:
    return bool(e) and not e.get("error") and isinstance(e.get("verdicts"), list) \
        and isinstance(e.get("faithfulness"), (int, float)) and e.get("relevance") in (0, 1)


def _cmp(name: str, good, bad) -> dict:
    return {"name": name, "good": good, "bad": bad, "ok": None if good is None or bad is None else good > bad}


def evaluate_run(scored: dict[str, dict], rows: list[dict]) -> dict:
    """1 回分の判定。scored: id → score_all の結果。rows: fixture の行。"""
    exp = {r["id"]: r for r in rows}
    broken = [i for i in IDS if not _entry_ok(scored.get(i))]
    val = {i: (scored.get(i) if i not in broken else None) for i in IDS}

    def get(i, key):
        return val[i][key] if val[i] is not None else None

    checks: list[dict] = []
    for a, b in FAITH_ORDER:
        checks.append(_cmp(f"faithfulness: {a} > {b}", get(a, "faithfulness"), get(b, "faithfulness")))
    for a, b in REL_ORDER:
        checks.append(_cmp(f"relevance: {a} > {b}", get(a, "relevance"), get(b, "relevance")))
    for i in EXACT_IDS:
        got, want = get(i, "verdicts"), exp[i]["expected_sentences"]
        checks.append({"name": f"verdicts: {i} == expected_sentences", "got": got, "want": want,
                       "ok": None if got is None else got == want})
    for i in REL_EXPECT_IDS:
        got, want = get(i, "relevance"), exp[i]["expected_relevance"]
        checks.append({"name": f"relevance: {i} == expected_relevance", "got": got, "want": want,
                       "ok": None if got is None else got == want})

    info = []
    for i in INFO_IDS:
        got, want = get(i, "verdicts"), exp[i]["expected_sentences"]
        info.append({"id": i, "got": got, "want": want, "match": None if got is None else got == want})

    if broken:
        result = "未判定"
    else:
        result = "pass" if all(c["ok"] for c in checks) else "fail"
    return {"result": result, "checks": checks, "info": info, "broken": broken}


def overall(results: list[str]) -> str:
    if any(r == "未判定" for r in results):
        return "未判定"
    return "pass" if all(r == "pass" for r in results) else "fail"


async def score_all(provider, rows: list[dict]) -> dict[str, dict]:
    """1 例ずつ judge_sentences で採点する。例ごとの例外は error に残して続ける。"""
    out: dict[str, dict] = {}
    for r in rows:
        t0 = time.monotonic()
        try:
            j = await judges_sentence.judge_sentences(provider, r["question"], r["chunks"], r["answer"])
            e = {
                "sentences": j.sentences, "verdicts": j.verdicts, "reasons": j.reasons,
                "faithfulness": j.faithfulness, "relevance": j.relevance, "raw": j.raw, "error": None,
            }
        except Exception as exc:
            e = {
                "sentences": None, "verdicts": None, "reasons": None, "faithfulness": None, "relevance": None,
                "raw": getattr(exc, "raw", {}), "error": f"{type(exc).__name__}: {exc}",
            }
        e["seconds"] = round(time.monotonic() - t0, 2)
        out[r["id"]] = e
    return out


def _get_settings():
    from app.config import settings

    return settings


async def _get_provider(name: str):
    """run_eval._resolve_provider と同じ。外部送信プロバイダ・資格情報は持たない。"""
    from app.services.providers import get_meta, get_provider

    if get_meta(name).requires_credentials:
        raise RuntimeError(f"Provider '{name}' は資格情報が必要だが本アプリは未対応")
    return await get_provider(name)


async def run(judge_name: str, fixture: Path, settings) -> dict:
    rows = load_fixture(fixture)
    provider = await _get_provider(judge_name)
    runs = []
    for k in range(REPEAT):
        scored = await score_all(provider, rows)
        verdict = evaluate_run(scored, rows)
        runs.append({"run": k + 1, **verdict, "scores": scored})
    return {
        "result": overall([r["result"] for r in runs]),
        "judge": judge_name,
        "llm_model": settings.llm_model,
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,   # 出力の上限。採点し直す側が一致を確かめる
        "vllm": run_meta.fetch_vllm_models(settings.vllm_base_url),
        "fixture": {"path": str(fixture), "sha256": run_meta.file_sha256(fixture), "rows": len(rows)},
        "prompt_sha256": judges_sentence.prompt_sha256(),
        "runs": runs,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="文ごと採点役の事前確認 (正解の分かっている 11 例を 2 回採点する)")
    p.add_argument("--judge", default="vllm", help="採点役のプロバイダ名")
    p.add_argument("--fixture", type=Path, default=FIXTURE)
    p.add_argument("--out", type=Path, required=True, help="結果の JSON (既にあれば拒否)")
    a = p.parse_args(argv)
    if a.out.exists():
        p.error(f"--out {a.out} が既にある (前の結果を上書きしないため拒否)")

    try:
        load_fixture(a.fixture)
    except ValueError as exc:
        print(f"fixture が不正で実行しない: {exc}", file=sys.stderr)
        return 2
    settings = _get_settings()
    try:
        temp_is_zero = float(settings.llm_temperature) == 0.0
    except (TypeError, ValueError):
        temp_is_zero = False
    if not temp_is_zero:
        print(
            f"llm_temperature={settings.llm_temperature!r} が 0 でないため実行しない "
            "(温度は設定で決まる。LLM_TEMPERATURE=0 で起動し直す)",
            file=sys.stderr,
        )
        return 2

    result = asyncio.run(run(a.judge, a.fixture, settings))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for r in result["runs"]:
        print(f"[{r['run']} 回目] {r['result']} (error のある例: {r['broken'] or 'なし'})")
        for c in r["checks"]:
            print(f"  {c['name']} → {c['ok']}")
    print(f"文ごと採点役の事前確認: {result['result']} → {a.out}", flush=True)
    return EXIT[result["result"]]


if __name__ == "__main__":
    sys.exit(main())
