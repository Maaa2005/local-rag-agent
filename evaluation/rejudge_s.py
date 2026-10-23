"""保存済みの回答を、採点役で「文ごとに 0/1」に採点し直す (M4-S)。

定義:
- 回答を metrics.split_sentences で分け、文ごとに 0/1。1 問の点は文の平均。
  1 回 (run_label) の点は、その回の全問の平均。0.90 以上で合格。
- 空の回答 (文が 0 個)・回答の失敗 (error が記録された行) は採点役を呼ばず、その問を 0 点にする
  (auto_zero として理由を残す)。
- 採点できない問が 1 つでもあれば、その回は「未判定」。切り詰め・補い・既定値での穴埋めはしない。
  (採点役の例外・文や 0/1 の数が回答と合わない返り値は、どちらも採点できない問として error に残す)

実行前に、採点役の確認結果 (judge_precheck_s の JSON) が今の採点役と同じ条件で pass していることを確かめる。
拒否 (終了コード 2): 温度が 0 でない / precheck が pass でない / llm_model・temperature・max_tokens・prompt_sha256 が違う /
precheck と今の vLLM のモデル root の集合が違う (どちらかが取れない・空のときも確かめられないので拒否)。

CLI (repo ルートから。採点役の vLLM を立てた後):
    python -m evaluation.rejudge_s --raw RAW.jsonl --precheck PRECHECK.json --out OUT.json
終了コード: 0=すべて pass / 1=どれかが fail / 3=どれかが未判定 / 2=引数の誤り・実行前の拒否。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Awaitable, Callable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))  # app は実行時にだけ import する

from evaluation import rejudge, run_meta  # noqa: E402
from evaluation.metrics import split_sentences  # noqa: E402

PASS_THRESHOLD = Fraction(9, 10)
EXIT = {"pass": 0, "fail": 1, "未判定": 3}
REFUSED = 2


# ─── 実行前の確認 ──────────────────────────────────────────────
def _roots(vllm) -> tuple[set[str] | None, str | None]:
    """vllm の記録から root の集合を取る。取れなければ (None, 理由)。"""
    if not isinstance(vllm, dict):
        return None, "記録が無い"
    if vllm.get("error"):
        return None, f"error: {vllm['error']}"
    models = vllm.get("models")
    if not isinstance(models, list) or not models:
        return None, "models が空"
    roots = [m.get("root") if isinstance(m, dict) else None for m in models]
    if any(not isinstance(r, str) or not r for r in roots):
        return None, "root が取れないモデルがある"
    return set(roots), None


def check_preconditions(
    *, settings, precheck: dict, prompt_sha: str, current_vllm: dict
) -> list[str]:
    """実行を拒否する理由の一覧。空なら実行してよい。"""
    reasons: list[str] = []
    if settings.llm_temperature != 0:
        reasons.append(f"llm_temperature が 0 でない: {settings.llm_temperature}")
    if precheck.get("result") != "pass":
        reasons.append(f"precheck の result が pass でない: {precheck.get('result')!r}")
    if precheck.get("llm_model") != settings.llm_model:
        reasons.append(
            f"llm_model が違う: precheck={precheck.get('llm_model')!r} 今={settings.llm_model!r}"
        )
    if precheck.get("temperature") != settings.llm_temperature:
        reasons.append(
            f"temperature が違う: precheck={precheck.get('temperature')!r} 今={settings.llm_temperature!r}"
        )
    if precheck.get("max_tokens") != settings.llm_max_tokens:
        reasons.append(
            f"max_tokens が違う: precheck={precheck.get('max_tokens')!r} 今={settings.llm_max_tokens!r}"
        )
    if precheck.get("prompt_sha256") != prompt_sha:
        reasons.append("prompt_sha256 が違う (採点のプロンプトが precheck のときと違う)")
    then, then_why = _roots(precheck.get("vllm"))
    now, now_why = _roots(current_vllm)
    if then is None:
        reasons.append(f"precheck の vllm モデルを確かめられない ({then_why})")
    if now is None:
        reasons.append(f"今の vllm モデルを確かめられない ({now_why})")
    if then is not None and now is not None and then != now:
        reasons.append(
            f"vllm のモデル root が違う: precheck={sorted(then)} 今={sorted(now)}"
        )
    return reasons


# ─── 採点 ──────────────────────────────────────────────────────
JudgeFn = Callable[..., Awaitable[object]]


def _validate(judgement, sentences: list[str]) -> list[int]:
    """採点役の返り値が回答の文と対応していなければ ValueError。0/1 の並びを返す。"""
    got = list(getattr(judgement, "sentences", None) or [])
    verdicts = list(getattr(judgement, "verdicts", None) or [])
    if got != sentences:
        raise ValueError(f"invalid_judgement: sentences が回答の文と合わない ({len(got)} 文 / 期待 {len(sentences)} 文)")
    if len(verdicts) != len(sentences):
        raise ValueError(f"invalid_judgement: verdicts が {len(verdicts)} 個 (文は {len(sentences)} 個)")
    for v in verdicts:
        if type(v) is not int or v not in (0, 1):
            raise ValueError(f"invalid_judgement: verdict が 0/1 でない: {v!r}")
    return verdicts


def _question_entry(r) -> dict:
    return {
        "id": r.id, "status": None, "auto_zero_reason": None,
        "sentences": [], "verdicts": [], "reasons": [],
        "faithfulness": None, "judge_faithfulness": None, "relevance": None, "error": None,
    }


async def judge_question(r, judge_fn: JudgeFn, provider) -> tuple[dict, Fraction | None]:
    """1 問を採点する。(記録, 点)。採点できなければ点は None で error に理由を残す。"""
    e = _question_entry(r)
    if r.error:
        e.update(status="auto_zero", auto_zero_reason=f"run_failed: {r.error}", faithfulness=0.0)
        return e, Fraction(0)
    sentences = split_sentences(r.answer or "")
    if not sentences:
        e.update(status="auto_zero", auto_zero_reason="empty_answer", faithfulness=0.0)
        return e, Fraction(0)
    try:
        j = await judge_fn(provider, r.question, r.retrieved_chunks, r.answer)
        verdicts = _validate(j, sentences)
    except Exception as exc:  # noqa: BLE001 - ScoringFailure を含め、すべてその問の error にする
        e.update(status="error", sentences=sentences, error=f"{type(exc).__name__}: {exc}")
        return e, None
    score = Fraction(sum(verdicts), len(verdicts))
    e.update(
        status="judged",
        sentences=sentences,
        verdicts=verdicts,
        reasons=list(getattr(j, "reasons", None) or []),
        faithfulness=float(score),
        judge_faithfulness=getattr(j, "faithfulness", None),
        relevance=getattr(j, "relevance", None),
    )
    return e, score


async def rejudge_groups(groups: dict, judge_fn: JudgeFn, provider) -> dict[str, dict]:
    """run_label ごとに全問を採点する。1 問でも採点できなければその回は未判定 (平均は出さない)。"""
    out: dict[str, dict] = {}
    for label, results in groups.items():
        entries, scores = [], []
        for r in results:
            entry, score = await judge_question(r, judge_fn, provider)
            entries.append(entry)
            scores.append(score)
        failed = sum(1 for s in scores if s is None)
        n_auto = sum(1 for e in entries if e["status"] == "auto_zero")
        if failed:
            result, mean = "未判定", None
        else:
            m = sum(scores) / len(scores) if scores else Fraction(0)
            mean = float(m)
            result = "pass" if scores and m >= PASS_THRESHOLD else "fail"
        out[label] = {
            "mean": mean,
            "n_questions": len(entries),
            "n_auto_zero": n_auto,
            "n_failed": failed,
            "result": result,
            "questions": entries,
        }
    return out


def overall_result(per_label: dict[str, dict]) -> str:
    results = {v["result"] for v in per_label.values()}
    if "未判定" in results or not results:
        return "未判定"
    return "fail" if "fail" in results else "pass"


# ─── CLI ───────────────────────────────────────────────────────
def main(
    argv: list[str] | None = None,
    *,
    judge_fn: JudgeFn | None = None,
    settings=None,
    prompt_sha: str | None = None,
    provider=None,
    vllm_fetch: Callable[[str], dict] | None = None,
) -> int:
    """judge_fn / settings / prompt_sha / provider / vllm_fetch はテストで差し替える。"""
    p = argparse.ArgumentParser(description="保存済み回答を文ごとに 0/1 で採点し直す (M4-S)")
    p.add_argument("--raw", type=Path, required=True, help="raw.jsonl")
    p.add_argument("--precheck", type=Path, required=True, help="採点役の確認結果 JSON")
    p.add_argument("--judge", default="vllm", help="採点役のプロバイダ名")
    p.add_argument("--out", type=Path, required=True, help="結果の JSON (既にあれば拒否)")
    a = p.parse_args(argv)

    def refuse(msg: str) -> int:
        print(f"拒否: {msg}", flush=True)
        return REFUSED

    if a.out.exists():
        return refuse(f"--out {a.out} が既にある (前の結果を上書きしないため拒否)")
    try:
        precheck = json.loads(a.precheck.read_text(encoding="utf-8"))
        if not isinstance(precheck, dict):
            raise ValueError("最上位が object でない")
    except (OSError, ValueError) as exc:
        return refuse(f"--precheck を読めない: {exc}")

    if settings is None:
        from app.config import settings as _settings

        settings = _settings
    if prompt_sha is None:
        from evaluation.judges_sentence import prompt_sha256

        prompt_sha = prompt_sha256()
    current_vllm = (vllm_fetch or run_meta.fetch_vllm_models)(settings.vllm_base_url)

    reasons = check_preconditions(
        settings=settings, precheck=precheck, prompt_sha=prompt_sha, current_vllm=current_vllm
    )
    if reasons:
        for r in reasons:
            print(f"拒否: {r}", flush=True)
        return REFUSED

    try:
        groups = rejudge.load_saved_results(a.raw)
    except (OSError, ValueError) as exc:
        return refuse(f"--raw を読めない: {exc}")

    if judge_fn is None:
        from evaluation.judges_sentence import judge_sentences

        judge_fn = judge_sentences
    if provider is None:
        from evaluation.judge_precheck import _get_provider

        provider = asyncio.run(_get_provider(a.judge))

    per_label = asyncio.run(rejudge_groups(groups, judge_fn, provider))
    result = overall_result(per_label)

    meta = run_meta.judged_meta(a.raw, a.judge, settings.llm_model, list(argv if argv is not None else sys.argv[1:]), {})
    run_meta.finish_run_meta(meta, {k: v["n_questions"] for k, v in per_label.items()})
    doc = {
        "result": result,
        "threshold": float(PASS_THRESHOLD),
        "raw": {"path": str(a.raw), "sha256": run_meta.file_sha256(a.raw)},
        "precheck": {"path": str(a.precheck), "sha256": run_meta.file_sha256(a.precheck)},
        "judge": a.judge,
        "llm_model": settings.llm_model,
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
        "prompt_sha256": prompt_sha,
        "vllm": current_vllm,
        "run_meta": meta,
        "runs": per_label,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for label, v in per_label.items():
        mean = "-" if v["mean"] is None else f"{v['mean']:.4f}"
        print(
            f"  {label}: {v['result']} 平均={mean} 問={v['n_questions']} "
            f"auto_zero={v['n_auto_zero']} 失敗={v['n_failed']}",
            flush=True,
        )
    print(f"M4-S 採点: {result} → {a.out}", flush=True)
    return EXIT[result]


if __name__ == "__main__":
    sys.exit(main())
