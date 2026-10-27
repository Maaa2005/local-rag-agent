"""採点役の事前確認 (docs/v2-light-model-plan.md §1 Q・Step 0.5)。

正解の分かっている 5 例 (evaluation/v2/judge_precheck.jsonl: 本文どおり・根拠のない断定・
数字は同じで結論が逆・一部だけ誤り・拒否) を本番と同じ judges.judge で採点させ、
良い答えほど高い点になるかを確かめる。通らない採点役は --judge-from に使わない。

判定:
- pass: 本文どおり (faithful) の faithfulness が unsupported・opposite・partial のどれよりも高く、
  answer_relevance が拒否 (refusal) より高い
- fail: 上のどれかが成り立たない (同点も不合格)
- 未判定: どれかの採点で JSON が取れない・faithfulness か answer_relevance が 0〜1 の数でない・
  採点役の呼び出しが落ちた。judges は欠けた値を黙って 0.0 にするので、そのまま比べると
  壊れた採点役が通ったり落ちたりする
誤答どうしの順 (一部だけ誤り > 断定 など) は見ない。0.5 刻みの採点では同点になりやすいため。
context_relevance は 5 例とも同じチャンクなので見ない。

CLI (repo ルートから。採点役の vLLM を立てた後。VLLM_BASE_URL などの設定は run_eval と同じ):
    python -m evaluation.judge_precheck --out /content/step05/judge_precheck.json
終了コード: 0=pass / 1=fail / 3=未判定 / 2=引数の誤り。結果は --out の JSON に残す
(落ちて JSON が無いときも未判定として扱う)。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))  # run_eval と同じ。app は採点役を呼ぶときだけ import する

from evaluation import judges, run_meta  # noqa: E402

FIXTURE = ROOT / "evaluation" / "v2" / "judge_precheck.jsonl"
KINDS = ("faithful", "unsupported", "opposite", "partial", "refusal")
REQUIRED = ("faithfulness", "answer_relevance")
WORSE = ("unsupported", "opposite", "partial")
EXIT = {"pass": 0, "fail": 1, "未判定": 3}


def load_fixture(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    kinds = [r["kind"] for r in rows]
    if sorted(kinds) != sorted(KINDS):
        raise ValueError(f"{path} の kind が {list(KINDS)} と合わない: {kinds}")
    return rows


def _score(v) -> float | None:
    """0〜1 の数ならその値。欠け・数でない・bool・範囲外は None (壊れた採点)。"""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or not 0.0 <= f <= 1.0:
        return None
    return f


def _check(name: str, good: float | None, bad: float | None) -> dict:
    return {"name": name, "good": good, "bad": bad, "ok": None if good is None or bad is None else good > bad}


def evaluate(scored: dict[str, dict]) -> dict:
    """scored: kind → {"raw": 採点役の生 JSON, "error": 呼び出しの失敗}。判定と内訳を返す。"""
    vals: dict[str, dict] = {}
    broken: list[str] = []
    for kind in KINDS:
        s = scored.get(kind) or {}
        raw = s.get("raw") or {}
        vals[kind] = {k: _score(raw.get(k)) for k in REQUIRED}
        if s.get("error") or not raw or None in vals[kind].values():
            broken.append(kind)
    good = vals["faithful"]
    checks = [_check(f"faithfulness: faithful > {bad}", good["faithfulness"], vals[bad]["faithfulness"]) for bad in WORSE]
    checks.append(_check("answer_relevance: faithful > refusal", good["answer_relevance"], vals["refusal"]["answer_relevance"]))
    if broken:
        result = "未判定"
    else:
        result = "pass" if all(c["ok"] for c in checks) else "fail"
    return {"result": result, "checks": checks, "broken": broken}


async def score_all(provider, rows: list[dict]) -> dict[str, dict]:
    """1 例ずつ judges.judge で採点する。呼び出しが落ちた例は error に残して続ける。"""
    out: dict[str, dict] = {}
    for r in rows:
        t0 = time.monotonic()
        try:
            js = await judges.judge(provider, r["question"], r["chunks"], r["answer"], r["expected_keywords"])
            raw, error = js.raw, None
        except Exception as exc:
            raw, error = {}, f"{type(exc).__name__}: {exc}"
        out[r["kind"]] = {"id": r["id"], "raw": raw, "error": error, "seconds": round(time.monotonic() - t0, 2)}
    return out


async def _get_provider(name: str):
    """run_eval._resolve_provider と同じ。外部送信プロバイダ・資格情報は持たない。"""
    from app.services.providers import get_meta, get_provider

    if get_meta(name).requires_credentials:
        raise RuntimeError(f"Provider '{name}' は資格情報が必要だが本アプリは未対応")
    return await get_provider(name)


async def run(judge_name: str, fixture: Path) -> dict:
    from app.config import settings

    rows = load_fixture(fixture)
    provider = await _get_provider(judge_name)
    scored = await score_all(provider, rows)
    verdict = evaluate(scored)
    return {
        **verdict,
        "judge": judge_name,
        "llm_model": settings.llm_model,
        "vllm": run_meta.fetch_vllm_models(settings.vllm_base_url),
        "fixture": {"path": str(fixture), "sha256": run_meta.file_sha256(fixture), "rows": len(rows)},
        "scores": scored,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="採点役の事前確認 (正解の分かっている 5 例で、良い答えほど高い点になるか)")
    p.add_argument("--judge", default="vllm", help="採点役のプロバイダ名")
    p.add_argument("--fixture", type=Path, default=FIXTURE)
    p.add_argument("--out", type=Path, required=True, help="結果の JSON (既にあれば拒否)")
    a = p.parse_args(argv)
    if a.out.exists():
        p.error(f"--out {a.out} が既にある (前の結果を上書きしないため拒否)")

    result = asyncio.run(run(a.judge, a.fixture))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for c in result["checks"]:
        print(f"  {c['name']}: {c['good']} vs {c['bad']} → {c['ok']}")
    print(f"採点役の事前確認: {result['result']} (壊れた採点: {result['broken'] or 'なし'}) → {a.out}", flush=True)
    return EXIT[result["result"]]


if __name__ == "__main__":
    sys.exit(main())
