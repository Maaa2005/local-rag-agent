from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRICS = [
    "json_parse_valid",
    "schema_valid",
    "status_correct",
    "intent_correct",
    "action_correct",
    "clarification_correct",
    "rejection_correct",
    "question_fields_exact",
    "question_fields_f1",
    "safety_filter_valid",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_table(base: dict[str, Any], ft: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in METRICS:
        base_value = base["overall"].get(metric)
        ft_value = ft["overall"].get(metric)
        result[metric] = {
            "base": base_value,
            "ft": ft_value,
            "delta": ft_value - base_value
            if base_value is not None and ft_value is not None
            else None,
            "count": ft["overall"].get(metric + "_count"),
        }
    return result


def failure_examples(path: Path, limit: int = 20) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            failed = [
                name
                for name in ["json_parse_valid", "schema_valid", "status_correct", "intent_correct", "action_correct"]
                if not row[name]
            ]
            if failed:
                failures.append(
                    {
                        "case_id": row["case_id"],
                        "scenario": row["scenario"],
                        "assistant_index": row["assistant_index"],
                        "failed_metrics": failed,
                        "target_status": row["target_status"],
                        "predicted_status": row["predicted_status"],
                        "raw_prediction": row["raw_prediction"][:1000],
                    }
                )
            if len(failures) >= limit:
                break
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root

    summaries: dict[str, dict[str, Any]] = {}
    for split in ("validation", "test"):
        summaries[f"base_{split}"] = load_json(args.root / f"base-{split}" / "summary.json")
        summaries[f"ft_{split}"] = load_json(args.root / f"ft-{split}" / "summary.json")

    expected = {"validation": (67, 103), "test": (81, 124)}
    for split, (records, targets) in expected.items():
        for model in ("base", "ft"):
            summary = summaries[f"{model}_{split}"]
            if summary["records"] != records or summary["assistant_targets"] != targets:
                raise ValueError(f"unexpected counts for {model} {split}: {summary}")

    comparison: dict[str, Any] = {
        "protocol": {
            "base_model": "google/gemma-4-E2B-it",
            "ft_adapter": "outputs/experiments/20260714-initial-100step/adapter",
            "decoding": "greedy",
            "max_sequence_length": 512,
            "max_new_tokens": 512,
            "batch_size": 8,
            "stop_tokens": ["<eos>", "<turn|>"],
            "evaluation_unit": "every assistant target, including intermediate multi-turn responses",
        },
        "artifact_hashes": {
            "adapter_model.safetensors": "be3b00b56b29b1ce5e0bd139a548bd53ec8a9933a9d8b8c9184b91b056f9857a",
            "validation.jsonl": "6fc7b3ce486722feea07c2e6c4658bd5596f882abef9732b1f262ae1dbe59f34",
            "test.jsonl": "20916738d1d7a2ff4defbb1517c866f455e45958e91282612afac5757428ae08",
        },
        "splits": {},
        "ft_failure_examples": {},
    }
    for split in ("validation", "test"):
        base = summaries[f"base_{split}"]
        ft = summaries[f"ft_{split}"]
        comparison["splits"][split] = {
            "records": ft["records"],
            "assistant_targets": ft["assistant_targets"],
            "base_elapsed_seconds": base["elapsed_seconds"],
            "ft_elapsed_seconds": ft["elapsed_seconds"],
            "metrics": metric_table(base, ft),
            "base_by_scenario": base["by_scenario"],
            "ft_by_scenario": ft["by_scenario"],
        }
        comparison["ft_failure_examples"][split] = failure_examples(
            args.root / f"ft-{split}" / "predictions.jsonl"
        )

    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = ["# Base vs FT evaluation", ""]
    for split in ("validation", "test"):
        section = comparison["splits"][split]
        lines.extend(
            [
                f"## {split}",
                "",
                f"{section['records']} conversations / {section['assistant_targets']} assistant targets.",
                "",
                "| Metric | Base | FT | Delta (pp) | N |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for metric, values in section["metrics"].items():
            def pct(value: float | None) -> str:
                return "n/a" if value is None else f"{value * 100:.2f}%"

            delta = values["delta"]
            lines.append(
                f"| {metric} | {pct(values['base'])} | {pct(values['ft'])} | "
                f"{'n/a' if delta is None else f'{delta * 100:+.2f}'} | {values['count']} |"
            )
        lines.extend(
            [
                "",
                f"Inference seconds: base={section['base_elapsed_seconds']:.2f}, "
                f"FT={section['ft_elapsed_seconds']:.2f}.",
                "",
                f"FT failures retained: {len(comparison['ft_failure_examples'][split])} (up to 20).",
                "",
            ]
        )
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
