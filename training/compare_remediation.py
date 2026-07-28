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
CORE = ["json_parse_valid", "schema_valid", "status_correct", "intent_correct", "action_correct"]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_predictions(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    return {(row["case_id"], row["assistant_index"]): row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report: dict[str, Any] = {
        "old_adapter": "20260714-initial-100step",
        "new_adapter": "20260714-remediation-120step",
        "new_adapter_sha256": "fbd057d6d8b9e310a9abfa95488ef422161b714f81dae5a5bdf6f179f25fcebd",
        "combined_train_sha256": "88c6e7f79ea9019eac9f91039a6d5921b776f583812cc84156d71f3fab0bb724",
        "added_records": 88,
        "splits": {},
    }
    for split, expected in (("validation", 103), ("test", 124)):
        old_summary = read_json(args.old_root / f"ft-{split}" / "summary.json")
        new_summary = read_json(args.new_root / f"ft-{split}" / "summary.json")
        old_rows = read_predictions(args.old_root / f"ft-{split}" / "predictions.jsonl")
        new_rows = read_predictions(args.new_root / f"ft-{split}" / "predictions.jsonl")
        if len(old_rows) != expected or old_rows.keys() != new_rows.keys():
            raise ValueError(f"prediction set mismatch for {split}")
        fixed = [
            {"case_id": key[0], "assistant_index": key[1]}
            for key in old_rows
            if not all(old_rows[key][metric] for metric in CORE)
            and all(new_rows[key][metric] for metric in CORE)
        ]
        regressed = [
            {"case_id": key[0], "assistant_index": key[1]}
            for key in old_rows
            if all(old_rows[key][metric] for metric in CORE)
            and not all(new_rows[key][metric] for metric in CORE)
        ]
        metrics = {}
        for metric in METRICS:
            old_value = old_summary["overall"][metric]
            new_value = new_summary["overall"][metric]
            metrics[metric] = {
                "old_ft": old_value,
                "remediation_ft": new_value,
                "delta": new_value - old_value
                if old_value is not None and new_value is not None
                else None,
            }
        report["splits"][split] = {
            "assistant_targets": expected,
            "metrics": metrics,
            "fixed_core_failures": fixed,
            "new_core_regressions": regressed,
        }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = ["# Remediation FT evaluation", ""]
    for split in ("validation", "test"):
        section = report["splits"][split]
        lines += [
            f"## {split}",
            "",
            f"Assistant targets: {section['assistant_targets']}",
            "",
            "| Metric | Old FT | Remediation FT | Delta (pp) |",
            "|---|---:|---:|---:|",
        ]
        for metric, values in section["metrics"].items():
            def pct(value: float | None) -> str:
                return "n/a" if value is None else f"{value * 100:.2f}%"

            delta = values["delta"]
            lines.append(
                f"| {metric} | {pct(values['old_ft'])} | {pct(values['remediation_ft'])} | "
                f"{'n/a' if delta is None else f'{delta * 100:+.2f}'} |"
            )
        lines += [
            "",
            f"Fixed core failures: {len(section['fixed_core_failures'])}",
            f"New core regressions: {len(section['new_core_regressions'])}",
            "",
        ]
    (args.output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
