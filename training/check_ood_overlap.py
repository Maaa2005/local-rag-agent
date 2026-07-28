from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def normalize(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.casefold(), flags=re.UNICODE)


def ngrams(text: str, n: int = 3) -> set[str]:
    value = normalize(text)
    return {value[index : index + n] for index in range(max(0, len(value) - n + 1))}


def similarity(left: str, right: str) -> float:
    a, b = ngrams(left), ngrams(right)
    return len(a & b) / len(a | b) if a or b else 1.0


def load(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def user_text(record: dict[str, Any]) -> str:
    return "\n".join(message["content"] for message in record["messages"] if message["role"] == "user")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("draft", type=Path)
    parser.add_argument(
        "--references",
        nargs="*",
        type=Path,
        default=[
            Path("training/generated/train.jsonl"),
            Path("training/generated/validation.jsonl"),
            Path("training/generated/test.jsonl"),
            Path("training/remediation/train.jsonl"),
        ],
    )
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--report", type=Path, default=Path("training/ood/overlap_report.json"))
    args = parser.parse_args()

    draft = load(args.draft)
    references = [row for path in args.references for row in load(path)]
    case_ids = [row["case_id"] for row in draft]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("duplicate OOD case_id")
    for row in draft:
        provenance = row.get("provenance")
        if provenance not in {"ai_draft_requires_human_review", "human_authored"}:
            raise ValueError(f"unsupported OOD provenance: {row['case_id']}")
        if provenance == "ai_draft_requires_human_review" and any(
            message["role"] == "assistant" for message in row["messages"]
        ):
            raise ValueError(f"draft already contains an assistant label: {row['case_id']}")

    results = []
    for row in draft:
        prompt = user_text(row)
        nearest = max(
            (
                (similarity(prompt, user_text(reference)), reference["case_id"], user_text(reference))
                for reference in references
            ),
            key=lambda item: item[0],
        )
        results.append(
            {
                "case_id": row["case_id"],
                "nearest_score": nearest[0],
                "nearest_case_id": nearest[1],
                "nearest_prompt": nearest[2],
                "passes_threshold": nearest[0] < args.threshold,
            }
        )
    args.report.write_text(
        json.dumps(
            {
                "draft_records": len(draft),
                "reference_records": len(references),
                "threshold": args.threshold,
                "all_pass": all(item["passes_threshold"] for item in results),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.report.read_text(encoding="utf-8"))
    if not all(item["passes_threshold"] for item in results):
        raise SystemExit("one or more OOD prompts are too similar to reference data")


if __name__ == "__main__":
    main()
