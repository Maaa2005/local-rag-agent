from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from hr_assistant.schema import OrchestratorResponse


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.dataset.open(encoding="utf-8") if line.strip()]
    scenarios = Counter()
    for row in rows:
        if row.get("provenance") != "human_authored":
            raise ValueError(f"strict OOD set must be human-authored: {row.get('case_id')}")
        review = row.get("review", {})
        if review.get("status") != "human_approved" or not review.get("reviewer_role"):
            raise ValueError(f"human approval missing: {row.get('case_id')}")
        assistants = [message for message in row["messages"] if message["role"] == "assistant"]
        if len(assistants) != 1 or row["messages"][-1]["role"] != "assistant":
            raise ValueError(f"exactly one final assistant target required: {row['case_id']}")
        OrchestratorResponse.model_validate_json(assistants[0]["content"])
        scenarios[row["scenario"]] += 1
    manifest = {
        "dataset": str(args.dataset),
        "sha256": sha256(args.dataset),
        "records": len(rows),
        "scenarios": dict(sorted(scenarios.items())),
        "provenance": "human_authored_and_approved",
        "blind_evaluation_required": True,
    }
    args.output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
