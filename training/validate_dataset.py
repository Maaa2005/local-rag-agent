from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from hr_assistant.schema import OrchestratorResponse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    failures: list[str] = []
    count = 0
    with args.dataset.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            count += 1
            try:
                record = json.loads(line)
                assistants = [message for message in record["messages"] if message["role"] == "assistant"]
                if not assistants or record["messages"][-1]["role"] != "assistant":
                    raise ValueError("conversation must end with an assistant response")
                for message in assistants:
                    OrchestratorResponse.model_validate_json(message["content"])
            except (json.JSONDecodeError, KeyError, ValueError, ValidationError) as exc:
                failures.append(f"line {line_number}: {exc}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"validated {count} records")


if __name__ == "__main__":
    main()
