from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate base or LoRA HR orchestrator")
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def extract_json(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    start = candidate.find("{")
    if start < 0:
        raise json.JSONDecodeError("no JSON object found", candidate, 0)
    value, _ = json.JSONDecoder().raw_decode(candidate[start:])
    if not isinstance(value, dict):
        raise ValueError("prediction is not a JSON object")
    return value


def iter_targets(dataset: Path) -> Iterable[dict[str, Any]]:
    with dataset.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            prefix: list[dict[str, str]] = []
            assistant_index = 0
            for message in record["messages"]:
                if message["role"] == "assistant":
                    yield {
                        "case_id": record["case_id"],
                        "scenario": record["scenario"],
                        "assistant_index": assistant_index,
                        "messages": list(prefix),
                        "target_text": message["content"],
                    }
                    assistant_index += 1
                prefix.append(message)


def first_action(value: dict[str, Any] | None) -> str | None:
    if not value:
        return None
    plan = value.get("execution_plan")
    if not isinstance(plan, list) or not plan or not isinstance(plan[0], dict):
        return None
    return plan[0].get("action")


def question_fields(value: dict[str, Any] | None) -> set[str]:
    if not value or not isinstance(value.get("questions"), list):
        return set()
    return {
        item.get("field")
        for item in value["questions"]
        if isinstance(item, dict) and isinstance(item.get("field"), str)
    }


def score_prediction(target: dict[str, Any], text: str) -> dict[str, Any]:
    from pydantic import ValidationError
    from hr_assistant.schema import OrchestratorResponse

    parsed: dict[str, Any] | None = None
    parse_error: str | None = None
    schema_error: str | None = None
    try:
        parsed = extract_json(text)
    except (json.JSONDecodeError, ValueError) as exc:
        parse_error = str(exc)

    schema_valid = False
    if parsed is not None:
        try:
            OrchestratorResponse.model_validate(parsed)
            schema_valid = True
        except ValidationError as exc:
            schema_error = str(exc)

    target_status = target.get("status")
    predicted_status = parsed.get("status") if parsed else None
    target_intent_value = target.get("intent")
    predicted_intent_value = parsed.get("intent") if parsed else None
    target_intent = (
        target_intent_value.get("category") if isinstance(target_intent_value, dict) else None
    )
    predicted_intent = (
        predicted_intent_value.get("category")
        if isinstance(predicted_intent_value, dict)
        else None
    )
    target_action = first_action(target)
    predicted_action = first_action(parsed)
    target_questions = question_fields(target)
    predicted_questions = question_fields(parsed)
    intersection = len(target_questions & predicted_questions)
    precision = intersection / len(predicted_questions) if predicted_questions else float(not target_questions)
    recall = intersection / len(target_questions) if target_questions else float(not predicted_questions)
    question_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    safety_filter_valid = None
    if target_action == "search_policy":
        search = (parsed or {}).get("search")
        filters = search.get("filters") if isinstance(search, dict) else {}
        filters = filters if isinstance(filters, dict) else {}
        safety_filter_valid = filters.get("access_level") == "employee" and filters.get("is_active") is True

    return {
        "json_parse_valid": parsed is not None,
        "schema_valid": schema_valid,
        "status_correct": predicted_status == target_status,
        "intent_correct": predicted_intent == target_intent,
        "action_correct": predicted_action == target_action,
        "clarification_correct": (predicted_status == "needs_clarification")
        == (target_status == "needs_clarification"),
        "rejection_correct": (predicted_status == "rejected") == (target_status == "rejected"),
        "question_fields_exact": predicted_questions == target_questions,
        "question_fields_f1": question_f1,
        "safety_filter_valid": safety_filter_valid,
        "target_status": target_status,
        "predicted_status": predicted_status,
        "target_intent": target_intent,
        "predicted_intent": predicted_intent,
        "target_action": target_action,
        "predicted_action": predicted_action,
        "prediction": parsed,
        "parse_error": parse_error,
        "schema_error": schema_error,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = [
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

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {"targets": len(items)}
        for name in metric_names:
            values = [item[name] for item in items if item[name] is not None]
            result[name] = sum(float(value) for value in values) / len(values) if values else None
            result[name + "_count"] = len(values)
        result["target_status_counts"] = dict(Counter(item["target_status"] for item in items))
        return result

    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scenario[row["scenario"]].append(row)
    return {
        "overall": summarize(rows),
        "by_scenario": {name: summarize(items) for name, items in sorted(by_scenario.items())},
    }


def batches(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def main() -> None:
    args = parse_args()
    from unsloth import FastLanguageModel

    import torch

    targets = list(iter_targets(args.dataset))
    if args.limit is not None:
        targets = targets[: args.limit]
    model_source = str(args.adapter) if args.adapter else args.model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_source,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        full_finetuning=False,
    )
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    end_of_turn_id = base_tokenizer.convert_tokens_to_ids("<turn|>")
    terminators = list(dict.fromkeys([tokenizer.eos_token_id, end_of_turn_id]))

    output_rows: list[dict[str, Any]] = []
    started = time.time()
    for batch_index, batch in enumerate(batches(targets, args.batch_size), 1):
        prompts = [
            tokenizer.apply_chat_template(
                item["messages"], tokenize=False, add_generation_prompt=True
            )
            for item in batch
        ]
        encoded = tokenizer(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_seq_length,
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=terminators,
            )
        continuation = generated[:, encoded["input_ids"].shape[1] :]
        predictions = tokenizer.batch_decode(continuation, skip_special_tokens=True)
        for item, prediction in zip(batch, predictions, strict=True):
            target = json.loads(item["target_text"])
            scored = score_prediction(target, prediction)
            output_rows.append(
                {
                    "case_id": item["case_id"],
                    "scenario": item["scenario"],
                    "assistant_index": item["assistant_index"],
                    "target": target,
                    "raw_prediction": prediction,
                    **scored,
                }
            )
        print(f"evaluated={len(output_rows)}/{len(targets)} batch={batch_index}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    summary = {
        "model": args.model,
        "adapter": str(args.adapter) if args.adapter else None,
        "dataset": str(args.dataset),
        "max_sequence_length": args.max_seq_length,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "decoding": "greedy",
        "stop_token_ids": terminators,
        "records": sum(1 for line in args.dataset.open(encoding="utf-8") if line.strip()),
        "assistant_targets": len(output_rows),
        "elapsed_seconds": time.time() - started,
        **aggregate(output_rows),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
