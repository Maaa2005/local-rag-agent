from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemma 4 text-only LoRA training")
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument("--dataset", default="docs/ft-data/train.jsonl")
    parser.add_argument("--output", default="outputs/gemma4-orchestrator-lora")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--load-in-16bit", action="store_true")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Unsloth must patch Transformers/TRL before those libraries are imported.
    from unsloth import FastLanguageModel

    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    dataset = load_dataset("json", data_files={"train": args.dataset}, split="train")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=not args.load_in_16bit,
        load_in_16bit=args.load_in_16bit,
        full_finetuning=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=args.max_seq_length,
    )

    def render(batch: dict[str, list]) -> dict[str, list[str]]:
        return {
            "text": [
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                for messages in batch["messages"]
            ]
        }

    dataset = dataset.map(render, batched=True)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_length=args.max_seq_length,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=10,
            max_steps=args.max_steps,
            learning_rate=1e-4,
            logging_steps=1,
            output_dir=args.output,
            optim="adamw_8bit",
            seed=3407,
            dataset_num_proc=1,
            report_to="none",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
        ),
    )
    result = trainer.train()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "training_summary.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "dataset": args.dataset,
                "max_sequence_length": args.max_seq_length,
                "max_steps": args.max_steps,
                "lora_rank": args.lora_rank,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "load_in_4bit": not args.load_in_16bit,
                "train_loss": result.training_loss,
                "metrics": result.metrics,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
