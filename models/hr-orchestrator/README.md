---
license: apache-2.0
base_model: unsloth/gemma-4-E2B-it-unsloth-bnb-4bit
base_model_relation: adapter
library_name: peft
pipeline_tag: text-generation
language:
- ja
- en
tags:
- gemma4
- lora
- peft
- qlora
- function-calling
- rag
---

# HR Orchestrator Adapter for Gemma 4 E2B

This repository contains the approved LoRA adapter from experiment
`20260714-ood-remediation-160step`. It was fine-tuned for pre-RAG orchestration:
interpreting Japanese HR requests, asking deterministic clarification questions,
rejecting unsafe requests, and returning a strict JSON execution plan.

## Base and inference models

- Training base: [`unsloth/gemma-4-E2B-it-unsloth-bnb-4bit`](https://huggingface.co/unsloth/gemma-4-E2B-it-unsloth-bnb-4bit)
- Original model: [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it)
- Tested inference quantization: [`cyankiwi/gemma-4-E2B-it-AWQ-INT4`](https://huggingface.co/cyankiwi/gemma-4-E2B-it-AWQ-INT4)

The adapter is a modification produced by QLoRA fine-tuning. It does not include
the Gemma base-model weights. See `NOTICE` for attribution and modification details.

## Training and evaluation

- Method: Unsloth 4-bit QLoRA, rank 8, alpha 16
- Final run: 160 steps
- Training examples: 884 synthetic conversations
- Human-authored OOD set: 30 held-out examples, not used for training
- OOD JSON validity: 96.7%
- OOD schema validity: 93.3%
- Safe-filter evaluation: 8/8

The training data is synthetic and does not contain real employee personal data or
real-company confidential policies.

## Intended use and limitations

This adapter is intended for an educational, local HR-assistant prototype. Backend
code, not this model, enforces document permissions and safety boundaries. The model
must not be used as the sole control for authorization, compliance, legal, labor, or
personnel decisions.

Held-out evaluation still contains premature `ready` decisions, missed rejections,
and one response outside the required JSON envelope. Human review and deterministic
backend validation remain required.

## Integrity

Expected SHA-256 for `adapter_model.safetensors`:

```text
5a8f318629bbb6fcc4f0131164ab6088299cac9eeec44a76463a32f37baa3a59
```

## License

The adapter is distributed under the Apache License 2.0. See `LICENSE` and `NOTICE`.
The referenced upstream models are also identified as Apache-2.0 on their respective
model cards. Users remain responsible for complying with applicable laws and all
upstream terms.
