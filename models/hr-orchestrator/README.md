# HR orchestrator adapter

This directory contains the approved files from experiment
`20260714-ood-remediation-160step/adapter`. The LoRA safetensors file is managed by
Git LFS; the roughly 7.58 GB Gemma base model is not stored in this repository.
The installer refuses to enable orchestration unless both `adapter_config.json` and
`adapter_model.safetensors` exist.

Expected adapter SHA-256:
`5a8f318629bbb6fcc4f0131164ab6088299cac9eeec44a76463a32f37baa3a59`.

After cloning, run `git lfs pull` before starting vLLM. Base-model download and use
remain subject to the upstream Gemma model terms.
