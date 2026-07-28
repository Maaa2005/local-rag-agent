# HR orchestrator adapter

This directory contains metadata for the approved files from experiment
`20260714-ood-remediation-160step/adapter`. The LoRA safetensors weight is
distributed through Hugging Face Hub and is not stored in Git. The roughly
7.58 GB Gemma base model is also not stored in this repository.

Set `ADAPTER_HF_REPO` (and preferably a pinned `ADAPTER_HF_REVISION`) in `.env`.
When `adapter_model.safetensors` is absent, `install.sh` and `install.ps1` download
the adapter snapshot before starting Docker. Both installers then require
`adapter_config.json` and `adapter_model.safetensors` and verify the approved
weight SHA-256.

Expected adapter SHA-256:
`5a8f318629bbb6fcc4f0131164ab6088299cac9eeec44a76463a32f37baa3a59`.

Private or gated Hub repositories also require `HF_TOKEN`. Base-model and adapter
download and use remain subject to their upstream model and repository terms.
