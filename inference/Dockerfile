FROM vllm/vllm-openai:v0.25.1

COPY patch_vllm_wsl.py /tmp/patch_vllm_wsl.py
RUN python3 /tmp/patch_vllm_wsl.py && rm /tmp/patch_vllm_wsl.py
