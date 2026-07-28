from pathlib import Path


TARGET = Path(
    '/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/buffer_utils.py'
)

source = TARGET.read_text(encoding='utf-8')

old_buffer = '''class UvaBuffer:
    def __init__(self, size: int | Sequence[int], dtype: torch.dtype):
        if not is_uva_available():
            raise RuntimeError("UVA is not available")
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu", pin_memory=True)
        self.np = self.cpu.numpy()
        self.uva = get_accelerator_view_from_cpu_tensor(self.cpu)
'''

new_buffer = '''class UvaBuffer:
    def __init__(self, size: int | Sequence[int], dtype: torch.dtype):
        # WSL2/WDDM may expose CUDA while UVA host mappings are unavailable.
        # These are small request-state buffers, so keep a normal CUDA mirror
        # and copy synchronously instead of failing engine initialization.
        self.fallback_to_gpu = not is_uva_available()
        self.cpu = torch.zeros(
            size,
            dtype=dtype,
            device="cpu",
            pin_memory=not self.fallback_to_gpu,
        )
        self.np = self.cpu.numpy()
        if self.fallback_to_gpu:
            self.uva = torch.zeros(size, dtype=dtype, device="cuda")
        else:
            self.uva = get_accelerator_view_from_cpu_tensor(self.cpu)
'''

old_copy = '''        dst[:n] = x
        return buf.uva[:n]
'''

new_copy = '''        dst[:n] = x
        if buf.fallback_to_gpu:
            buf.uva[:n].copy_(buf.cpu[:n], non_blocking=False)
        return buf.uva[:n]
'''

if source.count(old_buffer) != 1:
    raise RuntimeError('Unexpected vLLM UvaBuffer source; refusing to patch')
if source.count(old_copy) != 1:
    raise RuntimeError('Unexpected vLLM copy_to_uva source; refusing to patch')

patched = source.replace(old_buffer, new_buffer).replace(old_copy, new_copy)
TARGET.write_text(patched, encoding='utf-8')
print('Applied WSL no-UVA request-buffer fallback')
