import hashlib
import re
import struct

_CJK = re.compile(r'[　-鿿豈-﫿゠-ヿ぀-ゟ]')
_ASCII = re.compile(r'[a-zA-Z0-9]+')
_VOCAB_SIZE = 131072  # 2^17


def _token_id(token: str) -> int:
    d = hashlib.sha256(token.encode()).digest()
    return struct.unpack_from(">I", d)[0] % _VOCAB_SIZE


def build_sparse(text: str) -> tuple[list[int], list[float]]:
    """CJK文字bigram + ASCII単語 → TF正規化スパースベクトル (indices, values)。"""
    tokens: list[str] = []

    cjk = _CJK.findall(text)
    for i in range(len(cjk) - 1):
        tokens.append(cjk[i] + cjk[i + 1])

    tokens.extend(w.lower() for w in _ASCII.findall(text))

    if not tokens:
        return [], []

    counts: dict[int, float] = {}
    for t in tokens:
        tid = _token_id(t)
        counts[tid] = counts.get(tid, 0.0) + 1.0

    total = float(len(tokens))
    pairs = sorted(counts.items())
    return [k for k, _ in pairs], [v / total for _, v in pairs]
