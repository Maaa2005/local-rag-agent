"""規程カードの下書きを流す小さな道具 (Issue #16・docs/v2-rule-card-procedure.md の流れ 1)。

prepare: 規程ごとにプロンプト (<規程名>.prompt.md) を作り、指紋を draft_manifest.json に書く。
collect : 答え (<規程名>.answer.md) から <cards>・<unable> を取り出し、parse_card と quote (doc.quote と、あれば条件・例外ごとの quote) の実在を確かめて
          通った物を <cards>/<id>.json に書く。形が崩れた物は直さず <drafts>/unable.json に足す (作り直さない)。

使い方 (PYTHONUTF8=1・PYTHONPATH=backend):
  python evaluation/v2/sealed/draft_cards.py prepare --corpus <フォルダ> --options <options.json> --prompt <draft_prompt.md> --out <フォルダ>
  python evaluation/v2/sealed/draft_cards.py collect --drafts <フォルダ> --cards <フォルダ> [--status draft]
標準出力は件数だけ (規程の文は出さない)。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[3] / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.append(str(_BACKEND))

from app.services.rule_cards import CardError, parse_card  # noqa: E402

OPTIONS_FORMAT = "conclusion-options@1"
MANIFEST_FORMAT = "draft-manifest@1"
PROMPT_SUFFIX = ".prompt.md"
ANSWER_SUFFIX = ".answer.md"
UNABLE_FILE = "unable.json"
_HEAD_COMMENT = re.compile(r"\A\s*<!--.*?-->\s*", re.DOTALL)
_PLACEHOLDER = re.compile(r"\{\{(SOURCE|REGULATION|OPTION_PAIRS)\}\}")
_DOC_OPEN = "<document_content>\n"
_DOC_CLOSE = "\n</document_content>\n</document>\n</documents>\n"
_CARD_ID = re.compile(r"^[a-z][a-z0-9-]*$")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _squash(s: str) -> str:
    """空白を全部除く (改行位置の違いだけでは quote を落とさない)。"""
    return re.sub(r"\s+", "", s)


def load_pairs(options_path: Path) -> list[list[str]]:
    """options.json から語の組を読む。形が違えば ValueError。"""
    doc = json.loads(Path(options_path).read_text(encoding="utf-8"))
    pairs = doc.get("conclusion_options") if isinstance(doc, dict) else None
    if not (isinstance(doc, dict) and doc.get("format") == OPTIONS_FORMAT and isinstance(pairs, list) and pairs
            and all(isinstance(p, list) and p and all(isinstance(w, str) and w for w in p) for p in pairs)):
        raise ValueError(f"options の形が違う (format={OPTIONS_FORMAT}・conclusion_options は語の組の配列)")
    return pairs


def render_prompt(template: str, source: str, regulation: str, pairs: list[list[str]]) -> str:
    """先頭の <!-- --> を外し、{{SOURCE}}・{{REGULATION}}・{{OPTION_PAIRS}} を 1 回の走査で埋める。"""
    body = _HEAD_COMMENT.sub("", template, count=1)
    values = {"SOURCE": source, "REGULATION": regulation.strip("\n"),
              "OPTION_PAIRS": "\n".join(" / ".join(p) for p in pairs)}
    return _PLACEHOLDER.sub(lambda m: values[m.group(1)], body)


def prepare(corpus: Path, options: Path, prompt: Path, out: Path) -> int:
    """規程ごとのプロンプトと draft_manifest.json を書く。規程の数を返す。"""
    pairs = load_pairs(options)
    template_bytes = Path(prompt).read_bytes()
    template = template_bytes.decode("utf-8")
    files = sorted(Path(corpus).glob("*.md"))
    if not files:
        raise ValueError("corpus に *.md が無い")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rules: dict[str, str] = {}
    prompts: dict[str, str] = {}
    for f in files:
        raw = f.read_bytes()
        text = render_prompt(template, f.stem, raw.decode("utf-8"), pairs)
        (out / f"{f.stem}{PROMPT_SUFFIX}").write_text(text, encoding="utf-8", newline="\n")
        rules[f.stem] = _sha(raw)
        prompts[f.stem] = _sha(text.encode("utf-8"))
    manifest = {"format": MANIFEST_FORMAT, "prompt_sha256": _sha(template_bytes),
                "options_sha256": _sha(Path(options).read_bytes()), "n_pairs": len(pairs),
                "rules_sha256": rules, "rendered_prompts_sha256": prompts}
    (out / "draft_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8")
    return len(files)


def regulation_of(prompt_text: str) -> str:
    """prepare が作ったプロンプトの <document_content> から規程の文を取り出す。"""
    start = prompt_text.find(_DOC_OPEN)
    end = prompt_text.rfind(_DOC_CLOSE) if start >= 0 else -1
    if start < 0 or end < 0:
        raise ValueError("プロンプトに <document_content> が無い")
    return prompt_text[start + len(_DOC_OPEN):end]


def _tag(text: str, name: str) -> str | None:
    m = re.search(rf"<{name}>(.*?)</{name}>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def _json_list(raw: str | None, what: str) -> tuple[list | None, str | None]:
    """(配列, 理由)。壊れていれば (None, 理由)。"""
    if raw is None:
        return None, f"{what} の欄が無い"
    try:
        v = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"{what} の JSON が壊れている: {e.msg}"
    if not isinstance(v, list):
        return None, f"{what} が配列でない"
    return v, None


def _check_card(doc, regulation: str, status: str, seen: set[str]) -> str | None:
    """カードの問題を理由で返す (通れば None)。"""
    if not isinstance(doc, dict):
        return "カードがオブジェクトでない"
    cid = doc.get("id")
    try:
        parse_card(doc)
    except CardError as e:
        return f"parse_card: {e}"
    except (TypeError, AttributeError, KeyError) as e:
        return f"parse_card で型が崩れている: {type(e).__name__}: {e}"
    if not isinstance(cid, str) or not _CARD_ID.match(cid) or cid == UNABLE_FILE.removesuffix(".json"):
        return f"id が不正: {cid!r}"
    if cid in seen:
        return f"id が重複: {cid}"
    if doc["status"] != status:
        return f"status が {status} でない: {doc['status']}"
    if _squash(doc["doc"]["quote"]) not in _squash(regulation):
        return "quote が規程の文に無い"
    bad = _missing_quote(doc.get("conditions"), regulation) or _missing_quote(doc.get("exceptions"), regulation)
    if bad:
        return f"条件の quote が規程の文に無い: {bad}"
    return None


def _missing_quote(conds, regulation: str) -> str | None:
    """条件・例外 (入れ子も) の quote が規程の文に無ければその条件の名前を返す。quote は任意で、あれば確かめる。"""
    reg = _squash(regulation)
    for c in conds or []:
        if "quote" in c and _squash(c["quote"]) not in reg:
            return str(c.get("name"))
        for k in ("all", "any"):
            if k in c:
                inner = _missing_quote(c[k], regulation)
                if inner:
                    return inner
    return None


def collect(drafts: Path, cards_dir: Path, status: str = "draft") -> tuple[int, int]:
    """(書いたカードの枚数, 作れなかった数)。unable.json は <drafts> に書く。"""
    drafts, cards_dir = Path(drafts), Path(cards_dir)
    cards_dir.mkdir(parents=True, exist_ok=True)
    prompts = sorted(drafts.glob(f"*{PROMPT_SUFFIX}"))
    if not prompts:
        raise ValueError("drafts に *.prompt.md が無い (prepare が先)")
    unable: list[dict] = []
    seen: set[str] = set()
    written = 0
    for pf in prompts:
        name = pf.name.removesuffix(PROMPT_SUFFIX)
        answer_path = drafts / f"{name}{ANSWER_SUFFIX}"
        if not answer_path.is_file():
            unable.append({"rule": name, "reason": "答えのファイルが無い"})
            continue
        answer = answer_path.read_text(encoding="utf-8")
        regulation = regulation_of(pf.read_text(encoding="utf-8"))
        cards, why = _json_list(_tag(answer, "cards"), "<cards>")
        if cards is None:
            unable.append({"rule": name, "reason": why})
        for doc in cards or []:
            reason = _check_card(doc, regulation, status, seen)
            if reason:
                rid = doc.get("id") if isinstance(doc, dict) and isinstance(doc.get("id"), str) else name
                unable.append({"rule": rid, "reason": reason})
                continue
            seen.add(doc["id"])
            (cards_dir / f"{doc['id']}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                                                         encoding="utf-8")
            written += 1
        items, why = _json_list(_tag(answer, "unable"), "<unable>")
        if items is None:
            if cards is not None or why != "<unable> の欄が無い":
                unable.append({"rule": name, "reason": why})
            continue
        for it in items:
            if isinstance(it, dict) and isinstance(it.get("rule"), str) and isinstance(it.get("reason"), str):
                unable.append({"rule": it["rule"], "reason": it["reason"]})
            else:
                unable.append({"rule": name, "reason": "<unable> の要素の形が違う"})
    (drafts / UNABLE_FILE).write_text(json.dumps(unable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return written, len(unable)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--corpus", required=True)
    a.add_argument("--options", required=True)
    a.add_argument("--prompt", required=True)
    a.add_argument("--out", required=True)
    b = sub.add_parser("collect")
    b.add_argument("--drafts", required=True)
    b.add_argument("--cards", required=True)
    b.add_argument("--status", default="draft")
    args = p.parse_args(argv)
    try:
        if args.cmd == "prepare":
            n = prepare(Path(args.corpus), Path(args.options), Path(args.prompt), Path(args.out))
            print(f"規程: {n}")
        else:
            n, m = collect(Path(args.drafts), Path(args.cards), args.status)
            print(f"カード: {n}・作れなかった: {m}")
    except (ValueError, OSError) as e:
        print(f"失敗: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
