"""説明表示（F8）のデモ用ページ。決まった場面を製品と同じ経路に通し、説明を 1 枚の HTML に並べる。

- 経路は integration.flow.run_contract_a / integration.flow_b.run_contract_b をそのまま呼ぶ
- C1 / C2 はルール版（judge.rules）。送信役は gateway.adapters.FakeAdapter（偽物）で、外部へは一切送らない
- 利用者確認は候補の digest をそのまま返し、確認画面に出した内容（confirmation_view）を説明に渡す
- 閲覧者は依頼者本人（閲覧レベルは users.json の値）
- ページは外部の読み込みもスクリプトも許さない（Content-Security-Policy）。応答は orchestrator.explain がエスケープする

使い方:
  python -m evalharness.explain_demo --out out/explain-demo.html
"""
from __future__ import annotations

import argparse
import html
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gateway.adapters import FakeAdapter
from gateway.service import GatewayService
from gateway.store import SendStore
from integration.flow import run_contract_a
from integration.flow_b import run_contract_b
from judge.rules import RuleC1Router, RuleC2Gate
from orchestrator.explain import Explanation, explain, to_html
from orchestrator.pipeline import confirmation_view
from orchestrator.policy import load_policy

A_REF = "expl-keihi-001@1"
A_OPTS = {"文体": "です・ます調", "長さ": "400字以内"}
B_SPEC = "spec-csv-001@1"
B_OPTS = {"言語": "Python 3.11"}
B_FREE = "category ごとに amount を合計し、合計の大きい順に並べた結果を新しい CSV に書き出す関数を書いてください。"


@dataclass(frozen=True)
class Scene:
    title: str
    contract: str  # "A" / "B"
    user: str
    mode: str = "ok"  # FakeAdapter の振る舞い


SCENES = (
    Scene("契約 A を送信した", "A", "u_general"),
    Scene("契約 B を送信した", "B", "u_manager"),
    Scene("送信役が送る前に失敗した（送っていないと確定）", "A", "u_general", "not_sent"),
    Scene("送信先が受け付けなかった", "A", "u_general", "rejected"),
    Scene("送信先から返事が無かった（届いたか分からない）", "A", "u_general", "timeout"),
    Scene("権限の無い利用者（Gateway の前で拒否）", "A", "u_nobody"),
)


class _Confirmer:
    def __init__(self) -> None:
        self.seen: list = []

    def confirm(self, candidate):
        self.seen.append(candidate)
        return candidate.digest


def run_scene(scene: Scene, workdir: Path) -> Explanation:
    fake = FakeAdapter(mode=scene.mode)
    gateway = GatewayService(SendStore(workdir / "gateway.db"), {"claude": fake, "codex": fake})
    conf = _Confirmer()
    common: dict[str, Any] = dict(c1=RuleC1Router(), c2=RuleC2Gate(), confirmer=conf, gateway=gateway)
    if scene.contract == "A":
        res = run_contract_a(scene.user, A_REF, A_OPTS, "claude", request_text="経費精算の説明を FAQ にして", **common)
    else:
        res = run_contract_b(scene.user, B_SPEC, B_FREE, B_OPTS, "codex", **common)
    view = confirmation_view(conf.seen[-1]) if conf.seen else None
    user = load_policy().users.get(scene.user)
    return explain(res.run, viewer_id=scene.user, viewer_level=user.level if user else None, view=view, c1=res.c1)


def render_page(items: list[tuple[str, Explanation]]) -> str:
    sections = "".join(f"<h2>{html.escape(title)}</h2>{to_html(e)}" for title, e in items)
    return (
        "<!doctype html><html lang=\"ja\"><head><meta charset=\"utf-8\">"
        "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'\">"
        "<title>説明表示のデモ</title>"
        "<style>body{font-family:sans-serif;max-width:56rem;margin:2rem auto;line-height:1.6}"
        # 64 文字の digest や要求 ID で値の列が枠からはみ出さないよう、どこでも折り返す
        "dl{display:grid;grid-template-columns:14rem minmax(0,1fr);gap:.2rem 1rem}dt{font-weight:bold}"
        "dd{margin:0;overflow-wrap:anywhere}"
        "pre{white-space:pre-wrap;background:#f4f4f4;padding:.5rem}"
        ".explain{border:1px solid #ccc;padding:1rem;margin-bottom:2rem}"
        "[data-send-status=unknown],[data-send-status=refused]{border-color:#c60}</style></head><body>"
        "<h1>説明表示のデモ（偽の送信役。外部へは送っていない）</h1>"
        f"{sections}</body></html>"
    )


def build(out: Path) -> list[tuple[str, Explanation]]:
    items = []
    with tempfile.TemporaryDirectory() as d:
        for i, scene in enumerate(SCENES):
            work = Path(d) / str(i)
            work.mkdir()
            items.append((scene.title, run_scene(scene, work)))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_page(items), encoding="utf-8")
    return items


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    for title, e in build(args.out):
        print(f"{e.send_status:9} {e.outcome:9} {title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
