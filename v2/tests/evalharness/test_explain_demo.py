"""説明表示のデモ用ページ: 決まった場面が期待どおりの送信の状態になり、ページにスクリプトが入らない。"""
from __future__ import annotations

from evalharness.explain_demo import SCENES, build, main


def test_each_scene_has_the_expected_send_status(tmp_path):
    out = tmp_path / "demo.html"
    items = build(out)
    assert [(e.send_status, e.outcome) for _, e in items] == [
        ("sent", "sent"),
        ("sent", "sent"),
        ("not_sent", "held"),
        ("refused", "rejected"),
        ("unknown", "held"),
        ("not_sent", "rejected"),
    ]
    assert [t for t, _ in items] == [s.title for s in SCENES]
    # 送った 2 場面は確認した本文と応答を出し、権限の無い場面は本文を出さない
    assert all(e.body and e.output_text for _, e in items[:2])
    assert items[-1][1].body is None and items[-1][1].output_text is None
    page = out.read_text(encoding="utf-8")
    assert "<script" not in page and "default-src 'none'" in page
    assert page.count('class="explain"') == len(SCENES)
    # 長い digest で値の列が枠を越えない（10/26 に目で見て直した。見た目そのものは画面で確かめる）
    assert "minmax(0,1fr)" in page and "overflow-wrap:anywhere" in page


def test_main_writes_page(tmp_path, capsys):
    out = tmp_path / "sub" / "demo.html"
    assert main(["--out", str(out)]) == 0
    assert out.exists() and "unknown" in capsys.readouterr().out
