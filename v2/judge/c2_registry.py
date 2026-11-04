"""承認済み本文の登録簿（Issue #17「ID とハッシュの許可表」）。

送信時の C2（judge.provenance_gate）は、承認済み本文の中身を手がかり語で検査し直さない。代わりに
この登録簿で「同じ ID・版・本文（sha256）が、この契約・用途・宛先で登録時の検査を通ったか」を引く。
ハッシュは同じ本文であることの証明で、安全な本文であることの証明ではない。そこで登録時に内容を
検査し、その結果を用途・宛先・版（ID@版と契約の版）に結びつけて残す。

登録時の検査は、送信時に依頼文へ当てる手がかり語・文字種（judge.request_cues）と、今の RuleC2Gate の
規則（金額は閾値の形でも当たる）の和を、正規化した本文の上の位置つきで出す（registration_findings）。
契約 A は送信内容がほぼ承認済み本文だけなので、ここが唯一の内容検査になる。

承認は本文の箇所に結ぶ（本人決定 11/3）。承認は {"text": 本文中の文字列, "reason": 理由} で、外せる種類
（WAIVABLE: 規程上の額・日付など、実データでない値）の当たりだけを、その text（本文に 1 回だけ出る物）の範囲にまるごと入る
場合に限って外す。同じ種類の別の箇所の当たりは通らない。それ以外の種類（個人情報・社名・社内のテーブル名・
M&A・注入など）、空白を除いた写しだけでの当たり、文字種の当たりは、承認では外せず登録を拒否する。
text が正規化した本文に出てこない承認は記録の誤りとして拒否する。当たらない承認は拒否しない。

登録には、人が全文を確かめる段を別に置く。行の reviewed_by は本文の確認者で、承認済み本文の approver・
当たりの承認をした人（checked_by）と別人でなければならない。fixture: で始まる確認者は開発用の架空本文だけ。
封印用の測定と本番では、本文の作者・実装担当・承認者と別の人の名前が要る。

依頼文に書いてよい値（ApprovedText.request_identifiers・request_values）も登録時の値を行に写す。
検査の規則を変えると REGISTRATION_CHECK_VERSION が変わり、古い版で登録した本文は送信時に保留になる
（登録し直すまで送らない）。

登録簿は policies/c2_registry.json。署名は無いので、読み込み時に行の形と当たり・承認の対応を確かめる。
作り直しと照合:
  python -m judge.c2_registry --write --reviewed-by 確認者 [--acknowledge 'ID@版=本文の文字列::理由' ...
                                                          --checked-by 承認をした人]
  python -m judge.c2_registry --check   # ファイルが今の本文・今の検査と一致するか
--acknowledge は箇所ごとに繰り返す。同じ使い道（ref・contract・destination）で sha256・acknowledged・
request_values が同じ行があれば、checked_by・reviewed_by はその行から引き継ぐ。引き継げない行があるのに
--reviewed-by が無ければ拒否する（終了コード 1）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from judge.base import question_version
from judge.normalize import normalize_text
from judge.request_cues import CUES_VERSION, cue_hits, cue_spans, unexpected_letters
from judge.rules import _C2_PATTERNS
from orchestrator.policy import DEFAULT_POLICY_DIR, ApprovedText, Contract, Policy, load_policy, sha256_text

DEFAULT_REGISTRY_PATH = DEFAULT_POLICY_DIR / "c2_registry.json"
FORMAT = "c2-registry@1"

# 承認で外せる当たりの種類。規程上の額・日付など、実データでない値だけ
WAIVABLE = frozenset({"amount", "real_date"})

REGISTRATION_CHECK_SPEC = {
    "request_cues": CUES_VERSION,  # 正規化の版を含む
    "rule_patterns": [(n, p.pattern, p.flags) for n, p in _C2_PATTERNS],
    "unexpected_script": "judge.request_cues.unexpected_letters",
    "waivable": sorted(WAIVABLE),  # これ以外の種類・空白を除いた写しだけの当たり・文字種の当たりは外せない
    "acknowledgement_unit": "span",  # 承認の text の出現の範囲に、当たりの範囲がまるごと入る
    "acknowledgement_unique": True,  # 承認の text は本文に 1 回だけ出る（1 か所に結ぶ）
    "human_review": "reviewed_by is non-empty and differs from approver and checked_by after normalize_text/case/space; review is pass",
    "acknowledgement_one_value": True,  # 承認の text 1 つに値は 1 つ
}
REGISTRATION_CHECK_VERSION = question_version("c2-registration@3", REGISTRATION_CHECK_SPEC)

_ENTRY_KEYS = ("ref", "sha256", "contract", "purpose", "destination", "check_version", "hits", "acknowledged",
               "request_values", "checked_by", "reviewed_by", "review", "result")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class RegistrationRefused(ValueError):
    """登録時の検査を通らない、または承認の範囲外の登録。"""


def registration_hits(body: str) -> list[str]:
    """当たった項目の名前だけ（空白を除いた写しだけでの当たりと文字種の当たりも含む）。位置は registration_findings。"""
    n = normalize_text(body)
    hits = [name for name, pat in _C2_PATTERNS if pat.search(n)] + cue_hits(body)
    if unexpected_letters(body):
        hits.append("unexpected_script")
    return list(dict.fromkeys(hits))


def registration_findings(body: str) -> tuple[list[tuple[str, int, int, str]], list[str], list[str]]:
    """登録時の検査の当たりを位置つきで返す: (当たり, 空白を除いた写しだけで当たった語, 文字種の当たり)。

    当たりは (名前, 始まり, 終わり, 文字列) で、正規化した写しの上の位置。手がかり語（cue_spans）と、
    RuleC2Gate の規則（_C2_PATTERNS）の和を、始まり・終わり・名前の順に並べる。
    """
    n = normalize_text(body)
    spans, compact_only = cue_spans(body)
    spans = spans + [(name, m.start(), m.end(), m.group(0)) for name, pat in _C2_PATTERNS for m in pat.finditer(n)]
    return sorted(spans, key=lambda s: (s[1], s[2], s[0])), compact_only, unexpected_letters(body)


def _same_person(a: str, b: str) -> bool:
    """全角半角・大文字小文字・空白・見えない文字の違いだけなら同じ人（「部長（架空）」と「部長(架空)」）。"""
    def key(s: str) -> str:
        return "".join(normalize_text(s).casefold().split())
    return key(a) == key(b)


def _ack_problem(a: Any) -> str | None:
    if not isinstance(a, dict) or set(a) != {"text", "reason"}:
        return "acknowledgement must be {text, reason}"
    if not all(isinstance(a[k], str) and a[k].strip() for k in ("text", "reason")):
        return "acknowledgement text and reason must be non-empty strings"
    return None


def _occurrences(n: str, text: str) -> list[tuple[int, int]]:
    """n の中の text のすべての出現の範囲 [s, e)（重なる出現も数える）。"""
    out, i = [], n.find(text)
    while i != -1:
        out.append((i, i + len(text)))
        i = n.find(text, i + 1)
    return out


def _checked_row(approved: ApprovedText, contract: Contract, destination: str,
                 acknowledged: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """人（checked_by・reviewed_by）以外のすべての検査をして、登録簿の行の元を返す。"""
    if sha256_text(approved.body) != approved.sha256:
        raise RegistrationRefused(f"{approved.key}: body does not match its approved sha256")
    if contract.key not in approved.contracts:
        raise RegistrationRefused(f"{approved.key}: contract {contract.key} is not in its approval")
    if approved.kind != contract.input_kind:
        raise RegistrationRefused(f"{approved.key}: kind {approved.kind} is not the contract input {contract.input_kind}")
    if approved.purpose != contract.purpose:
        raise RegistrationRefused(f"{approved.key}: purpose {approved.purpose} differs from {contract.purpose}")
    if destination not in approved.destinations or destination not in contract.destinations:
        raise RegistrationRefused(f"{approved.key}: destination {destination} is outside the approval or contract")
    for a in acknowledged:
        why = _ack_problem(a)
        if why is not None:
            raise RegistrationRefused(f"{approved.key}: {why}: {a!r}")
    ack = sorted({(a["text"], a["reason"]) for a in acknowledged})
    n = normalize_text(approved.body)
    absent = [t for t, _ in ack if t not in n]
    if absent:
        raise RegistrationRefused(f"{approved.key}: acknowledged text is not in the normalized body: {absent}")
    # 承認は本文の 1 か所に結ぶ。2 回以上出る文字列は、前後の語を足して 1 か所に決める（Codex 11/3）
    many = [f"{t} ({len(_occurrences(n, t))} times)" for t, _ in ack if len(_occurrences(n, t)) > 1]
    if many:
        raise RegistrationRefused(f"{approved.key}: acknowledged text must point to one place; it occurs {many}. "
                                  "Include the words around it")

    spans, compact_only, scripts = registration_findings(approved.body)
    fixed = sorted({s[0] for s in spans if s[0] not in WAIVABLE} | set(compact_only)
                   | ({"unexpected_script"} if scripts else set()))
    if fixed:
        raise RegistrationRefused(f"{approved.key}: registration check hit {fixed}, which acknowledgement cannot waive")
    ranges = [r for t, _ in ack for r in _occurrences(n, t)]
    # 承認は 1 か所の値に結ぶ。本文全体を text に書いて別の実額まで外すのを止める（別の目 11/4）
    for t, _ in ack:
        rs, re_ = _occurrences(n, t)[0]
        inside = sorted({(s, e) for _, s, e, _ in spans if rs <= s and e <= re_})
        merged = [sp for i, sp in enumerate(inside) if i == 0 or sp[0] >= max(x[1] for x in inside[:i])]
        if len(merged) > 1:
            raise RegistrationRefused(f"{approved.key}: acknowledged text {t!r} covers {len(merged)} values; "
                                      "acknowledge each value on its own")
    open_hits = [f"{name}:{text}" for name, s, e, text in spans if not any(rs <= s and e <= re_ for rs, re_ in ranges)]
    if open_hits:
        raise RegistrationRefused(
            f"{approved.key}: registration check hit {list(dict.fromkeys(open_hits))} outside the acknowledged text")
    hits = list(dict.fromkeys((name, text) for name, _, _, text in spans))
    return {"ref": approved.key, "sha256": approved.sha256, "contract": contract.key, "purpose": contract.purpose,
            "destination": destination, "check_version": REGISTRATION_CHECK_VERSION,
            "hits": [{"cue": c, "text": t} for c, t in hits],
            "acknowledged": [{"text": t, "reason": r} for t, r in ack],
            "request_values": {"identifiers": list(approved.request_identifiers),
                               "values": [t for t, _ in approved.request_values]},
            "review": "pass", "result": "pass"}


def _people_problem(approved: ApprovedText, checked_by: str, reviewed_by: str) -> str | None:
    if not isinstance(reviewed_by, str) or not reviewed_by.strip():
        return "reviewed_by must be a non-empty string"
    if _same_person(reviewed_by, approved.approver):
        return f"reviewed_by {reviewed_by!r} is the approver of the text"
    if _same_person(reviewed_by, checked_by):
        return f"reviewed_by {reviewed_by!r} is the same as checked_by"
    return None


def register(approved: ApprovedText, contract: Contract, destination: str, *,
             acknowledged: Sequence[Mapping[str, str]] = (), checked_by: str, reviewed_by: str) -> dict[str, Any]:
    """1 つの使い道（契約・用途・宛先）について登録時の検査をし、登録簿の 1 行を返す。

    acknowledged は [{"text": 本文中の文字列, "reason": 理由}]。承認の範囲外（契約・用途・宛先が承認済み
    本文または契約と合わない）、本文が承認時の sha256 と違う、外せない当たり、承認の範囲に入らない当たり、
    承認の text が正規化した本文に無い、確認者（reviewed_by）が空または承認者・checked_by と同じ、
    のどれかなら RegistrationRefused。
    """
    row = _checked_row(approved, contract, destination, acknowledged)
    why = _people_problem(approved, checked_by, reviewed_by)
    if why is not None:
        raise RegistrationRefused(f"{approved.key}: {why}")
    return dict(row, checked_by=checked_by, reviewed_by=reviewed_by)


def build_entries(policy: Policy, *, checked_by: str, reviewed_by: str | None = None,
                  acknowledged: Mapping[str, Sequence[Mapping[str, str]]] | None = None,
                  previous: Iterable[Mapping[str, Any]] = (), re_review: bool = False) -> list[dict[str, Any]]:
    """policies/ の承認済み本文のすべての使い道（承認と契約の両方が許す契約×宛先）を登録する。

    previous（今の登録簿）に、同じ使い道・本文・承認・依頼文に書いてよい値の行があれば、その checked_by と
    reviewed_by を引き継ぐ。引き継げない行は reviewed_by（引数）が要り、無ければ RegistrationRefused。
    """
    def same(e: Mapping[str, Any]) -> tuple[Any, ...]:
        return (e["ref"], e["contract"], e["destination"], e["sha256"],
                tuple((a["text"], a["reason"]) for a in e["acknowledged"]),
                (tuple(e["request_values"]["identifiers"]), tuple(e["request_values"]["values"])))

    prev = {same(e): (e["checked_by"], e["reviewed_by"]) for e in previous}
    # 承認者が替わったら前の確認は引き継がない（全文確認は承認者と別の人である事を見た時の物なので）
    for e in previous:
        a0 = policy.approved.get(e["ref"])
        if a0 is not None and _same_person(e["reviewed_by"], a0.approver):
            prev.pop(same(e), None)
    out = []
    for a in sorted(policy.approved.values(), key=lambda x: x.key):
        for ck in a.contracts:
            c = policy.contracts.get(ck)
            if c is None:
                continue
            for d in c.destinations:
                if d in a.destinations:
                    e = _checked_row(a, c, d, (acknowledged or {}).get(a.key, ()))
                    inherited = prev.get(same(e))
                    if inherited is None and reviewed_by is None:
                        raise RegistrationRefused(
                            f"{a.key} ({ck}/{d}): reviewed_by is required for a row that cannot carry one over")
                    if re_review and reviewed_by is None:
                        raise RegistrationRefused("re_review needs reviewed_by")
                    # re_review（全部の本文を確かめ直した）なら、引き継げる行も reviewed_by に替える。
                    # 開発用の fixture:dev を、封印用・本番で実際の確認者に替えるため（Codex 11/3）
                    e["checked_by"] = inherited[0] if inherited else checked_by
                    e["reviewed_by"] = reviewed_by if (re_review or inherited is None) else inherited[1]
                    why = _people_problem(a, e["checked_by"], e["reviewed_by"])
                    if why is not None:
                        raise RegistrationRefused(f"{a.key} ({ck}/{d}): {why}")
                    out.append(e)
    return out


def _canon(e: Mapping[str, Any]) -> dict[str, Any]:
    """行の鍵の並びを決まった形にする（入れ子の鍵の並びで登録簿の版が変わらないように）。"""
    row = {k: e[k] for k in _ENTRY_KEYS}
    row["hits"] = [{"cue": h["cue"], "text": h["text"]} for h in e["hits"]]
    row["acknowledged"] = [{"text": a["text"], "reason": a["reason"]} for a in e["acknowledged"]]
    row["request_values"] = {"identifiers": list(e["request_values"]["identifiers"]),
                             "values": list(e["request_values"]["values"])}
    return row


def _dump(entries: Iterable[Mapping[str, Any]]) -> str:
    rows = [_canon(e) for e in entries]
    rows.sort(key=lambda e: (e["ref"], e["contract"], e["destination"]))
    return json.dumps({"format": FORMAT, "entries": rows}, ensure_ascii=False, indent=2) + "\n"


def write_registry(path: Path | str, entries: Iterable[Mapping[str, Any]]) -> None:
    Path(path).write_bytes(_dump(entries).encode("utf-8"))


@dataclass(frozen=True)
class Registry:
    entries: tuple[Mapping[str, Any], ...]
    sha256: str  # 登録簿の中身の sha256（質問の版に入れる）

    def find(self, ref: str, contract: str, purpose: str, destination: str) -> Mapping[str, Any] | None:
        for e in self.entries:
            if (e["ref"], e["contract"], e["purpose"], e["destination"]) == (ref, contract, purpose, destination):
                return e
        return None

    def entries_list(self) -> list[dict[str, Any]]:
        return [dict(e) for e in self.entries]


def _nonblank(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def _row_problem(e: Any) -> str | None:
    if not isinstance(e, dict) or set(e) != set(_ENTRY_KEYS):
        return f"keys {sorted(e) if isinstance(e, dict) else type(e).__name__} != {sorted(_ENTRY_KEYS)}"
    for k in ("ref", "contract", "purpose", "destination", "check_version", "checked_by", "reviewed_by", "review",
              "result"):
        if not isinstance(e[k], str) or not e[k].strip():
            return f"{k} must be a non-empty string"
    if not _SHA256.fullmatch(e["sha256"] if isinstance(e["sha256"], str) else ""):
        return "sha256 must be 64 lowercase hex digits"
    if e["result"] != "pass":
        return f"result {e['result']!r} is not pass"
    if e["review"] != "pass":
        return f"review {e['review']!r} is not pass"
    if _same_person(e["reviewed_by"], e["checked_by"]):
        return "reviewed_by must differ from checked_by"
    if not isinstance(e["acknowledged"], list) or any(_ack_problem(a) for a in e["acknowledged"]):
        return "acknowledged must be a list of {text, reason} (non-empty strings)"
    if not isinstance(e["hits"], list) or not all(
            isinstance(h, dict) and set(h) == {"cue", "text"} and _nonblank(h["cue"]) and _nonblank(h["text"])
            for h in e["hits"]):
        return "hits must be a list of {cue, text} (non-empty strings)"
    for h in e["hits"]:
        if h["cue"] not in WAIVABLE:
            return f"hit {h['cue']!r} cannot be waived by acknowledgement"
        if not any(h["text"] in a["text"] for a in e["acknowledged"]):
            return f"hit {h['text']!r} is not inside any acknowledged text"
    rv = e["request_values"]
    if not isinstance(rv, dict) or set(rv) != {"identifiers", "values"} or not all(
            isinstance(rv[k], list) and all(_nonblank(x) for x in rv[k]) for k in ("identifiers", "values")):
        return "request_values must be {identifiers, values}: lists of non-empty strings"
    return None


def load_registry(path: Path | str = DEFAULT_REGISTRY_PATH) -> Registry:
    doc = json.loads(Path(path).read_bytes().decode("utf-8"))
    if not isinstance(doc, dict) or doc.get("format") != FORMAT:
        raise ValueError(f"registry format != {FORMAT}")
    if not isinstance(doc.get("entries"), list):
        raise ValueError("registry entries must be a list")
    for i, e in enumerate(doc["entries"]):
        why = _row_problem(e)
        if why is not None:
            raise ValueError(f"registry entry {i}: {why}")
    entries = doc["entries"]
    if len({(e["ref"], e["contract"], e["purpose"], e["destination"]) for e in entries}) != len(entries):
        raise ValueError("registry has duplicate uses")
    # 改行コードや並びの違いで版が変わらないよう、正規化した書き出しのハッシュを使う
    return Registry(tuple(entries), hashlib.sha256(_dump(entries).encode("utf-8")).hexdigest())


def _parse_ack(items: Sequence[str]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for s in items:
        ref, sep, rest = s.partition("=")
        text, sep2, reason = rest.partition("::")
        if not sep or not sep2 or not ref or not text.strip() or not reason.strip():
            raise SystemExit(f"--acknowledge needs ID@版=本文の文字列::理由: {s!r}")
        out.setdefault(ref, []).append({"text": text, "reason": reason})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true", help="policies/approved/ から登録簿を作り直す")
    g.add_argument("--check", action="store_true", help="登録簿が今の本文・今の検査と一致するか")
    ap.add_argument("--policy-dir", type=Path, default=DEFAULT_POLICY_DIR)
    ap.add_argument("--registry", type=Path, default=None)
    ap.add_argument("--acknowledge", action="append", default=[],
                    help="承認者が承知の上で残す当たりの箇所。ID@版=本文の文字列::理由（箇所ごとに繰り返す・--write のとき）")
    ap.add_argument("--checked-by", default=None, help="新しく登録・承認した行に残す承認をした人")
    ap.add_argument("--reviewed-by", default=None,
                    help="新しく登録した行に残す、本文の全文を確かめた人（承認者・--checked-by と別人。--write のとき）")
    ap.add_argument("--re-review", action="store_true",
                    help="--reviewed-by の人が全部の本文を確かめ直した。引き継げる行の確認者も替える（--write のとき）")
    a = ap.parse_args(argv)
    if not a.write and (a.acknowledge or a.reviewed_by is not None or a.re_review):
        raise SystemExit("--acknowledge, --reviewed-by and --re-review are only for --write")
    if a.re_review and a.reviewed_by is None:
        raise SystemExit("--re-review needs --reviewed-by")
    if a.acknowledge and not a.checked_by:
        raise SystemExit("--acknowledge needs --checked-by (the person who approved the hits)")
    if a.reviewed_by is not None and not a.re_review:
        print("note: rows that carry over keep their reviewer; use --re-review if this person read every body",
              file=sys.stderr)
    ack_new = _parse_ack(a.acknowledge)
    path = a.registry or a.policy_dir / "c2_registry.json"
    pol = load_policy(a.policy_dir)
    old = None
    if path.exists():
        try:
            old = load_registry(path)
        except (ValueError, OSError) as e:  # 古い形・壊れた登録簿。照合はできず、作り直しは引き継ぎなしで進む
            print(f"unreadable registry {path}: {e}", file=sys.stderr)
            if a.check:
                return 1
    # 承認（acknowledged）は今の登録簿から引き継ぐ。本文が変わった物は引き継がない。--acknowledge が優先
    ack = {e["ref"]: e["acknowledged"] for e in (old.entries if old else ())
           if e["ref"] in pol.approved and pol.approved[e["ref"]].sha256 == e["sha256"]}
    ack.update(ack_new)
    try:
        entries = build_entries(pol, checked_by=a.checked_by or "registry-build", reviewed_by=a.reviewed_by, acknowledged=ack,
                                previous=old.entries if old else (), re_review=a.re_review)
    except RegistrationRefused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 1
    if a.write:
        write_registry(path, entries)
        print(f"wrote {len(entries)} entries to {path}")
        return 0
    if old is None:
        print(f"missing: {path}", file=sys.stderr)
        return 1
    if _dump(entries) != _dump(old.entries):
        print("registry differs from the current approved texts / registration check", file=sys.stderr)
        return 1
    print(f"ok: {len(old.entries)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
