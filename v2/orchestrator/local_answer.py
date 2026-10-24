"""ローカル回答（F2・F3）: 権限付き検索 → ローカル LLM で根拠付きの回答。

C1 が local を選んだ質問だけをここで扱う。外部へは何も送らない:
この関数は Gateway を引数に取らず、検索結果を送信候補にも入れない（設計書「契約 B の可変入力は本人が書いた依頼文に限る」）。

- 権限は検索の前に適用する: 利用者の閲覧レベルを Retriever に渡し、Qdrant 側のフィルタで絞る（F2）
- 返ってきた根拠も閲覧レベルで再確認し、レベル不明・超過は捨てる（二重の確認。捨てた件数は監査に残す）
- 未登録の利用者・検索失敗・根拠 0 件・生成失敗はすべて held（N4）。外部への代替送信はしない
- 回答の閲覧は依頼者本人かつ根拠の最高レベル以上に限る（RunResult.can_view）
- 監査には件数・document_id・字数だけを残す。質問・根拠・回答の本文は残さない（N7）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from orchestrator.pipeline import AuditEvent, RunResult
from orchestrator.policy import Policy

MAX_ANSWER_CHARS = 4000


@dataclass(frozen=True)
class Chunk:
    content: str
    source: str  # ファイル名（basename）。表示用
    access_level: int | None
    document_id: str | None = None
    chunk_index: int | None = None


class Retriever(Protocol):
    def retrieve(self, question: str, level: int) -> list[Chunk]:
        """閲覧レベル level 以下の文書だけから検索する。"""
        ...


class Generator(Protocol):
    def answer(self, question: str, chunks: Sequence[Chunk]) -> str: ...


class Decider(Protocol):
    """「コードが決める」方式（v2/decide の DecisionEngine）。結果は kind で見分ける:
    decided（answer_text・decision_record・used_chunks）/ held（decision_record）/ no_record。"""

    def decide(self, question: str, chunks: Sequence[Chunk]) -> object: ...


@dataclass
class LocalResult:
    run: RunResult
    sources: tuple[str, ...] = ()  # 回答の根拠（[n] の順）
    bodies: tuple[str, ...] = ()  # 監査の本文除去用（質問・根拠・回答）。表示や保存には使わない
    # decider が走った時だけ入る。判断の記録（decided / held / no_record の verdict.decision_record の写し）。
    # 最終の答えの記録（ask の on_final）の元になる。監査（AuditEvent）には入れない
    decision_record: dict | None = None
    # decider が返した kind（decided / held / no_record）。走らなかった・失敗した・不明な結果は None / "error"
    decider_kind: str | None = None


def _record_of(verdict) -> dict | None:
    rec = getattr(verdict, "decision_record", None)
    return dict(rec) if isinstance(rec, dict) else None


def _held(stage: str, reason: str, audit: list[AuditEvent], bodies: Sequence[str], *,
          decision_record: dict | None = None, decider_kind: str | None = None) -> LocalResult:
    audit.append(AuditEvent(stage, "error", reason))
    audit.append(AuditEvent("result", "held", f"stopped_at={stage}: {reason}"))
    return LocalResult(RunResult(outcome="held", stopped_at=stage, reason=reason, audit=audit), bodies=tuple(bodies),
                       decision_record=decision_record, decider_kind=decider_kind)


def _answer_by_code(user: str, verdict, events: list[AuditEvent], bodies: list[str], max_answer_chars: int) -> LocalResult:
    text = verdict.answer_text
    used = list(verdict.used_chunks)
    if not isinstance(text, str) or not text.strip() or not used or len(text) > max_answer_chars:
        return _held("decide", "コードによる判定の答えが使えない形", events, bodies,
                     decision_record=_record_of(verdict), decider_kind="decided")
    rec = verdict.decision_record
    bodies.append(text)
    events.append(AuditEvent("decide", "ok", f"rule={rec.get('rule_id')} outcome={rec.get('outcome_key')} chars={len(text)}"))
    events.append(AuditEvent("result", "answered", "stopped_at=none: コードが規程から判定（外部送信なし）"))
    run = RunResult(outcome="answered", stopped_at="none", reason="コードが規程から判定", output_text=text, audit=events,
                    response_visibility="source_level_requester_only", visible_to=user,
                    min_view_level=max(c.access_level for c in used))
    return LocalResult(run, sources=tuple(c.source for c in used), bodies=tuple(bodies),
                       decision_record=_record_of(verdict), decider_kind="decided")


def answer_locally(
    user: str,
    question: str,
    pol: Policy,
    retriever: Retriever,
    generator: Generator,
    *,
    audit: Sequence[AuditEvent] = (),
    max_answer_chars: int = MAX_ANSWER_CHARS,
    decider: Decider | None = None,
) -> LocalResult:
    events = list(audit)
    bodies: list[str] = [question]
    u = pol.users.get(user)
    if u is None:
        return _held("eligibility", "利用者が登録されていない（閲覧レベル不明のため検索しない）", events, bodies)

    try:
        found = list(retriever.retrieve(question, u.level))
    except Exception as e:  # noqa: BLE001
        return _held("retrieve", f"検索に失敗: {type(e).__name__}", events, bodies)

    allowed = [c for c in found if isinstance(c.access_level, int) and not isinstance(c.access_level, bool)
               and c.access_level <= u.level]
    dropped = len(found) - len(allowed)
    bodies.extend(c.content for c in found)
    ids = ",".join(str(c.document_id) for c in allowed)
    events.append(AuditEvent("retrieve", "ok", f"level={u.level} hits={len(found)} dropped={dropped} docs=[{ids}]"))
    if not allowed:
        return _held("retrieve", "閲覧できる根拠が見つからない（根拠なしでは生成しない）", events, bodies)

    no_record: dict | None = None
    decider_kind: str | None = None
    if decider is not None:
        try:
            verdict = decider.decide(question, allowed)
            kind = getattr(verdict, "kind", None)
            if kind == "decided":
                return _answer_by_code(user, verdict, events, bodies, max_answer_chars)
            if kind == "held":
                code = str(getattr(verdict, "decision_record", {}).get("hold_code") or "unknown")
                return _held("decide", f"コードによる判定を保留（{code}）", events, bodies,
                             decision_record=_record_of(verdict), decider_kind="held")
            if kind != "no_record":
                return _held("decide", "コードによる判定の結果が不明", events, bodies,
                             decision_record=_record_of(verdict), decider_kind="error")
            no_record, decider_kind = _record_of(verdict) or {}, "no_record"
        except Exception as e:  # noqa: BLE001
            return _held("decide", f"コードによる判定に失敗: {type(e).__name__}", events, bodies, decider_kind="error")
        # no_record: 数の規程ではない問は、下の通常のローカル回答へ

    try:
        text = generator.answer(question, allowed)
    except Exception as e:  # noqa: BLE001
        return _held("generate", f"ローカル LLM の生成に失敗: {type(e).__name__}", events, bodies,
                     decision_record=no_record, decider_kind=decider_kind)
    if not isinstance(text, str) or not text.strip():
        return _held("generate", "ローカル LLM の回答が空", events, bodies,
                     decision_record=no_record, decider_kind=decider_kind)
    truncated = len(text) > max_answer_chars
    if truncated:
        text = text[:max_answer_chars]
    bodies.append(text)
    events.append(AuditEvent("generate", "ok", f"chars={len(text)} truncated={truncated}"))
    events.append(AuditEvent("result", "answered", "stopped_at=none: ローカルで回答（外部送信なし）"))
    # 回答は根拠の最高レベル以上の閲覧者で、かつ依頼者本人だけが見られる
    run = RunResult(outcome="answered", stopped_at="none", reason="ローカルで回答", output_text=text, audit=events,
                    response_visibility="source_level_requester_only", visible_to=user,
                    min_view_level=max(c.access_level for c in allowed))
    return LocalResult(run, sources=tuple(c.source for c in allowed), bodies=tuple(bodies),
                       decision_record=no_record, decider_kind=decider_kind)
