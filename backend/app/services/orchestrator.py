from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.config import settings

SYSTEM_PROMPT = (Path(__file__).parent.parent / 'prompts' / 'orchestrator.txt').read_text(encoding='utf-8').strip()

STATUSES = {'needs_clarification', 'ready', 'rejected'}
CATEGORIES = {
    'hr_policy', 'expense', 'attendance', 'payroll', 'data_processing',
    'general_document', 'unknown',
}
ACTIONS = {
    'search_policy', 'process_file', 'request_approval', 'create_ticket',
    'answer', 'escalate',
}

DECISION_JSON_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'additionalProperties': False,
    'required': [
        'schema_version', 'status', 'intent', 'missing_fields', 'questions',
        'execution_plan', 'search',
    ],
    'properties': {
        'schema_version': {'type': 'string', 'const': '1.0'},
        'status': {'type': 'string', 'enum': sorted(STATUSES)},
        'intent': {
            'type': 'object',
            'additionalProperties': False,
            'required': ['category', 'summary', 'confidence'],
            'properties': {
                'category': {'type': 'string', 'enum': sorted(CATEGORIES)},
                'summary': {'type': 'string', 'minLength': 1},
                'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
            },
        },
        'missing_fields': {'type': 'array', 'items': {'type': 'string', 'minLength': 1}},
        'questions': {
            'type': 'array',
            'items': {
                'type': 'object',
                'additionalProperties': False,
                'required': ['field', 'question'],
                'properties': {
                    'field': {'type': 'string', 'minLength': 1},
                    'question': {'type': 'string', 'minLength': 1},
                },
            },
        },
        'execution_plan': {
            'type': 'array',
            'items': {
                'type': 'object',
                'additionalProperties': False,
                'required': ['step', 'action', 'parameters', 'requires_confirmation'],
                'properties': {
                    'step': {'type': 'integer', 'minimum': 1},
                    'action': {'type': 'string', 'enum': sorted(ACTIONS)},
                    'parameters': {'type': 'object'},
                    'requires_confirmation': {'type': 'boolean'},
                },
            },
        },
        'search': {
            'anyOf': [
                {'type': 'null'},
                {
                    'type': 'object',
                    'additionalProperties': False,
                    'required': ['query', 'top_k', 'filters'],
                    'properties': {
                        'query': {'type': 'string', 'minLength': 1},
                        'top_k': {'type': 'integer', 'minimum': 1},
                        'filters': {'type': 'object'},
                    },
                },
            ],
        },
    },
}

DECISION_RESPONSE_FORMAT = {
    'type': 'json_schema',
    'json_schema': {
        'name': 'orchestrator_decision',
        'strict': True,
        'schema': DECISION_JSON_SCHEMA,
    },
}


def extract_json(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith('```'):
        lines = candidate.splitlines()[1:]
        if lines and lines[-1].strip() == '```':
            lines.pop()
        candidate = '\n'.join(lines).strip()
    start = candidate.find('{')
    if start < 0:
        raise ValueError('orchestrator response did not contain JSON')
    value, end = json.JSONDecoder().raw_decode(candidate[start:])
    if not isinstance(value, dict):
        raise ValueError('orchestrator response must be a JSON object')
    trailing = candidate[start + end:].strip()
    if trailing and trailing != '```':
        raise ValueError('orchestrator response contained trailing content')
    return value


def validate_decision(value: dict[str, Any]) -> dict[str, Any]:
    required = {
        'schema_version', 'status', 'intent', 'missing_fields', 'questions',
        'execution_plan', 'search',
    }
    if set(value) != required or value['schema_version'] != '1.0':
        raise ValueError('invalid top-level schema')
    status, intent = value['status'], value['intent']
    if status not in STATUSES or not isinstance(intent, dict):
        raise ValueError('invalid status or intent')
    if set(intent) != {'category', 'summary', 'confidence'}:
        raise ValueError('invalid intent schema')
    if intent.get('category') not in CATEGORIES:
        raise ValueError('invalid intent category')
    confidence = intent.get('confidence')
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError('invalid intent confidence')
    if not isinstance(intent.get('summary'), str) or not intent['summary']:
        raise ValueError('invalid intent summary')
    missing_fields = value['missing_fields']
    questions = value['questions']
    plan = value['execution_plan']
    search = value['search']
    if not isinstance(missing_fields, list) or not all(
        isinstance(item, str) and item for item in missing_fields
    ):
        raise ValueError('invalid missing fields')
    if not isinstance(questions, list) or not all(
        isinstance(item, dict)
        and set(item) == {'field', 'question'}
        and isinstance(item['field'], str) and item['field']
        and isinstance(item['question'], str) and item['question']
        for item in questions
    ):
        raise ValueError('invalid questions')
    if not isinstance(plan, list):
        raise ValueError('invalid execution plan')
    for step in plan:
        if not isinstance(step, dict) or set(step) != {
            'step', 'action', 'parameters', 'requires_confirmation',
        }:
            raise ValueError('invalid plan step schema')
        if step.get('action') not in ACTIONS:
            raise ValueError('invalid plan action')
        if not isinstance(step['step'], int) or step['step'] < 1:
            raise ValueError('invalid plan step number')
        if not isinstance(step['parameters'], dict):
            raise ValueError('invalid plan parameters')
        if not isinstance(step['requires_confirmation'], bool):
            raise ValueError('invalid confirmation flag')
    if search is not None:
        if not isinstance(search, dict) or set(search) != {'query', 'top_k', 'filters'}:
            raise ValueError('invalid search schema')
        if not isinstance(search['query'], str) or not search['query']:
            raise ValueError('invalid search query')
        if not isinstance(search['top_k'], int) or search['top_k'] < 1:
            raise ValueError('invalid search top_k')
        if not isinstance(search['filters'], dict):
            raise ValueError('invalid search filters')
    if status == 'needs_clarification':
        if not missing_fields or not questions:
            raise ValueError('clarification requires fields and questions')
        if plan or search is not None:
            raise ValueError('clarification must not plan tools')
    if status == 'ready' and (
        missing_fields or questions or not plan
    ):
        raise ValueError('ready requires a plan and no questions')
    if status == 'rejected' and (missing_fields or questions or search is not None):
        raise ValueError('rejection must not ask questions or search')
    return value


def build_messages(question: str, history: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    for item in history or []:
        role = item.get('role')
        content = item.get('content')
        if role in {'user', 'assistant'} and isinstance(content, str) and content:
            messages.append({'role': role, 'content': content})
    messages.append({'role': 'user', 'content': question})
    return messages


def apply_required_field_guard(
    decision: dict[str, Any],
    question: str,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Enforce deterministic required fields after the probabilistic model."""
    user_text = " ".join(
        [
            str(item.get("content", ""))
            for item in history or []
            if item.get("role") == "user"
        ]
        + [question]
    )
    category = decision["intent"]["category"]
    if category == "expense" or any(
        keyword in user_text for keyword in ("領収書", "レシート", "経費精算", "立替")
    ):
        missing: list[str] = []
        questions: list[dict[str, str]] = []
        if not re.search(r"\d[\d,]*(?:円|万円)", user_text):
            missing.append("amount")
            questions.append({"field": "amount", "question": "経費の金額はいくらですか？"})
        if not re.search(
            r"(?:\d{4}年)?\d{1,2}月\d{1,2}日|今日|昨日|先週|今月|先月",
            user_text,
        ):
            missing.append("date")
            questions.append(
                {"field": "date", "question": "利用日または購入日はいつですか？"}
            )
        if missing:
            return {
                "schema_version": "1.0",
                "status": "needs_clarification",
                "intent": {
                    "category": "expense",
                    "summary": decision["intent"]["summary"],
                    "confidence": decision["intent"]["confidence"],
                },
                "missing_fields": missing,
                "questions": questions,
                "execution_plan": [],
                "search": None,
            }

    bypass_terms = ("アクセスフィルタ", "権限を変更", "権限変更", "指示を無視")
    if any(term in user_text for term in bypass_terms):
        return {
            "schema_version": "1.0",
            "status": "rejected",
            "intent": {
                "category": "general_document",
                "summary": decision["intent"]["summary"],
                "confidence": decision["intent"]["confidence"],
            },
            "missing_fields": [],
            "questions": [],
            "execution_plan": [
                {
                    "step": 1,
                    "action": "escalate",
                    "parameters": {"reason": "アクセス制御の回避は許可されていません"},
                    "requires_confirmation": False,
                }
            ],
            "search": None,
        }
    return decision


async def orchestrate(
    question: str, history: list[dict[str, str]] | None = None
) -> dict[str, Any] | None:
    if not settings.orchestrator_enabled:
        return None
    preflight_seed = {
        'schema_version': '1.0',
        'status': 'ready',
        'intent': {
            'category': 'unknown',
            'summary': question.strip() or 'request',
            'confidence': 1.0,
        },
        'missing_fields': [],
        'questions': [],
        'execution_plan': [
            {
                'step': 1,
                'action': 'answer',
                'parameters': {},
                'requires_confirmation': False,
            }
        ],
        'search': None,
    }
    preflight = apply_required_field_guard(preflight_seed, question, history)
    if preflight is not preflight_seed:
        return validate_decision(preflight)
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=settings.vllm_base_url, api_key='NONE')
    response = await client.chat.completions.create(
        model=settings.orchestrator_model,
        messages=build_messages(question, history),
        max_tokens=settings.orchestrator_max_tokens,
        temperature=0,
        response_format=DECISION_RESPONSE_FORMAT,
    )
    content = response.choices[0].message.content or ''
    decision = validate_decision(extract_json(content))
    return validate_decision(apply_required_field_guard(decision, question, history))


def terminal_message(decision: dict[str, Any]) -> str | None:
    if decision['status'] == 'needs_clarification':
        return '\n'.join('- ' + str(item.get('question', '')) for item in decision['questions'])
    if decision['status'] == 'rejected':
        return 'この依頼は安全上または権限上の理由で実行できません。必要な場合は管理者へ確認してください。'
    return None
