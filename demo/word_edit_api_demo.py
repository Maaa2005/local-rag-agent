from __future__ import annotations

import json
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'demo' / 'assets' / 'word_edit_sample.docx'
OUTPUT = ROOT / 'demo' / 'assets' / 'word_edit_sample_edited.docx'
BASE_URL = 'http://localhost:8000'

OPERATIONS = [
    {'find': '[所属]', 'replacement': '営業部'},
    {'find': '[氏名]', 'replacement': '山田 太郎'},
    {'find': '[紛失日]', 'replacement': '2026年7月25日'},
    {'find': '[証憑内容]', 'replacement': '取引先訪問時のタクシー領収書'},
    {'find': '[金額]', 'replacement': '4,280円'},
    {
        'find': '[理由]',
        'replacement': '移動中に封筒から取り出した後、誤って廃棄した可能性があります。',
    },
    {
        'find': '[再発防止策]',
        'replacement': '受領直後にスマートフォンで撮影し、当日中に経費精算システムへ登録します。',
    },
]


def main() -> None:
    OUTPUT.unlink(missing_ok=True)
    with httpx.Client(base_url=BASE_URL, timeout=30) as client:
        auth = client.post(
            '/api/auth/token',
            data={'username': 'demo_employee', 'password': 'DemoEmployee2026!'},
        )
        auth.raise_for_status()
        token = auth.json()['access_token']
        headers = {'Authorization': f'Bearer {token}'}

        with INPUT.open('rb') as source:
            draft_response = client.post(
                '/api/documents/docx/drafts',
                headers=headers,
                files={
                    'file': (
                        INPUT.name,
                        source,
                        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                    )
                },
                data={'operations_json': json.dumps(OPERATIONS, ensure_ascii=False)},
            )
        draft_response.raise_for_status()
        draft = draft_response.json()
        draft_url = '/api/documents/docx/drafts/{}/confirm'.format(draft['draft_id'])
        assert not OUTPUT.exists(), '承諾前に出力ファイルが作成されました'

        rejected = client.post(
            draft_url,
            headers=headers,
            json={'confirmed': False},
        )
        assert rejected.status_code == 400, rejected.text
        assert not OUTPUT.exists(), '承諾拒否後に出力ファイルが作成されました'

        confirmed = client.post(
            draft_url,
            headers=headers,
            json={'confirmed': True},
        )
        confirmed.raise_for_status()
        assert confirmed.headers['x-document-status'] == 'user-confirmed-draft'
        OUTPUT.write_bytes(confirmed.content)

        reused = client.post(
            draft_url,
            headers=headers,
            json={'confirmed': True},
        )
        assert reused.status_code == 404, reused.text

    print(json.dumps({
        'draft_id': draft['draft_id'],
        'replacements': draft['total_replacements'],
        'saved_after_confirmation': OUTPUT.exists(),
        'one_time_download_enforced': True,
        'output': str(OUTPUT),
        'bytes': OUTPUT.stat().st_size,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
