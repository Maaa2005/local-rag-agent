import asyncio
import hashlib
import json
from pathlib import Path

from app.config import settings
from app.database import db
from app.services.qdrant import close_client, get_client
from app.services.retriever import basename_source, retrieve


CASES = [
    ('G01', '入社半年後の有給休暇は何日ですか', 1, '01_就業規則.md', None),
    ('G02', '所定労働時間と休憩時間を教えて', 1, '01_就業規則.md', None),
    ('G03', '緊急時の有休は当日申請できますか', 1, '01_就業規則.md', None),
    ('G04', '領収書を紛失した場合の経費精算手続きは', 1, '02_経費精算規程.md', None),
    ('G05', '経費の申請期限はいつですか', 1, '02_経費精算規程.md', None),
    ('G06', 'タクシー代を申請できる条件は', 1, '02_経費精算規程.md', None),
    ('G07', '在宅勤務はいつまでに申請しますか', 1, '03_在宅勤務ガイド.md', None),
    ('G08', '会社PCを社外へ持ち出す手続きを教えて', 1, '03_在宅勤務ガイド.md', None),
    ('G09', 'PCを紛失したら何分以内に連絡しますか', 1, '03_在宅勤務ガイド.md', None),
    ('M01', '管理職の月次勤怠締めはいつまでですか', 2, '04_管理職向け勤怠承認マニュアル.md', None),
    ('M02', '残業40時間到達時の対応を教えて', 2, '04_管理職向け勤怠承認マニュアル.md', None),
    ('M03', '部下の実績時刻を推測入力してよいですか', 2, '04_管理職向け勤怠承認マニュアル.md', None),
    ('M04', '人事評価の上期と下期の期間は', 2, '05_評価運用ガイド.md', None),
    ('M05', '人事評価で使ってはいけない属性は', 2, '05_評価運用ガイド.md', None),
    ('M06', '評価キャリブレーションでは何を確認しますか', 2, '05_評価運用ガイド.md', None),
    ('E01', '役員報酬は何で構成されますか', 3, '06_役員報酬方針.md', None),
    ('E02', '重大なコンプライアンス違反時の役員賞与は', 3, '06_役員報酬方針.md', None),
    ('E03', '役員報酬の個別金額はどう決定しますか', 3, '06_役員報酬方針.md', None),
    ('P01', '役員報酬の構成を教えて', 1, None, '06_役員報酬方針.md'),
    ('P02', '管理職の月次勤怠締めを教えて', 1, None, '04_管理職向け勤怠承認マニュアル.md'),
    ('P03', '役員賞与の減額条件を教えて', 2, None, '06_役員報酬方針.md'),
    ('N01', '社長の血液型を教えて', 3, None, None),
    ('N02', '社員食堂の今日のメニューは', 3, None, None),
    ('N03', '会社の創業者の出身大学を教えて', 3, None, None),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


async def integrity() -> dict:
    rows = await db.fetchall(
        'SELECT id, source_path, file_hash, access_level, status, unclassified '
        'FROM documents ORDER BY source_path'
    )
    file_checks = []
    for row in rows:
        path = Path(row['source_path'])
        file_checks.append({
            'source': basename_source(row['source_path']),
            'exists': path.exists(),
            'hash_match': path.exists() and sha256(path) == row['file_hash'],
            'level': row['access_level'],
            'status': row['status'],
            'unclassified': bool(row['unclassified']),
        })
    info = await get_client().get_collection(settings.qdrant_collection)
    return {
        'sqlite_documents': len(rows),
        'qdrant_points': info.points_count,
        'qdrant_status': str(info.status),
        'files': file_checks,
    }


async def evaluate_mode(rerank_enabled: bool) -> dict:
    settings.rerank_enabled = rerank_enabled
    details = []
    for case_id, question, level, expected, forbidden in CASES:
        chunks = await retrieve(question, level)
        sources = [basename_source(item.get('source_file')) for item in chunks]
        rank = sources.index(expected) + 1 if expected in sources else None
        permission_ok = all((item.get('access_level') or 99) <= level for item in chunks)
        forbidden_absent = forbidden not in sources if forbidden else True
        details.append({
            'id': case_id,
            'question': question,
            'level': level,
            'expected': expected,
            'forbidden': forbidden,
            'sources': sources,
            'rank': rank,
            'hit_at_1': rank == 1 if expected else None,
            'hit_at_5': rank is not None if expected else None,
            'permission_ok': permission_ok and forbidden_absent,
        })
    answerable = [item for item in details if item['expected']]
    reciprocal = [1 / item['rank'] if item['rank'] else 0 for item in answerable]
    return {
        'rerank_enabled': rerank_enabled,
        'cases': len(details),
        'answerable_cases': len(answerable),
        'hit_at_1': sum(item['hit_at_1'] for item in answerable) / len(answerable),
        'recall_at_5': sum(item['hit_at_5'] for item in answerable) / len(answerable),
        'mrr': sum(reciprocal) / len(reciprocal),
        'permission_accuracy': sum(item['permission_ok'] for item in details) / len(details),
        'details': details,
    }


async def main() -> None:
    await db.connect()
    original = settings.rerank_enabled
    try:
        result = {
            'integrity': await integrity(),
            'retrieval': [await evaluate_mode(True), await evaluate_mode(False)],
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        settings.rerank_enabled = original
        await close_client()
        await db.close()


if __name__ == '__main__':
    asyncio.run(main())
