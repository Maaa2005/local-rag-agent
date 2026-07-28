import asyncio

from app.services.qdrant import close_client
from app.services.retriever import retrieve
from app.database import db


async def main() -> None:
    cases = [
        ('有給休暇は何日', 1),
        ('勤怠承認の月次締め', 1),
        ('勤怠承認の月次締め', 2),
        ('役員報酬方針', 2),
        ('役員報酬方針', 3),
    ]
    await db.connect()
    try:
        for question, level in cases:
            results = await retrieve(question, level)
            sources = [
                (item.get('source_file'), round(item.get('score', 0), 3))
                for item in results
            ]
            print(f'Lv{level} {question} => {sources}')
    finally:
        await close_client()
        await db.close()


if __name__ == '__main__':
    asyncio.run(main())
