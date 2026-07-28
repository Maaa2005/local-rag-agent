"""dataset.jsonl と evaluation/corpus の整合性検証 (Phase 4: 評価50問化)。

GPU 実行機の /watched に配置される実文書の代わりに、evaluation/corpus/ に
架空企業の Markdown 文書を同梱している。ここでは:

- 50 問揃っているか、ID の重複がないか
- expected_sources が指す文書が evaluation/corpus に実在するか
- should_refuse=false の質問について、expected_answer_keywords が対応する
  コーパス文書の本文に実際に含まれているか (キーワードが宙に浮いていないか)
- category / user_access_level の組み合わせが妥当か

を機械的に確認する。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset.jsonl"
CORPUS_ROOT = ROOT / "corpus"

VALID_CATEGORIES = {"general", "manager", "executive", "permission_test", "hallucination_test"}
# フォルダ名(=access_level) の対応。README にも明記する。
CATEGORY_ACCESS_LEVEL = {"general": 1, "manager": 2, "executive": 3}


def _load_dataset() -> list[dict]:
    with DATASET.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _source_to_corpus_path(source: str) -> Path:
    """/watched/<category>/<file>.md -> evaluation/corpus/<category>/<file>.md"""
    # Parse this logical container path independently of the host OS.
    parts = source.replace("\\", "/").split("/")
    rel_parts = [part for part in parts if part and part != "watched"]
    return CORPUS_ROOT.joinpath(*rel_parts)


@pytest.fixture(scope="module")
def dataset() -> list[dict]:
    return _load_dataset()


def test_dataset_has_exactly_50_questions(dataset):
    assert len(dataset) == 50


def test_ids_are_unique_and_sequential(dataset):
    ids = [row["id"] for row in dataset]
    assert len(ids) == len(set(ids)), "重複 ID がある"
    expected_ids = {f"Q{i:03d}" for i in range(1, 51)}
    assert set(ids) == expected_ids


def test_category_distribution(dataset):
    from collections import Counter

    counts = Counter(row["category"] for row in dataset)
    assert counts["general"] == 20
    assert counts["manager"] == 8
    assert counts["executive"] == 6
    assert counts["permission_test"] == 8
    assert counts["hallucination_test"] == 8


@pytest.mark.parametrize("row_index", range(50))
def test_row_schema_and_category_validity(dataset, row_index):
    row = dataset[row_index]
    required = {
        "id", "question", "expected_answer_keywords", "expected_sources",
        "user_access_level", "category", "should_refuse",
    }
    assert required <= set(row.keys()), f"missing fields in {row}"
    assert row["category"] in VALID_CATEGORIES
    assert row["user_access_level"] in (1, 2, 3)
    assert isinstance(row["should_refuse"], bool)
    assert isinstance(row["expected_answer_keywords"], list)
    assert isinstance(row["expected_sources"], list)

    # permission_test / hallucination_test は「拒否すべき」質問なので
    # expected_sources は空 (コーパス中に正解ソースを期待しない) が既定の設計。
    if row["category"] in ("permission_test", "hallucination_test"):
        assert row["should_refuse"] is True
    else:
        # general/manager/executive はコーパスに接地した質問である前提。
        assert row["expected_sources"], f"{row['id']}: expected_sources が空"


def test_expected_sources_exist_in_corpus(dataset):
    missing = []
    for row in dataset:
        for src in row["expected_sources"]:
            path = _source_to_corpus_path(src)
            if not path.is_file():
                missing.append((row["id"], src))
    assert not missing, f"corpus に存在しない expected_sources: {missing}"


def test_keywords_are_grounded_in_corpus_text(dataset):
    """should_refuse=false の質問は、キーワードが対応文書の本文に実在すること。"""
    missing = []
    for row in dataset:
        if row["should_refuse"]:
            continue
        texts = []
        for src in row["expected_sources"]:
            path = _source_to_corpus_path(src)
            if path.is_file():
                texts.append(path.read_text(encoding="utf-8"))
        combined = "\n".join(texts)
        for kw in row["expected_answer_keywords"]:
            if kw not in combined:
                missing.append((row["id"], kw))
    assert not missing, f"コーパス本文に存在しないキーワード: {missing}"


def test_permission_test_uses_lower_access_level_than_target_category(dataset):
    """permission_test は「低権限ユーザーが上位文書を聞く」設計であること。"""
    permission_rows = [r for r in dataset if r["category"] == "permission_test"]
    assert permission_rows
    for row in permission_rows:
        # 一般(1)しか持たないユーザーが manager/executive 文書を聞く、
        # または管理職(2)が executive 文書を聞くケースを想定するため、
        # 最上位(3)ではないこと (=誰かにとって"上位"の情報である)。
        assert row["user_access_level"] < 3, f"{row['id']}: permission_test の access_level が想定外"


def test_corpus_documents_are_non_trivial(dataset):
    """全 corpus 文書がある程度のボリューム (1000 文字以上) を持つこと。"""
    md_files = list(CORPUS_ROOT.rglob("*.md"))
    assert len(md_files) >= 9, "既存9文書相当のコーパスが揃っていない"
    for path in md_files:
        text = path.read_text(encoding="utf-8")
        assert len(text) >= 1000, f"{path} が短すぎる ({len(text)} 文字)"


def test_every_corpus_document_is_referenced_by_at_least_one_question(dataset):
    referenced = set()
    for row in dataset:
        for src in row["expected_sources"]:
            referenced.add(_source_to_corpus_path(src).resolve())
    all_docs = {p.resolve() for p in CORPUS_ROOT.rglob("*.md")}
    unreferenced = all_docs - referenced
    assert not unreferenced, f"どの質問からも参照されていない文書: {unreferenced}"
