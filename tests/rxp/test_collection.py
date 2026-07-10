"""Tests for RXP ChromaDB collection management."""

from __future__ import annotations

from unittest.mock import ANY, MagicMock, call, patch

import pytest

pytest.importorskip("chromadb")
pytest.importorskip("sentence_transformers")

from q_ai.rxp.collection import RetrievalCollection
from q_ai.rxp.embedder import get_embedder
from q_ai.rxp.models import CorpusDocument

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture()
def collection() -> RetrievalCollection:
    """Create a fresh collection for testing."""
    embedder = get_embedder(MODEL_NAME)
    return RetrievalCollection(name="test-collection", embedder=embedder)


@pytest.fixture()
def docs() -> list[CorpusDocument]:
    """Three corpus documents."""
    return [
        CorpusDocument(id="doc-1", text="Remote work policy for employees", source="test"),
        CorpusDocument(id="doc-2", text="Time off and vacation request process", source="test"),
        CorpusDocument(id="doc-3", text="Expense reimbursement procedures", source="test"),
    ]


class TestRetrievalCollectionConfiguration:
    """Tests for security-sensitive ChromaDB collection configuration."""

    def test_disables_chromadb_embedding_function(self) -> None:
        client = MagicMock()
        first_collection = MagicMock()
        first_collection.name = "test-collection"
        client.create_collection.return_value = first_collection

        with patch("q_ai.rxp.collection.chromadb.Client", return_value=client):
            collection = RetrievalCollection(name="test-collection", embedder=MagicMock())

        collection.reset()

        assert client.create_collection.call_args_list == [
            call(name=ANY, embedding_function=None),
            call(name="test-collection", embedding_function=None),
        ]


@pytest.mark.integration
class TestRetrievalCollection:
    """Tests for RetrievalCollection (requires chromadb + sentence-transformers)."""

    def test_ingest_adds_documents(
        self, collection: RetrievalCollection, docs: list[CorpusDocument]
    ) -> None:
        count = collection.ingest(docs)
        assert count == 3
        assert collection.count == 3

    def test_query_returns_hits(
        self, collection: RetrievalCollection, docs: list[CorpusDocument]
    ) -> None:
        collection.ingest(docs)
        hits = collection.query("remote work policy", top_k=3)
        assert len(hits) == 3
        assert hits[0].rank == 1

    def test_query_top_k_limits_results(
        self, collection: RetrievalCollection, docs: list[CorpusDocument]
    ) -> None:
        collection.ingest(docs)
        hits = collection.query("remote work", top_k=2)
        assert len(hits) == 2

    def test_query_poison_flag_preserved(self, collection: RetrievalCollection) -> None:
        docs = [
            CorpusDocument(id="clean-1", text="Normal document about policies", source="test"),
            CorpusDocument(
                id="poison-1", text="Malicious policy update", source="test", is_poison=True
            ),
        ]
        collection.ingest(docs)
        hits = collection.query("policy update", top_k=2)
        poison_hits = [h for h in hits if h.is_poison]
        assert len(poison_hits) == 1
        assert poison_hits[0].document_id == "poison-1"

    def test_reset_clears_collection(
        self, collection: RetrievalCollection, docs: list[CorpusDocument]
    ) -> None:
        collection.ingest(docs)
        assert collection.count == 3
        collection.reset()
        assert collection.count == 0
