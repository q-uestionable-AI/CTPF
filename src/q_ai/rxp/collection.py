"""ChromaDB collection management for retrieval testing."""

from __future__ import annotations

import uuid

import chromadb

from q_ai.rxp.embedder import Embedder
from q_ai.rxp.models import CorpusDocument, RetrievalHit


class RetrievalCollection:
    """Manages a ChromaDB collection for retrieval testing.

    Wraps a ChromaDB ephemeral client and collection. Documents are
    embedded via the specified Embedder (not ChromaDB's built-in
    embedding function) so we control which model is used.

    Each instance creates a unique collection to avoid name collisions.

    Args:
        name: Collection name prefix.
        embedder: Embedder instance for this collection.
    """

    def __init__(self, name: str, embedder: Embedder) -> None:
        self._embedder = embedder
        self._client = chromadb.Client()
        # Append UUID to avoid name collisions across instances
        unique_name = f"{name}-{uuid.uuid4().hex[:8]}"
        self._collection = self._client.create_collection(
            name=unique_name,
            embedding_function=None,
        )
        self._poison_ids: set[str] = set()

    def ingest(self, documents: list[CorpusDocument]) -> int:
        """Add documents to the collection.

        Embeds document text via self.embedder, stores in ChromaDB
        with metadata including is_poison flag.

        Args:
            documents: Documents to ingest.

        Returns:
            Number of documents added.
        """
        if not documents:
            return 0

        texts = [doc.text for doc in documents]
        ids = [doc.id for doc in documents]
        metadatas = [
            {**doc.metadata, "is_poison": str(doc.is_poison), "source": doc.source}
            for doc in documents
        ]

        embeddings = self._embedder.encode(texts)

        self._collection.add(
            ids=ids,
            embeddings=embeddings,  # type: ignore[arg-type]
            documents=texts,
            metadatas=metadatas,  # type: ignore[arg-type]
        )

        for doc in documents:
            if doc.is_poison:
                self._poison_ids.add(doc.id)

        return len(documents)

    def query(self, query_text: str, top_k: int = 5) -> list[RetrievalHit]:
        """Query the collection and return ranked results.

        Embeds the query text, queries ChromaDB, maps results
        to RetrievalHit objects with rank and distance.

        Args:
            query_text: The query to search for.
            top_k: Number of results to return.

        Returns:
            List of RetrievalHit in rank order (1-based).
        """
        query_embedding = self._embedder.encode([query_text])[0]

        results = self._collection.query(
            query_embeddings=[query_embedding],  # type: ignore[arg-type]
            n_results=min(top_k, self.count),
        )

        hits: list[RetrievalHit] = []
        if results["ids"] and results["ids"][0]:
            doc_ids = results["ids"][0]
            distances = results["distances"][0] if results["distances"] else [0.0] * len(doc_ids)

            for rank_idx, (doc_id, distance) in enumerate(zip(doc_ids, distances, strict=True)):
                hits.append(
                    RetrievalHit(
                        document_id=doc_id,
                        rank=rank_idx + 1,
                        distance=float(distance),
                        is_poison=doc_id in self._poison_ids,
                    )
                )

        return hits

    def reset(self) -> None:
        """Delete all documents from the collection."""
        # ChromaDB doesn't have a clear method; delete and recreate
        name = self._collection.name
        self._client.delete_collection(name)
        self._collection = self._client.create_collection(
            name=name,
            embedding_function=None,
        )
        self._poison_ids.clear()

    @property
    def count(self) -> int:
        """Number of documents in the collection."""
        return int(self._collection.count())
