"""Knowledge base management for the qai assistant.

Handles document ingestion, MDX preprocessing, chunking, embedding,
ChromaDB persistent storage, hash-based change detection, and retrieval.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import chromadb

from q_ai.rxp.embedder import get_embedder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_TARGET_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
CHARS_PER_TOKEN = 4

_ASSIST_DIR = Path.home() / ".qai" / "assist"
_CHROMA_DIR = _ASSIST_DIR / "chroma"
_MANIFEST_PATH = _ASSIST_DIR / "index_manifest.json"

PRODUCT_COLLECTION = "product_knowledge"
USER_COLLECTION = "user_knowledge"

# Supported file extensions for ingestion
_INGESTABLE_EXTENSIONS = {".md", ".mdx", ".txt", ".yaml", ".yml"}

# JSX tag pattern: lines that are purely opening/closing JSX tags
_JSX_TAG_RE = re.compile(
    r"^\s*</?(?:Card|CardGroup|Tabs|Tab|CodeGroup|Tip|Warning|Note|Steps|Step"
    r"|Info|Accordion|AccordionGroup|ResponseField|Expandable|Param"
    r"|ParamField|Snippet|Frame|Icon|Check|RequestExample|ResponseExample)"
    r"(?:\s[^>]*)?\s*/?\s*>\s*$"
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DocumentChunk:
    """A chunk of document text with metadata.

    Attributes:
        text: The chunk text content.
        source: Source file path (relative).
        heading: Heading hierarchy (e.g. "## Quick start").
        content_class: Either "product" or "user".
        chunk_id: Unique identifier for this chunk.
    """

    text: str
    source: str
    heading: str
    content_class: str
    chunk_id: str = ""


@dataclass
class RetrievalResult:
    """A retrieved chunk with relevance metadata.

    Attributes:
        chunk: The document chunk.
        distance: Distance score from the query.
    """

    chunk: DocumentChunk
    distance: float


@dataclass
class _FileManifest:
    """Tracks file hashes for change detection.

    Attributes:
        product_hashes: SHA-256 hashes for product doc files.
        user_hashes: SHA-256 hashes for user knowledge files.
    """

    product_hashes: dict[str, str] = field(default_factory=dict)
    user_hashes: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MDX preprocessing
# ---------------------------------------------------------------------------


def strip_mdx_jsx(text: str) -> str:
    """Strip JSX component tags from MDX content.

    Removes lines that are purely JSX opening/closing tags while
    preserving text content inside JSX blocks and frontmatter.

    Args:
        text: Raw MDX file content.

    Returns:
        Cleaned text with JSX tags removed.
    """
    lines = text.splitlines()
    cleaned: list[str] = []
    for line in lines:
        if _JSX_TAG_RE.match(line):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Extract YAML frontmatter from markdown/MDX content.

    Args:
        text: File content potentially starting with --- frontmatter ---.

    Returns:
        Tuple of (frontmatter dict, remaining content).
    """
    if not text.startswith("---"):
        return {}, text

    end_idx = text.find("---", 3)
    if end_idx == -1:
        return {}, text

    fm_block = text[3:end_idx].strip()
    body = text[end_idx + 3 :].strip()

    metadata: dict[str, str] = {}
    for line in fm_block.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            metadata[key.strip()] = value.strip().strip('"').strip("'")

    return metadata, body


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    """Estimate token count using character-based heuristic.

    Args:
        text: Text to estimate.

    Returns:
        Estimated token count.
    """
    return len(text) // CHARS_PER_TOKEN


def _split_on_headings(text: str) -> list[tuple[str, str]]:
    """Split markdown text on heading boundaries.

    Args:
        text: Markdown text content.

    Returns:
        List of (heading, section_text) tuples.
    """
    heading_re = re.compile(r"^(#{2,4}\s+.+)$", re.MULTILINE)
    parts = heading_re.split(text)

    sections: list[tuple[str, str]] = []
    current_heading = ""

    i = 0
    while i < len(parts):
        part = parts[i]
        if heading_re.match(part):
            current_heading = part.strip()
            i += 1
            continue
        content = part.strip()
        if content:
            sections.append((current_heading, content))
        i += 1

    if not sections and text.strip():
        sections.append(("", text.strip()))

    return sections


def _split_long_chunk(text: str, max_tokens: int) -> list[str]:
    """Split a long text chunk on paragraph boundaries.

    Args:
        text: Text that exceeds max_tokens.
        max_tokens: Maximum tokens per sub-chunk.

    Returns:
        List of sub-chunks.
    """
    paragraphs = re.split(r"\n\n+", text)
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if _estimate_tokens(candidate) > max_tokens and current:
            chunks.append(current)
            current = para
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks


def _add_overlap(chunks: list[str], overlap_chars: int) -> list[str]:
    """Add overlap from the end of the previous chunk to the start of the next.

    Args:
        chunks: List of text chunks.
        overlap_chars: Number of characters to overlap.

    Returns:
        Chunks with overlap prepended.
    """
    if len(chunks) <= 1 or overlap_chars <= 0:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap_chars:]
        # Find a word boundary in the overlap to avoid splitting mid-word
        space_idx = prev_tail.find(" ")
        if space_idx != -1:
            prev_tail = prev_tail[space_idx + 1 :]
        result.append(f"{prev_tail} {chunks[i]}")

    return result


def chunk_document(
    text: str,
    source: str,
    content_class: str,
) -> list[DocumentChunk]:
    """Chunk a document into embedding-sized pieces.

    Splits on heading boundaries first, then on paragraph boundaries
    for oversized chunks, with overlap between chunks.

    Args:
        text: Document text (after MDX stripping if applicable).
        source: Source file path for metadata.
        content_class: "product" or "user".

    Returns:
        List of DocumentChunk objects.
    """
    frontmatter, body = _parse_frontmatter(text)
    sections = _split_on_headings(body)

    raw_chunks: list[tuple[str, str]] = []
    for heading, section_text in sections:
        if _estimate_tokens(section_text) > CHUNK_TARGET_TOKENS:
            sub_chunks = _split_long_chunk(section_text, CHUNK_TARGET_TOKENS)
            raw_chunks.extend((heading, sc) for sc in sub_chunks)
        else:
            raw_chunks.append((heading, section_text))

    # Apply overlap
    texts = [t for _, t in raw_chunks]
    overlap_chars = CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN
    overlapped = _add_overlap(texts, overlap_chars)

    # Prepend frontmatter title/description to first chunk if present
    title = frontmatter.get("title", "")
    description = frontmatter.get("description", "")
    prefix = ""
    if title:
        prefix = f"# {title}\n"
    if description:
        prefix += f"{description}\n\n"

    chunks: list[DocumentChunk] = []
    for i, chunk_text in enumerate(overlapped):
        heading = raw_chunks[i][0] if i < len(raw_chunks) else ""
        full_text = f"{prefix}{chunk_text}" if i == 0 and prefix else chunk_text
        source_hash = hashlib.md5(f"{source}:{i}".encode(), usedforsecurity=False).hexdigest()[:12]
        chunks.append(
            DocumentChunk(
                text=full_text,
                source=source,
                heading=heading,
                content_class=content_class,
                chunk_id=f"{content_class}-{source_hash}",
            )
        )

    return chunks


# ---------------------------------------------------------------------------
# File hashing and change detection
# ---------------------------------------------------------------------------


def _hash_file(path: Path) -> str:
    """Compute SHA-256 hash of a file.

    Args:
        path: Path to the file.

    Returns:
        Hex digest of the file's SHA-256 hash.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def _scan_directory(directory: Path) -> dict[str, str]:
    """Scan a directory and compute hashes for all ingestable files.

    Args:
        directory: Directory to scan.

    Returns:
        Dict mapping relative file paths to their SHA-256 hashes.
    """
    if not directory.exists():
        return {}

    hashes: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in _INGESTABLE_EXTENSIONS:
            rel = str(path.relative_to(directory))
            hashes[rel] = _hash_file(path)
    return hashes


def _load_manifest() -> _FileManifest:
    """Load the index manifest from disk.

    Returns:
        FileManifest with stored hashes, or empty manifest if not found.
    """
    if not _MANIFEST_PATH.exists():
        return _FileManifest()

    try:
        data = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return _FileManifest()
        return _FileManifest(
            product_hashes=data.get("product_hashes", {}),
            user_hashes=data.get("user_hashes", {}),
        )
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load index manifest: %s", exc)
        return _FileManifest()


def _save_manifest(manifest: _FileManifest) -> None:
    """Save the index manifest to disk.

    Args:
        manifest: Manifest to persist.
    """
    _MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "product_hashes": manifest.product_hashes,
        "user_hashes": manifest.user_hashes,
    }
    _MANIFEST_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Product docs path resolution
# ---------------------------------------------------------------------------


def _resolve_product_docs_dir() -> Path:
    """Resolve the path to the product documentation directory.

    Tries the development layout first (relative to this file), then
    falls back to importlib.resources for installed packages.

    Returns:
        Path to the documentation directory.

    Raises:
        FileNotFoundError: If the documentation directory cannot be found.
    """
    # Development layout: src/q_ai/assist/knowledge.py -> documentation/
    dev_path = Path(__file__).resolve().parent.parent.parent.parent / "documentation"
    if dev_path.is_dir():
        return dev_path

    # Installed package: try importlib.resources
    try:
        from importlib import resources

        pkg_path = resources.files("q_ai").joinpath("../../documentation")
        if hasattr(pkg_path, "__fspath__"):
            resolved = Path(str(pkg_path))
            if resolved.is_dir():
                return resolved
    except Exception as exc:
        logger.debug("importlib.resources fallback failed: %s", exc)

    raise FileNotFoundError(
        "Cannot find product documentation directory. Expected at: " + str(dev_path)
    )


def _resolve_extra_product_files() -> list[Path]:
    """Resolve paths to extra product files (README, Architecture, etc.).

    Returns:
        List of existing extra product file paths.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    candidates = [
        repo_root / "README.md",
        repo_root / "docs" / "Architecture.md",
        repo_root / "docs" / "Roadmap.md",
        Path(__file__).resolve().parent.parent / "core" / "data" / "frameworks.yaml",
    ]
    return [p for p in candidates if p.is_file()]


# ---------------------------------------------------------------------------
# Knowledge Base Manager
# ---------------------------------------------------------------------------


class KnowledgeBase:
    """Manages document ingestion, storage, and retrieval for the assistant.

    Uses ChromaDB persistent client for vector storage and sentence-transformers
    for embedding. Supports two collections: product_knowledge and user_knowledge.

    Args:
        embedding_model: Sentence-transformers model name.
        knowledge_dir: Path to user knowledge directory.
    """

    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        knowledge_dir: Path | None = None,
    ) -> None:
        self._embedding_model = embedding_model
        self._knowledge_dir = knowledge_dir or (Path.home() / ".qai" / "knowledge")
        self._chroma_dir = _CHROMA_DIR
        self._chroma_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(self._chroma_dir))
        self._embedder = get_embedder(embedding_model)

    def _get_or_create_collection(self, name: str) -> chromadb.Collection:
        """Get or create a ChromaDB collection.

        Args:
            name: Collection name.

        Returns:
            ChromaDB Collection instance.
        """
        return self._client.get_or_create_collection(
            name=name,
            embedding_function=None,
        )

    def ensure_indexed(self, force: bool = False) -> None:
        """Ensure the knowledge base is up to date.

        Checks file hashes against the stored manifest and reindexes
        only changed collections. On first run, indexes everything.

        Args:
            force: If True, reindex everything regardless of changes.
        """
        manifest = _load_manifest()

        product_changed = self._check_and_index_product(manifest, force=force)
        user_changed = self._check_and_index_user(manifest, force=force)

        if product_changed or user_changed:
            _save_manifest(manifest)

    def _check_and_index_product(self, manifest: _FileManifest, force: bool = False) -> bool:
        """Check and reindex product documentation if changed.

        Args:
            manifest: Current file manifest (mutated on reindex).
            force: Force reindex regardless of changes.

        Returns:
            True if reindexing occurred.
        """
        try:
            docs_dir = _resolve_product_docs_dir()
        except FileNotFoundError as exc:
            logger.warning("Product docs not found: %s", exc)
            return False

        current_hashes = _scan_directory(docs_dir)

        # Add extra product files
        extra_files = _resolve_extra_product_files()
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        for ef in extra_files:
            try:
                rel = str(ef.relative_to(repo_root))
            except ValueError:
                rel = ef.name
            current_hashes[f"__extra__/{rel}"] = _hash_file(ef)

        if not force and current_hashes == manifest.product_hashes:
            logger.debug("Product knowledge is up to date, skipping reindex.")
            return False

        logger.info("Reindexing product knowledge (%d files)...", len(current_hashes))
        chunks = self._ingest_product_docs(docs_dir, extra_files)
        self._store_chunks(PRODUCT_COLLECTION, chunks)
        manifest.product_hashes = current_hashes
        return True

    def _check_and_index_user(self, manifest: _FileManifest, force: bool = False) -> bool:
        """Check and reindex user knowledge if changed.

        Args:
            manifest: Current file manifest (mutated on reindex).
            force: Force reindex regardless of changes.

        Returns:
            True if reindexing occurred.
        """
        if not self._knowledge_dir.exists():
            if manifest.user_hashes:
                # Directory was removed — clear the collection
                self._store_chunks(USER_COLLECTION, [])
                manifest.user_hashes = {}
                return True
            return False

        current_hashes = _scan_directory(self._knowledge_dir)

        if not force and current_hashes == manifest.user_hashes:
            logger.debug("User knowledge is up to date, skipping reindex.")
            return False

        logger.info("Reindexing user knowledge (%d files)...", len(current_hashes))
        chunks = self._ingest_directory(self._knowledge_dir, "user")
        self._store_chunks(USER_COLLECTION, chunks)
        manifest.user_hashes = current_hashes
        return True

    def _ingest_product_docs(self, docs_dir: Path, extra_files: list[Path]) -> list[DocumentChunk]:
        """Ingest all product documentation files.

        Args:
            docs_dir: Path to the documentation/ directory.
            extra_files: Additional product files (README, etc.).

        Returns:
            List of document chunks.
        """
        chunks = self._ingest_directory(docs_dir, "product")

        # Ingest extra product files
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        for ef in extra_files:
            try:
                rel = str(ef.relative_to(repo_root))
            except ValueError:
                rel = ef.name
            file_chunks = self._ingest_file(ef, f"__extra__/{rel}", "product")
            chunks.extend(file_chunks)

        return chunks

    def _ingest_directory(self, directory: Path, content_class: str) -> list[DocumentChunk]:
        """Ingest all supported files from a directory.

        Args:
            directory: Directory to scan.
            content_class: "product" or "user".

        Returns:
            List of document chunks.
        """
        chunks: list[DocumentChunk] = []
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in _INGESTABLE_EXTENSIONS:
                continue
            rel = str(path.relative_to(directory))
            try:
                file_chunks = self._ingest_file(path, rel, content_class)
                chunks.extend(file_chunks)
            except Exception as exc:
                logger.warning("Failed to ingest %s: %s", rel, exc)
        return chunks

    def _ingest_file(self, path: Path, source: str, content_class: str) -> list[DocumentChunk]:
        """Ingest a single file into document chunks.

        Args:
            path: Absolute path to the file.
            source: Relative source path for metadata.
            content_class: "product" or "user".

        Returns:
            List of document chunks.
        """
        text = path.read_text(encoding="utf-8", errors="replace")

        # Strip JSX from MDX files
        if path.suffix.lower() == ".mdx":
            text = strip_mdx_jsx(text)

        return chunk_document(text, source, content_class)

    def _store_chunks(self, collection_name: str, chunks: list[DocumentChunk]) -> None:
        """Replace a collection's contents with new chunks.

        Deletes the existing collection and creates a new one with the
        provided chunks.

        Args:
            collection_name: Name of the ChromaDB collection.
            chunks: Document chunks to store.
        """
        # Delete and recreate to ensure clean state
        with contextlib.suppress(Exception):
            self._client.delete_collection(name=collection_name)

        collection = self._client.create_collection(
            name=collection_name,
            embedding_function=None,
        )

        if not chunks:
            return

        # Batch embed and store
        batch_size = 100
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [c.text for c in batch]
            ids = [c.chunk_id for c in batch]
            metadatas = [
                {
                    "source": c.source,
                    "heading": c.heading,
                    "content_class": c.content_class,
                }
                for c in batch
            ]

            embeddings = self._embedder.encode(texts)
            collection.add(
                ids=ids,
                embeddings=embeddings,  # type: ignore[arg-type]
                documents=texts,
                metadatas=metadatas,  # type: ignore[arg-type]
            )

        logger.info("Stored %d chunks in collection '%s'.", len(chunks), collection_name)

    def retrieve(self, query: str, max_chunks: int = 20) -> list[RetrievalResult]:
        """Retrieve relevant chunks from both collections.

        Embeds the query and searches both product and user knowledge
        collections, returning results sorted by relevance.

        Args:
            query: User query text.
            max_chunks: Maximum total chunks to return.

        Returns:
            List of RetrievalResult objects sorted by distance.
        """
        query_embedding = self._embedder.encode([query])[0]
        results: list[RetrievalResult] = []

        # Allocate roughly 2/3 to product, 1/3 to user
        product_k = max(1, (max_chunks * 2) // 3)
        user_k = max(1, max_chunks - product_k)

        for coll_name, top_k in [
            (PRODUCT_COLLECTION, product_k),
            (USER_COLLECTION, user_k),
        ]:
            try:
                collection = self._client.get_collection(
                    name=coll_name,
                    embedding_function=None,
                )
            except Exception:
                logger.debug("Collection '%s' not found, skipping.", coll_name)
                continue

            count = collection.count()
            if count == 0:
                continue

            query_results = collection.query(
                query_embeddings=[query_embedding],  # type: ignore[arg-type]
                n_results=min(top_k, count),
            )

            if not query_results["ids"] or not query_results["ids"][0]:
                continue

            doc_ids = query_results["ids"][0]
            documents = query_results["documents"][0] if query_results["documents"] else []
            distances = query_results["distances"][0] if query_results["distances"] else []
            metadatas = query_results["metadatas"][0] if query_results["metadatas"] else []

            for i, doc_id in enumerate(doc_ids):
                meta = metadatas[i] if i < len(metadatas) else {}
                chunk = DocumentChunk(
                    text=documents[i] if i < len(documents) else "",
                    source=str(meta.get("source", "")),
                    heading=str(meta.get("heading", "")),
                    content_class=str(meta.get("content_class", coll_name.split("_")[0])),
                    chunk_id=doc_id,
                )
                dist = float(distances[i]) if i < len(distances) else 0.0
                results.append(RetrievalResult(chunk=chunk, distance=dist))

        results.sort(key=lambda r: r.distance)
        return results[:max_chunks]
