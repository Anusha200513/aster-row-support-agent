"""Knowledge-base loading, frontmatter parsing, chunking, embeddings, retrieval."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

# Expected frontmatter fields to preserve and validate
EXPECTED_FRONTMATTER_FIELDS = [
    "document_id",
    "title",
    "status",
    "effective_date",
    "last_reviewed",
    "audience",
    "policy_authority",
    "supersedes",
    "superseded_by",
    "customer_answering",
]

# Shared embedding model cache
_MODEL_CACHE: SentenceTransformer | None = None
_DEFAULT_KB: KnowledgeBase | None = None


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load or retrieve the cached SentenceTransformer model instance."""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = SentenceTransformer(model_name)
    return _MODEL_CACHE


def parse_frontmatter(content: str, source_identifier: str = "") -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter enclosed in '---' at the start of a Markdown file.

    Returns:
        tuple[dict[str, Any], str]: (metadata dictionary, markdown body)

    Raises:
        ValueError: If frontmatter is missing, unclosed, or contains malformed lines.
    """
    if not content.startswith("---"):
        raise ValueError(
            f"Malformed document ({source_identifier}): Missing opening '---' frontmatter delimiter."
        )

    pattern = r"^---\r?\n(.*?)\r?\n---\r?\n?(.*)$"
    match = re.match(pattern, content, re.DOTALL)
    if not match:
        raise ValueError(
            f"Malformed document ({source_identifier}): Frontmatter not closed with '---' delimiter."
        )

    raw_frontmatter, body = match.group(1), match.group(2)
    metadata: dict[str, Any] = {}

    for line_num, line in enumerate(raw_frontmatter.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(
                f"Malformed frontmatter in {source_identifier} at line {line_num}: "
                f"expected 'key: value' format, got: {line!r}"
            )
        key, val = line.split(":", 1)
        key = key.strip()
        val = val.strip()

        # Type conversion for booleans, quotes, numbers, and nulls
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        elif val.lower() == "true":
            val = True
        elif val.lower() == "false":
            val = False
        elif val.lower() in ("null", "~"):
            val = None
        elif val == "":
            val = ""
        else:
            if re.match(r"^-?\d+$", val):
                val = int(val)
            elif re.match(r"^-?\d+\.\d+$", val):
                val = float(val)

        metadata[key] = val

    return metadata, body


def chunk_markdown_document(
    metadata: dict[str, Any],
    body: str,
    filename: str,
    file_path: str,
) -> list[dict[str, Any]]:
    """Split a Markdown document by level-2 headings (## sections) and attach metadata.

    Returns:
        list[dict[str, Any]]: List of chunk dicts containing text, embed_text, heading, and metadata.
    """
    sections = re.split(r"(?m)^##\s+(.+)$", body)
    chunks: list[dict[str, Any]] = []
    title = metadata.get("title", "")

    # re.split produces: [preamble, heading1, section1, heading2, section2, ...]
    for i in range(1, len(sections), 2):
        heading = sections[i].strip()
        section_text = sections[i + 1].strip() if (i + 1) < len(sections) else ""
        if not section_text:
            continue

        chunk_metadata = dict(metadata)
        chunk_metadata["heading"] = heading
        chunk_metadata["section_heading"] = heading
        chunk_metadata["filename"] = filename
        chunk_metadata["source"] = filename
        chunk_metadata["file_path"] = file_path

        chunk_full_text = f"## {heading}\n\n{section_text}"
        embed_text = f"{title} - {heading}\n\n{section_text}"

        chunks.append({
            "text": chunk_full_text,
            "embed_text": embed_text,
            "heading": heading,
            "metadata": chunk_metadata,
        })

    return chunks


def compute_metadata_adjustment(metadata: dict[str, Any]) -> float:
    """Compute a deterministic ranking adjustment based on status and authority metadata.

    Score adjustments:
    - Active official policies: +0.05 boost (favors current authoritative guidance)
    - Superseded policies: -0.15 penalty (deprioritized below active policies)
    - Draft documents: -0.20 penalty
    - Policy authority none: -0.20 penalty
    - customer_answering == False: -0.15 penalty
    """
    status = str(metadata.get("status", "")).lower()
    authority = str(metadata.get("policy_authority", "")).lower()
    customer_answering = metadata.get("customer_answering", None)

    adjustment = 0.0

    # Boost active official documents
    if status == "active" and authority == "official":
        adjustment += 0.05

    # Penalize superseded documents
    if status == "superseded":
        adjustment -= 0.15

    # Penalize draft documents
    if status == "draft":
        adjustment -= 0.20

    # Penalize non-authoritative documents
    if authority == "none":
        adjustment -= 0.20

    # Penalize documents excluded from customer answering
    if customer_answering is False:
        adjustment -= 0.15

    return adjustment


class KnowledgeBase:
    """In-memory vector search knowledge base over Markdown policy documents."""

    def __init__(
        self,
        kb_dir: str | Path | None = None,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        if kb_dir is None:
            # Default to 'knowledge-base' in the project root
            project_root = Path(__file__).resolve().parent.parent
            self.kb_dir = project_root / "knowledge-base"
        else:
            self.kb_dir = Path(kb_dir)

        if not self.kb_dir.exists() or not self.kb_dir.is_dir():
            raise FileNotFoundError(f"Knowledge-base directory does not exist: {self.kb_dir}")

        self.model = get_embedding_model(model_name)
        self.documents: list[dict[str, Any]] = []
        self.chunks: list[dict[str, Any]] = []
        self.embeddings: np.ndarray = np.empty((0, 384), dtype=np.float32)

        self._load_and_index()

    def _load_and_index(self) -> None:
        """Load all .md files, parse frontmatter, create chunks, and compute embeddings."""
        md_files = sorted(list(self.kb_dir.glob("*.md")))
        if not md_files:
            raise ValueError(f"No markdown documents found in knowledge-base directory: {self.kb_dir}")

        all_chunks: list[dict[str, Any]] = []
        loaded_docs: list[dict[str, Any]] = []

        for file_path in md_files:
            filename = file_path.name
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            metadata, body = parse_frontmatter(content, source_identifier=filename)
            loaded_docs.append({
                "filename": filename,
                "file_path": str(file_path),
                "metadata": metadata,
                "body": body,
            })

            doc_chunks = chunk_markdown_document(
                metadata=metadata,
                body=body,
                filename=filename,
                file_path=str(file_path),
            )
            all_chunks.extend(doc_chunks)

        if not all_chunks:
            raise ValueError(f"No sections found across documents in: {self.kb_dir}")

        self.documents = loaded_docs
        self.chunks = all_chunks

        # Compute normalized embeddings in memory
        embed_texts = [c["embed_text"] for c in all_chunks]
        self.embeddings = self.model.encode(
            embed_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve top_k most relevant chunks for a query using cosine similarity + metadata ranking.

        Returns:
            list[dict[str, Any]]: Top matching results with chunk text, metadata, similarity, and ranking scores.
        """
        if not query or not query.strip():
            return []

        # Embed query vector with unit normalization
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        # Cosine similarity is simply the dot product between normalized vectors
        similarity_scores = np.dot(self.embeddings, query_embedding)

        results: list[dict[str, Any]] = []
        for chunk, sim in zip(self.chunks, similarity_scores):
            sim_score = float(sim)
            metadata = chunk["metadata"]
            adjustment = compute_metadata_adjustment(metadata)
            final_score = float(sim_score + adjustment)

            results.append({
                "text": chunk["text"],
                "chunk": chunk["text"],
                "heading": chunk["heading"],
                "metadata": metadata,
                "similarity_score": sim_score,
                "final_ranking_score": final_score,
                "score": final_score,
                "source": metadata.get("source", ""),
                "filename": metadata.get("filename", ""),
            })

        # Rank by final adjusted score descending
        results.sort(key=lambda r: r["final_ranking_score"], reverse=True)
        return results[:top_k]


def get_default_kb() -> KnowledgeBase:
    """Get or create the singleton KnowledgeBase instance."""
    global _DEFAULT_KB
    if _DEFAULT_KB is None:
        _DEFAULT_KB = KnowledgeBase()
    return _DEFAULT_KB


def retrieve(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Retrieve top_k chunks from the default knowledge base."""
    return get_default_kb().retrieve(query=query, top_k=top_k)
