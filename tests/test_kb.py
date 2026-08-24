"""Tests for the KnowledgeBase module (app.kb)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.kb import (
    EXPECTED_FRONTMATTER_FIELDS,
    KnowledgeBase,
    chunk_markdown_document,
    compute_metadata_adjustment,
    parse_frontmatter,
    retrieve,
)


@pytest.fixture(scope="module")
def kb() -> KnowledgeBase:
    """Fixture providing a loaded KnowledgeBase instance over the repository documents."""
    return KnowledgeBase()


def test_all_14_markdown_files_loaded(kb: KnowledgeBase):
    """Verify that all 14 knowledge-base Markdown files load successfully."""
    assert len(kb.documents) == 14, f"Expected 14 documents, got {len(kb.documents)}"
    assert len(kb.chunks) > 14, "Expected multiple chunks across documents"
    assert kb.embeddings.shape[0] == len(kb.chunks)
    assert kb.embeddings.shape[1] == 384  # all-MiniLM-L6-v2 embedding dimension


def test_frontmatter_fields_parsed(kb: KnowledgeBase):
    """Verify that expected frontmatter fields are correctly parsed from the documents."""
    for doc in kb.documents:
        meta = doc["metadata"]
        assert "document_id" in meta, f"Missing document_id in {doc['filename']}"
        assert "title" in meta, f"Missing title in {doc['filename']}"
        assert "status" in meta, f"Missing status in {doc['filename']}"
        assert "effective_date" in meta, f"Missing effective_date in {doc['filename']}"
        assert "last_reviewed" in meta, f"Missing last_reviewed in {doc['filename']}"
        assert "audience" in meta, f"Missing audience in {doc['filename']}"
        assert "policy_authority" in meta, f"Missing policy_authority in {doc['filename']}"

    # Verify specific metadata values for key documents
    docs_by_name = {d["filename"]: d["metadata"] for d in kb.documents}

    ret_current = docs_by_name["01-returns-policy-current.md"]
    assert ret_current["document_id"] == "RET-2026-01"
    assert ret_current["status"] == "active"
    assert ret_current["policy_authority"] == "official"
    assert ret_current["supersedes"] == "RET-2024-01"

    ret_legacy = docs_by_name["02-returns-policy-legacy.md"]
    assert ret_legacy["document_id"] == "RET-2024-01"
    assert ret_legacy["status"] == "superseded"
    assert ret_legacy["superseded_by"] == "RET-2026-01"

    migration = docs_by_name["14-internal-content-migration-notes.md"]
    assert migration["status"] == "draft"
    assert migration["policy_authority"] == "none"
    assert migration["customer_answering"] is False


def test_ranking_active_current_above_superseded_legacy(kb: KnowledgeBase):
    """Verify that current returns policy ranks above legacy superseded policy."""
    query = "return window for a regular customer"
    results = kb.retrieve(query=query, top_k=10)

    # Find the top ranks for 01-returns-policy-current.md and 02-returns-policy-legacy.md
    current_rank = None
    legacy_rank = None

    for rank, res in enumerate(results):
        fn = res["metadata"]["filename"]
        if fn == "01-returns-policy-current.md" and current_rank is None:
            current_rank = rank
        elif fn == "02-returns-policy-legacy.md" and legacy_rank is None:
            legacy_rank = rank

    assert current_rank is not None, "01-returns-policy-current.md should appear in top results"
    if legacy_rank is not None:
        assert current_rank < legacy_rank, (
            f"01-returns-policy-current.md (rank {current_rank}) must rank above "
            f"02-returns-policy-legacy.md (rank {legacy_rank})"
        )


def test_retrieval_surfaces_both_tumbler_documents(kb: KnowledgeBase):
    """Verify that 'dishwasher safe tumbler' surfaces BOTH product care and product card."""
    query = "dishwasher safe tumbler"
    results = kb.retrieve(query=query, top_k=5)

    filenames = [r["metadata"]["filename"] for r in results]
    assert "11-product-care.md" in filenames, (
        f"Expected 11-product-care.md in top results for '{query}', got: {filenames}"
    )
    assert "12-breeze-tumbler-product-card.md" in filenames, (
        f"Expected 12-breeze-tumbler-product-card.md in top results for '{query}', got: {filenames}"
    )


def test_retrieval_result_structure_and_scores(kb: KnowledgeBase):
    """Verify that retrieval results contain all required metadata and scoring fields."""
    results = kb.retrieve(query="standard shipping time", top_k=3)
    assert len(results) == 3

    for item in results:
        # Check top-level result fields
        assert "text" in item and len(item["text"]) > 0
        assert "chunk" in item
        assert "heading" in item and len(item["heading"]) > 0
        assert "metadata" in item and isinstance(item["metadata"], dict)
        assert "similarity_score" in item and isinstance(item["similarity_score"], float)
        assert "final_ranking_score" in item and isinstance(item["final_ranking_score"], float)

        # Check metadata fields
        meta = item["metadata"]
        assert "document_id" in meta
        assert "title" in meta
        assert "status" in meta
        assert "policy_authority" in meta
        assert "heading" in meta
        assert "filename" in meta


def test_module_level_retrieve_convenience():
    """Verify that the module-level retrieve() function works seamlessly."""
    results = retrieve(query="warranty coverage on zippers", top_k=2)
    assert len(results) == 2
    assert results[0]["similarity_score"] > 0
    assert results[0]["final_ranking_score"] is not None


def test_metadata_adjustment_logic():
    """Verify deterministic scoring adjustments for various statuses and authorities."""
    active_official = {"status": "active", "policy_authority": "official"}
    assert compute_metadata_adjustment(active_official) == pytest.approx(0.05)

    superseded = {"status": "superseded", "policy_authority": "official"}
    assert compute_metadata_adjustment(superseded) == pytest.approx(-0.15)

    draft_none = {
        "status": "draft",
        "policy_authority": "none",
        "customer_answering": False,
    }
    # -0.20 (draft) + -0.20 (none) + -0.15 (customer_answering=False) = -0.55
    assert compute_metadata_adjustment(draft_none) == pytest.approx(-0.55)


def test_frontmatter_parser_errors():
    """Verify frontmatter parser raises clear errors for malformed content."""
    # Missing opening delimiter
    with pytest.raises(ValueError, match="Missing opening '---'"):
        parse_frontmatter("# Document without frontmatter\nBody")

    # Unclosed delimiter
    with pytest.raises(ValueError, match="Frontmatter not closed"):
        parse_frontmatter("---\ntitle: Test\nbody without close")

    # Missing colon
    with pytest.raises(ValueError, match="expected 'key: value'"):
        parse_frontmatter("---\ntitle Test\n---\nBody")


def test_kb_nonexistent_directory():
    """Verify KnowledgeBase raises FileNotFoundError for non-existent path."""
    with pytest.raises(FileNotFoundError):
        KnowledgeBase(kb_dir="non_existent_directory_xyz")


def test_kb_empty_directory():
    """Verify KnowledgeBase raises ValueError for directory without markdown files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="No markdown documents found"):
            KnowledgeBase(kb_dir=tmpdir)
