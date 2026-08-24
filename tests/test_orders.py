"""Tests for the Orders module (app.orders)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.orders import (
    ALLOWED_ITEM_FIELDS,
    ALLOWED_TOP_LEVEL_FIELDS,
    get_snapshot_at,
    lookup_order,
    normalize_order_id,
)


def test_valid_lookup_ord_1007():
    """Verify valid lookup of ORD-1007 returns expected order and only allowed fields."""
    result = lookup_order("ORD-1007")

    assert result["found"] is True
    assert result["order_id"] == "ORD-1007"
    assert result["status"] == "shipped"
    assert result["carrier"] == "UPS"
    assert result["tracking_number"] == "1ZAR100700000007"
    assert result["estimated_delivery"] == "2026-08-22"
    assert result["membership_tier"] == "standard"

    # Verify top-level keys are strictly within the allowed set
    for key in result.keys():
        assert key in ALLOWED_TOP_LEVEL_FIELDS, f"Disallowed field '{key}' found in lookup result"

    # Verify items structure
    assert isinstance(result["items"], list)
    assert len(result["items"]) == 1
    item = result["items"][0]
    assert set(item.keys()) <= ALLOWED_ITEM_FIELDS
    assert item["name"] == "Atlas Weekender"
    assert item["quantity"] == 1
    assert item["final_sale"] is False


def test_order_id_normalization_whitespace_and_case():
    """Verify that leading/trailing whitespace and lowercase resolve correctly."""
    result_raw = lookup_order("ORD-1007")
    result_spaced = lookup_order("  ord-1007 ")
    assert result_spaced["found"] is True
    assert result_spaced["order_id"] == result_raw["order_id"]
    assert result_spaced["tracking_number"] == result_raw["tracking_number"]


def test_order_id_normalization_punctuation_and_spacing():
    """Verify that minor spacing and punctuation variations resolve correctly."""
    expected_id = "ORD-1007"
    test_variants = [
        "ord 1007",
        "ORD 1007",
        "ord_1007",
        "ORD_1007",
        "ord.1007",
        "ORD.1007",
        "ORD1007",
        "ord1007",
        " ORD - 1007 ",
    ]
    for variant in test_variants:
        res = lookup_order(variant)
        assert res["found"] is True, f"Failed to resolve variant '{variant}'"
        assert res["order_id"] == expected_id, f"Variant '{variant}' resolved to '{res.get('order_id')}'"


def test_unknown_order_not_found():
    """Verify that unknown order ID ORD-9999 returns a clean not-found result without guessing."""
    result = lookup_order("ORD-9999")
    assert result["found"] is False
    assert "error" in result
    assert "not found" in result["error"].lower()
    assert result["order_id"] == "ORD-9999"


def test_malformed_order_ids_handled_cleanly():
    """Verify that malformed or non-string inputs are handled cleanly without exceptions."""
    malformed_inputs = ["", "   ", "???", "INVALID-FORMAT-XYZ", "12345", "!@#$%"]
    for val in malformed_inputs:
        res = lookup_order(val)
        assert res["found"] is False
        assert "error" in res


def test_cancelled_order_ord_1004():
    """Verify ORD-1004 status precedence: cancelled suppresses estimated_delivery and sets safety flag."""
    result = lookup_order("ORD-1004")

    assert result["found"] is True
    assert result["order_id"] == "ORD-1004"
    assert result["status"] == "cancelled"
    assert result.get("requires_no_arrival_claim") is True
    # Stale estimated delivery from carrier must be suppressed
    assert result["estimated_delivery"] is None


def test_shipped_order_without_eta_ord_1011():
    """Verify ORD-1011: shipped with null ETA sets eta_unavailable = True without inventing ETA."""
    result = lookup_order("ORD-1011")

    assert result["found"] is True
    assert result["order_id"] == "ORD-1011"
    assert result["status"] == "shipped"
    assert result.get("eta_unavailable") is True
    assert result["estimated_delivery"] is None
    assert result["carrier"] == "Canada Post"
    assert result["tracking_number"] == "AR1011CA00001"


def test_exception_order_ord_1010():
    """Verify ORD-1010: exception status sets needs_human_handoff = True."""
    result = lookup_order("ORD-1010")

    assert result["found"] is True
    assert result["order_id"] == "ORD-1010"
    assert result["status"] == "exception"
    assert result.get("needs_human_handoff") is True


def test_pii_and_internal_fields_never_leak_across_all_orders():
    """Verify that customer PII and internal fields never appear in any lookup result."""
    project_root = Path(__file__).resolve().parent.parent
    orders_path = project_root / "data" / "orders.json"

    with open(orders_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    for raw_order in raw_data["orders"]:
        order_id = raw_order["order_id"]
        result = lookup_order(order_id)

        assert result["found"] is True, f"Failed lookup for {order_id}"

        # Assert absence of PII and internal dictionary structures
        assert "customer" not in result, f"'customer' object leaked in {order_id}"
        assert "internal" not in result, f"'internal' object leaked in {order_id}"

        # Assert absence of specific sensitive field names
        assert "name" not in result, f"Customer name leaked in root of {order_id}"
        assert "email" not in result, f"Customer email leaked in {order_id}"
        assert "shipping_address" not in result, f"Shipping address leaked in {order_id}"
        assert "risk_score" not in result, f"risk_score leaked in {order_id}"
        assert "warehouse_note" not in result, f"warehouse_note leaked in {order_id}"
        assert "support_tags" not in result, f"support_tags leaked in {order_id}"

        # Check all top-level keys against strict allowlist
        for k in result.keys():
            assert k in ALLOWED_TOP_LEVEL_FIELDS, f"Disallowed field '{k}' found in {order_id}"

        # Check item keys against item allowlist (e.g. sku must NOT leak)
        for item in result["items"]:
            for item_k in item.keys():
                assert item_k in ALLOWED_ITEM_FIELDS, f"Disallowed item field '{item_k}' in {order_id}"
            assert "sku" not in item, f"Item SKU leaked in {order_id}"


def test_status_precedence_on_returned_order_ord_1008():
    """Verify ORD-1008 status precedence: returned status suppresses ETA and sets safety flag."""
    result = lookup_order("ORD-1008")

    assert result["found"] is True
    assert result["order_id"] == "ORD-1008"
    assert result["status"] == "returned"
    assert result.get("requires_no_arrival_claim") is True
    assert result["estimated_delivery"] is None


def test_snapshot_at_loaded_and_authoritative():
    """Verify snapshot_at is loaded from orders.json and matches mock current time."""
    snapshot = get_snapshot_at()
    assert snapshot == "2026-08-15T12:00:00Z"
    assert isinstance(snapshot, str)
    assert len(snapshot) > 0
