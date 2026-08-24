"""Sanitized order lookup over data/orders.json."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Allowed customer-safe top-level fields and safety flags
ALLOWED_TOP_LEVEL_FIELDS = {
    "found",
    "order_id",
    "membership_tier",
    "items",
    "placed_at",
    "status",
    "status_updated_at",
    "shipped_at",
    "delivered_at",
    "carrier",
    "tracking_number",
    "estimated_delivery",
    "customer_safe_message",
    "requires_no_arrival_claim",
    "eta_unavailable",
    "needs_human_handoff",
}

ALLOWED_ITEM_FIELDS = {"name", "quantity", "final_sale"}

# In-memory orders store
_ORDERS_MAP: dict[str, dict[str, Any]] = {}
_SNAPSHOT_AT: str = ""
_DATASET_NAME: str = ""


def normalize_order_id(raw_id: Any) -> str:
    """Normalize an order ID string by trimming, uppercasing, and standardizing delimiters.

    Handles minor variations such as '  ord-1007 ', 'ord 1007', 'ORD_1007', 'ORD.1007', 'ORD1007'.
    Does not guess or fuzzy match substantially different IDs.
    """
    if not isinstance(raw_id, str):
        return ""

    cleaned = raw_id.strip().upper()
    if not cleaned:
        return ""

    # Standardize space, underscore, dot to hyphen
    cleaned = re.sub(r"[\s_.]+", "-", cleaned)
    cleaned = re.sub(r"-+", "-", cleaned)

    # Insert hyphen if missing between letters and numbers: 'ORD1007' -> 'ORD-1007'
    cleaned = re.sub(r"^([A-Z]+)(\d+)$", r"\1-\2", cleaned)

    return cleaned


def load_orders(data_path: str | Path | None = None) -> None:
    """Load orders.json into in-memory storage and capture snapshot timestamp."""
    global _ORDERS_MAP, _SNAPSHOT_AT, _DATASET_NAME

    if data_path is None:
        project_root = Path(__file__).resolve().parent.parent
        data_path = project_root / "data" / "orders.json"
    else:
        data_path = Path(data_path)

    if not data_path.exists() or not data_path.is_file():
        raise FileNotFoundError(f"Orders dataset file not found: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    _SNAPSHOT_AT = data.get("snapshot_at", "")
    _DATASET_NAME = data.get("dataset_name", "")

    orders_list = data.get("orders", [])
    orders_map: dict[str, dict[str, Any]] = {}

    for order in orders_list:
        raw_id = order.get("order_id")
        if raw_id:
            normalized_key = normalize_order_id(raw_id)
            orders_map[normalized_key] = order

    _ORDERS_MAP = orders_map


def get_snapshot_at() -> str:
    """Get the authoritative snapshot_at timestamp representing 'now' for the dataset."""
    return _SNAPSHOT_AT


def lookup_order(order_id: str) -> dict[str, Any]:
    """Look up an order by ID and return a sanitized, customer-safe dictionary.

    Ensures that PII (customer name, email, shipping address) and internal fields
    (risk score, warehouse notes, internal support tags, item SKUs) are never exposed.

    Enforces status precedence and attaches deterministic safety flags:
    - cancelled / returned: estimated_delivery is set to None, requires_no_arrival_claim = True
    - shipped with null ETA: eta_unavailable = True, estimated_delivery is None
    - exception: needs_human_handoff = True
    """
    normalized_id = normalize_order_id(order_id)
    if not normalized_id or normalized_id not in _ORDERS_MAP:
        return {
            "found": False,
            "error": f"Order '{order_id}' not found.",
            "order_id": normalized_id if normalized_id else str(order_id),
        }

    raw_order = _ORDERS_MAP[normalized_id]
    status = str(raw_order.get("status", "")).lower()

    # Sanitize items list - only include customer-safe fields
    sanitized_items: list[dict[str, Any]] = []
    for item in raw_order.get("items", []):
        sanitized_items.append({
            "name": item.get("name"),
            "quantity": item.get("quantity"),
            "final_sale": bool(item.get("final_sale", False)),
        })

    sanitized: dict[str, Any] = {
        "found": True,
        "order_id": raw_order.get("order_id", normalized_id),
        "membership_tier": raw_order.get("membership_tier"),
        "items": sanitized_items,
        "placed_at": raw_order.get("placed_at"),
        "status": status,
        "status_updated_at": raw_order.get("status_updated_at"),
        "shipped_at": raw_order.get("shipped_at"),
        "delivered_at": raw_order.get("delivered_at"),
        "carrier": raw_order.get("carrier"),
        "tracking_number": raw_order.get("tracking_number"),
        "estimated_delivery": raw_order.get("estimated_delivery"),
        "customer_safe_message": raw_order.get("customer_safe_message"),
    }

    # Status precedence & deterministic safety flags
    if status in ("cancelled", "returned"):
        # Suppress stale carrier ETA on cancelled/returned orders
        sanitized["estimated_delivery"] = None
        sanitized["requires_no_arrival_claim"] = True
    elif status == "shipped" and raw_order.get("estimated_delivery") is None:
        sanitized["estimated_delivery"] = None
        sanitized["eta_unavailable"] = True
    elif status == "exception":
        sanitized["needs_human_handoff"] = True

    return sanitized


# Load dataset at module startup
load_orders()
