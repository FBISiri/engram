"""Scan stage (R1): Qdrant scroll + pairwise cosine similarity.

Upgrades the read-only patterns from cmd/consolidation_scanner/main.py:
returns typed MemoryPoint objects and a normalized similarity matrix.
READ-ONLY against Qdrant.
"""

from typing import List, Optional, Tuple

import numpy as np
import requests

from .types import MemoryPoint


def qdrant_scroll_all(qdrant_url: str, name: str, timeout: int = 30) -> List[dict]:
    """Scroll all points from a collection (vectors + payloads)."""
    all_points: List[dict] = []
    offset = None
    while True:
        body = {"limit": 100, "with_payload": True, "with_vector": True}
        if offset:
            body["offset"] = offset
        r = requests.post(
            qdrant_url + "/collections/" + name + "/points/scroll",
            json=body,
            timeout=timeout,
        )
        if r.status_code == 404:
            break  # collection doesn't exist
        r.raise_for_status()
        data = r.json()["result"]
        all_points.extend(data["points"])
        offset = data.get("next_page_offset")
        if offset is None:
            break
    return all_points


def qdrant_healthy(qdrant_url: str, timeout: int = 5) -> bool:
    """Qdrant REST health gate."""
    try:
        r = requests.get(qdrant_url + "/healthz", timeout=timeout)
        if r.status_code == 200:
            return True
        # older Qdrant exposes root collections list instead of /healthz
        r = requests.get(qdrant_url + "/collections", timeout=timeout)
        return r.status_code == 200
    except requests.RequestException:
        return False


def collect_points(qdrant_url: str, collections: List[str]) -> List[MemoryPoint]:
    """Return all ACTIVE memory points across the given collections.

    Skips points without a plain vector and soft-deleted (archived/deprecated)
    memories — the Engram REST delete is a soft delete, so without this filter a
    later run would re-cluster memories a previous run already merged away.
    """
    points: List[MemoryPoint] = []
    for coll in collections:
        for p in qdrant_scroll_all(qdrant_url, coll):
            vec = p.get("vector")
            if not isinstance(vec, list):
                continue  # skip points without a plain (single) vector
            payload = p.get("payload", {}) or {}
            if payload.get("lifecycle_status") in ("archived", "deprecated"):
                continue
            points.append(MemoryPoint.from_qdrant_point(coll, p))
    return points


def similarity_matrix(points: List[MemoryPoint]) -> Optional[np.ndarray]:
    """Cosine similarity via normalized dot product. None if no points."""
    if not points:
        return None
    V = np.asarray([p.vector for p in points], dtype=np.float64)
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    V = V / norms
    return V @ V.T


def scan(qdrant_url: str, collections: List[str]) -> Tuple[List[MemoryPoint], Optional[np.ndarray]]:
    points = collect_points(qdrant_url, collections)
    return points, similarity_matrix(points)
