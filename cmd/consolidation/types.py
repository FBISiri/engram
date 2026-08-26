"""Shared data classes and constants for the consolidation agent."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Zone bands (classification by a cluster's max internal cosine similarity)
# ---------------------------------------------------------------------------
ZONE_DISTINCT = "distinct"            # < llm_adjudicate_threshold
ZONE_LLM_ADJUDICATE = "llm_adjudicate"  # [llm, auto_merge)
ZONE_AUTO_MERGE = "auto_merge"        # [auto_merge, dedup_anomaly)
ZONE_DEDUP_ANOMALY = "dedup_anomaly"  # >= dedup_anomaly_threshold

# ---------------------------------------------------------------------------
# Decision sources (what action a cluster/pair resolves to)
# ---------------------------------------------------------------------------
DS_AUTO_MERGE = "auto_merge"
DS_LLM_ADJUDICATE = "llm_adjudicate"
DS_DEDUP_ANOMALY = "dedup_anomaly"
DS_SKIP = "skip"

# ---------------------------------------------------------------------------
# LLM decisions
# ---------------------------------------------------------------------------
DECISION_MERGE = "MERGE"
DECISION_KEEP_SEPARATE = "KEEP_SEPARATE"
DECISION_PARTIAL_MERGE = "PARTIAL_MERGE"

# ---------------------------------------------------------------------------
# Trust hierarchy for source_type (highest trust first). Mirrors the Go
# sourceTypeTrust map (C2 source-type-aware dedup).
# ---------------------------------------------------------------------------
TRUST_HIERARCHY = [
    "user_input",
    "tool_output",
    "web_search",
    "document",
    "calendar",
    "reflection",
    "unknown",
]


def trust_rank(source_type: Optional[str]) -> int:
    """Return trust rank (0 = highest trust). Unknown/missing sorts last."""
    if source_type in TRUST_HIERARCHY:
        return TRUST_HIERARCHY.index(source_type)
    return len(TRUST_HIERARCHY)  # below 'unknown'


def highest_trust_source(source_types: List[Optional[str]]) -> str:
    """Return the highest-trust source_type among the inputs."""
    best = None
    best_rank = len(TRUST_HIERARCHY) + 1
    for st in source_types:
        r = trust_rank(st)
        if r < best_rank:
            best_rank = r
            best = st
    return best if best else "unknown"


@dataclass
class MemoryPoint:
    """A single memory point (id + vector + payload) from Qdrant."""
    id: str
    collection: str
    vector: List[float]
    type: str
    content: str
    importance: Optional[float]
    tags: List[str]
    source_type: Optional[str]
    created_at: Optional[float]  # UTC unix seconds
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_payload: Dict[str, Any] = field(default_factory=dict)  # original Qdrant payload (for undo)

    @property
    def provenance_history(self) -> List[Dict[str, Any]]:
        ph = self.metadata.get("provenance_history")
        return ph if isinstance(ph, list) else []

    @property
    def content_preview(self) -> str:
        return (self.content or "")[:200]

    @classmethod
    def from_qdrant_point(cls, collection: str, point: Dict[str, Any]) -> "MemoryPoint":
        payload = point.get("payload", {}) or {}
        metadata = payload.get("metadata", {}) or {}
        source_type = metadata.get("source_type") if isinstance(metadata, dict) else None
        return cls(
            id=str(point.get("id")),
            collection=collection,
            vector=point.get("vector") or [],
            type=payload.get("type", "") or "",
            content=payload.get("content", "") or "",
            importance=payload.get("importance"),
            tags=list(payload.get("tags", []) or []),
            source_type=source_type,
            created_at=payload.get("created_at"),
            metadata=metadata if isinstance(metadata, dict) else {},
            raw_payload=payload,
        )


@dataclass
class Pair:
    """An undirected similarity edge between two point indices."""
    a: int
    b: int
    similarity: float


@dataclass
class Cluster:
    """A connected component (or a single fringe pair) of memory points.

    members_idx / pairs index into the global points list passed alongside.
    """
    members_idx: List[int]
    pairs: List[Pair] = field(default_factory=list)
    max_similarity: float = 0.0
    is_fringe_pair: bool = False

    # Filled in by the decision engine (cluster.classify_cluster):
    zone: Optional[str] = None            # band label for reporting
    decision_source: Optional[str] = None  # DS_*
    skip_reason: Optional[str] = None
    guard: Optional[str] = None           # which guard fired (recency/identity/...)

    def members(self, points: List[MemoryPoint]) -> List[MemoryPoint]:
        return [points[i] for i in self.members_idx]
