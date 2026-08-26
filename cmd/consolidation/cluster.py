"""Cluster algorithm v2 (R7) + 4-zone decision engine (R2).

Union-find clustering at the 0.88 edge threshold (fixes Scanner v1's 0.82
transitive-closure explosion), fringe-pair extraction for the 0.82-0.88 LLM
band, oversized-cluster splitting, then per-cluster zone classification with
recency / type / cross-type / temporal-evolution guards.
"""

from typing import List, Optional

import numpy as np

from .config import Config
from .types import (
    Cluster,
    MemoryPoint,
    Pair,
    ZONE_AUTO_MERGE,
    ZONE_DEDUP_ANOMALY,
    ZONE_DISTINCT,
    ZONE_LLM_ADJUDICATE,
    DS_AUTO_MERGE,
    DS_DEDUP_ANOMALY,
    DS_LLM_ADJUDICATE,
    DS_SKIP,
)


# ---------------------------------------------------------------------------
# Native union-find (networkx is not installed)
# ---------------------------------------------------------------------------
class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# ---------------------------------------------------------------------------
# Zone band (pure numeric classification by max cosine similarity)
# ---------------------------------------------------------------------------
def zone_band(sim: float, cfg: Config) -> str:
    if sim >= cfg.dedup_anomaly_threshold:
        return ZONE_DEDUP_ANOMALY
    if sim >= cfg.auto_merge_threshold:
        return ZONE_AUTO_MERGE
    if sim >= cfg.llm_adjudicate_threshold:
        return ZONE_LLM_ADJUDICATE
    return ZONE_DISTINCT


def is_temporal_evolution(members: List[MemoryPoint], cfg: Config) -> bool:
    """Heuristic: 3+ members spanning > 7 days => same-topic time evolution."""
    if len(members) < 3:
        return False
    cats = [m.created_at for m in members if m.created_at]
    if len(cats) < 2:
        return False
    span_seconds = max(cats) - min(cats)
    return span_seconds > 7 * 86400


# ---------------------------------------------------------------------------
# Cluster building (v2)
# ---------------------------------------------------------------------------
def _edges_above(sim, points, threshold_lo, threshold_hi):
    """Upper-triangle SAME-COLLECTION edges with threshold_lo <= s < threshold_hi.

    threshold_hi = None means +inf (>= threshold_lo).
    """
    n = len(points)
    iu, ju = np.triu_indices(n, k=1)
    s = sim[iu, ju]
    if threshold_hi is None:
        mask = s >= threshold_lo
    else:
        mask = (s >= threshold_lo) & (s < threshold_hi)
    edges = []
    for a, b, sv in zip(iu[mask].tolist(), ju[mask].tolist(), s[mask].tolist()):
        if points[a].collection == points[b].collection:  # within-collection only
            edges.append((a, b, float(sv)))
    return edges


def _components_at(members: List[int], edges, threshold: float):
    """Connected components among `members` using edges with sim >= threshold.

    Returns list of member-index lists (singletons dropped)."""
    idx_of = {m: i for i, m in enumerate(members)}
    uf = UnionFind(len(members))
    for a, b, s in edges:
        if s >= threshold and a in idx_of and b in idx_of:
            uf.union(idx_of[a], idx_of[b])
    groups = {}
    for m in members:
        root = uf.find(idx_of[m])
        groups.setdefault(root, []).append(m)
    return [sorted(g) for g in groups.values() if len(g) > 1]


def _split_oversized(members: List[int], edges, max_size: int) -> List[List[int]]:
    """Split a > max_size cluster by raising the internal similarity threshold
    (removing the lowest-similarity 'max gap' edges first)."""
    if len(members) <= max_size:
        return [members]
    sims = sorted({s for _, _, s in edges}, reverse=True)  # high -> low
    last_good: Optional[List[List[int]]] = None
    for t in sims:
        comps = _components_at(members, edges, t)
        if comps and all(len(c) <= max_size for c in comps):
            last_good = comps
        elif last_good is not None:
            break
    if last_good is not None:
        return last_good
    # Fallback: even the tightest edges leave an oversized blob -> hard chunk.
    return [members[i:i + max_size] for i in range(0, len(members), max_size)]


def build_clusters(points: List[MemoryPoint], sim, cfg: Config):
    """Return (clusters, fringe_pairs).

    clusters: tight connected components at >= cluster_edge_threshold, each
              split to <= max_cluster_size.
    fringe_pairs: individual [llm, edge) pairs (each a 2-member Cluster) for
                  LLM adjudication.
    """
    n = len(points)
    if n == 0 or sim is None:
        return [], []

    edge_lo = cfg.cluster_edge_threshold
    tight_edges = _edges_above(sim, points, edge_lo, None)

    uf = UnionFind(n)
    for a, b, _ in tight_edges:
        uf.union(a, b)

    groups = {}
    for a, b, _ in tight_edges:
        for node in (a, b):
            groups.setdefault(uf.find(node), set()).add(node)

    clusters: List[Cluster] = []
    tight_membership = set()
    for node_set in groups.values():
        members = sorted(node_set)
        member_pos = set(members)
        internal = [(a, b, s) for (a, b, s) in tight_edges
                    if a in member_pos and b in member_pos]
        for sub in _split_oversized(members, internal, cfg.max_cluster_size):
            sub_pos = set(sub)
            sub_edges = [Pair(a, b, s) for (a, b, s) in internal
                         if a in sub_pos and b in sub_pos]
            max_sim = max((p.similarity for p in sub_edges), default=0.0)
            clusters.append(Cluster(members_idx=sub, pairs=sub_edges, max_similarity=max_sim))
            tight_membership.update(sub)

    # Fringe pairs: [llm, edge) same-collection pairs, treated individually.
    fringe_edges = _edges_above(sim, points, cfg.llm_adjudicate_threshold, edge_lo)
    fringe_pairs: List[Cluster] = []
    for a, b, s in fringe_edges:
        fringe_pairs.append(Cluster(
            members_idx=[a, b],
            pairs=[Pair(a, b, s)],
            max_similarity=s,
            is_fringe_pair=True,
        ))

    clusters.sort(key=lambda c: c.max_similarity, reverse=True)
    fringe_pairs.sort(key=lambda c: c.max_similarity, reverse=True)
    return clusters, fringe_pairs


# ---------------------------------------------------------------------------
# 4-zone decision engine + guards (R2)
# ---------------------------------------------------------------------------
def classify_cluster(cluster: Cluster, points: List[MemoryPoint], cfg: Config, now: float) -> None:
    """Set cluster.zone / decision_source / skip_reason / guard in place.

    Guard precedence (spec §2.1): recency -> identity -> directive ->
    temporal-evolution -> cross-type -> raw band.
    """
    members = cluster.members(points)
    max_sim = cluster.max_similarity
    band = zone_band(max_sim, cfg)
    cluster.zone = band

    # Recency guard: skip if ANY member < recency_guard_hours old.
    recency_seconds = cfg.recency_guard_hours * 3600
    if any(m.created_at and (now - m.created_at) < recency_seconds for m in members):
        cluster.decision_source = DS_SKIP
        cluster.skip_reason = "recency_guard"
        cluster.guard = "recency"
        return

    if band == ZONE_DISTINCT:
        cluster.decision_source = DS_SKIP
        cluster.skip_reason = "distinct"
        return

    types = {m.type for m in members}
    has_identity = "identity" in types
    has_directive = "directive" in types
    cross_type = len(types) > 1

    # Identity guard: never auto-merge. Only route to LLM in the dedup-anomaly
    # band (full-duplicate check, LLM confidence must clear the bar downstream).
    if has_identity:
        cluster.guard = "identity"
        if band == ZONE_DEDUP_ANOMALY and not cfg.identity_merge_enabled:
            cluster.decision_source = DS_LLM_ADJUDICATE
        elif band == ZONE_DEDUP_ANOMALY and cfg.identity_merge_enabled:
            cluster.decision_source = DS_LLM_ADJUDICATE
        else:
            cluster.decision_source = DS_SKIP
            cluster.skip_reason = "identity_protected"
        return

    # Directive guard: only mergeable at >= directive_merge_threshold, and
    # always via LLM (never auto-merge).
    if has_directive:
        cluster.guard = "directive"
        if max_sim >= cfg.directive_merge_threshold:
            cluster.decision_source = DS_LLM_ADJUDICATE
        else:
            cluster.decision_source = DS_SKIP
            cluster.skip_reason = "directive_below_threshold"
        return

    # Temporal-evolution guard: downgrade to LLM (never lose an evolution chain).
    if is_temporal_evolution(members, cfg):
        cluster.guard = "temporal"
        cluster.decision_source = DS_LLM_ADJUDICATE
        return

    # Cross-type guard: different types never auto-merge -> LLM.
    if cross_type:
        cluster.guard = "cross_type"
        cluster.decision_source = DS_LLM_ADJUDICATE
        return

    # Normal same-type band resolution.
    if band == ZONE_AUTO_MERGE:
        cluster.decision_source = DS_AUTO_MERGE
    elif band == ZONE_LLM_ADJUDICATE:
        cluster.decision_source = DS_LLM_ADJUDICATE
    else:  # ZONE_DEDUP_ANOMALY
        cluster.decision_source = DS_DEDUP_ANOMALY


def classify_all(clusters: List[Cluster], points: List[MemoryPoint], cfg: Config, now: float) -> None:
    for c in clusters:
        classify_cluster(c, points, cfg, now)
