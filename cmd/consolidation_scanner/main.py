#!/usr/bin/env python3
"""
consolidation_scanner/main.py -- Engram Consolidation Scanner CLI

Scans all Engram memories across collections, computes pairwise cosine
similarity, and reports connected-component clusters at >= threshold.

READ-ONLY: never mutates Qdrant. No LLM calls -- pure vector similarity.

Usage:
    python3 engram/cmd/consolidation_scanner/main.py \
        [--qdrant-url URL] [--collections c1,c2] [--threshold 0.82]
"""
import argparse
import datetime
import json
import os
from pathlib import Path

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Config / paths
# ---------------------------------------------------------------------------
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
DEFAULT_COLLECTIONS = ["engram_user", "engram_reflection", "engram_pigo"]
DEFAULT_THRESHOLD = 0.82

JSON_REPORT = Path("/data/armyoftheagent/engram/eval/reports/consolidation_scanner_report.json")
MD_REPORT = Path("/data/obsidian-vault/Research/consolidation-scanner-report-2026-08-04.md")

# Zone thresholds (classification by a cluster's max internal similarity)
ZONE_AUTO_MERGE = 0.88
ZONE_DEDUP_ANOMALY = 0.92


# ---------------------------------------------------------------------------
# Qdrant access (read-only) -- mirrors eval/harness/harness.py qdrant_scroll_all
# ---------------------------------------------------------------------------
def qdrant_scroll_all(qdrant_url, name):
    """Scroll all points from a collection. Returns list of point dicts."""
    all_points = []
    offset = None
    while True:
        body = {"limit": 100, "with_payload": True, "with_vector": True}
        if offset:
            body["offset"] = offset
        r = requests.post(
            qdrant_url + "/collections/" + name + "/points/scroll",
            json=body,
            timeout=30,
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


# ---------------------------------------------------------------------------
# Native union-find (networkx is NOT installed)
# ---------------------------------------------------------------------------
class UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int64)

    def find(self, x):
        # iterative with path compression
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def zone_for(max_sim):
    if max_sim >= ZONE_DEDUP_ANOMALY:
        return "dedup_anomaly"
    if max_sim >= ZONE_AUTO_MERGE:
        return "auto_merge"
    return "llm_adjudicate"


# ---------------------------------------------------------------------------
# Core scan
# ---------------------------------------------------------------------------
def collect_points(qdrant_url, collections):
    """Return parallel lists: metas (dict per point) and vectors (list of lists)."""
    metas = []
    vectors = []
    for coll in collections:
        pts = qdrant_scroll_all(qdrant_url, coll)
        for p in pts:
            vec = p.get("vector")
            if not isinstance(vec, list):
                continue  # skip points without a plain vector
            payload = p.get("payload", {}) or {}
            content = payload.get("content", "") or ""
            metas.append({
                "id": str(p.get("id")),
                "collection": coll,
                "type": payload.get("type", ""),
                "content_preview": content[:200],
                "importance": payload.get("importance"),
                "tags": payload.get("tags", []),
                "created_at": payload.get("created_at"),
            })
            vectors.append(vec)
    return metas, vectors


def build_clusters(vectors, threshold):
    """Return (clusters, sim_matrix). clusters: list of dicts with members idx + pairs."""
    n = len(vectors)
    if n == 0:
        return [], None

    V = np.asarray(vectors, dtype=np.float64)
    # Defensive re-normalization (vectors are already cosine-normalized by Voyage;
    # this guards against float drift producing sim > 1).
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    V = V / norms

    sim = V @ V.T  # (N, N) cosine similarity

    # Upper-triangle pairs >= threshold
    iu, ju = np.triu_indices(n, k=1)
    mask = sim[iu, ju] >= threshold
    edge_i = iu[mask]
    edge_j = ju[mask]

    uf = UnionFind(n)
    for a, b in zip(edge_i.tolist(), edge_j.tolist()):
        uf.union(a, b)

    # Group nodes that participate in >=1 edge, by root
    groups = {}
    for a, b in zip(edge_i.tolist(), edge_j.tolist()):
        for node in (a, b):
            root = int(uf.find(node))
            groups.setdefault(root, set()).add(node)

    clusters = []
    for root, node_set in groups.items():
        members = sorted(node_set)
        member_pos = set(members)
        # collect edges fully inside this cluster
        pairs = []
        for a, b, s in zip(edge_i.tolist(), edge_j.tolist(), sim[edge_i, edge_j].tolist()):
            if a in member_pos and b in member_pos:
                pairs.append((a, b, float(s)))
        max_sim = max(p[2] for p in pairs) if pairs else 0.0
        clusters.append({
            "members_idx": members,
            "pairs_idx": pairs,
            "max_similarity": max_sim,
        })

    # Sort by max internal similarity, highest first
    clusters.sort(key=lambda c: c["max_similarity"], reverse=True)
    return clusters, sim


def render_json(metas, clusters, scan_date):
    zone_summary = {
        "auto_merge": {"count": 0, "memory_count": 0},
        "llm_adjudicate": {"count": 0, "memory_count": 0},
        "dedup_anomaly": {"count": 0, "memory_count": 0},
    }
    out_clusters = []
    for cid, c in enumerate(clusters, start=1):
        zone = zone_for(c["max_similarity"])
        members = [metas[i] for i in c["members_idx"]]
        pairs = [
            {"id_a": metas[a]["id"], "id_b": metas[b]["id"], "similarity": round(s, 6)}
            for (a, b, s) in c["pairs_idx"]
        ]
        pairs.sort(key=lambda p: p["similarity"], reverse=True)
        out_clusters.append({
            "cluster_id": cid,
            "zone": zone,
            "max_similarity": round(c["max_similarity"], 6),
            "members": members,
            "pairs": pairs,
        })
        zone_summary[zone]["count"] += 1
        zone_summary[zone]["memory_count"] += len(members)

    return {
        "scan_date": scan_date,
        "total_memories": len(metas),
        "total_clusters": len(out_clusters),
        "clusters": out_clusters,
        "zone_summary": zone_summary,
    }


def render_markdown(report):
    lines = []
    lines.append("# Engram Consolidation Scanner Report")
    lines.append("")
    lines.append("- **Scan date:** {}".format(report["scan_date"]))
    lines.append("- **Total memories scanned:** {}".format(report["total_memories"]))
    lines.append("- **Total clusters (>= threshold):** {}".format(report["total_clusters"]))
    lines.append("")
    lines.append("## Zone Summary")
    lines.append("")
    lines.append("| Zone | Clusters | Memories |")
    lines.append("|------|----------|----------|")
    zs = report["zone_summary"]
    zone_labels = [
        ("dedup_anomaly", "Dedup anomaly (>= 0.92)"),
        ("auto_merge", "Auto-merge (0.88-0.92)"),
        ("llm_adjudicate", "LLM adjudicate (0.82-0.88)"),
    ]
    for key, label in zone_labels:
        lines.append("| {} | {} | {} |".format(label, zs[key]["count"], zs[key]["memory_count"]))
    lines.append("")
    lines.append("## Clusters")
    lines.append("")
    if not report["clusters"]:
        lines.append("_No clusters found at the given threshold._")
    for c in report["clusters"]:
        lines.append("### Cluster {} — zone: `{}` — max sim: {:.4f}".format(
            c["cluster_id"], c["zone"], c["max_similarity"]))
        lines.append("")
        lines.append("**Members ({}):**".format(len(c["members"])))
        lines.append("")
        for m in c["members"]:
            preview = m["content_preview"].replace("\n", " ").strip()
            lines.append("- `{}` [{}] type=`{}` imp={} tags={}".format(
                m["id"], m["collection"], m["type"], m["importance"], m["tags"]))
            lines.append("  > {}".format(preview))
        lines.append("")
        lines.append("**Similar pairs:**")
        lines.append("")
        for p in c["pairs"]:
            lines.append("- {:.4f} — `{}` ↔ `{}`".format(p["similarity"], p["id_a"], p["id_b"]))
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Engram consolidation similarity scanner")
    parser.add_argument("--qdrant-url", default=QDRANT_URL)
    parser.add_argument("--collections", default=",".join(DEFAULT_COLLECTIONS),
                        help="comma-separated collection names")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    collections = [c.strip() for c in args.collections.split(",") if c.strip()]
    scan_date = datetime.datetime.now(
        datetime.timezone(datetime.timedelta(hours=8))
    ).replace(microsecond=0).isoformat()

    print("Scanning {} collection(s) at threshold {} ...".format(len(collections), args.threshold))
    metas, vectors = collect_points(args.qdrant_url, collections)
    print("Collected {} memories with vectors.".format(len(metas)))

    clusters, _ = build_clusters(vectors, args.threshold)
    print("Found {} cluster(s).".format(len(clusters)))

    report = render_json(metas, clusters, scan_date)

    JSON_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(JSON_REPORT, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("Wrote JSON report: {}".format(JSON_REPORT))

    md = render_markdown(report)
    MD_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(MD_REPORT, "w") as f:
        f.write(md)
    print("Wrote Markdown report: {}".format(MD_REPORT))

    print("Zone summary: {}".format(report["zone_summary"]))


if __name__ == "__main__":
    main()
