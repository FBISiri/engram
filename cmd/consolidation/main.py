#!/usr/bin/env python3
"""Engram Consolidation Agent — CLI runner (R1).

Modes:
  --scan-only   scan + cluster + zone-classify, report only (READ-ONLY, no LLM)
  --dry-run     full plan incl. LLM adjudication, no mutation (DEFAULT)
  --live        execute merges (writes to Engram + Qdrant)
  --restore P   delegate to undo.restore

Feature flag: ENGRAM_CONSOLIDATION_ENABLED must be true.
See spec: /data/obsidian-vault/Engram/consolidation-agent-spec-v1.md
"""

import argparse
import datetime
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow running as a direct script (python3 cmd/consolidation/main.py) as well
# as a package module (python3 -m cmd.consolidation.main).
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import consolidation  # noqa: F401
    __package__ = "consolidation"

from . import adjudicate as adj
from . import report as report_mod
from . import scanner
from .cluster import build_clusters, classify_all
from .config import load_config, Config
from .merge import EngramClient, MergeExecutor, MergeOp, build_merge_op, pick_superset
from .metrics import Metrics, append_jsonl, last_run_timestamp
from .types import (
    Cluster,
    MemoryPoint,
    DS_AUTO_MERGE,
    DS_DEDUP_ANOMALY,
    DS_LLM_ADJUDICATE,
    DS_SKIP,
    DECISION_MERGE,
    DECISION_KEEP_SEPARATE,
)
from .undo import UndoLog, restore

LOCK_PATH = "/tmp/engram-consolidation.lock"


# ---------------------------------------------------------------------------
# Gating (spec §3.4)
# ---------------------------------------------------------------------------
class GateError(Exception):
    pass


class PidLock:
    def __init__(self, path: str = LOCK_PATH):
        self.path = Path(path)
        self.acquired = False

    def acquire(self) -> None:
        if self.path.exists():
            try:
                pid = int(self.path.read_text().strip())
            except (ValueError, OSError):
                pid = None
            if pid and _pid_alive(pid):
                raise GateError(f"another consolidation run holds the lock (pid {pid})")
        self.path.write_text(str(os.getpid()))
        self.acquired = True

    def release(self) -> None:
        if self.acquired:
            try:
                self.path.unlink()
            except OSError:
                pass
            self.acquired = False


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------
def _top_pair(cluster: Cluster, points: List[MemoryPoint]):
    if cluster.pairs:
        best = max(cluster.pairs, key=lambda p: p.similarity)
        return points[best.a], points[best.b], best.similarity
    ms = cluster.members(points)
    return ms[0], ms[1], cluster.max_similarity


def _member_summary(m: MemoryPoint) -> Dict[str, Any]:
    return {
        "id": m.id,
        "type": m.type,
        "importance": m.importance,
        "content_preview": m.content_preview,
    }


def _cluster_entry(cluster: Cluster, points: List[MemoryPoint]) -> Dict[str, Any]:
    return {
        "cluster_id": None,  # assigned by caller
        "zone": cluster.zone,
        "decision_source": cluster.decision_source,
        "guard": cluster.guard,
        "skip_reason": cluster.skip_reason,
        "is_fringe_pair": cluster.is_fringe_pair,
        "max_similarity": round(cluster.max_similarity, 6),
        "members": [_member_summary(m) for m in cluster.members(points)],
        "decision": None,
        "reasoning": None,
        "proposed_merged_content": None,
        "keep_id": None,
        "delete_ids": [],
        "would_reduce_by": 0,
    }


def _priority_key(cluster: Cluster, points: List[MemoryPoint]):
    members = cluster.members(points)
    imps = [m.importance or 0.0 for m in members]
    avg_imp = sum(imps) / len(imps) if imps else 0.0
    # larger clusters first, then lower average importance first (§6.3)
    return (-len(members), avg_imp)


def run_pipeline(cfg: Config, mode: str, api_key: str) -> Dict[str, Any]:
    metrics = Metrics()
    run_id = uuid.uuid4().hex
    ts = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
    now = time.time()

    # --- Scan ---
    t0 = time.time()
    if not scanner.qdrant_healthy(cfg.qdrant_url):
        raise GateError(f"Qdrant not healthy at {cfg.qdrant_url}")
    points, sim = scanner.scan(cfg.qdrant_url, cfg.collections)
    metrics.scan_duration_seconds = round(time.time() - t0, 3)
    metrics.total_memories_scanned = len(points)

    # min memory count gate (enforced for live; warn otherwise)
    if len(points) < cfg.min_memory_count:
        msg = f"only {len(points)} memories (< min_memory_count {cfg.min_memory_count})"
        if mode == "live":
            raise GateError(msg)
        print(f"[warn] {msg} — continuing in {mode} mode", file=sys.stderr)

    # --- Cluster ---
    t1 = time.time()
    clusters, fringe = build_clusters(points, sim, cfg)
    all_clusters = clusters + fringe
    classify_all(all_clusters, points, cfg, now)
    metrics.cluster_duration_seconds = round(time.time() - t1, 3)
    metrics.total_clusters_found = len(all_clusters)

    # zone + guard breakdown
    zone_counts: Dict[str, int] = {}
    guard_counts: Dict[str, int] = {"recency": 0, "type": 0, "temporal": 0}
    for c in all_clusters:
        if c.decision_source and c.decision_source != DS_SKIP:
            zone_counts[c.decision_source] = zone_counts.get(c.decision_source, 0) + 1
        if c.guard == "recency":
            guard_counts["recency"] += 1
        elif c.guard in ("identity", "directive"):
            guard_counts["type"] += 1
        elif c.guard == "temporal":
            guard_counts["temporal"] += 1
    metrics.clusters_by_zone = zone_counts
    metrics.clusters_skipped_by_guard = guard_counts

    # sort candidates by merge priority
    all_clusters.sort(key=lambda c: _priority_key(c, points))

    cluster_entries: List[Dict[str, Any]] = []
    ops: List[MergeOp] = []

    if mode == "scan-only":
        for i, c in enumerate(all_clusters, start=1):
            e = _cluster_entry(c, points)
            e["cluster_id"] = i
            cluster_entries.append(e)
        metrics.run_duration_seconds = round(time.time() - now, 3)
        return _finish(cfg, mode, run_id, ts, metrics, cluster_entries,
                       {"executed": [], "failed": [], "remaining": []})

    # --- Adjudicate + build merge ops (dry-run + live) ---
    # The max_merges cap doubles as an LLM-call budget: we only adjudicate
    # candidates (in priority order) while we still have room for more merge
    # ops. This keeps a run within the spec §5.4 cost target (~$0.03/run)
    # instead of paying for an LLM call on every fringe pair.
    for i, c in enumerate(all_clusters, start=1):
        entry = _cluster_entry(c, points)
        entry["cluster_id"] = i
        members = c.members(points)

        if c.decision_source == DS_SKIP:
            entry["decision"] = DECISION_KEEP_SEPARATE
            cluster_entries.append(entry)
            continue

        budget_left = len(ops) < cfg.max_merges_per_run
        llm_budget_left = metrics.llm_calls < cfg.max_merges_per_run

        if c.decision_source == DS_AUTO_MERGE:
            merged = pick_superset(members).content
            entry["decision"] = DECISION_MERGE
            entry["proposed_merged_content"] = merged
            entry["would_reduce_by"] = len(members) - 1
            if budget_left:
                ops.append(build_merge_op(i, members, DS_AUTO_MERGE, c.zone, now, merged_content=merged))
            else:
                entry["decision"] = "MERGE (capped by max_merges)"
            cluster_entries.append(entry)
            continue

        # LLM adjudication (llm_adjudicate + dedup_anomaly). Bounded by BOTH the
        # merge-op cap and a hard LLM-call budget (<= max_merges calls) so a run
        # stays within the spec §5.4 cost target even when most verdicts are
        # KEEP_SEPARATE.
        if not (budget_left and llm_budget_left):
            entry["decision"] = "NOT_ADJUDICATED (budget reached)"
            cluster_entries.append(entry)
            continue

        ma, mb, s = _top_pair(c, points)
        verdict = adj.adjudicate_pair(ma, mb, s, cfg, api_key=api_key)
        metrics.add_llm_usage(verdict.get("usage", {}))
        entry["decision"] = verdict["decision"]
        entry["reasoning"] = verdict.get("reasoning")

        merge_ok = verdict["decision"] == DECISION_MERGE
        # identity full-dup requires very high confidence (spec §4.3)
        is_identity = any(m.type == "identity" for m in (ma, mb))
        if is_identity and verdict.get("confidence", 0.0) < 0.95:
            merge_ok = False
            entry["decision"] = DECISION_KEEP_SEPARATE
            entry["reasoning"] = "identity requires confidence >= 0.95"

        if merge_ok:
            merged = verdict.get("merged_content") or pick_superset([ma, mb]).content
            entry["proposed_merged_content"] = merged
            entry["would_reduce_by"] = 1
            ops.append(build_merge_op(i, [ma, mb], c.decision_source, c.zone, now, merged_content=merged))
        cluster_entries.append(entry)

    metrics.merges_proposed = len(ops)

    # --- Execute (live only) ---
    execution = {"executed": [], "failed": [], "remaining": []}
    if mode == "live" and ops:
        client = EngramClient(cfg.engram_url, cfg.engram_api_key)
        if not client.healthy():
            raise GateError(f"Engram server not healthy at {cfg.engram_url}")
        undo = UndoLog(cfg, run_id, mode, timestamp=ts)
        executor = MergeExecutor(client, undo)
        result = executor.run(ops)
        metrics.merges_executed = len(result.completed)
        metrics.merges_failed = len(result.failed)
        metrics.memories_removed = result.memories_removed
        metrics.memories_created = result.memories_created
        execution = {
            "undo_log_path": str(undo.path),
            "executed": [_op_result(o) for o in result.completed],
            "failed": [_op_result(o) for o in result.failed],
            "remaining": [_op_result(o) for o in result.remaining],
        }

    metrics.run_duration_seconds = round(time.time() - now, 3)
    return _finish(cfg, mode, run_id, ts, metrics, cluster_entries, execution)


def _op_result(op: MergeOp) -> Dict[str, Any]:
    return {
        "cluster_id": op.cluster_id,
        "collection": op.collection,
        "decision_source": op.decision_source,
        "source_ids": op.source_ids,
        "new_memory_id": op.new_memory_id,
        "status": op.status,
        "error": op.error,
    }


def _finish(cfg: Config, mode: str, run_id: str, ts, metrics: Metrics,
            cluster_entries: List[Dict[str, Any]], execution: Dict[str, Any]) -> Dict[str, Any]:
    metrics.finalize()
    run_meta = {
        "run_id": run_id,
        "run_timestamp": ts.isoformat(),
        "mode": mode,
        "collections": cfg.collections,
    }
    report = report_mod.build_report(run_meta, metrics.to_dict(), cluster_entries, execution)
    json_path, md_path = report_mod.write_reports(cfg, report, ts)

    # longitudinal metrics
    summary = dict(metrics.to_dict())
    summary["run_id"] = run_id
    summary["run_timestamp"] = ts.isoformat()
    summary["run_timestamp_unix"] = ts.timestamp()
    summary["mode"] = mode
    append_jsonl(cfg.metrics_path, summary)

    return {
        "run_id": run_id,
        "mode": mode,
        "report_json": json_path,
        "report_md": md_path,
        "metrics": metrics.to_dict(),
        "execution": execution,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="consolidation",
        description="Engram Consolidation Agent — offline memory merge pass.",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="plan only, no mutation (default)")
    mode.add_argument("--live", action="store_true", help="execute merges")
    mode.add_argument("--scan-only", action="store_true", help="scan + cluster report only")

    p.add_argument("--max-merges", type=int, default=None)
    p.add_argument("--collections", default=None, help="comma-separated collection names")
    p.add_argument("--auto-merge-threshold", type=float, default=None)
    p.add_argument("--llm-threshold", type=float, default=None)
    p.add_argument("--config", default=None, help="path to consolidation.yaml")

    p.add_argument("--restore", default=None, help="restore from an undo log path")
    p.add_argument("--operation-id", type=int, default=None)
    return p.parse_args(argv)


def _cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if args.max_merges is not None:
        overrides["max_merges_per_run"] = args.max_merges
    if args.collections is not None:
        overrides["collections"] = [c.strip() for c in args.collections.split(",") if c.strip()]
    if args.auto_merge_threshold is not None:
        overrides["auto_merge_threshold"] = args.auto_merge_threshold
    if args.llm_threshold is not None:
        overrides["llm_adjudicate_threshold"] = args.llm_threshold
    return overrides


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    cfg = load_config(config_path=args.config, cli_overrides=_cli_overrides(args))

    # Restore path is independent of feature flag / gates.
    if args.restore:
        result = restore(args.restore, cfg, operation_id=args.operation_id)
        print_json(result)
        return 0 if not result["errors"] else 1

    mode = "live" if args.live else ("scan-only" if args.scan_only else "dry-run")

    # Feature flag gate (all modes).
    if not cfg.enabled:
        print("ENGRAM_CONSOLIDATION_ENABLED is not true — refusing to run "
              "(set ENGRAM_CONSOLIDATION_ENABLED=true or enable in config).",
              file=sys.stderr)
        return 2

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    lock = PidLock()
    try:
        if mode == "live":
            # min interval gate
            last = last_run_timestamp(cfg.metrics_path)
            if last and (time.time() - last) < cfg.min_interval_hours * 3600:
                print(f"last live run was < {cfg.min_interval_hours}h ago — skipping.",
                      file=sys.stderr)
                return 2
            lock.acquire()
        result = run_pipeline(cfg, mode, api_key)
    except GateError as e:
        print(f"gate failed: {e}", file=sys.stderr)
        return 2
    finally:
        lock.release()

    print_json(result)
    return 0


def print_json(obj: Dict[str, Any]) -> None:
    import json
    print(json.dumps(obj, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
