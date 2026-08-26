"""Undo log write + restore CLI (R5).

Undo log: /data/armyoftheagent/engram/consolidation/undo/<YYYY-MM-DD>-<HH-MM>.json
Schema per spec §4.2 — full memory snapshots including vectors, so restore can
re-insert originals without re-embedding.

Restore: delete the merged memory (Engram REST) -> re-insert each original
point straight into Qdrant with its original vector + payload (bypasses embed).
"""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# Allow running as a direct script (python3 cmd/consolidation/undo.py ...).
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import consolidation  # noqa: F401
    __package__ = "consolidation"

from .config import Config, load_config
from .merge import EngramClient, MergeOp


def _now_beijing() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))


class UndoLog:
    def __init__(self, cfg: Config, run_id: str, run_mode: str,
                 timestamp: Optional[datetime.datetime] = None):
        self.cfg = cfg
        self.run_id = run_id
        self.run_mode = run_mode
        self.timestamp = timestamp or _now_beijing()
        fname = self.timestamp.strftime("%Y-%m-%d-%H-%M") + ".json"
        self.path = Path(cfg.undo_dir) / fname
        self.operations: List[Dict[str, Any]] = []

    def _snapshot_member(self, m) -> Dict[str, Any]:
        return {
            "id": m.id,
            "collection": m.collection,
            "content": m.content,
            "type": m.type,
            "importance": m.importance,
            "tags": m.tags,
            "source_type": m.source_type,
            "metadata": m.metadata,
            "created_at": m.created_at,
            "vector": m.vector,
            "payload": m.raw_payload,
        }

    def build(self, ops: List[MergeOp]) -> None:
        self.operations = []
        for i, op in enumerate(ops, start=1):
            self.operations.append({
                "operation_id": i,
                "type": "merge",
                "cluster_id": op.cluster_id,
                "collection": op.collection,
                "zone": op.zone,
                "decision_source": op.decision_source,
                "source_memories": [self._snapshot_member(m) for m in op.source_members],
                "new_memory": {
                    "id": op.new_memory_id,
                    "content": op.merged_content,
                    "type": op.resolved["type"],
                    "importance": op.resolved["importance"],
                    "tags": op.resolved["tags"],
                    "source_type": op.resolved["source_type"],
                },
                "status": op.status,
                "error": op.error,
            })

    def update_statuses(self, ops: List[MergeOp]) -> None:
        for entry, op in zip(self.operations, ops):
            entry["status"] = op.status
            entry["error"] = op.error
            entry["new_memory"]["id"] = op.new_memory_id

    def _summary(self) -> Dict[str, Any]:
        completed = [o for o in self.operations if o["status"] == "completed"]
        removed = sum(len(o["source_memories"]) for o in completed)
        created = len(completed)
        return {
            "total_merges": len(completed),
            "total_memories_removed": removed,
            "total_memories_created": created,
            "net_reduction": removed - created,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_timestamp": self.timestamp.isoformat(),
            "run_mode": self.run_mode,
            "config": {
                "auto_merge_threshold": self.cfg.auto_merge_threshold,
                "llm_adjudicate_threshold": self.cfg.llm_adjudicate_threshold,
                "dedup_anomaly_threshold": self.cfg.dedup_anomaly_threshold,
                "max_merges": self.cfg.max_merges_per_run,
            },
            "operations": self.operations,
            "summary": self._summary(),
        }

    def write(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return self.path


# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------
def qdrant_upsert_point(qdrant_url: str, collection: str, point_id: str,
                        vector: List[float], payload: Dict[str, Any], timeout: int = 30) -> None:
    """Re-insert a single point into Qdrant with its original vector + payload."""
    body = {"points": [{"id": point_id, "vector": vector, "payload": payload}]}
    r = requests.put(
        qdrant_url + "/collections/" + collection + "/points?wait=true",
        json=body, timeout=timeout,
    )
    r.raise_for_status()


def restore(undo_path: str, cfg: Config, operation_id: Optional[int] = None) -> Dict[str, Any]:
    with open(undo_path, "r") as f:
        log = json.load(f)

    client = EngramClient(cfg.engram_url, cfg.engram_api_key)
    restored_ops = 0
    reinserted = 0
    deleted_merged = 0
    errors: List[str] = []

    for op in log.get("operations", []):
        if operation_id is not None and op.get("operation_id") != operation_id:
            continue
        if op.get("status") != "completed":
            continue  # only completed merges mutated data

        collection = op.get("collection", cfg.collections[0])
        new_mem = op.get("new_memory") or {}
        new_id = new_mem.get("id")

        # 1. delete the merged memory
        if new_id:
            try:
                client.delete_memory(collection, new_id)
                deleted_merged += 1
            except Exception as e:  # noqa: BLE001
                errors.append(f"op {op['operation_id']}: delete merged {new_id} failed: {e}")

        # 2. re-insert originals with original vectors
        for src in op.get("source_memories", []):
            try:
                qdrant_upsert_point(
                    cfg.qdrant_url, src["collection"], src["id"],
                    src["vector"], src.get("payload") or {},
                )
                reinserted += 1
            except Exception as e:  # noqa: BLE001
                errors.append(f"op {op['operation_id']}: reinsert {src['id']} failed: {e}")

        # 3. verify originals exist in Qdrant
        for src in op.get("source_memories", []):
            try:
                rr = requests.get(
                    cfg.qdrant_url + "/collections/" + src["collection"] + "/points/" + str(src["id"]),
                    timeout=10,
                )
                if rr.status_code != 200:
                    errors.append(f"op {op['operation_id']}: verify {src['id']} status {rr.status_code}")
            except Exception as e:  # noqa: BLE001
                errors.append(f"op {op['operation_id']}: verify {src['id']} failed: {e}")

        restored_ops += 1

    return {
        "undo_path": undo_path,
        "restored_operations": restored_ops,
        "merged_deleted": deleted_merged,
        "originals_reinserted": reinserted,
        "errors": errors,
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Restore memories from a consolidation undo log")
    p.add_argument("--restore", required=True, help="path to undo log JSON")
    p.add_argument("--operation-id", type=int, default=None, help="restore only this operation")
    p.add_argument("--config", default=None, help="path to consolidation.yaml")
    args = p.parse_args(argv)

    cfg = load_config(config_path=args.config)
    result = restore(args.restore, cfg, operation_id=args.operation_id)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if not result["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
