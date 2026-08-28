"""Merge executor (R4).

Field resolution (importance=max, tags=union+consolidated, source_type=trust
hierarchy, created_at=min), provenance inheritance, and the spec §2.4 execution
order: generate merged content -> write undo log (BEFORE any mutation) ->
memory_add -> verify -> memory_delete sources. On failure it stops and reports
completed + remaining operations.

Writes go through the Engram REST API. NOTE: the live server routes are
`/collections/{name}/memories` (POST) and `/collections/{name}/memories/{id}`
(DELETE) with an `X-Caller-Type` header — NOT `/api/v1/memories`. Base URL,
auth and paths are configurable so this can adapt if the surface changes.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from .config import Config
from .types import MemoryPoint, highest_trust_source


# collection name -> X-Caller-Type value (server maps caller type -> collection)
CALLER_TYPE_FOR_COLLECTION = {
    "engram_user": "user",
    "engram_reflection": "reflection",
    "engram_pigo": "pigo",
    "engram_agent_self": "agent-self",
}


@dataclass
class MergeOp:
    cluster_id: int
    collection: str
    zone: str
    decision_source: str
    source_members: List[MemoryPoint]
    merged_content: str
    resolved: Dict[str, Any]
    reasoning: str = ""
    status: str = "pending"           # pending | completed | failed | skipped
    error: Optional[str] = None
    new_memory_id: Optional[str] = None

    @property
    def source_ids(self) -> List[str]:
        return [m.id for m in self.source_members]


# ---------------------------------------------------------------------------
# Field resolution (spec §2.3)
# ---------------------------------------------------------------------------
def pick_superset(members: List[MemoryPoint]) -> MemoryPoint:
    """Auto-merge content selection: highest importance, tie-break longest."""
    return max(members, key=lambda m: ((m.importance or 0.0), len(m.content or "")))


def resolve_fields(members: List[MemoryPoint], decision_source: str, now: float) -> Dict[str, Any]:
    importances = [m.importance for m in members if m.importance is not None]
    importance = max(importances) if importances else 5.0

    tags: List[str] = []
    for m in members:
        for t in m.tags:
            if t not in tags:
                tags.append(t)
    if "consolidated" not in tags:
        tags.append("consolidated")

    source_type = highest_trust_source([m.source_type for m in members])

    created_ats = [m.created_at for m in members if m.created_at]
    created_at = min(created_ats) if created_ats else now

    # type: same-type clusters share it; cross-type takes the anchor's type.
    anchor = pick_superset(members)
    mem_type = anchor.type or "insight"

    # Provenance inheritance: preserve every source's provenance_history, then
    # append one consolidation record (spec §2.3.2).
    provenance_history: List[Dict[str, Any]] = []
    for m in members:
        for entry in m.provenance_history:
            provenance_history.append(entry)
    provenance_history.append({
        "source_type": "consolidation",
        "merged_at": int(now),
        "merged_from": [m.id for m in members],
    })

    metadata: Dict[str, Any] = {
        "source_type": source_type,
        "merged_from": [m.id for m in members],
        "merged_at": int(now),
        "merge_reason": decision_source,
        "caller": "consolidation",
        "provenance_history": provenance_history,
    }

    return {
        "type": mem_type,
        "importance": importance,
        "tags": tags,
        "source_type": source_type,
        "created_at": created_at,
        "metadata": metadata,
    }


def build_merge_op(
    cluster_id: int,
    members: List[MemoryPoint],
    decision_source: str,
    zone: str,
    now: float,
    merged_content: Optional[str] = None,
) -> MergeOp:
    resolved = resolve_fields(members, decision_source, now)
    if not merged_content:
        merged_content = pick_superset(members).content
    return MergeOp(
        cluster_id=cluster_id,
        collection=members[0].collection,
        zone=zone,
        decision_source=decision_source,
        source_members=members,
        merged_content=merged_content,
        resolved=resolved,
    )


# ---------------------------------------------------------------------------
# Engram REST client
# ---------------------------------------------------------------------------
class EngramClient:
    def __init__(self, base_url: str, api_key: str = "", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()

    def _headers(self, collection: str) -> Dict[str, str]:
        headers = {"content-type": "application/json"}
        caller = CALLER_TYPE_FOR_COLLECTION.get(collection, "user")
        headers["X-Caller-Type"] = caller
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        return headers

    def healthy(self) -> bool:
        try:
            r = self.session.get(self.base_url + "/health", timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def add_memory(self, collection: str, op: MergeOp) -> str:
        body = {
            "content": op.merged_content,
            "type": op.resolved["type"],
            "importance": op.resolved["importance"],
            "tags": op.resolved["tags"],
            "source": "agent",
            "metadata": op.resolved["metadata"],
        }
        url = self.base_url + "/collections/" + collection + "/memories"
        r = self.session.post(url, json=body, headers=self._headers(collection), timeout=self.timeout)
        if r.status_code == 409:
            try:
                dedup = r.json()
            except ValueError:
                dedup = {}
            existing_id = dedup.get("existing_id")
            if existing_id:
                logging.getLogger("consolidation.merge").info(
                    "server-side dedup: reusing existing memory %s (similarity=%s)",
                    existing_id, dedup.get("similarity"),
                )
                return str(existing_id)
        r.raise_for_status()
        data = r.json()
        new_id = str(data.get("id", ""))
        if not new_id:
            raise RuntimeError("add_memory returned no id: " + r.text[:200])
        return new_id

    def get_memory(self, collection: str, mem_id: str) -> Optional[dict]:
        url = self.base_url + "/collections/" + collection + "/memories/" + mem_id
        r = self.session.get(url, headers=self._headers(collection), timeout=self.timeout)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    def delete_memory(self, collection: str, mem_id: str) -> None:
        url = self.base_url + "/collections/" + collection + "/memories/" + mem_id
        r = self.session.delete(url, headers=self._headers(collection), timeout=self.timeout)
        if r.status_code not in (200, 204):
            r.raise_for_status()


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------
@dataclass
class ExecutionResult:
    completed: List[MergeOp] = field(default_factory=list)
    failed: List[MergeOp] = field(default_factory=list)
    remaining: List[MergeOp] = field(default_factory=list)

    @property
    def memories_removed(self) -> int:
        return sum(len(op.source_members) for op in self.completed)

    @property
    def memories_created(self) -> int:
        return len(self.completed)


class MergeExecutor:
    """Executes merge ops with an undo log written before any mutation."""

    def __init__(self, client: EngramClient, undo_log, verify: bool = True):
        self.client = client
        self.undo_log = undo_log
        self.verify = verify

    def _execute_one(self, op: MergeOp) -> None:
        # 1. content already generated. 2. undo already on disk (whole run).
        # 3. add merged memory
        new_id = self.client.add_memory(op.collection, op)
        op.new_memory_id = new_id
        # 4. verify
        if self.verify:
            got = self.client.get_memory(op.collection, new_id)
            if not got:
                raise RuntimeError(f"verify failed: new memory {new_id} not found")
        # 5. delete sources by ID
        for m in op.source_members:
            self.client.delete_memory(op.collection, m.id)
        op.status = "completed"

    def run(self, ops: List[MergeOp]) -> ExecutionResult:
        result = ExecutionResult()
        # Persist the full undo log BEFORE any mutation (spec §4.2).
        self.undo_log.build(ops)
        self.undo_log.write()

        for i, op in enumerate(ops):
            try:
                self._execute_one(op)
                result.completed.append(op)
            except Exception as e:  # stop on first failure, report the rest
                op.status = "failed"
                op.error = str(e)
                result.failed.append(op)
                result.remaining = ops[i + 1:]
                for r in result.remaining:
                    r.status = "skipped"
                self.undo_log.update_statuses(ops)
                self.undo_log.write()
                return result
            self.undo_log.update_statuses(ops)
            self.undo_log.write()
        return result
