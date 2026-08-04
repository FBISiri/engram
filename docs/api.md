# Engram API Reference

> Complete parameter documentation for all MCP tools and REST API endpoints.
>
> **Version**: v0.2 (W23, 2026-06-03)

---

## Table of Contents

- [MCP Tools (stdio transport)](#mcp-tools-stdio-transport)
  - [memory_search](#memory_search)
  - [memory_add](#memory_add)
  - [memory_update](#memory_update)
  - [memory_delete](#memory_delete)
  - [reflection_check](#reflection_check)
  - [reflection_run](#reflection_run)
- [REST API Endpoints](#rest-api-endpoints)
  - [GET /health](#get-health)
  - [POST /reflect](#post-reflect)
  - [GET /reflect/check](#get-reflectcheck)
  - [GET /memories/expiry-candidates](#get-memoriesexpiry-candidates)
  - [DELETE /memories/expired](#delete-memoriesexpired)
  - [POST /memories](#post-memories)
  - [GET /memories/{id}](#get-memoriesid)
  - [PATCH /memories/{id}](#patch-memoriesid)
  - [PUT /memories/{id}](#put-memoriesid)
  - [DELETE /memories/{id}](#delete-memoriesid)
  - [POST /memories/{id}/reset](#post-memoriesid-reset)
  - [POST /memories/search](#post-memoriessearch)
  - [POST /memories/cross-search](#post-memoriescross-search)
  - [Provenance / EU AI Act Compliance](#provenance--eu-ai-act-compliance)
  - [POST /collections](#post-collections)
  - [GET /collections](#get-collections)
  - [Collection-scoped CRUD](#collection-scoped-crud)
  - [GET /metrics](#get-metrics)
- [CLI (`engram-cli`)](cli.md) — thin shell wrapper over these REST routes
- [Data Types](#data-types)
  - [Memory Object](#memory-object)
  - [Memory Types](#memory-types)
  - [Lifecycle FSM](#lifecycle-fsm)
  - [TTL Matrix](#ttl-matrix)
  - [Scoring Algorithm](#scoring-algorithm)

---

## MCP Tools (stdio transport)

The MCP transport is the primary interface used by LLM agents (Siri, BMO) via `mcp-go` stdio.

### memory_search

Semantic search over stored memories. Returns scored results combining relevance, recency, and importance.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `query` | string | ✅ | — | Search query text for semantic similarity matching. |
| `limit` | number | — | 5 | Maximum results to return. Range: 1–100. |
| `types` | string[] | — | all | Filter by memory types: `identity`, `event`, `insight`, `directive`. |
| `tags` | string[] | — | — | Filter by tags. Memories must have at least one matching tag. |
| `time_start` | number | — | — | Filter memories created after this Unix timestamp. |
| `time_end` | number | — | — | Filter memories created before this Unix timestamp. |
| `collections` | string[] | — | all (fan-out) | Filter by collection names (e.g. `engram_user`, `engram_reflection`). Unknown name → error. |

**How it works:**

1. Query text is embedded into a vector via the configured embedder.
2. Over-fetches `3 × limit` candidates from Qdrant (minimum 10).
3. Applies 3-component scoring: `relevance × w_r + recency × w_rec + importance × w_i` (with time decay).
4. Re-ranks using MMR (Maximal Marginal Relevance, λ configurable) for relevance + diversity.
5. Truncates to `limit`.
6. Asynchronously increments `access_count` and `last_accessed_at` on returned memories.

**Request example (MCP):**

```json
{
  "query": "Frank's cycling preferences",
  "limit": 3,
  "types": ["identity", "directive"],
  "tags": ["frank", "cycling"],
  "time_start": 1717200000
}
```

**Response example:**

```json
[
  {
    "id": "a1b2c3d4-...",
    "type": "identity",
    "content": "Frank prefers road cycling in the morning before 8am.",
    "source": "user",
    "importance": 7,
    "tags": ["frank", "cycling", "preferences"],
    "created_at": 1717200000,
    "updated_at": 1717200000,
    "score": 1.823,
    "valid_until": 0,
    "access_count": 42,
    "last_accessed_at": 1717300000,
    "source_collection": "engram_user"
  }
]
```

---

### memory_add

Store a new memory. Automatically deduplicates against existing memories (cosine similarity ≥ 0.92 threshold).

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `content` | string | ✅ | — | The memory content text. |
| `type` | string | — | `"event"` | Memory type: `identity`, `event`, `insight`, `directive`. |
| `importance` | number | — | 5 | Importance score. Clamped to 1–10. |
| `tags` | string[] | — | `[]` | Tags for classification. |
| `source` | string | — | `"agent"` | Source of the memory: `user`, `agent`, or `system`. |
| `valid_until` | number | — | auto | Expiration time as Unix timestamp. 0 or omitted = auto-computed via TTL matrix (see [TTL Matrix](#ttl-matrix)). |

**Deduplication behavior:**

Before inserting, the tool searches for the top 3 most similar memories within the same collection. If any match has cosine similarity ≥ 0.92 (configurable via `ENGRAM_DEDUP_THRESHOLD`), the insert is **skipped** and a `"duplicate"` response is returned with the existing memory's ID and content.

**Request example:**

```json
{
  "content": "Frank prefers flat white coffee with oat milk.",
  "type": "identity",
  "importance": 6,
  "tags": ["frank", "preferences", "coffee"],
  "source": "user"
}
```

**Response (created):**

```json
{
  "status": "created",
  "memory": {
    "id": "e5f6a7b8-...",
    "type": "identity",
    "content": "Frank prefers flat white coffee with oat milk.",
    "source": "user",
    "importance": 6,
    "tags": ["frank", "preferences", "coffee"],
    "created_at": 1717401000,
    "updated_at": 1717401000,
    "valid_until": 0,
    "collection": "engram_user"
  }
}
```

**Response (duplicate):**

```json
{
  "status": "duplicate",
  "message": "A very similar memory already exists. Skipped.",
  "existing": {
    "id": "prev-id-...",
    "content": "Frank likes flat white coffee with oat milk.",
    "score": 0.95
  }
}
```

---

### memory_update

Update memories by semantic search. Finds old memories matching `old_content`, deletes them, and stores `new_content`.

**⚠️ SAFETY:** `similarity_threshold` must be ≥ 0.85. Values below this are **rejected** (3 prior mass-delete incidents at lower values). Recommended: 0.92 for single targeted replacement.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `old_content` | string | ✅ | — | Search query to find old memories to replace. |
| `new_content` | string | ✅ | — | New memory content to store. |
| `type` | string | — | `"event"` | Memory type for the new memory: `identity`, `event`, `insight`, `directive`. |
| `importance` | number | — | 5 | Importance score for the new memory. Clamped to 1–10. |
| `tags` | string[] | — | `[]` | Tags for the new memory. |
| `similarity_threshold` | number | — | 0.7 (**rejected** — must use ≥ 0.85) | Minimum cosine similarity for deletion. Hard floor: 0.85. |
| `valid_until` | number | — | inherited | Expiration timestamp. 0 or omitted = inherits from the first deleted memory, or auto-computed if no match. |
| `dry_run` | boolean | — | `false` | Preview which memories would be deleted/added without making changes. |

**How it works:**

1. Embeds both `old_content` and `new_content` in a single batch call.
2. Searches for up to 20 candidates matching `old_content`.
3. Filters candidates by `similarity_threshold`.
4. Deletes matching memories.
5. Inserts a new memory with `new_content`, inheriting `valid_until` from the first deleted memory if not explicitly set.

**Request example:**

```json
{
  "old_content": "Frank prefers oat milk in coffee",
  "new_content": "Frank switched from oat milk to almond milk in coffee as of 2026-05.",
  "type": "identity",
  "importance": 6,
  "tags": ["frank", "preferences", "coffee"],
  "similarity_threshold": 0.92
}
```

**Response:**

```json
{
  "status": "updated",
  "deleted_count": 1,
  "deleted": [
    {
      "id": "e5f6a7b8-...",
      "content": "Frank prefers flat white coffee with oat milk.",
      "score": 0.94,
      "valid_until": 0
    }
  ],
  "new_memory": {
    "id": "new-id-...",
    "type": "identity",
    "content": "Frank switched from oat milk to almond milk in coffee as of 2026-05.",
    "importance": 6,
    "tags": ["frank", "preferences", "coffee"],
    "created_at": 1717500000,
    "updated_at": 1717500000
  }
}
```

**Dry run response:**

```json
{
  "status": "dry_run",
  "would_delete_count": 1,
  "would_delete": [
    { "id": "e5f6a7b8-...", "content": "...", "score": 0.94, "valid_until": 0 }
  ],
  "new_content": "Frank switched from oat milk to almond milk in coffee as of 2026-05."
}
```

---

### memory_delete

Delete memories by semantic search. Finds memories matching the query above the similarity threshold and removes them.

**⚠️ SAFETY:** When `limit > 1`, `similarity_threshold` must be ≥ 0.85 (enforced server-side). For single deletion (`limit=1`), any threshold is accepted.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `query` | string | ✅ | — | Search query to find memories to delete. |
| `similarity_threshold` | number | — | 0.7 | Minimum cosine similarity for deletion. Must be ≥ 0.85 when `limit > 1`. |
| `limit` | number | — | 20 | Maximum number of memories to delete. Minimum: 1. |
| `dry_run` | boolean | — | `false` | Preview which memories would be deleted without removing them. |

**Request example:**

```json
{
  "query": "outdated weather data from last week",
  "similarity_threshold": 0.85,
  "limit": 5,
  "dry_run": true
}
```

**Response (no matches):**

```json
{
  "status": "no_matches",
  "deleted_count": 0,
  "deleted": []
}
```

**Response (deleted):**

```json
{
  "status": "deleted",
  "deleted_count": 2,
  "deleted": [
    { "id": "id-1", "content": "Weather was 28°C on 2026-05-25...", "score": 0.91, "valid_until": 1717200000 },
    { "id": "id-2", "content": "Weather was 30°C on 2026-05-26...", "score": 0.88, "valid_until": 1717300000 }
  ]
}
```

---

### reflection_check

Check whether the Reflection Engine should run now. Returns trigger status, accumulated importance, unreflected memory count, and skip reason if not triggered.

**Parameters:** None.

**Response example:**

```json
{
  "should_trigger": false,
  "skip_reason": "importance accumulation 25.0 < threshold 40.0 (8 unreflected memories)",
  "unreflected_count": 8,
  "accumulated_importance": 25.0,
  "threshold": 40.0,
  "hours_since_last_run": 1.5,
  "runs_today": 1
}
```

**Trigger gates (evaluated in order):**

1. **Time interval** — Must be ≥ 2h since last run (configurable via `MinIntervalH`).
2. **Daily limit** — Max 3 runs per calendar day.
3. **Importance accumulation** — Sum of `importance` of unreflected memories must be ≥ 40 (configurable via `Threshold`).

---

### reflection_run

Run one Reflection Engine cycle. Synthesizes insights from unreflected memories using Haiku LLM. Respects min-interval (2h) and daily limit (3x/day).

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `dry_run` | boolean | — | `false` | Simulate the run without writing any changes. |

**Response (RunResult):**

```json
{
  "triggered": true,
  "input_count": 15,
  "insights_created": 3,
  "drafts_written": 0,
  "sources_marked": 15,
  "sources_orphaned": 0,
  "duration": "4.2s",
  "trigger_importance": 85.0,
  "dry_run": false,
  "mode": "v1-flat",
  "llm_calls": 1,
  "llm_cost_estimate_usd": 0.002,
  "runs_today": 2,
  "valid_until_set": true,
  "valid_until": "2026-09-01T00:00:00Z",
  "errors": []
}
```

**Key RunResult fields:**

| Field | Type | Description |
|-------|------|-------------|
| `triggered` | bool | Whether the run actually executed (false if gates blocked). |
| `skip_reason` | string | Why the run was skipped (only when `triggered=false`). |
| `input_count` | int | Number of unreflected memories consumed. |
| `insights_created` | int | New insight memories written to store. |
| `drafts_written` | int | Low-confidence insights (< 0.6) sent to Obsidian draft. |
| `sources_marked` | int | Source memories marked with `reflected_at` timestamp. |
| `sources_orphaned` | int | Source IDs that were deleted between fetch and mark (TOCTOU race). |
| `mode` | string | Algorithm used: `"v1-flat"` or `"v2-focal"`. |
| `llm_calls` | int | Total Haiku LLM calls made. |
| `llm_cost_estimate_usd` | float | Estimated cost in USD. |
| `runs_today` | int | How many reflection runs occurred today (CST). |
| `valid_until_set` | bool | Whether all produced insights have a non-zero TTL. |
| `valid_until` | string? | Earliest expiry across all insights (RFC 3339 UTC). Null when `valid_until_set=false`. |

---

## REST API Endpoints

The REST API runs on a configurable HTTP port alongside the MCP stdio transport. All endpoints except `/health` and `/metrics` require Bearer token authentication:

```
Authorization: Bearer <API_KEY>
```

### GET /health

Liveness/readiness probe. **No authentication required.**

Pings the Qdrant backend via `store.Stats()` and returns system status.

**Response (200 OK):**

```json
{
  "status": "ok",
  "qdrant": "green",
  "point_count": 1250,
  "uptime_seconds": 86400.5,
  "memory_count": {
    "engram_user": 800,
    "engram_reflection": 400,
    "engram_agent_self": 50,
    "total": 1250
  },
  "last_reflection": {
    "should_trigger": false,
    "skip_reason": "too soon: last run 0.5h ago (min interval 2.0h)",
    "unreflected_count": 0,
    "accumulated_importance": 0,
    "threshold": 40,
    "hours_since_last_run": 0.5,
    "runs_today": 1
  },
  "embedding_latency": {
    "p50_seconds": 0.045,
    "p99_seconds": 0.210
  }
}
```

**Response (503 Service Unavailable):**

```json
{
  "status": "degraded",
  "qdrant": "unreachable",
  "error": "connection refused"
}
```

---

### POST /reflect

Run one Reflection Engine cycle.

**Request body (optional):**

```json
{ "dry_run": true }
```

**Response:** [RunResult](#reflection_run) JSON (same as MCP `reflection_run`).

---

### GET /reflect/check

Check trigger conditions without running a reflection cycle.

**Response:** [CheckResult](#reflection_check) JSON (same as MCP `reflection_check`).

---

### GET /memories/expiry-candidates

List memories eligible for policy-based deletion. Capped at 50 results.

**Expiry policy criteria** (both must be satisfied):

| Memory Type | Importance < N | Age > N days | Notes |
|-------------|---------------|-------------|-------|
| `event` | < 4 | > 30 days | — |
| `insight` | < 5 | > 90 days | — |
| `directive` | < 6 | > 180 days | — |
| `identity` | — | — | **Never auto-deleted.** |

**Additional safety rule:** Memories with `importance ≥ 8` are **always protected** regardless of age.

**Response:**

```json
[
  {
    "id": "abc-123",
    "type": "event",
    "importance": 3,
    "age_days": 45.2,
    "content_preview": "Weather was 28°C on...",
    "tags": ["weather", "location"]
  }
]
```

---

### DELETE /memories/expired

Execute policy-based cleanup.

**Query parameters:**

| Param | Default | Description |
|-------|---------|-------------|
| `dry_run` | `true` | Must set `dry_run=false&confirm=true` to actually delete. |
| `confirm` | `false` | Safety confirmation flag. Required when `dry_run=false`. |

**Response:**

```json
{
  "deleted_count": 5,
  "skipped_count": 2,
  "snapshot_path": "/path/to/snapshot.json"
}
```

---

### POST /memories

Create a new memory (REST equivalent of `memory_add`).

**Request body:**

```json
{
  "type": "insight",
  "content": "Siri's event loop processes ~60 emails per day on average.",
  "source": "agent",
  "importance": 6,
  "tags": ["siri", "metrics"],
  "valid_until": 0,
  "metadata": { "sprint": "W23", "source_type": "tool_output" }
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `content` | string | ✅ | — | Memory content text. |
| `type` | string | — | `"event"` | `identity`, `event`, `insight`, `directive`. |
| `source` | string | — | `"agent"` | `user`, `agent`, `system`. |
| `importance` | number | — | 5 | Clamped to 1–10. |
| `tags` | string[] | — | `[]` | Classification tags. |
| `valid_until` | number | — | auto | Unix timestamp. 0 = auto via TTL matrix. |
| `metadata` | object | — | `{}` | Arbitrary key-value pairs. |
| `metadata.source_type` | string | — | `"unknown"` | Fine-grained provenance tag. Stored in `metadata.source_type`. Enum: `reflection`, `user_input`, `web_search`, `tool_output`, `calendar`, `document`. Invalid value → `400`. Behaviour when omitted depends on `ENGRAM_PROVENANCE_MODE` (see [Provenance](#provenance--eu-ai-act-compliance)). |

**Response:** `201 Created` with the full [Memory object](#memory-object). Provenance is returned nested as `metadata.source_type` (there is no top-level `source_type` field on the POST 201 body).

---

### GET /memories/{id}

Retrieve a single memory by ID.

**Response (200):** Full [Memory object](#memory-object). Provenance is carried in `metadata.source_type` and mirrored as a top-level `source_type` field.

**Response (404):**

```json
{ "error": "not found" }
```

---

### PATCH /memories/{id}

Partial update. **Content field is forbidden** (use PUT for content changes).

Supports FSM lifecycle transitions via `lifecycle_status`.

**Request body** (all fields optional):

```json
{
  "tags": ["updated-tag"],
  "importance": 8,
  "source": "user",
  "metadata": { "reviewed": true },
  "lifecycle_status": "deprecated"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `tags` | string[] | Replace tags. |
| `importance` | number | Update importance. |
| `source` | string | Update source. |
| `metadata` | object | Replace metadata. May include `source_type` (validated against the enum; invalid → `400`). |
| `lifecycle_status` | string | Transition lifecycle state (see [FSM](#lifecycle-fsm)). |

**Lifecycle transitions allowed via PATCH:**

- `active` → `deprecated`
- `active` → `archived`
- `deprecated` → `archived`

Transition `archived → *` is **forbidden** via PATCH — use `POST /memories/{id}/reset` instead.

**Response (200):** Updated [Memory object](#memory-object).

**Response (409 Conflict):**

```json
{ "error": "lifecycle transition active→archived is not allowed" }
```

---

### PUT /memories/{id}

Full replacement. Content is allowed and will be **re-embedded**. Preserves lifecycle status, `created_at`, `access_count`, and other health fields from the existing memory.

**Request body:**

```json
{
  "type": "insight",
  "content": "Updated insight content with new observations.",
  "source": "agent",
  "importance": 7,
  "tags": ["updated"],
  "valid_until": 0,
  "metadata": {}
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `content` | string | ✅ | — | New content (will be re-embedded). |
| `type` | string | — | inherits | Defaults to previous type if omitted. |
| `source` | string | — | inherits | Defaults to previous source. |
| `importance` | number | — | inherits | Defaults to previous importance. |
| `tags` | string[] | — | inherits | Defaults to previous tags. |
| `valid_until` | number | — | — | Explicit TTL override. |
| `metadata` | object | — | — | Replaces metadata (not merged). May include `source_type` (validated against the enum; invalid → `400`). |

**Response (200):** Updated [Memory object](#memory-object).

> `source_type` provenance semantics on PATCH/PUT match POST /memories — invalid enum → `400`, and omission is governed by `ENGRAM_PROVENANCE_MODE` (see [Provenance](#provenance--eu-ai-act-compliance)).

---

### DELETE /memories/{id}

Soft delete: sets `lifecycle_status` to `"archived"` and stamps `archived_at`. Does **not** physically remove the memory from the store.

Also stamps `reflected_at` if the memory was previously unreflected (prevents archived memories from appearing in the unreflected pool on every reflection run).

**Response (200):**

```json
{ "id": "abc-123", "lifecycle_status": "archived" }
```

---

### POST /memories/{id}/reset

Restore an archived or deprecated memory to active status.

**Precondition:** Memory must be in `archived` or `deprecated` state. Attempting to reset an `active` memory returns `409 Conflict`.

**Response (200):**

```json
{ "id": "abc-123", "lifecycle_status": "active" }
```

**Response (409):**

```json
{ "error": "memory is already active — reset is not allowed on active memories" }
```

---

### POST /memories/search

Vector search with lifecycle filtering (REST equivalent of MCP `memory_search`).

**Request body:**

```json
{
  "query": "Siri's daily routine",
  "collection": "engram_user",
  "limit": 10,
  "include_archived": false,
  "types": ["identity", "directive"],
  "tags": ["siri"],
  "source_type": ["user_input"]
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | ✅ | — | Search query text. |
| `collection` | string | — | resolved from X-Caller-Type | Explicit collection name. Unknown → 400. |
| `limit` | number | — | 10 | Range: 1–100. |
| `include_archived` | bool | — | `false` | Include archived memories in results. |
| `types` | string[] | — | all | Filter by memory type. |
| `tags` | string[] | — | — | Filter by tags (any match). |
| `source_type` | string[] | — | — | Filter by provenance. Each value validated against the enum; invalid → `400`. |

**Response (200):** Array of memory objects with `score`, `resolved_collection`, and `source_type` fields.

---

### POST /memories/cross-search

Cross-collection search (strict mode). The `collections` array is **required** — there is no implicit all-collection fallback.

Each collection is searched independently against its physical Qdrant collection, results are merged and re-sorted by score.

**Request body:**

```json
{
  "query": "architecture decisions",
  "collections": ["engram_user", "engram_reflection"],
  "limit": 5,
  "include_archived": false,
  "types": ["insight"],
  "tags": [],
  "source_type": ["reflection"]
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | ✅ | — | Search query text. |
| `collections` | string[] | ✅ | — | Collection names to search. Unknown name → 400. |
| `limit` | number | — | 10 | Range: 1–100. |
| `include_archived` | bool | — | `false` | Include archived memories. |
| `types` | string[] | — | all | Filter by memory type. |
| `tags` | string[] | — | — | Filter by tags. |
| `source_type` | string[] | — | — | Filter by provenance. Each value validated against the enum; invalid → `400`. |

**Response (200):** Array of objects with `score`, `collection`, and `source_type` fields.

---

## Provenance / EU AI Act Compliance

Every memory carries a fine-grained provenance classifier in `metadata.source_type`.
This complements the coarse `source` field (`user`/`agent`/`system`) and provides an
auditable record of *what kind of input* produced each memory — required for EU AI Act
traceability of AI-generated vs. human-sourced content.

### source_type values

| Value | Use case |
|-------|----------|
| `user_input` | Content that came directly from user input (chat, dictation, forms). |
| `web_search` | Content derived from a web search result. |
| `tool_output` | Content derived from a tool/function-call result. |
| `reflection` | Insight synthesized by the Reflection Engine. |
| `calendar` | Content derived from calendar data. |
| `document` | Content extracted from an ingested document. |
| `unknown` | Sentinel for legacy memories / omitted provenance (see modes below). Not a value a caller should send in strict mode. |

Any value outside this enum is rejected with `400` on every write path
(`POST /memories`, `PUT`/`PATCH /memories/{id}`, and the MCP `memory_add`/`memory_update` tools).

### Enforcement modes

Provenance enforcement at write time is controlled by the `ENGRAM_PROVENANCE_MODE`
environment variable. Invalid values fall back to `warn`.

| `ENGRAM_PROVENANCE_MODE` | Behaviour when `source_type` is **omitted** | Behaviour when `source_type` is **present** |
|--------------------------|---------------------------------------------|---------------------------------------------|
| `warn` (default) | Accepted; `metadata.source_type` defaults to `"unknown"` and a warning is logged. | Accepted if in the enum; otherwise `400`. |
| `default` | Same as `warn` at write time (defaults to `"unknown"`, logs a warning). | Accepted if in the enum; otherwise `400`. |
| `strict` | **Rejected** with `422 Unprocessable Entity`. | `400` if not in the enum; `422` if the value is `"unknown"` or (when `ENGRAM_ALLOWED_PROVENANCES` is set) not in that allow-list; otherwise accepted. |

> Note: `warn` and `default` behave identically on the write gate. The distinction
> only matters for the Reflection Engine's evidence filter (see below).

### Related environment variables

| Variable | Type | Default | Effect |
|----------|------|---------|--------|
| `ENGRAM_PROVENANCE_MODE` | `warn` \| `strict` \| `default` | `warn` | Write-time enforcement (table above). |
| `ENGRAM_ALLOWED_PROVENANCES` | comma-separated list | empty | In `strict` mode, an explicit `source_type` outside this list is rejected with `422`. Empty = any enum value except `unknown` is accepted. |
| `ENGRAM_REQUIRE_PROVENANCE` | bool | `false` | Legacy flag. Enables the **Reflection Engine** provenance evidence filter (restricting which memories the reflector may cite). It does **not** by itself gate the REST/MCP write path — write rejection is governed solely by `ENGRAM_PROVENANCE_MODE=strict`. |

---

### POST /collections

Register a new collection (in-memory registry, W20 Day2 Phase 1).

**Request body:**

```json
{
  "name": "engram_custom",
  "ttl": "30d"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | ✅ | Collection name. |
| `ttl` | string | — | Default TTL for memories in this collection. Accepts Go duration (`720h`) or day shorthand (`30d`). Empty = no TTL. |

**Response (201):**

```json
{
  "name": "engram_custom",
  "created_at": "2026-06-03T10:00:00Z",
  "ttl": "30d"
}
```

**Response (409):** Duplicate collection name.

---

### GET /collections

List all registered collections.

**Response (200):** Array of collection objects.

---

### Collection-scoped CRUD

All CRUD operations are also available scoped to a specific collection:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/collections/{name}/memories` | Create in specific collection |
| `POST` | `/collections/{name}/memories/search` | Search within specific collection |
| `GET` | `/collections/{name}/memories/{id}` | Get by ID (validates collection ownership) |
| `PATCH` | `/collections/{name}/memories/{id}` | Partial update |
| `PUT` | `/collections/{name}/memories/{id}` | Full replace |
| `DELETE` | `/collections/{name}/memories/{id}` | Soft delete |
| `POST` | `/collections/{name}/memories/{id}/reset` | Restore |

Each route validates that the URL collection matches the caller's resolved collection (from `X-Caller-Type` header).

---

### GET /metrics

Prometheus metrics endpoint. **No authentication required.**

Exposes:
- `engram_search_duration_seconds` — search latency histogram
- `engram_embed_duration_seconds` — embedding latency histogram
- Embed cache counters (hit/miss)
- Per-collection memory counts

---

## Data Types

### Memory Object

```json
{
  "id": "uuid-v4",
  "type": "identity|event|insight|directive",
  "content": "The memory text content.",
  "source": "user|agent|system",
  "importance": 5.0,
  "tags": ["tag1", "tag2"],
  "created_at": 1717200000.0,
  "updated_at": 1717200000.0,
  "metadata": { "source_type": "user_input" },
  "valid_until": 0,
  "superseded_by": "",
  "access_count": 0,
  "last_accessed_at": 0,
  "last_accessed_source": "",
  "reflected_at": 0,
  "confidence": 0,
  "archived_at": 0,
  "archive_reason": "",
  "lifecycle_status": "active",
  "collection": "engram_user"
}
```

### Memory Types

| Type | Description | Auto-delete eligible? |
|------|-------------|----------------------|
| `identity` | Stable facts about the user (name, job, preferences, relationships). | **Never.** |
| `event` | Something that happened or was observed. | Yes (importance < 4, age > 30d). |
| `insight` | Inferred patterns, reflections, or conclusions. | Yes (importance < 5, age > 90d). |
| `directive` | Explicit instructions from the user. | Yes (importance < 6, age > 180d). |

### Lifecycle FSM

```
         ┌─────────────┐
         │   active     │ (default)
         └──────┬───────┘
                │
       ┌────────┴────────┐
       ▼                 ▼
 ┌───────────┐    ┌────────────┐
 │deprecated │───▶│  archived  │
 └───────────┘    └─────┬──────┘
                        │
                  POST /{id}/reset
                        │
                        ▼
                 ┌─────────────┐
                 │   active    │
                 └─────────────┘
```

**Allowed transitions via PATCH:**
- `active` → `deprecated`
- `active` → `archived`
- `deprecated` → `archived`

**Restore via POST:** `archived` → `active`, `deprecated` → `active`.

### TTL Matrix

Automatically computed if `valid_until` is not explicitly set. Based on type × importance band:

| Type | importance < 5 | importance 5–7 | importance ≥ 8 |
|------|----------------|----------------|----------------|
| `identity` | permanent | permanent | permanent |
| `directive` | 90 days | permanent | permanent |
| `insight` | 30 days | 90 days | permanent |
| `event` | 3 days | 7 days | 30 days |

**Tag overrides:**
- `"permanent"` tag → never expires (overrides matrix).
- `"time-sensitive"` or `"location"` tag → forces 7-day TTL (unless `"permanent"` tag is also present).
- Explicit `valid_until` value always takes precedence over all auto-computation.

### Scoring Algorithm

The final score for each search result combines three components:

```
final_score = cosine_similarity × w_relevance
            + recency_factor    × w_recency
            + importance_norm   × w_importance
```

Where:
- `recency_factor` uses exponential time decay (configurable half-life).
- `importance_norm` = `importance / 10.0`.
- `confidence` (0–1) from the Reflection Engine acts as a multiplier on the final score.
- Weights (`w_relevance`, `w_recency`, `w_importance`) and decay parameters are configurable via `engram.yaml`.

After scoring, results are re-ranked using **MMR (Maximal Marginal Relevance)** to balance relevance with diversity (configurable λ parameter, default: 0.7).
