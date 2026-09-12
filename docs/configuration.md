# Configuration Reference

All Engram configuration is done via environment variables. No config files are needed.

## Table of Contents

- [Storage (Qdrant)](#storage-qdrant)
- [Embedding](#embedding)
- [Scoring & Retrieval](#scoring--retrieval)
- [Deduplication](#deduplication)
- [Server / Transport](#server--transport)
- [Reflection Engine](#reflection-engine)
- [State Directory](#state-directory)
- [Lifecycle & Admission (Evaporation / A-MAC / Write Checkpoints)](#lifecycle--admission-evaporation--a-mac--write-checkpoints)
- [Observability (OpenTelemetry)](#observability-opentelemetry)
- [TTL Auto-Calculator](#ttl-auto-calculator)
- [Multi-Collection Architecture](#multi-collection-architecture)
- [Deprecated Variables](#deprecated-variables)
- [Quick-Start Examples](#quick-start-examples)

---

## Storage (Qdrant)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENGRAM_QDRANT_URL` | `string` | `localhost:6334` | Qdrant gRPC address. Use port 6334 for gRPC (not 6333, which is REST). |
| `ENGRAM_QDRANT_API_KEY` | `string` | _(empty)_ | Qdrant API key. Required if your Qdrant instance has authentication enabled. |
| `ENGRAM_QDRANT_USE_TLS` | `bool` | `false` | Enable TLS for the Qdrant gRPC connection. Set to `true` when connecting to Qdrant Cloud or any TLS-secured instance. |

### Usage Example

```bash
# Local Qdrant (default)
export ENGRAM_QDRANT_URL=localhost:6334

# Remote Qdrant Cloud
export ENGRAM_QDRANT_URL=abc123.us-east4-0.gcp.cloud.qdrant.io:6334
export ENGRAM_QDRANT_API_KEY=your-api-key-here
export ENGRAM_QDRANT_USE_TLS=true
```

---

## Embedding

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENGRAM_EMBEDDER_PROVIDER` | `string` | `openai` | Embedding provider. Supported values: `openai`, `voyage`. |
| `ENGRAM_EMBEDDING_MODEL` | `string` | `text-embedding-3-small` | Embedding model name. Must match the provider. |
| `ENGRAM_EMBEDDING_DIMENSION` | `int` | `1536` | Embedding vector size. Must match the model's output dimension. |
| `ENGRAM_OPENAI_API_KEY` | `string` | _(empty)_ | **Required** when `ENGRAM_EMBEDDER_PROVIDER=openai`. Your OpenAI (or compatible) API key. |
| `ENGRAM_OPENAI_BASE_URL` | `string` | `https://api.openai.com/v1` | OpenAI-compatible API base URL. Change this to use OpenRouter, Azure OpenAI, or other compatible providers. |
| `ENGRAM_VOYAGE_API_KEY` | `string` | _(empty)_ | **Required** when `ENGRAM_EMBEDDER_PROVIDER=voyage`. Your Voyage AI API key. |

### Provider Configurations

**OpenAI (default)**:
```bash
export ENGRAM_EMBEDDER_PROVIDER=openai
export ENGRAM_OPENAI_API_KEY=sk-...
export ENGRAM_EMBEDDING_MODEL=text-embedding-3-small
export ENGRAM_EMBEDDING_DIMENSION=1536
```

**OpenAI with `text-embedding-3-large`**:
```bash
export ENGRAM_EMBEDDER_PROVIDER=openai
export ENGRAM_OPENAI_API_KEY=sk-...
export ENGRAM_EMBEDDING_MODEL=text-embedding-3-large
export ENGRAM_EMBEDDING_DIMENSION=3072
```

**Voyage AI**:
```bash
export ENGRAM_EMBEDDER_PROVIDER=voyage
export ENGRAM_VOYAGE_API_KEY=pa-...
export ENGRAM_EMBEDDING_MODEL=voyage-3.5
export ENGRAM_EMBEDDING_DIMENSION=1024
```

**OpenRouter (via OpenAI-compatible base URL)**:
```bash
export ENGRAM_EMBEDDER_PROVIDER=openai
export ENGRAM_OPENAI_API_KEY=sk-or-v1-...
export ENGRAM_OPENAI_BASE_URL=https://openrouter.ai/api/v1
export ENGRAM_EMBEDDING_MODEL=openai/text-embedding-3-small
export ENGRAM_EMBEDDING_DIMENSION=1536
```

> **⚠️ Important**: Changing the embedding model or dimension on an existing Qdrant collection will cause errors. You must recreate the collection (delete and re-index) when switching embedding configurations.

---

## Scoring & Retrieval

Engram scores memories using a three-component formula:

```
score = W_relevance × relevance + W_recency × recency + W_importance × importance
```

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENGRAM_WEIGHT_RELEVANCE` | `float64` | `1.0` | Weight for cosine similarity between query and memory embedding. Higher values prioritize semantic match. |
| `ENGRAM_WEIGHT_RECENCY` | `float64` | `0.5` | Weight for temporal recency. Higher values favor recently created/accessed memories. |
| `ENGRAM_WEIGHT_IMPORTANCE` | `float64` | `0.3` | Weight for user-assigned importance (1–10 scale, normalized). Higher values favor high-importance memories. |
| `ENGRAM_MMR_LAMBDA` | `float64` | `0.5` | Maximal Marginal Relevance (MMR) diversity factor. `0.0` = maximum diversity (results are as different from each other as possible). `1.0` = maximum relevance (standard similarity ranking, no diversity penalty). |

### Recency Decay

Recency uses per-type exponential decay factors (not currently configurable via env vars — hardcoded defaults):

| Memory Type | Decay Factor | Approximate Half-Life |
|-------------|-------------|----------------------|
| `identity` | `1.0` | Permanent (no decay) |
| `event` | `0.99` | ~3 days |
| `insight` | `0.9998` | ~90 days |
| `directive` | `1.0` | Permanent (no decay) |

### Usage Example

```bash
# Favor relevance heavily, ignore recency
export ENGRAM_WEIGHT_RELEVANCE=2.0
export ENGRAM_WEIGHT_RECENCY=0.0
export ENGRAM_WEIGHT_IMPORTANCE=0.5

# Balanced retrieval with high diversity
export ENGRAM_MMR_LAMBDA=0.3
```

---

## Deduplication

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENGRAM_DEDUP_THRESHOLD` | `float64` | `0.92` | Cosine similarity threshold for automatic deduplication. When a new memory's embedding is ≥ this threshold similar to an existing memory, the add is silently skipped. Range: `0.0` – `1.0`. |

### Tuning Guide

| Threshold | Behavior |
|-----------|----------|
| `0.95` – `1.0` | Very strict: only near-exact duplicates are caught |
| `0.90` – `0.95` | **Recommended range**. Catches paraphrases and minor rewording |
| `0.80` – `0.90` | Aggressive: may block semantically similar but distinct memories |
| `< 0.80` | Too aggressive — will likely cause data loss |

### Usage Example

```bash
# Strict dedup (only catch near-exact matches)
export ENGRAM_DEDUP_THRESHOLD=0.96

# Default (good balance)
export ENGRAM_DEDUP_THRESHOLD=0.92
```

---

## Server / Transport

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENGRAM_TRANSPORT` | `string` | `stdio` | Server transport mode. `stdio` = MCP over stdin/stdout (for MCP clients like Claude Desktop). `http` = REST API only. `both` = MCP + HTTP simultaneously. |
| `ENGRAM_HTTP_PORT` | `int` | `8080` | Port for the HTTP REST API. Only used when `ENGRAM_TRANSPORT` is `http` or `both`. |
| `ENGRAM_API_KEY` | `string` | _(empty)_ | API key for HTTP Bearer token authentication. All HTTP requests must include `Authorization: Bearer <key>`. If empty, HTTP auth is disabled. The `/health` endpoint always bypasses auth. |

### Transport Modes

**MCP stdio** (default — for AI agent integration):
```bash
export ENGRAM_TRANSPORT=stdio
# No port needed; communicates via stdin/stdout
```

**HTTP REST API** (for web services, scripts, or debugging):
```bash
export ENGRAM_TRANSPORT=http
export ENGRAM_HTTP_PORT=8080
export ENGRAM_API_KEY=my-secret-key
```

**Both** (MCP + HTTP simultaneously):
```bash
export ENGRAM_TRANSPORT=both
export ENGRAM_HTTP_PORT=8080
export ENGRAM_API_KEY=my-secret-key
```

### HTTP Endpoints

When HTTP transport is enabled:

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/reflect` | Yes | Run one Reflection Engine cycle. Body: `{"dry_run": true}` (optional) |
| `GET` | `/reflect/check` | Yes | Check reflection trigger conditions |
| `GET` | `/health` | **No** | Deep liveness check (pings Qdrant). Safe for load balancers / Kubernetes probes. |

---

## Reflection Engine

The Reflection Engine periodically synthesizes high-level insights from unreflected memories.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENGRAM_REFLECTION_ENABLED` | `bool` | `false` | Enable the Reflection Engine. When disabled, `reflection_run` MCP tool still works but the automatic trigger is off. |
| `ENGRAM_REFLECTION_TRIGGER` | `string` | `count` | Trigger mode. `count` = trigger when unreflected memory count reaches threshold. `cron` = time-based schedule. `manual` = only via explicit `reflection_run` calls. |
| `ENGRAM_REFLECTION_COUNT` | `int` | `10` | Minimum number of unreflected memories required to trigger reflection (only applies when `ENGRAM_REFLECTION_TRIGGER=count`). |
| `ENGRAM_REFLECTION_MODEL` | `string` | `claude-sonnet-4-20250514` | LLM model used for synthesis. Must be accessible via Anthropic API. |
| `ENGRAM_REFLECTION_MODE` | `string` | `v1` | Reflection algorithm. `v1` = flat synthesis (default). `v2` = 4-stage focal-point pipeline (focal selection → evidence gathering → dialectic → synthesis). See [`docs/reflection.md`](reflection.md). |
| `ENGRAM_LLM_MAX_TOKENS` | `int` | `1500` | Max output tokens per reflection/dream LLM call. |
| `ENGRAM_LLM_DIALECTIC_MAX_TOKENS` | `int` | `4000` | Max output tokens for the dialectic reflection stage (needs more headroom for reasoning tokens). |
| `ENGRAM_LLM_MAX_TOKENS_CEILING` | `int` | `8000` | Upper cap when a truncated (`finish_reason=length`) call is retried at double budget. |
| `ENGRAM_DIALECTIC_TIMEOUT` | `duration` | `45s` | V2 only. Per-call timeout for the dialectic LLM stage. Values above `90s` are clamped to `90s`. |

### Guardrails

Regardless of trigger mode, the Reflection Engine enforces:

- **Minimum interval**: 2 hours between runs
- **Daily limit**: Maximum 3 runs per calendar day (CST timezone)
- **Accumulated importance threshold**: Default 50 (sum of importance scores of unreflected memories)

### Usage Example

```bash
# Enable with count-based trigger
export ENGRAM_REFLECTION_ENABLED=true
export ENGRAM_REFLECTION_TRIGGER=count
export ENGRAM_REFLECTION_COUNT=15

# Use a lighter model for reflection
export ENGRAM_REFLECTION_MODEL=claude-haiku-4-20250514
```

---

## State Directory

The Reflection Engine trigger (`pkg/reflection/trigger.go`) and the dream-engine gate (`pkg/dream/gate.go`) persist their run-state (last-run timestamps, daily counters) on local disk so cadence survives restarts. Both resolve the directory through `pkg/statedir`.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENGRAM_STATE_DIR` | `string` | _(empty — see resolution order)_ | Explicit state directory. When set, the value is used **as-is** (no `.siri` suffix appended). Directory is created automatically (`0755`) if missing. Introduced in the 2026-09-04 hotfix (`a62c256`, "tolerate missing $HOME") so root/systemd units without `$HOME` still keep reflection & dream cadence. |

### Resolution Order

1. `$ENGRAM_STATE_DIR` — used directly as the state dir
2. `$SIRI_HOME` → `$SIRI_HOME/.siri`
3. `os.UserHomeDir()` → `$HOME/.siri`
4. Fallback → `/root/.siri` (logs a `[WARN]`)

Resolution never hard-fails because `$HOME` is missing; the only error path is a failed `MkdirAll`.

### Usage Example

```bash
# systemd unit without $HOME: pin state explicitly
export ENGRAM_STATE_DIR=/var/lib/engram/state
```

---

## Lifecycle & Admission (Evaporation / A-MAC / Write Checkpoints)

Three feature-flagged subsystems govern what gets *into* the store and how long it stays *relevant*.
All three default **off** (except the individual checkpoint sub-flags, which are only consulted when the
parent flag is on), so a fresh install behaves exactly like a plain vector store with dedup.

### Evaporation — runtime importance decay

Evaporation lowers a memory's stored `importance` over time (per-type half-life) and evicts it once
importance falls below the threshold. It is distinct from the *search-time* recency decay in
[Scoring & Retrieval](#scoring--retrieval): evaporation mutates the record, recency decay only affects ranking.
Recent reads boost the memory (`AccessBoostAlpha`), so memories that keep getting recalled survive.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENGRAM_EVAPORATION_ENABLED` | `bool` | `false` | Enable the background sweep. |
| `ENGRAM_EVAPORATION_HALF_LIFE_EVENT` | `float64` (days) | `30` | Half-life for `event` memories. |
| `ENGRAM_EVAPORATION_HALF_LIFE_INSIGHT` | `float64` (days) | `180` | Half-life for `insight` memories. |
| `ENGRAM_EVAPORATION_HALF_LIFE_DIRECTIVE` | `float64` (days) | `0` | Half-life for `directive` memories. `0` = never decays (v2: was `365`). |
| `ENGRAM_EVAPORATION_HALF_LIFE_IDENTITY` | `float64` (days) | `0` | Half-life for `identity` memories. `0` = never decays. |
| `ENGRAM_EVAPORATION_ACCESS_BOOST_ALPHA` | `float64` | `0.15` | Importance boost applied on each recall (counteracts decay for actively-used memories). |
| `ENGRAM_EVAPORATION_ACCESS_BOOST_MAX` | `float64` | `1.3` | Upper bound on the multiplicative access boost `1 + α·ln(1+access_count)`; caps how much frequent recall can inflate importance (default: at most +30%). `0` disables the cap (legacy unbounded). |
| `ENGRAM_EVAPORATION_EVICTION_THRESHOLD` | `float64` | `1.0` | Memories whose decayed importance drops below this are evicted by the sweep. |
| `ENGRAM_EVAPORATION_SWEEP_INTERVAL_H` | `int` (hours) | `6` | How often the sweep runs. |
| `ENGRAM_EVAPORATION_SWEEP_BATCH_LIMIT` | `int` | `100` | Max memories processed per sweep. |
| `ENGRAM_EVAPORATION_DRY_RUN` | `bool` | `true` | When true the sweep computes and reports candidates but performs **no** `store.Update` (observe-only, fail-safe). |
| `ENGRAM_EVAPORATION_DECAY_BASIS` | `string` | `last_access` | Reinforcement clock: `last_access` decays from `max(created_at, last_accessed_at)`; `created` is the legacy created-only basis. Invalid values fall back to `last_access`. |
| `ENGRAM_EVAPORATION_MIN_AGE_DAYS` | `float64` (days) | `14` | P8: never evaporate a memory younger than this, whatever the math says. |
| `ENGRAM_EVAPORATION_OBSERVATION_DAYS` | `float64` (days) | `30` | Window between soft-deprecate and hard-delete eligibility. |
| `ENGRAM_EVAPORATION_PROTECT_TYPES` | csv | `identity,directive` | P1: types that are structurally exempt from evaporation, regardless of half-life config. |
| `ENGRAM_EVAPORATION_PROTECT_IMPORTANCE` | `float64` | `8` | P2: memories at or above this importance are exempt. |
| `ENGRAM_EVAPORATION_PROTECT_ACCESS_COUNT` | `int` | `5` | P3: memories with at least this many accesses are exempt. |
| `ENGRAM_EVAPORATION_PROTECT_RECENT_ACCESS_DAYS` | `float64` (days) | `30` | P4: memories accessed within this window are exempt. |
| `ENGRAM_EVAPORATION_PROTECT_TAGS` | csv | `permanent,frank-feedback,directive,identity` | P5: memories carrying any of these tags are exempt. |
| `ENGRAM_EVAPORATION_PROTECT_CORROBORATED` | `bool` | `true` | P7: memories with a non-empty `provenance_history` (dedup merge target) are exempt. |

### A-MAC — type-aware admission control

A-MAC (Agent Memory Admission Control) replaces the single global dedup threshold with per-type
policy: per-type dedup thresholds, importance defaults and clamping bounds, and a 1-hour
sliding-window write rate limit per (collection, type). When `ENGRAM_AMAC_ENABLED=false` the
global `ENGRAM_DEDUP_THRESHOLD` and caller-supplied importance are used unchanged.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENGRAM_AMAC_ENABLED` | `bool` | `false` | Enable per-type admission policy. |
| `ENGRAM_DEDUP_THRESHOLD_IDENTITY` | `float64` | `0.95` | Per-type dedup threshold (protective — only near-identical merge). |
| `ENGRAM_DEDUP_THRESHOLD_DIRECTIVE` | `float64` | `0.90` | Per-type dedup threshold (aggressive — prevents directive pile-up). |
| `ENGRAM_DEDUP_THRESHOLD_INSIGHT` | `float64` | `0.92` | Per-type dedup threshold. |
| `ENGRAM_DEDUP_THRESHOLD_EVENT` | `float64` | `0.92` | Per-type dedup threshold. |
| `ENGRAM_IMPORTANCE_DEFAULT_{IDENTITY,DIRECTIVE,INSIGHT,EVENT}` | `float64` | `7` / `7` / `5` / `4` | Importance applied when the caller omits it. |
| `ENGRAM_IMPORTANCE_MIN_{TYPE}` / `ENGRAM_IMPORTANCE_MAX_{TYPE}` | `float64` | identity `7–9`, directive `6–10`, insight `5–8`, event `3–7` | Per-type importance bounds; out-of-range values are clamped silently. |
| `ENGRAM_RATE_LIMIT_{IDENTITY,DIRECTIVE,INSIGHT,EVENT}` | `int` (writes/hour) | `5` / `10` / `20` / `50` | Sliding-window write cap per (collection, type). Counters reset on restart. |

Precedence for dedup thresholds: per-type env `ENGRAM_DEDUP_THRESHOLD_{TYPE}` > global
`ENGRAM_DEDUP_THRESHOLD` > A-MAC built-in default.

### Write Checkpoints — advisory signals on the write path

Write Checkpoints attach non-blocking advisory warnings to `memory_add` responses (they never reject a
write). Four checkpoints exist; each has its own switch that is only consulted when the parent flag is on.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENGRAM_WRITE_CHECKPOINTS_ENABLED` | `bool` | `false` | Master switch for all checkpoints. |
| `ENGRAM_CP_DEDUP_ADVISORY_ENABLED` | `bool` | `true` | CP1: warn when a near-duplicate exists above `MIN_SCORE` but below the hard dedup threshold. |
| `ENGRAM_CP_DEDUP_ADVISORY_MIN_SCORE` | `float64` | `0.70` | CP1 lower bound of the advisory band. |
| `ENGRAM_CP_IMPORTANCE_MONITOR_ENABLED` | `bool` | `true` | CP2: warn on importance inflation (rolling average over the last N writes exceeds the threshold). |
| `ENGRAM_CP_IMPORTANCE_WINDOW` | `int` | `50` | CP2 rolling window size. |
| `ENGRAM_CP_IMPORTANCE_THRESHOLD` | `float64` | `7.0` | CP2 average-importance threshold. |
| `ENGRAM_CP_RATE_LIMIT_WARNING_ENABLED` | `bool` | `true` | CP3: warn when a (collection, type) approaches its A-MAC rate limit. |
| `ENGRAM_CP_RATE_LIMIT_WARNING_FRACTION` | `float64` | `0.80` | CP3 fraction of the limit at which to warn. |
| `ENGRAM_CP_CONTENT_CHECK_ENABLED` | `bool` | `true` | CP4: warn on content-shape issues (length, missing tags, missing `source_type`). |
| `ENGRAM_CP_CONTENT_MIN_LENGTH` | `int` | `20` | CP4 minimum content length. |
| `ENGRAM_CP_CONTENT_MAX_LENGTH` | `int` | `2000` | CP4 maximum content length. |
| `ENGRAM_CP_REQUIRE_TAGS` | `bool` | `true` | CP4 warn when `tags` is empty. |
| `ENGRAM_CP_REQUIRE_SOURCE_TYPE` | `bool` | `true` | CP4 warn when `source_type` is missing. |

### Usage Example

```bash
# Production-style lifecycle: all three on, shorter event half-life
export ENGRAM_EVAPORATION_ENABLED=true
export ENGRAM_EVAPORATION_HALF_LIFE_EVENT=14
export ENGRAM_AMAC_ENABLED=true
export ENGRAM_WRITE_CHECKPOINTS_ENABLED=true
export ENGRAM_REFLECTION_MODE=v2
```

---

## Observability (OpenTelemetry)

Engram uses OpenTelemetry for distributed tracing. Configured in `internal/otel/config.go`.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENGRAM_OTEL_ENABLED` | `bool` | `true` | Enable/disable OpenTelemetry tracing. Set to `false` to disable all trace collection. |
| `ENGRAM_OTEL_EXPORTER` | `string` | `file` | Trace exporter. `file` = daily-rotating JSONL files. `stdout` = print to stdout (useful for debugging). `none` = traces are generated but discarded. |
| `ENGRAM_OTEL_FILE_DIR` | `string` | `/tmp/siri-state/engram-traces` | Directory for JSONL trace files. Only used when `ENGRAM_OTEL_EXPORTER=file`. Directory is created automatically if it doesn't exist. |
| `ENGRAM_OTEL_FILE_ROTATION` | `string` | `daily` | File rotation strategy. `daily` = one file per day. `size` = rotate based on file size. |
| `ENGRAM_OTEL_SAMPLE_RATIO` | `float64` | `1.0` | Sampling ratio (`0.0` – `1.0`). `1.0` = trace every operation. `0.1` = trace 10% of operations. Lower values reduce I/O overhead in high-throughput deployments. |

### Instrumented Spans

| Span Name | Description | Key Attributes |
|-----------|-------------|----------------|
| `engram.memory.search` | Memory search operation | `query.length`, `tags.count`, `limit`, `result.count`, `latency_ms`, `embedder.provider` |
| `engram.memory.add` | Memory add operation | `content.length`, `tags.count`, `type`, `importance`, `dedup.hit` |
| `engram.memory.dedup_check` | Dedup check (child of `add`) | `query.length`, `threshold`, `top_score`, `decision` |
| `engram.reflection.run` | Reflection Engine cycle | `engram.memory.valid_until_set`, `engram.memory.valid_until` |

### Usage Example

```bash
# Disable tracing entirely (production, minimal overhead)
export ENGRAM_OTEL_ENABLED=false

# Debug: print traces to stdout
export ENGRAM_OTEL_EXPORTER=stdout

# Custom trace directory with 50% sampling
export ENGRAM_OTEL_FILE_DIR=/var/log/engram/traces
export ENGRAM_OTEL_SAMPLE_RATIO=0.5
```

---

## TTL Auto-Calculator

When `valid_until` is not explicitly set on `memory_add` or `memory_update`, Engram automatically computes a TTL based on a **type × importance** matrix:

| Type | Importance < 5 | Importance 5–7 | Importance ≥ 8 |
|------|---------------|----------------|----------------|
| `identity` | Permanent | Permanent | Permanent |
| `directive` | 90 days | Permanent | Permanent |
| `insight` | 30 days | 90 days | Permanent |
| `event` | 3 days | 7 days | 30 days |

### Special Tag Overrides

| Tag | Effect |
|-----|--------|
| `permanent` | Memory never expires (overrides TTL matrix) |
| `time-sensitive` | Forces max 7-day TTL (unless matrix gives shorter) |
| `location` | Same as `time-sensitive` — forces max 7-day TTL |

### Precedence

1. **Explicit `valid_until`** (caller-provided) → always wins
2. **`permanent` tag** → never expires
3. **`time-sensitive` / `location` tag** → cap at 7 days
4. **TTL matrix** (type × importance) → default calculation

> The TTL matrix is not currently configurable via environment variables. To customize, modify `DefaultTTLConfig()` in `pkg/memory/ttl.go`.

---

## Multi-Collection Architecture

Engram uses three hardcoded Qdrant collections to isolate writes from different caller types:

| Collection | Caller Type (HTTP header `X-Caller-Type`) | Purpose |
|-----------|------------------------------------------|---------|
| `engram_user` | `user` (default) | User-facing memories — the primary store |
| `engram_agent_self` | `agent-self` | Agent self-reflection and internal state |
| `engram_reflection` | `reflection` | Reflection Engine outputs |

Collections are registered at startup via `pkg/collection/registry.go`. The `X-Caller-Type` header is resolved to the target collection; unknown or empty values default to `engram_user`.

> **Note**: The `ENGRAM_COLLECTION_NAME` environment variable has been **removed** as of the multi-collection migration. See [Deprecated Variables](#deprecated-variables).

---

## Deprecated Variables

| Variable | Status | Replacement |
|----------|--------|-------------|
| `ENGRAM_COLLECTION_NAME` | **Removed** | Multi-collection architecture uses three hardcoded collection names (`engram_user`, `engram_agent_self`, `engram_reflection`). The variable is still referenced in `docker-compose.yml` and `integration_test.sh` for backward compatibility but is ignored by the Go binary. |

---

## Quick-Start Examples

### Minimal (MCP stdio with OpenAI)

```bash
export ENGRAM_OPENAI_API_KEY=sk-...
./engram serve
```

### Production (HTTP + MCP, with auth and tracing)

```bash
export ENGRAM_OPENAI_API_KEY=sk-...
export ENGRAM_QDRANT_URL=qdrant.internal:6334
export ENGRAM_QDRANT_API_KEY=qdrant-secret
export ENGRAM_QDRANT_USE_TLS=true
export ENGRAM_TRANSPORT=both
export ENGRAM_HTTP_PORT=8080
export ENGRAM_API_KEY=engram-bearer-token
export ENGRAM_REFLECTION_ENABLED=true
export ENGRAM_REFLECTION_MODEL=claude-sonnet-4-20250514
export ENGRAM_OTEL_FILE_DIR=/var/log/engram/traces
export ENGRAM_OTEL_SAMPLE_RATIO=0.5
./engram serve
```

### Docker Compose

```bash
# .env file
ENGRAM_OPENAI_API_KEY=sk-...
ENGRAM_REFLECTION_ENABLED=true

docker-compose up -d
```

### Development

```bash
export ENGRAM_OPENAI_API_KEY=sk-...
export ENGRAM_TRANSPORT=http
export ENGRAM_HTTP_PORT=9090
export ENGRAM_OTEL_EXPORTER=stdout
export ENGRAM_DEDUP_THRESHOLD=0.85  # more aggressive dedup for testing
./engram serve
```

---

## Complete Variable Reference

| # | Variable | Type | Default | Section |
|---|----------|------|---------|---------|
| 1 | `ENGRAM_QDRANT_URL` | string | `localhost:6334` | Storage |
| 2 | `ENGRAM_QDRANT_API_KEY` | string | _(empty)_ | Storage |
| 3 | `ENGRAM_QDRANT_USE_TLS` | bool | `false` | Storage |
| 4 | `ENGRAM_EMBEDDER_PROVIDER` | string | `openai` | Embedding |
| 5 | `ENGRAM_EMBEDDING_MODEL` | string | `text-embedding-3-small` | Embedding |
| 6 | `ENGRAM_EMBEDDING_DIMENSION` | int | `1536` | Embedding |
| 7 | `ENGRAM_OPENAI_API_KEY` | string | _(empty)_ | Embedding |
| 8 | `ENGRAM_OPENAI_BASE_URL` | string | `https://api.openai.com/v1` | Embedding |
| 9 | `ENGRAM_VOYAGE_API_KEY` | string | _(empty)_ | Embedding |
| 10 | `ENGRAM_WEIGHT_RELEVANCE` | float64 | `1.0` | Scoring |
| 11 | `ENGRAM_WEIGHT_RECENCY` | float64 | `0.5` | Scoring |
| 12 | `ENGRAM_WEIGHT_IMPORTANCE` | float64 | `0.3` | Scoring |
| 13 | `ENGRAM_MMR_LAMBDA` | float64 | `0.5` | Scoring |
| 14 | `ENGRAM_DEDUP_THRESHOLD` | float64 | `0.92` | Deduplication |
| 15 | `ENGRAM_TRANSPORT` | string | `stdio` | Server |
| 16 | `ENGRAM_HTTP_PORT` | int | `8080` | Server |
| 17 | `ENGRAM_API_KEY` | string | _(empty)_ | Server |
| 18 | `ENGRAM_REFLECTION_ENABLED` | bool | `false` | Reflection |
| 19 | `ENGRAM_REFLECTION_TRIGGER` | string | `count` | Reflection |
| 20 | `ENGRAM_REFLECTION_COUNT` | int | `10` | Reflection |
| 21 | `ENGRAM_REFLECTION_MODEL` | string | `claude-sonnet-4-20250514` | Reflection |
| 22 | `ENGRAM_OTEL_ENABLED` | bool | `true` | Observability |
| 23 | `ENGRAM_OTEL_EXPORTER` | string | `file` | Observability |
| 24 | `ENGRAM_OTEL_FILE_DIR` | string | `/tmp/siri-state/engram-traces` | Observability |
| 25 | `ENGRAM_OTEL_FILE_ROTATION` | string | `daily` | Observability |
| 26 | `ENGRAM_OTEL_SAMPLE_RATIO` | float64 | `1.0` | Observability |
| 27 | `ENGRAM_REFLECTION_MODE` | string | `v1` | Reflection |
| 28 | `ENGRAM_DIALECTIC_TIMEOUT` | duration | `45s` | Reflection |
| 29 | `ENGRAM_EVAPORATION_ENABLED` | bool | `false` | Lifecycle |
| 30 | `ENGRAM_EVAPORATION_HALF_LIFE_{EVENT,INSIGHT,DIRECTIVE,IDENTITY}` | float64 | `30`/`180`/`365`/`0` | Lifecycle |
| 31 | `ENGRAM_EVAPORATION_ACCESS_BOOST_ALPHA` | float64 | `0.15` | Lifecycle |
| 32 | `ENGRAM_EVAPORATION_EVICTION_THRESHOLD` | float64 | `1.0` | Lifecycle |
| 33 | `ENGRAM_EVAPORATION_SWEEP_INTERVAL_H` | int | `6` | Lifecycle |
| 34 | `ENGRAM_EVAPORATION_SWEEP_BATCH_LIMIT` | int | `100` | Lifecycle |
| 35 | `ENGRAM_AMAC_ENABLED` | bool | `false` | Lifecycle |
| 36 | `ENGRAM_DEDUP_THRESHOLD_{IDENTITY,DIRECTIVE,INSIGHT,EVENT}` | float64 | `0.95`/`0.90`/`0.92`/`0.92` | Lifecycle |
| 37 | `ENGRAM_IMPORTANCE_DEFAULT_{TYPE}` | float64 | `7`/`7`/`5`/`4` | Lifecycle |
| 38 | `ENGRAM_IMPORTANCE_MIN_{TYPE}` / `ENGRAM_IMPORTANCE_MAX_{TYPE}` | float64 | per-type | Lifecycle |
| 39 | `ENGRAM_RATE_LIMIT_{IDENTITY,DIRECTIVE,INSIGHT,EVENT}` | int | `5`/`10`/`20`/`50` | Lifecycle |
| 40 | `ENGRAM_WRITE_CHECKPOINTS_ENABLED` | bool | `false` | Lifecycle |
| 41 | `ENGRAM_CP_*` (12 sub-flags, see section) | mixed | see section | Lifecycle |
| 42 | `ENGRAM_STATE_DIR` | string | _(empty; falls back to `$SIRI_HOME/.siri` > `$HOME/.siri` > `/root/.siri`)_ | State Directory |
| 43 | `ENGRAM_EVAPORATION_DRY_RUN` | bool | `true` | Lifecycle |
| 44 | `ENGRAM_EVAPORATION_DECAY_BASIS` | string | `last_access` | Lifecycle |
| 45 | `ENGRAM_EVAPORATION_MIN_AGE_DAYS` | float64 | `14` | Lifecycle |
| 46 | `ENGRAM_EVAPORATION_OBSERVATION_DAYS` | float64 | `30` | Lifecycle |
| 47 | `ENGRAM_EVAPORATION_PROTECT_TYPES` | csv | `identity,directive` | Lifecycle |
| 48 | `ENGRAM_EVAPORATION_PROTECT_IMPORTANCE` | float64 | `8` | Lifecycle |
| 49 | `ENGRAM_EVAPORATION_PROTECT_ACCESS_COUNT` | int | `5` | Lifecycle |
| 50 | `ENGRAM_EVAPORATION_PROTECT_RECENT_ACCESS_DAYS` | float64 | `30` | Lifecycle |
| 51 | `ENGRAM_EVAPORATION_PROTECT_TAGS` | csv | `permanent,frank-feedback,directive,identity` | Lifecycle |
| 52 | `ENGRAM_EVAPORATION_PROTECT_CORROBORATED` | bool | `true` | Lifecycle |
| 53 | `ENGRAM_EVAPORATION_ACCESS_BOOST_MAX` | float64 | `1.3` | Lifecycle |
