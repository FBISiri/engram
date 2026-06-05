# Engram

Long-term memory for AI agents. Simple, fast, and storage-agnostic.

Engram provides a vector-based memory system that lets AI agents store, retrieve, update, and delete memories with semantic search. It also includes two autonomous cognitive engines — **Reflection** and **Dream** — that synthesize higher-order insights from raw memories. Designed for use via [MCP](https://modelcontextprotocol.io/) (Model Context Protocol) or REST API.

## Features

- **4 memory types** — `identity`, `event`, `insight`, `directive` (+ free-form tags for specificity)
- **Three-component scoring** — Relevance × Recency × Importance for intelligent retrieval
- **MMR reranking** — Balances relevance with diversity in results
- **Automatic deduplication** — Prevents storing near-identical memories (configurable threshold)
- **Memory expiry** — Optional `valid_until` timestamp for auto-cleanup of time-bound memories
- **Reflection Engine** — Lightweight, frequent synthesis (1–3×/day) that discovers cross-domain patterns from unreflected memories
- **Dream Engine** — Deep consolidation (1×/day) with 4-phase pipeline: Orient → Gather → Consolidate → Prune
- **Storage-agnostic** — Qdrant backend with pluggable interface for others
- **Embedding-agnostic** — OpenAI (default) or Voyage AI, bring your own embedder
- **Dual transport** — MCP (stdio) and HTTP REST API with Bearer-token auth
- **Write-through + Ring Buffer** — BoltDB-backed commit log for durability and crash recovery
- **No LLM in hot path** — Store and retrieve are pure vector operations. Reflection and Dream are optional and async.

## Quick Start

### Option A: Docker Compose (recommended)

```bash
# Clone the repo
git clone https://github.com/FBISiri/engram.git
cd engram

# Create .env file
cat > .env << EOF
ENGRAM_OPENAI_API_KEY=sk-...
EOF

# Start Qdrant + build Engram image
docker-compose up -d qdrant

# Wait for Qdrant to be healthy, then run Engram interactively (MCP stdio)
docker-compose run --rm engram serve
```

### Option B: Binary + Docker Qdrant

```bash
# 1. Start Qdrant
docker run -d --name engram-qdrant \
  --security-opt seccomp=unconfined \
  -p 6333:6333 -p 6334:6334 \
  -v engram_qdrant_data:/qdrant/storage \
  qdrant/qdrant:v1.9.7

# 2. Build Engram
go build -o engram ./cmd/engram/

# 3. Configure
export ENGRAM_QDRANT_URL=localhost:6334
export ENGRAM_OPENAI_API_KEY=sk-...

# 4. Run as MCP server (stdio)
./engram serve
```

### Option C: Go Install

```bash
# Prerequisites: Qdrant running on localhost:6334
go install github.com/FBISiri/engram/cmd/engram@latest

export ENGRAM_QDRANT_URL=localhost:6334
export ENGRAM_OPENAI_API_KEY=sk-...

engram serve
```

### Using with MCP Clients

Add Engram to your MCP client config (e.g., Claude Desktop, Army of the Agent):

```json
{
  "mcpServers": {
    "engram": {
      "command": "/path/to/engram",
      "args": ["serve"],
      "env": {
        "ENGRAM_QDRANT_URL": "localhost:6334",
        "ENGRAM_OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

Or in YAML (Army of the Agent):

```yaml
mcp_servers:
  - name: "engram"
    transport: stdio
    command: /path/to/engram
    args: ["serve"]
    env:
      ENGRAM_QDRANT_URL: "localhost:6334"
      ENGRAM_OPENAI_API_KEY: "sk-..."
```

## Integration Test

Run the full end-to-end integration test against a live Qdrant instance:

```bash
# Ensure Qdrant is running on localhost:6333/6334
ENGRAM_OPENAI_API_KEY=sk-... ./integration_test.sh
```

This tests all 6 MCP tools (search, add, update, delete, reflection_check, reflection_run) including dedup detection.

## Memory Types

| Type | Description | Decay | Examples |
|------|-------------|-------|----------|
| `identity` | Stable facts about the user | Permanent | Name, job, preferences, relationships |
| `event` | Something that happened | ~3 day half-life | Conversations, observations, plans |
| `insight` | Inferred patterns | ~90 day half-life | Reflections, behavioral tendencies |
| `directive` | Explicit user instructions | Permanent | "Always respond in Chinese" |

Use **tags** for further classification: `["relationship", "person:Alice"]`, `["study", "golang"]`, `["preference", "food"]`.

### Memory Expiry

Memories can have an optional `valid_until` field (Unix timestamp). A background goroutine runs every 10 minutes to clean up expired memories automatically. Set `valid_until` to `0` or omit it for memories that never expire.

### TTL Auto-Calculator

When `valid_until` is not explicitly set on `memory_add` / `memory_update`, Engram derives a sensible TTL from a **type × importance** matrix (e.g., low-importance `event` memories expire within days, high-importance `insight` memories within months, `identity` / `directive` types never expire by default). Explicit `valid_until` values always win — the auto-calculator only fills in when the caller omits it.

## API

### MCP Tools

| Tool | Description |
|------|-------------|
| `memory_search` | Semantic search with type/tag/time filters |
| `memory_add` | Store a memory (auto-deduplicates) |
| `memory_update` | Find old memories by meaning → replace with new |
| `memory_delete` | Find memories by meaning → delete |
| `reflection_check` | Check Reflection Engine trigger conditions without running |
| `reflection_run` | Run one Reflection Engine cycle (supports `dry_run` mode) |

### Transport Modes: stdio vs HTTP

Engram supports two transport modes, configurable via `ENGRAM_TRANSPORT`:

| Mode | Value | Use Case | How It Works |
|------|-------|----------|-------------|
| **MCP stdio** | `stdio` (default) | MCP clients (Claude Desktop, Army of the Agent) | Reads JSON-RPC from stdin, writes to stdout. The MCP client manages the process lifecycle. |
| **HTTP REST** | `http` | Direct API integration, dashboards, scripts | Starts an HTTP server on `ENGRAM_HTTP_PORT` (default 8080). RESTful JSON API with Bearer auth. |
| **Both** | `both` | Run MCP + REST simultaneously | Useful when you need MCP for an agent AND REST for monitoring/admin. |

**When to use stdio**: You're integrating with an MCP-compatible client. The client spawns Engram as a subprocess and communicates via stdin/stdout. Zero network config needed.

**When to use HTTP**: You need programmatic access from scripts, dashboards, or non-MCP clients. Also required for the `/health` endpoint used by container orchestrators (Kubernetes, Docker health checks).

```bash
# stdio mode (default) — for MCP clients
./engram serve

# HTTP mode — for REST API access
ENGRAM_TRANSPORT=http ENGRAM_API_KEY=your-secret ./engram serve

# Both — MCP + REST simultaneously
ENGRAM_TRANSPORT=both ENGRAM_API_KEY=your-secret ./engram serve
```

### REST API (HTTP Transport)

Enable with `ENGRAM_TRANSPORT=http` or `ENGRAM_TRANSPORT=both` (MCP + HTTP).

Authentication: set `ENGRAM_API_KEY` to require `Authorization: Bearer <key>` on all HTTP requests. The `/health` and `/metrics` endpoints bypass auth for monitoring.

**Core Memory Operations**

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/memories` | Store a new memory (auto-deduplicates) |
| `POST` | `/memories/search` | Semantic search with filters |
| `GET` | `/memories/{id}` | Get a specific memory by ID |
| `PATCH` | `/memories/{id}` | Partial update of a memory |
| `PUT` | `/memories/{id}` | Full replace of a memory |
| `DELETE` | `/memories/{id}` | Delete a memory |
| `POST` | `/memories/{id}/reset` | Reset memory access stats |
| `POST` | `/memories/cross-search` | Search across multiple collections |

**Collection Management**

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/collections` | Create a new collection |
| `GET` | `/collections` | List all collections |
| `POST` | `/collections/{name}/memories` | Add memory to a specific collection |
| `POST` | `/collections/{name}/memories/search` | Search within a specific collection |

**Reflection & Maintenance**

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reflect` | Run one Reflection Engine cycle (optional: `{"dry_run": true}`) |
| `GET` | `/reflect/check` | Check reflection trigger conditions |
| `GET` | `/memories/expiry-candidates` | List memories approaching expiration |
| `DELETE` | `/memories/expired` | Purge expired memories |

**Observability**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Deep liveness — pings Qdrant (no auth required) |
| `GET` | `/metrics` | Prometheus metrics (no auth required) |

## Reflection Engine

The Reflection Engine periodically synthesizes high-level insights from unreflected memories, inspired by the *Generative Agents* paper. It runs lighter and more frequently than the Dream Engine.

**Trigger**: accumulated importance of unreflected memories ≥ threshold (default: 40). Min interval: 2 hours. Max per day: 3 runs.

**Output**: `insight`-type memories with `source="system"`, tagged with reflection source IDs. Source memories are marked as `reflected=true`.

Configure via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGRAM_REFLECTION_ENABLED` | `false` | Enable the Reflection Engine |
| `ENGRAM_REFLECTION_TRIGGER` | `count` | Trigger mode: `count`, `cron`, `manual` |
| `ENGRAM_REFLECTION_COUNT` | `10` | Min unreflected memories to trigger |
| `ENGRAM_REFLECTION_MODEL` | `claude-sonnet-4-20250514` | LLM model for synthesis |

## Dream Engine

The Dream Engine performs deep memory consolidation — autonomous insight generation with a 4-phase pipeline:

1. **Orient** — Assess current memory landscape, identify clusters
2. **Gather** — Collect related memories for consolidation candidates
3. **Consolidate** — Merge redundant/related memories into higher-order insights
4. **Prune** — Remove superseded memories to keep the store clean

**Gate conditions** (all must pass):
- **Gate 1 (Time)**: ≥ 20 hours since last run
- **Gate 2 (Volume)**: ≥ 20 new memories since last run
- **Gate 3 (PID)**: No other dream process currently running (stale PID timeout: 2 hours)

State is persisted in `~/.siri/` (last run timestamp, PID lock file).

## Scoring

```
score = 1.0 × relevance + 0.5 × recency + 0.3 × importance
```

- **Relevance**: Cosine similarity between query and memory embeddings
- **Recency**: Exponential decay based on memory type (configurable)
- **Importance**: User-assigned 1–10 scale, normalized

Weights are configurable via `ENGRAM_WEIGHT_RELEVANCE`, `ENGRAM_WEIGHT_RECENCY`, `ENGRAM_WEIGHT_IMPORTANCE`.

## Configuration

All via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGRAM_QDRANT_URL` | `localhost:6334` | Qdrant gRPC address |
| `ENGRAM_QDRANT_API_KEY` | — | Qdrant API key (if secured) |
| `ENGRAM_EMBEDDER_PROVIDER` | `openai` | Embedding provider: `openai` or `voyage` |
| `ENGRAM_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model name |
| `ENGRAM_EMBEDDING_DIMENSION` | `1536` | Embedding vector size |
| `ENGRAM_OPENAI_API_KEY` | — | OpenAI API key |
| `ENGRAM_OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI API base URL |
| `ENGRAM_VOYAGE_API_KEY` | — | Voyage AI API key |
| `ENGRAM_DEDUP_THRESHOLD` | `0.92` | Cosine similarity for dedup |
| `ENGRAM_MMR_LAMBDA` | `0.5` | MMR diversity factor (0=max diversity, 1=max relevance) |
| `ENGRAM_WEIGHT_RELEVANCE` | `1.0` | Scoring weight for relevance |
| `ENGRAM_WEIGHT_RECENCY` | `0.5` | Scoring weight for recency |
| `ENGRAM_WEIGHT_IMPORTANCE` | `0.3` | Scoring weight for importance |
| `ENGRAM_TRANSPORT` | `stdio` | Server transport: `stdio`, `http`, `both` |
| `ENGRAM_HTTP_PORT` | `8080` | REST API port |
| `ENGRAM_API_KEY` | — | API key for HTTP Bearer auth |
| `ENGRAM_REFLECTION_ENABLED` | `false` | Enable Reflection Engine |
| `ENGRAM_REFLECTION_TRIGGER` | `count` | Reflection trigger mode |
| `ENGRAM_REFLECTION_COUNT` | `10` | Min unreflected memories to trigger |
| `ENGRAM_REFLECTION_MODEL` | `claude-sonnet-4-20250514` | LLM model for reflection |

See [full configuration reference](docs/configuration.md) for all options.

## Architecture

```
engram/
├── cmd/engram/          CLI entry point
├── internal/
│   └── otel/            OpenTelemetry tracing (file/stdout exporters)
├── pkg/
│   ├── collection/      Multi-collection registry (user, agent_self, reflection)
│   ├── config/          Configuration from env vars
│   ├── dream/           Dream Engine — deep 4-phase memory consolidation
│   ├── embedding/       Embedder interface + OpenAI + Voyage AI + LRU cache
│   ├── memory/          Core types, scoring, dedup, MMR, expiry, TTL
│   ├── metrics/         Prometheus metrics
│   ├── qdrant/          Qdrant Store + MultiStore implementation
│   ├── reflection/      Reflection Engine — V1 (flat) + V2 (focal point)
│   ├── server/          MCP server (stdio) + HTTP server (REST + CRUD)
│   └── sync/            Write-through + Ring Buffer (BoltDB commit log)
├── docs/                API reference + configuration guide
├── Dockerfile           Multi-stage build
├── docker-compose.yml   Engram + Qdrant
└── integration_test.sh  End-to-end MCP test
```

## Observability

Engram uses OpenTelemetry for distributed tracing. Spans are exported to daily-rotating JSONL files by default.

### Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `ENGRAM_OTEL_ENABLED` | `true` | Enable/disable tracing |
| `ENGRAM_OTEL_EXPORTER` | `file` | Exporter: `file`, `stdout`, `none` |
| `ENGRAM_OTEL_FILE_DIR` | `/tmp/siri-state/engram-traces` | Directory for JSONL trace files |
| `ENGRAM_OTEL_SAMPLE_RATIO` | `1.0` | Sampling ratio (0–1) |

### §1 Reflection instrumentation (R-S3)

| Span name | Entry point | Attributes |
|-----------|-------------|------------|
| `engram.reflection.run` | `pkg/reflection/engine.go::Run` | `engram.memory.valid_until_set` (bool), `engram.memory.valid_until` (string, RFC 3339) |

### §2 Reflection V2 instrumentation (R-S3)

Same span name `engram.reflection.run` emitted by `RunV2` and `RunSingleEvent`.

### §3 Memory instrumentation (R-S5)

| Span name | Entry point | Attributes |
|-----------|-------------|------------|
| `engram.memory.search` | `pkg/server/server.go::handleSearch` | `query.length`, `tags.count`, `limit`, `result.count`, `latency_ms`, `embedder.provider` |
| `engram.memory.add` | `pkg/server/server.go::handleAdd` | `content.length`, `tags.count`, `type`, `importance`, `dedup.hit` (bool) |
| `engram.memory.dedup_check` | `pkg/server/server.go::checkDedup` | `query.length`, `threshold`, `top_score`, `decision` (add/skip), `latency_ms` |

Error paths set `span.RecordError(err)` + `span.SetStatus(codes.Error, ...)`.

`engram.memory.dedup_check` is a child span of `engram.memory.add`.

## Background

Engram (noun): *The hypothetical physical or biochemical change in neural tissue that represents a memory.* — from neuroscience.

This project was born from operating [chat2mem](https://github.com/FBISiri/chat2mem) (a Python MCP memory server) in production for months. Engram is a ground-up redesign in Go, informed by real-world agent memory usage patterns and research from systems like Graphiti, Hindsight, MemOS, and Mem0.

## License

MIT
