# Engram

Long-term memory for AI agents. Simple, fast, and storage-agnostic.

Engram provides a vector-based memory system that lets AI agents store, retrieve, update, and delete memories with semantic search. Designed for use via [MCP](https://modelcontextprotocol.io/) (Model Context Protocol) or REST API.

## Features

- **4 memory types** — `identity`, `event`, `insight`, `directive` (+ free-form tags for specificity)
- **Three-component scoring** — Relevance × Recency × Importance for intelligent retrieval
- **MMR reranking** — Balances relevance with diversity in results
- **Automatic deduplication** — Prevents storing near-identical memories
- **Storage-agnostic** — Qdrant backend with pluggable interface for others
- **Embedding-agnostic** — OpenAI default, bring your own embedder
- **Dual transport** — MCP (stdio/HTTP) and REST API
- **No LLM in hot path** — Store and retrieve are pure vector operations. Reflection is optional and async.

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

This tests all 4 MCP tools (search, add, update, delete) including dedup detection.

## Memory Types

| Type | Description | Decay | Examples |
|------|-------------|-------|----------|
| `identity` | Stable facts about the user | Permanent | Name, job, preferences, relationships |
| `event` | Something that happened | ~3 day half-life | Conversations, observations, plans |
| `insight` | Inferred patterns | ~90 day half-life | Reflections, behavioral tendencies |
| `directive` | Explicit user instructions | Permanent | "Always respond in Chinese" |

Use **tags** for further classification: `["relationship", "person:Alice"]`, `["study", "golang"]`, `["preference", "food"]`.

## API

### MCP Tools

| Tool | Description |
|------|-------------|
| `memory_search` | Semantic search with type/tag/time filters |
| `memory_add` | Store a memory (auto-deduplicates) |
| `memory_update` | Find old memories by meaning → replace with new |
| `memory_delete` | Find memories by meaning → delete |

### REST API (planned)

```
POST   /v1/memory/search    Search memories
POST   /v1/memory           Add memories
PUT    /v1/memory           Update memories
DELETE /v1/memory           Delete memories
GET    /v1/health           Health check
GET    /v1/stats            Collection stats
```

## Scoring

```
score = 1.0 × relevance + 0.5 × recency + 0.3 × importance
```

- **Relevance**: Cosine similarity between query and memory embeddings
- **Recency**: Exponential decay based on memory type (configurable)
- **Importance**: User-assigned 1-10 scale, normalized

## Configuration

All via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGRAM_QDRANT_URL` | `localhost:6334` | Qdrant gRPC address |
| `ENGRAM_COLLECTION_NAME` | `engram` | Qdrant collection name |
| `ENGRAM_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `ENGRAM_EMBEDDING_DIMENSION` | `1536` | Embedding vector size |
| `ENGRAM_OPENAI_API_KEY` | — | OpenAI API key |
| `ENGRAM_OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI API base URL |
| `ENGRAM_DEDUP_THRESHOLD` | `0.92` | Cosine similarity for dedup |
| `ENGRAM_TRANSPORT` | `stdio` | Server transport: stdio, http, both |
| `ENGRAM_HTTP_PORT` | `8080` | REST API port |
| `ENGRAM_API_KEY` | — | API key for HTTP auth |

See [full configuration reference](docs/configuration.md) for all options.

## Architecture

```
engram/
├── cmd/engram/          CLI entry point
├── pkg/
│   ├── memory/          Core types, scoring, dedup, MMR
│   ├── embedding/       Embedder interface + OpenAI + Voyage AI
│   ├── qdrant/          Qdrant Store implementation
│   ├── server/          MCP server (stdio transport)
│   ├── reflection/      Optional reflection engine
│   └── config/          Configuration from env vars
├── Dockerfile           Multi-stage build
├── docker-compose.yml   Engram + Qdrant
└── integration_test.sh  End-to-end MCP test
```

## Background

Engram (noun): *The hypothetical physical or biochemical change in neural tissue that represents a memory.* — from neuroscience.

This project was born from operating [chat2mem](https://github.com/FBISiri/chat2mem) (a Python MCP memory server) in production for months. Engram is a ground-up redesign in Go, informed by real-world agent memory usage patterns and research from systems like Graphiti, Hindsight, MemOS, and Mem0.

## License

MIT
