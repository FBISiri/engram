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

```bash
# Prerequisites: Qdrant running locally
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Install
go install github.com/anthropics/engram/cmd/engram@latest

# Configure
export ENGRAM_QDRANT_URL=localhost:6334
export ENGRAM_OPENAI_API_KEY=sk-...

# Run as MCP server (stdio)
engram serve --transport stdio

# Or as REST API
engram serve --transport http --port 8080
```

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
| `memory.search` | Semantic search with type/tag/time filters |
| `memory.add` | Store one or more memories (batch supported) |
| `memory.update` | Find old memories by meaning → replace with new |
| `memory.delete` | Find memories by meaning → delete |

### REST API

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
| `ENGRAM_OPENAI_API_KEY` | — | OpenAI API key |
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
│   ├── memory/          Core types, scoring, dedup
│   ├── embedding/       Embedder interface + OpenAI
│   ├── qdrant/          Qdrant Store implementation
│   ├── server/          MCP + REST servers
│   ├── reflection/      Optional reflection engine
│   └── config/          Configuration
```

## Background

Engram (noun): *The hypothetical physical or biochemical change in neural tissue that represents a memory.* — from neuroscience.

This project was born from operating [chat2mem](https://github.com/anthropics/chat2mem) (a Python MCP memory server) in production for months. Engram is a ground-up redesign in Go, informed by real-world agent memory usage patterns and research from systems like Graphiti, Hindsight, MemOS, and Mem0.

## License

MIT
