# External Agent Quickstart (REST)

If your agent isn't an MCP client — a custom Python service, another framework, or anything
that speaks HTTP — you can use Engram as a memory backend over its REST transport instead of
running `engram serve` as a local MCP stdio process.

## 1 — Start Engram with the HTTP transport

Follow the [main Quickstart](../README.md#quickstart) to get Qdrant and Engram running, then
enable the HTTP transport and set an API key:

```bash
export ENGRAM_QDRANT_URL=localhost:6334
export ENGRAM_OPENAI_API_KEY=sk-...
export ENGRAM_TRANSPORT=http           # or "both" to serve MCP stdio + REST together
export ENGRAM_API_KEY=my-secret-key    # required for all endpoints except /health
./engram serve
```

Engram listens on `:8080` by default.

## 2 — Health check (no auth)

```bash
curl http://localhost:8080/health
```

`/health` performs a deep liveness check (it pings Qdrant) and needs no auth — good for
Kubernetes probes.

## 3 — Store a memory

```bash
curl -X POST http://localhost:8080/memories \
  -H "Authorization: Bearer my-secret-key" -H "Content-Type: application/json" \
  -d '{
        "content": "Frank prefers concise commit messages, imperative mood, no emoji.",
        "type": "identity",
        "importance": 8,
        "tags": ["frank", "preference"]
      }'
```

## 4 — Search it back

```bash
curl -X POST http://localhost:8080/memories/search \
  -H "Authorization: Bearer my-secret-key" -H "Content-Type: application/json" \
  -d '{"query": "how does Frank like commit messages", "limit": 3}'
```

Results come back ranked by the fused `score` (`1.0×relevance + 0.5×recency +
0.3×importance`), not raw cosine — same ranking as the MCP `memory_search` tool.

## Endpoint cheat sheet

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | deep liveness (no auth) |
| `POST` | `/memories` | create (≡ `memory_add`) |
| `GET`/`PATCH`/`PUT`/`DELETE` | `/memories/{id}` | read / partial update / replace / delete |
| `POST` | `/memories/search` | semantic search (≡ `memory_search`) |
| `POST` | `/memories/cross-search` | search across collections and merge |
| `POST` / `GET` | `/collections` | create / list collections |
| `POST` / `GET` | `/reflect`, `/reflect/check` | trigger / check reflection |
| `GET` / `DELETE` | `/memories/expiry-candidates`, `/memories/expired` | TTL management |
| `GET` | `/metrics` | Prometheus metrics |

All endpoints except `/health` require `Authorization: Bearer <ENGRAM_API_KEY>`. Full request
and response bodies are in [`api.md`](api.md).
