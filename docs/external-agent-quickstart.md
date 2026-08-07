# External Agent Quickstart

> **Goal:** From zero to "store a memory → search it back → see the fused ranking" in **5 minutes**.
>
> This guide is for agents and services that talk HTTP — not MCP stdio.
> If you're an MCP client (Claude Desktop, Army of the Agent), see the [main Quickstart](https://github.com/FBISiri/engram#quickstart).

---

## Prerequisites

- **Docker** (to run Qdrant, the storage backend)
- **One embedding API key** — either:
  - OpenAI: `sk-...`
  - Voyage AI: `pa-...` (used in production)
- **curl** or **Python 3** (for the examples below)

---

## Step 1 — Start the stack

### Option A: Docker Compose (recommended)

```bash
git clone https://github.com/FBISiri/engram.git && cd engram
cp .env.example .env
```

Edit `.env` — set your API key and enable the HTTP transport:

```bash
# .env (minimum viable config)
ENGRAM_OPENAI_API_KEY=sk-...           # or use Voyage (see .env.example)
ENGRAM_TRANSPORT=http                  # expose REST API on :8080
ENGRAM_API_KEY=pick-a-secret-token     # protects all endpoints except /health
```

```bash
docker compose up -d
```

This starts both Qdrant (vector store) and Engram (memory layer).

### Option B: Binary + standalone Qdrant

```bash
# 1. Start Qdrant
docker run -d --name engram-qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v engram_qdrant_data:/qdrant/storage \
  qdrant/qdrant:v1.9.7

# 2. Build Engram from source
git clone https://github.com/FBISiri/engram.git && cd engram
go build -o engram ./cmd/engram/

# 3. Run with HTTP transport
export ENGRAM_QDRANT_URL=localhost:6334
export ENGRAM_OPENAI_API_KEY=sk-...
export ENGRAM_TRANSPORT=http
export ENGRAM_API_KEY=pick-a-secret-token
./engram serve
```

Engram listens on **`:8080`** by default.

---

## Step 2 — Health check

```bash
curl http://localhost:8080/health
```

Expected response:

```json
{"status": "ok", "qdrant": "connected"}
```

`/health` performs a deep liveness check (pings Qdrant) and requires **no auth** — use it for uptime probes.

---

## Step 3 — Store your first memory

<details>
<summary><strong>curl</strong></summary>

```bash
curl -X POST http://localhost:8080/memories \
  -H "Authorization: Bearer pick-a-secret-token" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "The deployment pipeline requires approval from two reviewers before merging to main.",
    "type": "directive",
    "importance": 8,
    "tags": ["deployment", "ci-cd", "policy"],
    "metadata": {
      "source_type": "user_input"
    }
  }'
```

</details>

<details>
<summary><strong>Python</strong></summary>

```python
import requests

ENGRAM_URL = "http://localhost:8080"
API_KEY = "pick-a-secret-token"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# Store a memory
resp = requests.post(f"{ENGRAM_URL}/memories", headers=HEADERS, json={
    "content": "The deployment pipeline requires approval from two reviewers before merging to main.",
    "type": "directive",
    "importance": 8,
    "tags": ["deployment", "ci-cd", "policy"],
    "metadata": {
        "source_type": "user_input"
    },
})
print(resp.json())
# → {"status": "created", "memory": {"id": "a1b2c3d4-...", ...}}
```

</details>

### Key parameters

| Param | Type | Required | Notes |
|---|---|---|---|
| `content` | string | ✅ | Memory text. **Use English** for consistent cross-session recall. |
| `type` | string | ✅ | `identity` (permanent) · `directive` (permanent) · `insight` (~144d half-life) · `event` (~2.9d half-life) |
| `importance` | number | — | 1–10, default 5. Accurate > high — mark everything 9 and you've marked nothing. |
| `tags` | string[] | — | Free-form labels for filtering. |
| `metadata.source_type` | string | — | Provenance: `user_input` · `reflection` · `web_search` · `tool_output` · `calendar` · `document` |

**Dedup:** Engram auto-rejects writes where an existing memory has cosine similarity ≥ 0.92. You'll get a `409` with the existing memory's ID — this is by design, not an error.

---

## Step 4 — Search it back

<details>
<summary><strong>curl</strong></summary>

```bash
curl -X POST http://localhost:8080/memories/search \
  -H "Authorization: Bearer pick-a-secret-token" \
  -H "Content-Type: application/json" \
  -d '{"query": "what is the merge policy", "limit": 3}'
```

</details>

<details>
<summary><strong>Python</strong></summary>

```python
results = requests.post(f"{ENGRAM_URL}/memories/search", headers=HEADERS, json={
    "query": "what is the merge policy",
    "limit": 3,
}).json()

for r in results:
    print(f"[{r['score']:.2f}] {r['content'][:80]}")
# → [1.42] The deployment pipeline requires approval from two reviewers before mer...
```

</details>

### Understanding the score

```
score = 1.0 × relevance + 0.5 × recency + 0.3 × importance
```

This is **not raw cosine similarity**. Two equally-relevant memories — the fresher, more important one ranks higher. This is the core difference from "just querying a vector DB."

### Search filters

| Param | Type | Notes |
|---|---|---|
| `query` | string | Natural-language question (required) |
| `limit` | number | Max results, default 5, max 100 |
| `types` | string[] | Filter by type, e.g. `["directive", "identity"]` |
| `tags` | string[] | Filter by tag (OR logic) |
| `source_type` | string[] | Filter by provenance, e.g. `["user_input", "web_search"]` |
| `time_start` / `time_end` | number | Unix timestamp range filter |

---

## Step 5 — Verify end-to-end

Run the built-in integration test to exercise every operation (add, search, update, delete, dedup):

```bash
ENGRAM_OPENAI_API_KEY=sk-... ./integration_test.sh
```

Or do a quick manual round-trip:

```bash
# 1. Store
ID=$(curl -s -X POST http://localhost:8080/memories \
  -H "Authorization: Bearer pick-a-secret-token" \
  -H "Content-Type: application/json" \
  -d '{"content":"Quick test memory","type":"event","importance":3}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['memory']['id'])")

# 2. Read back by ID
curl -s http://localhost:8080/memories/$ID \
  -H "Authorization: Bearer pick-a-secret-token" | python3 -m json.tool

# 3. Search
curl -s -X POST http://localhost:8080/memories/search \
  -H "Authorization: Bearer pick-a-secret-token" \
  -H "Content-Type: application/json" \
  -d '{"query":"quick test","limit":1}' | python3 -m json.tool

# 4. Clean up
curl -s -X DELETE http://localhost:8080/memories/$ID \
  -H "Authorization: Bearer pick-a-secret-token"
```

If all four steps return `200`, your install is working.

---

## What's next

### Update a memory

```bash
# PATCH — change metadata only (no re-embedding)
curl -X PATCH http://localhost:8080/memories/<id> \
  -H "Authorization: Bearer pick-a-secret-token" \
  -H "Content-Type: application/json" \
  -d '{"importance": 9, "tags": ["deployment", "ci-cd", "critical"]}'

# PUT — replace content (triggers re-embedding)
curl -X PUT http://localhost:8080/memories/<id> \
  -H "Authorization: Bearer pick-a-secret-token" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Deployments to main now require THREE reviewers (updated Q3 2026).",
    "type": "directive",
    "importance": 9
  }'
```

### Write-gate discipline (recommended)

Before every write, search for near-duplicates:

```python
# Before adding, check for semantic overlap
existing = requests.post(f"{ENGRAM_URL}/memories/search", headers=HEADERS, json={
    "query": new_content,
    "limit": 3,
}).json()

top_score = existing[0]["score"] if existing else 0

if top_score > 0.82:
    # Too similar — update the existing memory instead
    requests.put(f"{ENGRAM_URL}/memories/{existing[0]['id']}", headers=HEADERS, json={
        "content": new_content, "type": "directive", "importance": 8,
    })
elif top_score < 0.70:
    # New enough — safe to add
    requests.post(f"{ENGRAM_URL}/memories", headers=HEADERS, json={
        "content": new_content, "type": "directive", "importance": 8,
    })
else:
    # Gray zone (0.70–0.82): compare semantics manually and decide
    pass
```

### Type-based forgetting (TTL)

Memories auto-expire based on `type × importance`:

| Type | importance <5 | 5–7 | ≥8 |
|---|---|---|---|
| `identity` | permanent | permanent | permanent |
| `directive` | 90 days | permanent | permanent |
| `insight` | 30 days | 90 days | permanent |
| `event` | 3 days | 7 days | 30 days |

Low-importance events clean themselves up. High-importance directives live forever. You don't need a cron job.

### Reflection Engine

Engram can periodically synthesize scattered memories into higher-level insights:

```bash
# Check if reflection should run
curl http://localhost:8080/reflect/check \
  -H "Authorization: Bearer pick-a-secret-token"

# Trigger a reflection cycle
curl -X POST http://localhost:8080/reflect \
  -H "Authorization: Bearer pick-a-secret-token"
```

Throttled to min 2h interval, max 3×/day. Configurable via env vars.

### Prometheus metrics

```bash
curl http://localhost:8080/metrics
```

Exposes op counts, latency histograms, and collection sizes.

---

## Endpoint cheat sheet

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `/health` | ✗ | Deep liveness check (pings Qdrant) |
| `POST` | `/memories` | ✓ | Create memory |
| `GET` | `/memories/{id}` | ✓ | Read memory by ID |
| `PATCH` | `/memories/{id}` | ✓ | Update metadata (no re-embed) |
| `PUT` | `/memories/{id}` | ✓ | Replace content (re-embeds) |
| `DELETE` | `/memories/{id}` | ✓ | Delete memory |
| `POST` | `/memories/search` | ✓ | Semantic search |
| `POST` | `/memories/cross-search` | ✓ | Search across collections |
| `POST` | `/collections` | ✓ | Create collection |
| `GET` | `/collections` | ✓ | List collections |
| `POST` | `/reflect` | ✓ | Trigger reflection |
| `GET` | `/reflect/check` | ✓ | Check reflection readiness |
| `GET` | `/metrics` | ✓ | Prometheus metrics |

Auth = `Authorization: Bearer <ENGRAM_API_KEY>` header.

---

## Error codes

| HTTP | Meaning | Common cause |
|---|---|---|
| `200` | OK | — |
| `400` | Bad request | Missing `content`, invalid `type` |
| `401` | Unauthorized | Missing or wrong Bearer token |
| `404` | Not found | Memory ID doesn't exist |
| `409` | Conflict (dedup) | Memory too similar to existing (≥0.92) — not an error, by design |
| `422` | Unprocessable entity | Provenance validation failed — strict mode rejected an unrecognized or omitted `source_type` |
| `429` | Throttled | Reflection called too soon (min 2h interval) |
| `500` | Internal error | Qdrant down, embedding API unreachable |

---

## Minimal Python wrapper

For agents that want a thin client without MCP:

```python
"""engram_client.py — minimal Engram REST client (stdlib only, no pip)."""
import json
import urllib.request

class EngramClient:
    def __init__(self, url="http://localhost:8080", api_key=""):
        self.url = url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _req(self, method, path, body=None):
        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(
            f"{self.url}{path}", data=data, headers=self.headers, method=method
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())

    def health(self):
        return self._req("GET", "/health")

    def add(self, content, type="event", importance=5, tags=None, source_type=None):
        body = {"content": content, "type": type, "importance": importance}
        if tags:
            body["tags"] = tags
        if source_type:
            body.setdefault("metadata", {})["source_type"] = source_type
        return self._req("POST", "/memories", body)

    def search(self, query, limit=5, types=None, tags=None, source_type=None):
        body = {"query": query, "limit": limit}
        if types:
            body["types"] = types
        if tags:
            body["tags"] = tags
        if source_type:
            body["source_type"] = source_type
        return self._req("POST", "/memories/search", body)

    def get(self, memory_id):
        return self._req("GET", f"/memories/{memory_id}")

    def update(self, memory_id, **kwargs):
        return self._req("PATCH", f"/memories/{memory_id}", kwargs)

    def replace(self, memory_id, content, type="event", importance=5):
        return self._req("PUT", f"/memories/{memory_id}",
                         {"content": content, "type": type, "importance": importance})

    def delete(self, memory_id):
        return self._req("DELETE", f"/memories/{memory_id}")
```

Usage:

```python
from engram_client import EngramClient

mem = EngramClient(url="http://localhost:8080", api_key="pick-a-secret-token")

# Store
result = mem.add(
    "The staging env resets every Sunday at 03:00 UTC.",
    type="event", importance=5, tags=["ops", "staging"],
    source_type="user_input",
)
print(result)  # {"status": "created", "memory": {...}}

# Search
hits = mem.search("when does staging reset", limit=3)
for h in hits:
    print(f"[{h['score']:.2f}] {h['content']}")
```

---

## Configuration reference

All config is via environment variables. Minimum viable:

```bash
ENGRAM_QDRANT_URL=localhost:6334        # Qdrant gRPC (not 6333 REST)
ENGRAM_OPENAI_API_KEY=sk-...            # embedding key (required)
ENGRAM_TRANSPORT=http                   # expose REST API
ENGRAM_API_KEY=your-secret              # auth for REST endpoints
```

Full reference: [`docs/configuration.md`](https://github.com/FBISiri/engram/blob/main/docs/configuration.md)

### Embedding providers

| Provider | Env vars | Default model |
|---|---|---|
| OpenAI (default) | `ENGRAM_OPENAI_API_KEY` | `text-embedding-3-small` (1536d) |
| Voyage AI | `ENGRAM_VOYAGE_API_KEY`, `ENGRAM_EMBEDDER_PROVIDER=voyage` | `voyage-3.5` (1024d) |
| OpenRouter | `ENGRAM_OPENAI_API_KEY`, `ENGRAM_OPENAI_BASE_URL=https://openrouter.ai/api/v1` | any compatible model |

> ⚠️ Changing the embedding model on an existing collection requires recreating the collection (delete + re-index).

---

## Provenance tracking

### What is `source_type`?

Every memory carries two orthogonal provenance fields:

- **`source`** (`user` / `agent` / `system`) — *who* wrote the memory.
- **`metadata.source_type`** — *what kind of input* produced the memory's content.

`source_type` answers the question: "Where did this information **originally** come
from?" This is required for EU AI Act traceability (distinguishing AI-generated from
human-sourced content) and enables provenance-filtered search at query time.

### The six valid values

| Value | When to use | Example |
|-------|-------------|---------|
| `user_input` | Content came directly from a human (chat, dictation, form). | User says "I prefer dark mode" → store with `user_input`. |
| `web_search` | Content derived from a web search result. | Agent searched "Python 3.12 release date" → store the finding. |
| `tool_output` | Content derived from a non-search tool or function call. | CI pipeline returned build status → store the result. |
| `reflection` | Insight synthesized by the agent itself (thinking, summarizing, connecting dots). | Agent realizes two directives conflict → store the insight. |
| `calendar` | Content derived from calendar/scheduling data. | "Frank's flight lands at 14:00" from a calendar event. |
| `document` | Content extracted from an ingested file (PDF, DOCX, code). | Key finding from an uploaded research paper. |

**Selection guide — when in doubt:**

1. Did a human type/say it? → `user_input`
2. Did it come from a web search? → `web_search`
3. Did it come from a tool call (API, CLI, database)? → `tool_output`
4. Did it come from a calendar event? → `calendar`
5. Did it come from reading a file/document? → `document`
6. Did the agent generate it by thinking/synthesizing? → `reflection`

Most agent-generated memories (summaries, lessons learned, synthesized insights) should
use `reflection`. When uncertain, default to `reflection`.

### Enforcement modes: `warn` vs `strict`

Provenance enforcement at write time is controlled by the `ENGRAM_PROVENANCE_MODE`
environment variable:

| Mode | `source_type` **omitted** | `source_type` **present** |
|------|---------------------------|---------------------------|
| `warn` (default) | Accepted; defaults to `"unknown"`, warning logged. | Accepted if valid enum value; otherwise `400`. |
| `strict` | **Rejected** with `422 Unprocessable Entity`. | `400` if not in enum; `422` if value is `"unknown"`. |

In `warn` mode (the default), omitting `source_type` works but produces a log warning
and stores `"unknown"` — this is fine for getting started but makes provenance auditing
unreliable. In `strict` mode, every write **must** include a valid `source_type` or it
will be rejected with HTTP `422`.

> **Tip:** Start with `warn` during development. Switch to `strict` in production to
> guarantee every memory has clean provenance.

Related environment variables:

| Variable | Default | Effect |
|----------|---------|--------|
| `ENGRAM_PROVENANCE_MODE` | `warn` | Write-time enforcement (table above). |
| `ENGRAM_ALLOWED_PROVENANCES` | *(empty)* | In `strict` mode, restricts accepted values to this comma-separated list. Empty = all six enum values accepted. |
| `ENGRAM_REQUIRE_PROVENANCE` | `false` | Legacy flag. Enables the Reflection Engine's provenance evidence filter. Does **not** gate the write path — use `ENGRAM_PROVENANCE_MODE=strict` for that. |

#### curl — storing with `source_type`

```bash
# Store a memory with explicit provenance
curl -X POST http://localhost:8080/memories \
  -H "Authorization: Bearer pick-a-secret-token" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Python 3.12 was released on October 2, 2023.",
    "type": "insight",
    "importance": 5,
    "tags": ["python", "releases"],
    "metadata": {
      "source_type": "web_search"
    }
  }'
```

In `strict` mode, omitting `metadata.source_type` returns:

```json
{"error": "source_type is required", "status": 422}
```

### Filtering search results by `source_type`

Pass `source_type` as a string array in your search request to restrict results to
specific provenances. This is useful when you only want human-provided facts, or only
agent-generated insights:

```bash
# Find only memories that came from user input
curl -X POST http://localhost:8080/memories/search \
  -H "Authorization: Bearer pick-a-secret-token" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "deployment policy",
    "limit": 5,
    "source_type": ["user_input"]
  }'
```

```bash
# Find memories from web search OR documents (multiple values = OR logic)
curl -X POST http://localhost:8080/memories/search \
  -H "Authorization: Bearer pick-a-secret-token" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Python release dates",
    "limit": 5,
    "source_type": ["web_search", "document"]
  }'
```

Invalid `source_type` filter values are rejected with `400`.

See [`api.md § Provenance`](https://github.com/FBISiri/engram/blob/main/docs/api.md#provenance--eu-ai-act-compliance) for the full enforcement matrix and edge cases.

---

## FAQ

**Q: Can I run Engram without Docker?**
A: Yes. Build from source (`go build -o engram ./cmd/engram/`) and run Qdrant however you like (binary, systemd, cloud). Engram itself is a single static binary.

**Q: What happens if I try to store a near-duplicate?**
A: Engram's server-side dedup gate rejects writes with cosine similarity ≥ 0.92 to an existing memory. You get back `409` with the existing memory's ID. This prevents the "same fact, ten phrasings" problem.

**Q: Do I need to manage memory expiry?**
A: No. Engram auto-computes TTL based on `type × importance` (see table above). Low-importance events expire in 3 days; permanent types (`identity`, `directive`) never expire. You can override with `valid_until` if needed.

**Q: Is there a Python SDK?**
A: Not yet — Engram is designed API-first. The minimal wrapper above (stdlib only, zero dependencies) covers all operations. A proper SDK is on the roadmap.

**Q: Can multiple agents share one Engram instance?**
A: Yes. Create separate collections (`POST /collections`) for each agent. Memories are physically isolated per collection. Use `cross-search` when you need to query across agents.

**Q: What's `source_type` and do I need it?**
A: It's a provenance tag that records *where* the memory came from (`user_input`, `web_search`, `reflection`, etc.). It's optional but strongly recommended — it enables compliance auditing and provenance-filtered search.
