# Engram Single-Agent Personal Memory — Config 1

**When to use**: One agent, one user. You want the agent to remember the user's
preferences, past decisions, and working patterns across sessions — without
the user re-explaining things every time.

This is the entry point for most personal AI assistants (Claude Code with project
memory, a personal productivity agent, an email assistant, etc.).

---

## Design decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Embedding model | `text-embedding-3-small` | Accurate and cheap; adequate for personal memory scale |
| Dedup threshold | `0.92` (default) | Standard gate — blocks near-identical writes |
| Reflection | Disabled initially | Enable after 30+ memories accumulate (see §Next steps) |
| TTL | Matrix-managed | `identity`/`directive` permanent; `event` expires in 3–30 days |
| Transport | `stdio` (MCP) | Agent connects directly; no HTTP server needed |

---

## Architecture

```
Your agent (MCP client)
        │ stdio
        ▼
  engram serve
        │ gRPC :6334
        ▼
  Qdrant (local)
        └─► my_assistant  (your collection)
```

The 9 seeded memories cover all four types:

| Count | Type | What it represents |
|-------|------|--------------------|
| 2 | `identity` | Who the user is, communication style, schedule |
| 2 | `directive` | Hard rules the agent must never break |
| 2 | `insight` | Distilled lessons from past sessions |
| 2 | `event` | Recent task history (fades in ~7 days) |
| 1 | `event` | A feedback moment (fades in ~7 days) |

---

## Setup (3 steps)

### Step 1 — Start Qdrant

```bash
docker run -d --name engram-qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v engram_qdrant_data:/qdrant/storage \
  qdrant/qdrant:v1.9.7
```

### Step 2 — Configure

```bash
cp .env.example .env
# Required: fill ENGRAM_OPENAI_API_KEY (or ENGRAM_VOYAGE_API_KEY)
# Optional: change ENGRAM_COLLECTION_NAME (default: my_assistant)
```

### Step 3 — Build Engram and seed example memories

```bash
# Build (from repo root)
go build -o engram ./cmd/engram/

# Start server in HTTP mode so bootstrap.py can reach POST /memories
ENGRAM_TRANSPORT=both \
ENGRAM_HTTP_PORT=8080 \
ENGRAM_OPENAI_API_KEY=$(grep ENGRAM_OPENAI_API_KEY .env | cut -d= -f2) \
./engram serve &

# Seed
pip install requests python-dotenv
python bootstrap.py

# Preview without sending (dry run)
python bootstrap.py --dry-run
```

After seeding, your collection has 9 example memories with real embeddings.

---

## Connecting your agent

**MCP (Claude Desktop / Claude Code)** — add to your MCP config:

```json
{
  "mcpServers": {
    "engram": {
      "command": "/path/to/engram",
      "args": ["serve"],
      "env": {
        "ENGRAM_QDRANT_URL": "localhost:6334",
        "ENGRAM_OPENAI_API_KEY": "sk-...",
        "ENGRAM_COLLECTION_NAME": "my_assistant"
      }
    }
  }
}
```

**Quick search test** — in your MCP session:

```
memory_search(query="how does Alex feel about filler text", limit=3)
```

You should get back Alex's communication preference (importance 9, type `identity`)
ranked first — that's the three-factor fusion (`relevance × recency × importance`)
working.

---

## Customise for your user

The bootstrap seeds memories for a fictional user "Alex Chen". Replace with your own:

1. Edit `bootstrap.py` — update the `MEMORIES` list at the top of the file.
2. Keep one `identity` memory for who the user is (importance 7–9).
3. Keep at least one `directive` for your agent's hardest constraints (importance 9–10).
4. Lean toward `insight` over `event` for things you want to remember long-term.

**Key rule**: set `importance` accurately, not uniformly high. A cluster of
importance-10 memories is equivalent to no importance signal.

---

## Next steps

- **After 30+ real memories**: enable Reflection to synthesise insights from accumulated
  events (`ENGRAM_REFLECTION_ENABLED=true`, see [long-cycle-reflection-heavy](../long-cycle-reflection-heavy/))
- **Multiple agents**: see [multi-agent-shared-memory](../multi-agent-shared-memory/)
- **Production deployment**: see [qdrant-cloud-production](../qdrant-cloud-production/)
- **Full config reference**: [`../../docs/configuration.md`](../../docs/configuration.md)

---

## Files

| File | Description |
|------|-------------|
| `.env.example` | Environment variable template — copy to `.env` |
| `bootstrap.py` | Seeds 9 example memories (all 4 types) via Engram HTTP API |
| `README.md` | This file |
