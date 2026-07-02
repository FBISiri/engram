# Engram Multi-Agent Shared Memory — Config 4

**Scenario**: Two or more autonomous agents (e.g. Siri + BMO) share one Engram
server but use **separate, physically isolated collections** — so their memories
can't cross-contaminate, yet both can optionally read a shared `team` collection
for org-wide facts.

```
  Siri (MCP client)               BMO (MCP client)
        │                                │
        │  memory_add/search             │  memory_add/search
        │  collection: engram_user       │  collection: bmo
        ▼                                ▼
  ┌────────────────────────────────────────────────────────┐
  │                  Engram MCP Server                     │
  │   ┌──────────────┐  ┌──────────┐  ┌────────────────┐  │
  │   │ engram_user  │  │   bmo    │  │  team_shared   │  │
  │   │  (Siri)      │  │  (BMO)   │  │  (read-only)   │  │
  │   └──────────────┘  └──────────┘  └────────────────┘  │
  └────────────────────────────────────────────────────────┘
                         │
                      Qdrant
```

## Why separate collections, not separate servers?

- **One embedder, one Qdrant** — cheaper, operationally simpler.
- **Physical isolation by design** — Engram fan-out search respects collection
  boundaries; `memory_search` in `bmo` never touches `engram_user` vectors.
- **Shared facts without copy-paste** — seed org-wide directives once into
  `team_shared`; both agents can `memory_search(collections=["team_shared"])`.

## Step 1: Start Qdrant

```bash
docker run -d --name engram-qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v engram_qdrant_data:/qdrant/storage \
  qdrant/qdrant:v1.9.7
```

## Step 2: Configure environment

Create a `.env` file (copy `.env.example`):

```bash
cp .env.example .env
# Fill in ENGRAM_OPENAI_API_KEY (or VOYAGE)
```

## Step 3: Seed shared team facts and agent-specific memories

```bash
pip install requests python-dotenv
python bootstrap.py
```

This seeds:
- 3 memories into `team_shared` (org-wide directives, same for both agents)
- 4 memories into `engram_user` (Siri-specific: identity, preferences)
- 4 memories into `bmo` (BMO-specific: daemon role, scheduled tasks)

## Step 4: Configure each agent's MCP client

**Siri** (`~/.claude.json` or Army of the Agent MCP config):
```json
{
  "mcpServers": {
    "engram": {
      "command": "/path/to/engram",
      "args": ["serve"],
      "env": {
        "ENGRAM_QDRANT_URL": "localhost:6334",
        "ENGRAM_OPENAI_API_KEY": "sk-...",
        "ENGRAM_COLLECTION_NAME": "engram_user"
      }
    }
  }
}
```

**BMO** (same binary, different `ENGRAM_COLLECTION_NAME`):
```json
{
  "mcpServers": {
    "engram": {
      "command": "/path/to/engram",
      "args": ["serve"],
      "env": {
        "ENGRAM_QDRANT_URL": "localhost:6334",
        "ENGRAM_OPENAI_API_KEY": "sk-...",
        "ENGRAM_COLLECTION_NAME": "bmo"
      }
    }
  }
}
```

## Step 5: Cross-agent shared search

Either agent can search the shared collection explicitly:

```
memory_search(
  query="what are the team's coding standards",
  collections=["team_shared"],
  limit=5
)
```

Or fan-out across all collections (agent's own + shared):
```
memory_search(
  query="Frank's email preferences",
  collections=["engram_user", "team_shared"],
  limit=5
)
```

## Key design decisions

| Decision | Rationale |
|---|---|
| One Engram server, N collections | Simpler ops than N servers; Qdrant handles isolation |
| `team_shared` is write-protected by convention | Only seed script / BMO admin writes here; agents read-only |
| No cross-collection `memory_add` by default | Each agent only writes to its own collection; prevents accidental pollution |
| Reflection runs per-collection | `reflection_run` synthesises insights from `engram_user` only when called from Siri's MCP session |

## Next steps

- Add a third agent (clone, sub-agent) — just new `ENGRAM_COLLECTION_NAME`, no server change
- Promote a team-wide insight from an agent collection to `team_shared` via `memory_search` → read → `memory_add(collection="team_shared")`
- See `../long-cycle-reflection-heavy/` for reflection-heavy config to apply per-agent
