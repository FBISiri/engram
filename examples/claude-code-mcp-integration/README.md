# Engram × Claude Code MCP Integration — Config 5

**Scenario**: Wire Engram as a persistent memory backend for a Claude Code
agent (or any MCP client) so the agent can **remember across sessions**,
accumulate project-specific directives, and surface past decisions without
relying on its context window.

```
  Claude Code session
  ┌──────────────────────────────────────────────────────┐
  │  claude --mcp-config .mcp.json                       │
  │                                                      │
  │  User: "Add pagination to the payments endpoint"     │
  │                                                      │
  │  [Claude internally calls memory_search]             │
  │  → retrieves: "payments service uses cursor-based    │
  │    pagination per ADR-014, not offset"               │
  │  → retrieves: "Alex wants type hints on all params"  │
  │                                                      │
  │  Claude writes code with correct pattern + hints.    │
  │                                                      │
  │  [After task] memory_add event: "Added pagination    │
  │   to GET /payments, used cursor pattern, PR #442"    │
  └──────────────────────────────────────────────────────┘
          │ MCP stdio (via mcpServers config)
          ▼
  ┌──────────────────────────┐
  │   engram serve           │
  │   (stdio transport)      │
  │   collection: my_project │
  └──────────────────────────┘
          │ gRPC
          ▼
       Qdrant
```

## Why this matters

A Claude Code session starts blank every time. Without memory:
- You re-explain project conventions each session.
- Past decisions (architecture choices, gotchas) are invisible.
- The agent can't learn your preferences over time.

With Engram as an MCP tool, every session can instantly retrieve:
- **Directives**: coding standards, safety rules, workflow preferences.
- **Insights**: distilled lessons from past sessions ("cursor pagination, not offset").
- **Events**: recent task history ("upgraded Stripe SDK last Tuesday, kept v1 webhooks 90 days").

## Step 1: Start Qdrant

```bash
docker run -d --name engram-qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v engram_qdrant_data:/qdrant/storage \
  qdrant/qdrant:v1.9.7
```

## Step 2: Configure environment

```bash
cp .env.example .env
# Fill in ENGRAM_OPENAI_API_KEY (or ENGRAM_VOYAGE_API_KEY)
# Set ENGRAM_COLLECTION_NAME to your project name
```

## Step 3: Build or download the Engram binary

```bash
# Build from source (Go 1.22+)
git clone https://github.com/your-org/engram
cd engram
go build -o engram ./cmd/engram
export ENGRAM_BIN=$(pwd)/engram

# Or use the pre-built release binary from GitHub Releases
```

## Step 4: Create the Claude Code MCP config

Copy `.mcp.json.example` to `.mcp.json` (or merge into your existing
`~/.claude.json` `mcpServers` section):

```bash
cp .mcp.json.example .mcp.json
# Edit ENGRAM_BIN path and env vars to match your setup
```

> **Important**: `.mcp.json` is project-local (Claude Code reads it from the
> working directory). If you want Engram available in *all* projects, put the
> `mcpServers.engram` block in `~/.claude.json` instead.

## Step 5: Seed initial project memory

```bash
pip install requests python-dotenv
python bootstrap.py
```

This seeds 10 example memories for a hypothetical "payments-service" project:
- 3 **directives**: coding standards your agent must always follow.
- 3 **insights**: architectural decisions distilled from past sessions.
- 2 **identity** memories: project context and team preferences.
- 2 **events**: recent task history.

For a real project, skip the bootstrap and let memories accumulate organically:
the agent will write memories as it works.

## Step 6: Run Claude Code with Engram memory

```bash
# From your project root (where .mcp.json lives)
claude --mcp-config .mcp.json

# Or if .mcp.json is in your home dir:
claude --mcp-config ~/.claude-mcp.json
```

Claude Code will now have access to `memory_search`, `memory_add`,
`memory_update`, and `memory_delete` as MCP tools.

## Recommended system prompt additions

Add these instructions to your Claude Code system prompt (or `.claude/CLAUDE.md`):

```markdown
## Memory protocol

You have access to Engram long-term memory tools. Use them as follows:

**Start of session (after task is clear):**
memory_search(query="<task or domain>", limit=5)
→ retrieve relevant directives, past decisions, and recent events.
Always do this before writing any code.

**During work (when you learn something durable):**
memory_add(type="insight", content="...", importance=6-7, tags=["<domain>"])
→ capture architectural decisions, gotchas, patterns worth remembering.

**After completing a task:**
memory_add(type="event", content="<what was done, key decisions, PR number>",
           importance=5, tags=["<domain>", "completed"])

**Do NOT store:**
- Transient scratchpad / intermediate reasoning
- User's current message (it's in context)
- Anything with importance < 4 unless it's a directive
```

## Key design decisions

| Decision | Rationale |
|---|---|
| stdio transport (not HTTP) | Claude Code MCP uses stdio; no port to manage, no auth headers |
| One collection per project | Keeps search scoped; avoids cross-project noise |
| `importance >= 6` for directives | High importance → longer TTL, survives pruning |
| `importance 4-5` for events | Events are transient; Dream Engine prunes stale ones |
| Bootstrap seeds real memories via REST | Real embeddings from day 1; not zero-vector placeholders |

## Connecting to an existing Engram server (HTTP mode)

If you already run `engram serve` in `both` or `http` transport mode on a server:

```bash
# bootstrap.py can seed via the HTTP API instead of Qdrant directly
python bootstrap.py --base-url http://your-engram-server:8080
```

For the MCP connection, the Claude Code process still needs a local `engram`
binary in stdio mode — each Claude Code session forks its own MCP server
subprocess. The local binary should point at the same Qdrant backing store
as your shared server:

```json
{
  "mcpServers": {
    "engram": {
      "command": "/path/to/engram",
      "args": ["serve"],
      "env": {
        "ENGRAM_QDRANT_URL": "your-qdrant-host:6334",
        "ENGRAM_OPENAI_API_KEY": "sk-...",
        "ENGRAM_COLLECTION_NAME": "payments_service"
      }
    }
  }
}
```

## Troubleshooting

**"memory_search: no results"** — Collection may be empty (run bootstrap.py)
or embedder is misconfigured (check `ENGRAM_OPENAI_API_KEY`).

**"engram: connection refused to Qdrant"** — Qdrant must be running before
Claude Code starts. Check `docker ps` and the port mapping.

**"MCP server not found"** — Verify the `command` path in `.mcp.json` is
absolute and the binary is executable (`chmod +x /path/to/engram`).

**Memories aren't persisting across sessions** — Confirm `ENGRAM_COLLECTION_NAME`
is the same between sessions. The default `my_project` is safe; just don't
change it mid-session.

## Next steps

- Add **Reflection Engine** to synthesise session insights weekly:
  set `ENGRAM_REFLECTION_ENABLED=true` and `ANTHROPIC_API_KEY`.
- For a team using the same project memory, see `../multi-agent-shared-memory/`
  to share one collection across multiple developers' Claude Code instances.
- For a long-running autonomous agent (weeks to months), see
  `../long-cycle-reflection-heavy/` for reflection + Dream Engine pruning.
