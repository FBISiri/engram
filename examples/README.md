# Engram — Example Configs

Six self-contained examples, each with a `README.md`, `bootstrap.py`, and `.env.example`.
Pick the one closest to your use case, run `bootstrap.py`, and you have a seeded Engram
collection to build on.

---

## Which config should I use?

| # | Config | When to use |
|---|--------|-------------|
| 1 | [single-agent-personal-memory](./single-agent-personal-memory/) | One agent, one user. The starting point for most personal AI assistants. |
| 3 | [long-cycle-reflection-heavy](./long-cycle-reflection-heavy/) | Agent that runs for weeks/months and should get smarter over time (Reflection + Dream Engine). |
| 4 | [multi-agent-shared-memory](./multi-agent-shared-memory/) | Two or more agents (e.g. Siri + BMO) sharing one Engram server with physically isolated collections. |
| 5 | [claude-code-mcp-integration](./claude-code-mcp-integration/) | Wire Engram as the persistent memory backend for a Claude Code or any other MCP client. |
| 6 | [qdrant-cloud-production](./qdrant-cloud-production/) | Ship to production with Qdrant Cloud (TLS), HTTP REST + MCP simultaneously, and OTel tracing. |
| 7 | [chatbot-session-memory](./chatbot-session-memory/) | Chatbot that needs short-lived per-user context (auto-expires in days, not months). |

> Numbers follow the internal Config numbering in `/Engram/example-configs.md`; Config 2 (episodic research agent) is in the backlog.

---

## Common prerequisites

All examples require:

- **Go 1.22+** (to build the `engram` binary) **or** Docker (for `docker-compose up`)
- **Qdrant** — local (`docker run qdrant/qdrant`) or cloud (see Config 6)
- One of:
  - OpenAI API key (`ENGRAM_OPENAI_API_KEY=sk-...`)  
  - Voyage AI key (`ENGRAM_VOYAGE_API_KEY=pa-...`) — used in production

Each directory has its own `.env.example`; copy to `.env` and fill in your keys.

---

## Quick recipe (any config)

```bash
# 1. Start Qdrant
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant

# 2. Build Engram
git clone https://github.com/FBISiri/engram.git
cd engram
go build -o engram ./cmd/engram/

# 3. Pick an example, configure, seed
cd examples/single-agent-personal-memory    # ← swap for your chosen config
cp .env.example .env && $EDITOR .env
pip install requests python-dotenv
python bootstrap.py

# 4. Run integration test to confirm everything works
cd ../..
ENGRAM_OPENAI_API_KEY=sk-... ./integration_test.sh
```

---

## What `bootstrap.py` does

Each script seeds a curated set of example memories (typically 8–15) into the Qdrant
collection via Engram's HTTP API (`POST /memories`). This means:

- Memories get **real server-side embeddings** (not placeholder zeros)
- Engram's **TTL matrix** sets automatic expiry based on type + importance
- Engram's **dedup gate** (≥0.92) prevents duplicate seeding — re-running is safe

After bootstrapping, your collection is a live, search-ready starting point.
Replace the example memories with your agent's real data as it accumulates.

---

## Related docs

- [Main README](../README.md) — problem statement, core concepts, architecture
- [docs/api.md](../docs/api.md) — full MCP tool + REST API reference
- [docs/configuration.md](../docs/configuration.md) — all `ENGRAM_*` env vars
