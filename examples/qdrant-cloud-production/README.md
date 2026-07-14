# Engram Config 6 — Qdrant Cloud Production

**When to use this config**: You're shipping Engram to production with a hosted Qdrant instance (Qdrant Cloud or self-hosted with TLS), need both HTTP API and MCP access simultaneously, and want observability via OpenTelemetry traces.

---

## Architecture

```
AI Agent (Claude / GPT)
        │  MCP stdio
        ▼
  engram serve  ──────────── HTTP REST ──────► scripts / dashboards
        │  ENGRAM_TRANSPORT=both
        │
        │  gRPC + TLS
        ▼
  Qdrant Cloud (remote)
        │
        └─► engram_user    (primary user memories)
            engram_agent_self   (agent self-state)
            engram_reflection   (Reflection Engine outputs)
```

The `both` transport runs MCP (stdin/stdout) and HTTP (`ENGRAM_HTTP_PORT`) simultaneously. Your AI agent connects via MCP; cron jobs, monitoring scripts, and dashboards use the REST API with Bearer auth.

---

## Checklist

### Prerequisites
- [ ] Qdrant Cloud account at [cloud.qdrant.io](https://cloud.qdrant.io) — or self-hosted Qdrant with TLS
- [ ] Voyage AI key at [voyageai.com](https://voyageai.com) (or OpenAI key if using `text-embedding-3-*`)
- [ ] Anthropic key for Reflection Engine (optional, but recommended)

### Step 1 — Copy and fill `.env`

```bash
cp .env.example .env
# Fill in: ENGRAM_QDRANT_URL, ENGRAM_QDRANT_API_KEY, ENGRAM_VOYAGE_API_KEY,
#          ENGRAM_API_KEY, ANTHROPIC_API_KEY
```

**Important**: `ENGRAM_QDRANT_URL` must point to the **gRPC port** (6334), not the REST port (6333). On Qdrant Cloud, the cluster URL typically looks like:
```
abc123.us-east4-0.gcp.cloud.qdrant.io:6334
```

### Step 2 — Run with environment loaded

```bash
# Export all vars from .env, then start
set -a && source .env && set +a
./engram serve
```

Or use Docker:
```bash
docker run --env-file .env -p 8080:8080 ghcr.io/bm0/engram:latest
```

### Step 3 — Verify connectivity

```bash
# Health endpoint (no auth required)
curl http://localhost:8080/health

# Seed a test memory via HTTP REST
curl -X POST http://localhost:8080/memories \
  -H "Authorization: Bearer $ENGRAM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"content": "Production deployment verified", "type": "event", "importance": 5}'
```

### Step 4 — Seed production memories

```bash
pip install requests python-dotenv
python bootstrap.py
```

---

## Configuration Notes

### Why Voyage AI over OpenAI?

For production deployments with >10k memories, `voyage-3.5` (1024-dim) typically outperforms `text-embedding-3-small` (1536-dim) on recall benchmarks while costing ~40% less per token. The tradeoff: you can't mix embedding models on an existing Qdrant collection without recreating it.

If you already have collections using OpenAI embeddings, stick with OpenAI and set `ENGRAM_EMBEDDING_DIMENSION=1536`.

### OTel sampling in production

`ENGRAM_OTEL_SAMPLE_RATIO=0.5` traces 50% of operations. For high-throughput deployments (>1k memory_search/hour), consider `0.1` to reduce I/O. Traces land in `ENGRAM_OTEL_FILE_DIR` as daily-rotating JSONL files.

### Dedup threshold

The default `0.92` catches near-identical memories (paraphrases, minor rewording) while allowing genuinely distinct memories through. In production, lower thresholds (`0.85`) risk silently blocking memories that only seem similar. Don't change this without reviewing `docs/configuration.md §Deduplication`.

---

## Files

| File | Description |
|------|-------------|
| `.env.example` | Environment variable template — copy to `.env` |
| `bootstrap.py` | Seeds 10 example memories (production agent scenario) |
| `README.md` | This file |

---

## Related Examples

| Config | Use Case |
|--------|----------|
| [single-agent-personal-memory](../single-agent-personal-memory/) | Local dev quickstart (Config 1) |
| [long-cycle-reflection-heavy](../long-cycle-reflection-heavy/) | Long-running autonomous agent with reflection tuning (Config 3) |
| [multi-agent-shared-memory](../multi-agent-shared-memory/) | Multiple agents sharing a memory layer (Config 4) |
| [claude-code-mcp-integration](../claude-code-mcp-integration/) | Claude Code × Engram via stdio MCP (Config 5) |
