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

## Config Examples

Unlike the bootstrap examples above (which seed memories), these three are **runtime
tuning profiles** — each is a `config.yaml` + `README.md` you apply via the
`memory_apply_config` MCP tool to shape how memories are ranked (`retrieve_config`) and
written/deduped (`update_config`) for a given deployment shape.

| Config | When to use |
|--------|-------------|
| [config-personal-agent](./config-personal-agent/) | Single high-value, low-volume agent. Recall of a rare-but-critical fact matters more than precision (召回 > 精确). |
| [config-team-knowledge](./config-team-knowledge/) | Shared knowledge base, many contributors. Dedup + provenance matter more than exhaustive recall (精确 > 召回). |
| [config-research-dedup](./config-research-dedup/) | Dense research notes with high semantic overlap. The most aggressive dedup profile — forces consolidation over fragment pile-up. |

### 对比速查

| 维度 | 1. 个人 agent | 2. 团队知识库 | 3. 研究笔记去重 |
|---|---|---|---|
| 主要风险 | 稀有重要记忆静默丢失 | 重复条目 + provenance 丢失 | 碎片堆积 + 反向优先级 |
| 取舍倾向 | 召回 > 精确 | 精确 > 召回 | 强制合并 |
| `dedupe_threshold` | 0.92（保守） | 0.85（激进） | 0.80（最激进） |
| `recency_weight` | 0.20（低） | 0.35（中高） | 0.30（中） |
| `importance_weight` | 0.35（高） | 0.25（低） | 0.20（最低） |
| `min_score` | 0.55（宽） | 0.65（紧） | 0.70（最紧） |
| `limit` | 6 | 10 | 8 |
| 特色开关 | per-type 严 dedup | provenance + require_source | supersede 链 + 离线 consolidation |

> Some per-type / merge / consolidation sub-fields are marked `# proposed` in each
> `config.yaml` — intended productization knobs not yet all wired server-side. Each
> README has a **⚠️ proposed fields** section flagging what is live vs roadmap.

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
