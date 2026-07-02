# Engram Long-Cycle Reflection-Heavy Memory — Quickstart

**Config 3.** A memory setup for a *long-running autonomous agent* (weeks to
months) that should get **smarter over time**, not just bigger. Two autonomic
engines keep the store high-signal:

```
  New memories flow in all day
       │
       ▼
  Reflection Engine (V2 focal-point, 1-3x/day)
    → generates focal questions, retrieves evidence per question,
      synthesises durable `insight` memories, marks sources reflected
       │
       ▼
  Dream Engine (nightly ~20:00)
    → consolidate: tag-group recent events into insights
    → prune: DELETE importance ≤ 3 AND access_count = 0 (stale events)
       │
       ▼
  High-signal, self-curated memory store
```

(See `/Engram/example-configs.md` Config 3 §1 for the full diagram and
rationale.) Events are *transient raw material*: Reflection lifts the signal
into higher-importance insights, then Dream reclaims the spent events. That is
why this seed set keeps event importance intentionally low (≤ 5).

**Why V2 reflection?** V1 feeds all unreflected memories into one flat LLM
call. V2 generates focal questions, does targeted semantic search per question,
then synthesises one high-quality insight per question — ~4× the Haiku calls
but dramatically better cross-domain synthesis for a large, diverse memory base.

---

## Step 1: Start Qdrant
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

## Step 2: Start the Engram server in HTTP mode

The engram binary is configured via `ENGRAM_*` environment variables — there is
no `--config` flag. Launch it in HTTP mode so `bootstrap.py` can reach
`POST /memories`, and enable reflection:

```bash
ENGRAM_TRANSPORT=both \
ENGRAM_HTTP_PORT=8080 \
ENGRAM_COLLECTION_NAME=siri_long_cycle \
ENGRAM_REFLECTION_ENABLED=true \
ENGRAM_REFLECTION_MODEL=claude-3-5-haiku-latest \
ANTHROPIC_API_KEY=sk-ant-... \
ENGRAM_OPENAI_API_KEY=sk-... \
./engram serve
```

Key env vars:
- `ENGRAM_TRANSPORT=both` — expose the HTTP API (memory_add) alongside stdio MCP; the default `stdio` does not serve `POST /memories`.
- `ENGRAM_REFLECTION_ENABLED=true` — default is `false`; must be true for reflection to run.
- `ANTHROPIC_API_KEY` — Reflection V2's Haiku calls; degrades to V1 if unset.
- `ENGRAM_OPENAI_API_KEY` — embedder for server-side embeddings.

> **`config.yaml` is a design reference, not a loaded file.** It documents the
> intended long-cycle tuning (reflection V2 / threshold 30 / Dream gates). The
> running binary takes its config from `ENGRAM_*` env vars; reflection V2
> focal-point mode + threshold are set on the reflection engine (struct /
> `--mode v2`), and Dream gates are hardcoded in `pkg/dream/gate.go`. Editing
> `config.yaml` does **not** change server behavior on its own.

## Step 3: Configure and install deps
```bash
cp .env.example .env
# edit ENGRAM_COLLECTION_NAME / ENGRAM_HOST / ENGRAM_HTTP_PORT if needed
pip install requests python-dotenv
```

## Step 4: Seed Aria's initial memory state
```bash
python3 bootstrap.py                  # requires Python 3.7+ (use python3.9 if
                                      # the default python3 is 3.6)
# preview the payloads without touching the network:
python3 bootstrap.py --dry-run
```

`bootstrap.py` seeds via the Engram **HTTP API** (`POST /memories`), *not* raw
Qdrant — so each memory gets a real server-side embedding and an automatic
`valid_until` (decay) based on its type and importance. Re-running is safe:
Engram's ~0.92 semantic auto-dedup prevents duplicate points.

The seed set covers all four memory types: 1 identity + 2 directives (durable,
never decay), 3 events (transient, importance ≤ 5), and 2 reflection-generated
examples (1 insight + 1 directive, tagged `source:reflection`) so a fresh
collection demonstrates the full lifecycle.

---

## Monthly health check

Run these monthly to confirm the lifecycle is alive (spec Config 3 §6):

```bash
# 1. Reflection is running
ls -la ~/.siri/reflection_last_run
cat ~/.siri/reflection_daily_count

# 2. Dream is running
ls -la ~/.siri/dream_last_run
cat /data/armyoftheagent/workspace/dream-run-$(date +%Y-%m-%d).md

# 3. Memory composition — want insight:event ratio > 1
engram stats

# 4. Bloat check — > 5000 points usually means prune is failing
curl http://localhost:6333/collections/siri_long_cycle | jq .result.points_count
```

### The key metric: insight:event ratio > 1

The single most informative health signal for a long-cycle store is the ratio
of `insight` memories to `event` memories.

- **Ratio > 1 (healthy):** events are being consolidated into insights and then
  pruned, exactly as designed. The agent is converting raw experience into
  durable knowledge.
- **Ratio < 0.5 (unhealthy):** Reflection or Dream isn't running, or the prune
  threshold is set too high. Events are piling up faster than they're
  consolidated — the store is getting *bigger*, not *smarter*. Check that the
  engines are firing (steps 1–2) and that `ANTHROPIC_API_KEY` is set so
  Reflection runs in V2 rather than degrading.

Blog-ready framing: *"Is your agent's memory getting smarter or just bigger?"*

---

## Next steps
- Read `/Engram/example-configs.md` Config 3 (§1–8) for the full design rationale.
- Watch the composition over 30 days: events spike then drop (pruned), insights
  accumulate. That curve is the proof the lifecycle works.
- See `../` for the other example configs (single-agent, multi-tenant HTTP).
