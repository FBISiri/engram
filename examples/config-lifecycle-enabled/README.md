# Config Example — Lifecycle ENABLED (Evaporation LIVE + Reflection + A-MAC)

**Shape:** a single agent that runs for months, whose memory store must actively
*age* rather than grow forever. This is the **fully enabled** sibling of
[`examples/lifecycle.env`](../lifecycle.env): same 58 `ENGRAM_*` keys, same
layout — but the evaporation sweep is switched out of observe-only mode so it
**actually mutates stored importance and evicts**.

## When to use this profile

- A long-lived agent whose store grows faster than it forgets, and you have
  **already reviewed the dry-run metrics** from `lifecycle.env` and are ready to
  let evaporation write.
- You want insights synthesized from what survives (Reflection V2) and per-type
  write admission (A-MAC) — all on, in one profile.

Do **not** start here on a fresh/production store. Start with
`lifecycle.env` (observe-only), inspect what *would* be evaporated, then adopt
this profile.

## What "lifecycle enabled" means

| Dimension | `lifecycle.env` (observe-only) | This profile (enabled) |
|---|---|---|
| Evaporation sweep | `ENGRAM_EVAPORATION_ENABLED=true` | same |
| Evaporation writes | `ENGRAM_EVAPORATION_DRY_RUN=true` → **reports only** | **`ENGRAM_EVAPORATION_DRY_RUN=false` → LIVE, mutates + evicts** |
| Reflection V2 | on | on |
| A-MAC admission | on | on |

The single most important line in `.env.example`:

```
ENGRAM_EVAPORATION_DRY_RUN=false
```

Flipping this from `true` to `false` is the **whole point** of this example. In
dry-run the background sweep computes decay candidates and reports them via
metrics/logs but performs **no writes**. Live, it writes the decayed importance
back and evicts anything whose effective importance falls below
`ENGRAM_EVAPORATION_EVICTION_THRESHOLD` (default `1.0`) — subject to all the
`ENGRAM_EVAPORATION_PROTECT_*` guards and the `MIN_AGE_DAYS` floor.

**Reflection V2** (`ENGRAM_REFLECTION_MODE=v2`) runs the 4-stage
focal→evidence→dialectic→synthesis pipeline over the memories evaporation leaves
standing. **A-MAC** (`ENGRAM_AMAC_ENABLED=true`) admits writes per type — dedup
threshold, importance clamp, and hourly rate limit — so noisy events cannot
crowd out directives/identity.

## ⚠️ Consolidation caveat — there is NO consolidation env var

**There are no `ENGRAM_*CONSOLIDATION*` environment keys in this repo.** Do not
look for one — evaporation aging is env-driven, but *consolidation* (folding
fragmented memories into one evolving entry) is expressed two other ways:

1. A **runtime tuning profile** applied via the `memory_apply_config` MCP tool —
   see [`config.yaml`](./config.yaml) here and the fuller example in
   [`examples/config-research-dedup/config.yaml`](../config-research-dedup/config.yaml).
   Its `merge_strategy` / `consolidation_pass` fields are marked `# proposed`
   (not yet wired server-side).
2. The **per-type dedup thresholds** `ENGRAM_DEDUP_THRESHOLD_{IDENTITY,DIRECTIVE,INSIGHT,EVENT}`
   — these ARE env vars (in `.env.example`) and control write-time merging, but
   they are *dedup*, not a full consolidation pass.

So: consolidation is **config.yaml + per-type dedup env keys**, NOT an env var of
its own.

## ⚠️ RISK — running evaporation non-dry-run on a real store

With `ENGRAM_EVAPORATION_DRY_RUN=false`, the sweep **permanently lowers
importance and evicts memories**. If your half-lives or protect settings are
mis-tuned, you can silently lose data on a live store. Mitigations already baked
into this profile: identity/directive have `HALF_LIFE=0` (never decay) and are
`PROTECT_TYPES`; `PROTECT_IMPORTANCE=8`, `PROTECT_ACCESS_COUNT=5`,
`PROTECT_RECENT_ACCESS_DAYS=30`, `PROTECT_TAGS`, `PROTECT_CORROBORATED=true`, and
`MIN_AGE_DAYS=14` all shield high-value / young / reinforced memories. Still,
**validate against dry-run first**.

## How to roll back

Flip the one line back and restart:

```
ENGRAM_EVAPORATION_DRY_RUN=true
```

The sweep returns to observe-only immediately — it keeps computing candidates but
stops writing. (Note: rollback stops *future* eviction; it does not resurrect
memories already evicted while live — which is why you validate in dry-run.) To
disable evaporation entirely, set `ENGRAM_EVAPORATION_ENABLED=false`.

## Usage

```bash
# 1. Start Qdrant
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant

# 2. Configure
cp .env.example .env && $EDITOR .env   # fill in real keys; keep placeholders out of git

# 3. Run Engram with this env
set -a; . ./.env; set +a
./engram serve
```

To apply the consolidation profile at runtime, call `memory_apply_config` with
the JSON equivalent of [`config.yaml`](./config.yaml) (see
`config-research-dedup/README.md` for the exact MCP-call shape).

## ⚠️ Proposed fields

Not yet wired server-side — roadmap; require review before mapping to
`MemoryConfig`:

- `update_config.merge_strategy: replace_or_supersede` (`# proposed`)
- `update_config.consolidation_pass: enabled` (`# proposed`)
