# Changelog

All notable changes to Engram are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once tagged
releases begin.

## [Unreleased]

### Added
- **`memory_list` now returns `source_type` (and `metadata`) per record.**
  The `memory_list` MCP tool projection dropped provenance: it emitted only
  `id/content/type/importance/created_at/tags/source_collection`. It now adds
  `source_type` (via the same `sourceTypeFromMetadata` helper as `memory_search`)
  and `metadata`, both `,omitempty`. The Scroll/`ListMemories` path already
  carried `Memory.Metadata` end-to-end (`qdrant.pointToMemory` maps the payload
  metadata field), so only the handler's local projection struct needed the fix —
  no change to `pkg/memory` or `pkg/qdrant`. Additive JSON fields → backward
  compatible. Requires recompiling/restarting `engram.service` to take effect.
- **Memory evaporation v2 — safety & gap closure** (spec
  `Engram/spec-memory-evaporation.md` §4). Builds on the shipped evaporation v1
  (`c170c0d`, `ca23b07`) to make the sweep *safe to enable*:
  - Structural protection rules P1..P8 (`memory.EvaporationExempt`) evaluated
    **before** any deprecation. P1 (protected types `identity`/`directive`) is
    structural — it holds even if a half-life is misconfigured `> 0`, so a
    config typo can never make identity/directive memories evaporable.
  - Reinforcement clock (`ENGRAM_EVAPORATION_DECAY_BASIS=last_access`, new
    default): decay now runs from `max(created_at, last_accessed_at)`, so a
    memory that keeps getting read stays alive. `created` preserves the legacy
    created-only basis.
  - Observation window (`ENGRAM_EVAPORATION_OBSERVATION_DAYS`, default `30`)
    between soft-deprecate and hard-delete eligibility.
  - **G1 fix**: an evaporation-deprecated memory is now a hard-delete candidate
    ONLY IF it is not exempt AND its observation window has elapsed — the
    evaporation flag no longer bypasses the `importance>=8` / tag / type guards
    in `isExpiryCandidate`.
  - **G8 fix**: the expiry snapshot directory now resolves relative to the
    state dir (`pkg/statedir`) instead of the non-existent hard-coded vault path.
  - 10 new `ENGRAM_EVAPORATION_*` vars (#43–52 in the config reference) and 4
    new Prometheus metrics (`engram_evaporation_scanned_total`,
    `_exempted_total{type,rule}`, `_dry_run_candidates_total{type}`,
    `_hard_deleted_total{type}`).

### Changed
- **BEHAVIOUR REDUCTION (fail-safe): `ENGRAM_EVAPORATION_DRY_RUN` now defaults
  to `true`.** An operator who already set `ENGRAM_EVAPORATION_ENABLED=true`
  (e.g. from `examples/lifecycle.env`) silently drops into **observe-only** mode
  on upgrade — the sweep computes and reports candidates + metrics but performs
  zero `store.Update` calls until `ENGRAM_EVAPORATION_DRY_RUN=false` is set
  explicitly. This is intentional: it moves existing live deployments in the
  fail-safe direction. Rollout order is now
  `enabled=false` → `enabled=true, dry_run=true` (observe ≥1 week) → `dry_run=false`.
- `ENGRAM_EVAPORATION_HALF_LIFE_DIRECTIVE` default changed `365` → `0`
  (directives never evaporate; blast radius ≫ storage saved). `examples/lifecycle.env`
  also corrects `ENGRAM_EVAPORATION_HALF_LIFE_EVENT` `14` → `30` (G6).

- **`pigo` caller-type + `engram_pigo` baseline collection** — pigo is now a
  first-class caller type (`X-Caller-Type: pigo`) that owns its own physically
  isolated collection `engram_pigo`, registered as a baseline store alongside
  `engram_user`, `engram_agent_self`, and `engram_reflection`. Writes routed to
  the pigo caller land in the dedicated `engram_pigo` Qdrant store.
- **`ENGRAM_PRINCIPAL_KEYS` per-principal API keys** — optional env var
  (`ENGRAM_PRINCIPAL_KEYS="pigo:key1,reflection:key2"`) mapping a caller type to
  a dedicated Bearer key. A request authenticated with a principal key has its
  caller type derived from the key itself (the `X-Caller-Type` header is
  ignored), making collection ownership enforceable rather than self-declared.
  Entries with an unknown/typo caller-type are skipped + logged (never default
  to `engram_user`). The legacy shared `ENGRAM_API_KEY` continues to work with
  the `X-Caller-Type` header.
- **Read isolation for isolated caller-types** — an isolated principal (today:
  `pigo`) may only READ its own store: `POST /memories/search` (and the
  per-collection search route) is force-scoped to the caller's own collection
  filter, so it can never fan out to other collections. Legacy
  `user`/`agent-self`/`reflection` callers keep cross-store fan-out search
  unchanged (Siri/BMO rely on it).
- **Security note — legacy `ENGRAM_API_KEY` is admin/master-scoped:** it can
  self-declare ANY `X-Caller-Type` (including `pigo`) and thus read/write any
  collection. This is intentional and unchanged for backward compat. Only
  per-principal keys are identity-bound and enforced; treat the shared legacy
  key as an admin credential.

## [0.2.0] — 2026-04-19 (Docker images: 2026-06-02)

> Docker Hub: `imsiri/engram:0.2.0`, `imsiri/engram:latest`, `imsiri/engram:0.2`
> Published: 2026-06-02 01:43 UTC (workflow run 26792927494)

First tagged release after `v0.1.0`. Consolidates ~6 months of work across the
Reflection Engine, Dream Engine, HTTP transport, TTL/expiry schema, and the
v1.1 Reflection upgrade (confidence, evidence grounding, event-driven trigger).

### Added — Reflection Engine v1.1 (2026-04-17)
- **Confidence field** (0-1) on `Memory` with `WithConfidence` option; `Score`
  multiplies by confidence when set (>0), else treated as 1.0 for backward
  compat with pre-v1.1 memories. Persisted through Qdrant payload
  (`1b3e482`).
- **Haiku prompt emits `CONFIDENCE:` line**; parser defaults to 0.8 when the
  line is absent (`1b3e482`).
- **Low-confidence draft diversion** — insights with `0 < conf < 0.6` bypass
  Engram and land as markdown drafts in
  `$HOME/siri-vault/Reflection/drafts/` (`1b3e482`).
- **30-day TTL + `source:reflection` tag** on every reflection-origin insight
  (`1b3e482`).
- **Dream Engine boundary isolation** — consolidation + usage-frequency loops
  skip `source:reflection` memories younger than 7 days, preventing premature
  compression (`1b3e482`).
- **`RunResult.DraftsWritten`** metric (`1b3e482`).
- **Evidence grounding gate** — `validateEvidenceGrounding()` rejects batches
  with `<2` source memories before calling Haiku (`8dd5200`).
- **Event-driven reflection** — new `RunSingleEvent` path for task-failure /
  user-correction triggers: bypasses accumulator + daily quota, caps output at
  1 insight, still enforces `source:reflection` tag, 30-day TTL, and
  low-confidence draft diversion (`8dd5200`).
- **MCP tool `reflection_run_event`** (`cause, summary, evidence_ids,
  importance, extra_tags, dry_run`) (`8dd5200`).

### Added — HTTP Transport & Dream Engine
- **HTTP transport** with `/reflect`, `/reflect/check`, `/health` endpoints
  (`0977860`) and README updates (`c50d2f4`).
- **Dream Engine** — autonomous 4-phase memory consolidation (Orient → Gather
  → Consolidate → Prune) with three gate conditions (time, volume, PID lock)
  (`95fccfe`).
- **Voyage AI embedder** as an alternative to OpenAI (`95fccfe`).
- **Memory expiry** — `valid_until` field with a background cleanup goroutine
  (`95fccfe`).
- **TTL auto-calculator** — `valid_until` is now derived from a
  type × importance matrix when callers omit it, and injected at the
  `handleAdd` / `handleUpdate` layer (`ec21d8d`).
- **Deep `/health` endpoint** — the liveness probe pings Qdrant and bypasses
  Bearer auth so external probes work without credentials (`8d185b1`).

### Added — Reflection Engine v1 & Schema
- **Reflection Engine v1** — lightweight periodic synthesis of `insight`
  memories from unreflected events, with count / cron / manual triggers
  (`f9ace98`) and Gate1/Gate2/Gate3 unit tests (`c87887a`).
- **Write-through + Ring Buffer** — BoltDB-backed commit log and
  `WriteThroughStore` for durability and crash recovery
  (`a0eeae3`, `cd36c70`).
- Extended memory schema with `valid_until`, `superseded_by`, `access_count`,
  `last_accessed_at` (`c195fd4`), exposed via the MCP server layer
  (`5fdf3c0`).
- `CONTRIBUTING.md` (`9eaad0c`).

### Changed
- **Dream Engine scroll pagination** — `Scroll` calls inside the Dream pipeline
  go through a new `scrollAll` helper to avoid truncated result windows
  (`acc7019`).
- **Reflection / Dream hot paths** — loop-hoisted time calculations and
  propagated `context.Context` through to `callHaiku` for cancellation
  (`a3e5e45`).
- **Consolidation semantics** — superseded memories are now marked and cleaned
  up consistently across `orient` and `gather` phases (`4b824a8`).
- **Unreflected-memory fetch** — indexed `reflected_at` filter replaces the
  prior full-scan path (`b544396`).
- Module path renamed from `anthropics/engram` to `FBISiri/engram`
  (`6808739`).

### Fixed
- **Provenance-merge dotted-key persistence bug** — `provenanceMerge` and the
  §5.3 legacy-backfill branch in `checkDedup` built the store `Update` payload
  with dotted literal keys (`metadata.source_type`,
  `metadata.provenance_history`). Qdrant `SetPayload` stored those verbatim as
  top-level keys, so the read path (`pointToMemory` → nested `metadata` dict)
  never saw them, breaking provenance visibility, idempotency, and evaporation
  P7. Both sites now read-modify-write the nested `metadata` map (via a new
  `cloneMetadata` helper), preserving unrelated sibling keys. Added a real-Qdrant
  regression test (throwaway collection) plus a sibling-preservation unit test.
  New `scripts/migrate-provenance-dotted-keys.py` (dry-run by default, `--apply`
  to mutate; dotted value wins on conflict; idempotent) folds the ~18 legacy
  dirty points in `engram_agent_self` into the nested metadata dict.
- **Reflection V2 write-back livelock** — a v2-focal run whose insights were all
  caught by pre-write dedup marked zero sources reflected, so the next run
  re-fetched the same batch → same insights → dedup again forever. Stage 5 now
  marks sources when `Written>0 || Drafts>0 || DedupSkipped>0`, and dedup hits
  are counted in a new `insights_dedup_skipped` counter on the V2 path (split
  out of the generic `insights_skipped`).
- Removed hardcoded 1536-dimension zero-vector fallback in Dream and
  Reflection (`629102d`).
- Gate2 now uses `new_memories_since_last_run` instead of `session_count`;
  Gate1 relaxed from 24h to 20h (`96570dd`).
- `mockStore` implements `DeleteExpired` to satisfy the `memory.Store`
  interface (`5a746f9`).
- MCP tool names use underscores instead of dots for Claude API compatibility
  (`af94f5a`).

### Infrastructure
- GitHub Actions workflow publishes Docker images to `imsiri/engram` on Docker
  Hub (`8c90c5a`, `19f51da`).
- **Docker Publish fix** — builder image upgraded from `golang:1.24-alpine` to
  `golang:1.25-alpine` to match `go.mod` requirement (`053a5ce`).

### Post-release additions (included in v0.2.0 Docker image)

These features were committed after the initial v0.2.0 tag but are included
in the final published Docker images.

#### Added
- **Reflection confidence counters + Gate3 threshold=40** — `runs_today`
  metric added to `RunResult` and OTel spans (`e485ccf`, `ff6197a`).
- **Embed cache** — P5-A1 LRU embedding cache with `/metrics` endpoint
  (`1a67541`).
- **Physical collection isolation (Phase 4)** — multi-collection architecture
  with `CollectionFromContext` routing, scoped dedup, migration tooling
  (`migrate-collections`, `drop-legacy`, `migrate-extra-collections`),
  and fan-out resilience (`274d9c3`, `461805a`, `75c0950`, `eff0989`,
  `175feab`, `4252a29`, `364a27a`, `324f464`).
- **Mass-delete guardrail** — `dry_run` support for `memory_update` /
  `memory_delete` to preview changes before applying (`f53b2f3`).
- **Enhanced /health endpoint** — `uptime_seconds`, `memory_count`,
  `last_reflection`, `embedding_latency` p50/p99, Prometheus metrics wiring
  (`13455b4`, `c046cd6`).

#### Fixed
- Multi-store fan-out: `NotFound` errors in `Update`, `SearchByIDs`, and
  `SetPayload` are now logged at WARN and do not fail the operation
  (`b7b6374`, `9afb8b1`, `b49bb01`).
- Reflection orphan-ID issue causing `sources_marked=0` resolved
  (`2e0af12`).
- Deprecated `GetData()` replaced with `GetDense().GetData()` for Qdrant
  vector extraction (`6632dea`).

#### Performance
- Batch embed `old+new` content in `handleUpdate` (`a67c09c`).

## [0.1.0] — Initial tagged release

CI / Docker publish workflow baseline.

## Older History

Earlier commits predate structured release notes. Highlights:

- Core memory types (`identity`, `event`, `insight`, `directive`) with
  relevance × recency × importance scoring.
- MMR reranking and automatic dedup (default cosine similarity threshold 0.92).
- Qdrant vector store backend with pluggable interface.
- OpenAI embedder (default).
- MCP server over stdio with `memory_search` / `memory_add` / `memory_update` /
  `memory_delete` tools.
