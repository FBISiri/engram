# Changelog

All notable changes to Engram are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once tagged
releases begin.

## [Unreleased]

_No unreleased changes._

## [0.2.0] — 2026-04-19

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
