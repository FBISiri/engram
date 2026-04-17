# Changelog

All notable changes to Engram are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once tagged
releases begin.

## [Unreleased]

### Added
- **TTL auto-calculator** — `valid_until` is now derived from a type × importance
  matrix when callers omit it, and injected at the `handleAdd` / `handleUpdate`
  layer (`ec21d8d`).
- **Deep `/health` endpoint** — the liveness probe now pings Qdrant and bypasses
  Bearer auth so external probes work without credentials (`8d185b1`).

### Changed
- **Dream Engine scroll pagination** — `Scroll` calls inside the Dream pipeline
  go through a new `scrollAll` helper to avoid truncated result windows
  (`acc7019`).
- **Reflection / Dream hot paths** — loop-hoisted time calculations and propagated
  `context.Context` through to `callHaiku` for cancellation (`a3e5e45`).
- **Consolidation semantics** — superseded memories are now marked and cleaned
  up consistently across `orient` and `gather` phases (`4b824a8`).
- **Unreflected-memory fetch** — indexed `reflected_at` filter replaces the prior
  full-scan path (`b544396`).

## [0.3.0] — HTTP Transport & Dream Engine

### Added
- **HTTP transport** with `/reflect`, `/reflect/check`, `/health` endpoints
  (`0977860`) and documentation updates (`c50d2f4`).
- **Dream Engine** — autonomous 4-phase memory consolidation (Orient → Gather →
  Consolidate → Prune) with three gate conditions (time, volume, PID lock)
  (`95fccfe`).
- **Voyage AI embedder** as an alternative to OpenAI (`95fccfe`).
- **Memory expiry** — `valid_until` field with a background cleanup goroutine
  (`95fccfe`).

### Fixed
- Removed hardcoded 1536-dimension zero-vector fallback in Dream and Reflection
  (`629102d`).
- Gate2 now uses `new_memories_since_last_run` instead of `session_count`;
  Gate1 relaxed from 24h to 20h (`96570dd`).
- `mockStore` implements `DeleteExpired` to satisfy the `memory.Store` interface
  (`5a746f9`).

### Infrastructure
- GitHub Actions workflow publishes Docker images to `imsiri/engram` on Docker
  Hub (`8c90c5a`, `19f51da`).

## [0.2.0] — Reflection Engine & Write-Through Sync

### Added
- **Reflection Engine v1** — lightweight periodic synthesis of `insight` memories
  from unreflected events, with count / cron / manual triggers (`f9ace98`) and
  Gate1/Gate2/Gate3 unit tests (`c87887a`).
- **Write-through + Ring Buffer** — BoltDB-backed commit log and
  `WriteThroughStore` for durability and crash recovery
  (`a0eeae3`, `cd36c70`).
- Extended memory schema with `valid_until`, `superseded_by`, `access_count`,
  `last_accessed_at` (`c195fd4`), exposed via the MCP server layer (`5fdf3c0`).
- `CONTRIBUTING.md` (`9eaad0c`).

### Fixed
- MCP tool names use underscores instead of dots for Claude API compatibility
  (`af94f5a`).

### Changed
- Module path renamed from `anthropics/engram` to `FBISiri/engram` (`6808739`).

## Older History

Earlier commits predate structured release notes. Highlights:

- Core memory types (`identity`, `event`, `insight`, `directive`) with
  relevance × recency × importance scoring.
- MMR reranking and automatic dedup (default cosine similarity threshold 0.92).
- Qdrant vector store backend with pluggable interface.
- OpenAI embedder (default).
- MCP server over stdio with `memory_search` / `memory_add` / `memory_update` /
  `memory_delete` tools.
