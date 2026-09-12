# Replay Harness — Candidate Mode & Write-Side `task_id`

Two write-side additions to the replay harness / MCP surface (S29 items 4 & 2).

## `replay --mode candidate` — offline admission recompute

A deterministic, **zero-LLM / zero-Qdrant / zero-network** report of how the
memory-admission decision set *would* change under overridden parameters, using
only the D4 `candidate` records already written to the trajectory JSONL (one per
`memory_add` / REST `POST /memories`). It never contacts a server — it is a pure
function over the log.

```
go run ./cmd/replay --mode candidate \
    --trace 'trajectories/2026-09-*.jsonl' \
    --dedup-threshold 0.88
```

### Flags (candidate mode)

| Flag | Meaning |
|------|---------|
| `--trace <file\|glob\|dir>` | Source JSONL. A directory loads all `*.jsonl` inside; a glob is expanded. |
| `--dedup-threshold <v>` | Global float (`0.88`) **or** per-type list (`insight=0.90,directive=0.92`). |
| `--importance-bounds <t=lo:hi,...>` | Per-type importance bounds (`insight=5:8,directive=6:10`). |
| `--output-dir <dir>` | Where the JSON+Markdown report is written (default `eval/reports/`). |
| `--ci` | Candidate mode never exits non-zero on the *diff*; only parse/IO errors exit non-zero. |

The read-time modes (`--trace` alone, `--baseline/--candidate`) are unchanged.

### What is resolvable vs. unresolvable (honesty contract)

- **`dedup_rejected` records** carry the dedup score in `gate_details`
  (`dedup score=%.4f against id=%s`). Under a **lower** override they flip to
  `newly_admitted`; under a higher/equal one they stay rejected. Fully resolvable.
- **`admitted` records carry NO dedup score** (the server logs the score only on
  rejection), so under any threshold change they **cannot** be re-decided →
  counted as **`unresolvable`**. Always-logging the top score is a separate job
  (S32 item **A-6**) and is deliberately *not* done here.
- **Importance is post-clamp.** An `--importance-bounds` change is only
  resolvable when the recorded value falls **outside** the new bounds (it would
  then be re-clamped to the new bound). In-bounds records are `unresolvable`
  because the original pre-clamp value is not recorded. Importance does not gate
  admission, so this is reported as a separate informational section.
- `newly_rejected` is therefore always empty here (flipping an admitted record to
  rejected needs its top score → A-6).

## Write-side `task_id`

`memory_add` (MCP) and `POST /memories` (REST) now accept an optional `task_id`
string with the same semantics as `memory_search`'s `task_id`: an event-loop task
identifier that joins writes to task outcomes (Memory Worth analysis / L3 valence
reinforcement). It is recorded on the `candidate` trajectory record
(`Record.TaskID`).

**What the client must send:**

- **MCP `memory_add`:** pass `"task_id": "<id>"` in the tool arguments.
- **REST `POST /memories`:** either a `"task_id"` JSON body field (wins) or an
  `X-Task-ID` request header (fallback).

The MCP path currently supports the `task_id` **argument** only; an `X-Task-ID`
header fallback for MCP would require wiring `WithHTTPContextFunc` into the
streamable-HTTP transport (`mcp_http.go`) and is not included here.
