# Reflection Engine

Reflection is how Engram moves from a *passive store* to *active curation*: instead of just
accumulating scattered `event`s, it periodically synthesizes them into higher-level
`insight`s that are reusable across sessions.

## What it does

`reflection_run` pulls memories that haven't been reflected yet, feeds them to an LLM to
synthesize higher-level `insight`s, writes those insights back, and marks the source memories
as reflected so they aren't re-processed. Reflection output lands in its own collection
(`engram_reflection`) so synthesized insight doesn't bleed into raw memory.

```python
reflection_check()
# → {triggered: bool, accumulated_importance: float, unreflected_count: int, skip_reason?: string}

reflection_run(dry_run=False)
# → {insights_created: int, sources_marked: int, errors: string[]}
```

Use `dry_run=True` to preview what a reflection pass would produce without writing anything.

## Trigger conditions & throttling

Reflection is throttled so it doesn't burn tokens on every write. All limits are configurable:

- **Min interval:** 2h between runs.
- **Daily cap:** at most 3 runs per day.
- **Accumulated importance threshold:** triggers once the summed importance of unreflected
  memories reaches **≥ 50**.

When a run is requested but throttled, `reflection_check` returns a `skip_reason`, and the REST
`/reflect` endpoint returns `429`.

## Configuration

Automatic triggering is **off by default** (`ENGRAM_REFLECTION_ENABLED=false`). Relevant env
vars:

| Env var | Default | Meaning |
|---|---|---|
| `ENGRAM_REFLECTION_ENABLED` | `false` | enable automatic reflection triggering |
| `ENGRAM_REFLECTION_TRIGGER` | `count` | `count` / `cron` / `manual` |
| `ENGRAM_REFLECTION_COUNT` | `10` | unreflected-memory count that triggers (count mode) |
| `ENGRAM_REFLECTION_MODEL` | `claude-sonnet-4-20250514` | LLM used for synthesis |

See [`configuration.md`](configuration.md) for the full reflection config block.

## V2 "focal" pipeline (default-off)

A 4-stage "focal" reflection pipeline exists in the codebase but ships **default-off**. It is
a more structured synthesis path (focus selection → gather → synthesize → write-back) intended
for long-running, reflection-heavy agents. Until it is promoted to default, the throttled
single-pass `reflection_run` above is the supported path. See the
[`long-cycle-reflection-heavy`](../examples/long-cycle-reflection-heavy/) example for a
reflection-tuned configuration.
