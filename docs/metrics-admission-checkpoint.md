# Admission & Checkpoint metrics (B#3)

Two Prometheus counters, registered in the existing registry inside
`metrics.New()` (pkg/metrics/metrics.go). Both are guarded at every call site by
`if s.metrics != nil`.

## engram_admission_total{type,decision}

- `type`     — memory type (event, directive, identity, …).
- `decision` — admission outcome, mirrors `trajectory.Record.AdmissionDecision`
  (pkg/trajectory/trajectory.go): `admitted | dedup_rejected | rate_limited | error`.

Incremented where the candidate trajectory record is built:
- pkg/server/server.go (handleAdd deferred candidate closure)
- pkg/server/crud.go (REST create deferred candidate closure)

One increment per write attempt, tagged with the final decision reached by the
admission gates.

## engram_checkpoint_total{kind,bucket}

- `kind`   — write-discipline checkpoint id: `cp1` (near-duplicate advisory) or
  `cp2` (importance-inflation advisory).
- `bucket` — outcome bucket.

Bucket taxonomy (derived directly from the advisory functions in
pkg/server/advisory.go, which return `*Advisory` or `nil`):

| bucket     | meaning                                                        |
|------------|----------------------------------------------------------------|
| `advisory` | the checkpoint produced an advisory (near-dup found / importance inflated) |
| `clean`    | the checkpoint ran and produced no advisory                    |

Only these two buckets are implemented, because they are the only outcomes the
current code can distinguish unambiguously (advisory emitted vs not). CP3/CP4
are intentionally out of scope for this counter. Sites:
- CP1: `cp1DedupAdvisory` call in handleAdd (server.go) and REST create (crud.go).
- CP2: `cp2Importance` call in the same two blocks.

Counters only increment when `WriteCheckpointsEnabled` (the checkpoint master
switch) is on, since that gate wraps the CP1/CP2 calls.
