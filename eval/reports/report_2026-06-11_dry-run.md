# Engram Eval Report -- 2026-06-11T01:41:09.299125Z

**Version**: core_v1
**Overall**: 6/26 = 23.1%
**Gate**: FAIL (min 80% overall, 65% per-category)

## Category Breakdown

| Category | Pass | Total | Score | Gate |
|---|---|---|---|---|
| retrieve_precision | 1 | 8 | 12.5% | FAIL |
| dedup_accuracy | 4 | 6 | 66.7% | PASS |
| recency_bias | 0 | 4 | 0.0% | FAIL |
| cross_collection | 1 | 4 | 25.0% | FAIL |
| trajectory_replay | 0 | 4 | 0.0% | FAIL |

## Task Results

| ID | Category | Pass | Reason |
|---|---|---|---|
| RP-01 | retrieve_precision | FAIL | missing required eval_id 'mem-rp01-target' (got: []) |
| RP-02 | retrieve_precision | FAIL | missing required eval_id 'mem-rp02-target' (got: []) |
| RP-03 | retrieve_precision | FAIL | missing required eval_id 'mem-rp03-target' (got: []) |
| RP-04 | retrieve_precision | FAIL | missing required eval_id 'mem-rp04-target' (got: []) |
| RP-05 | retrieve_precision | FAIL | missing required eval_id 'mem-rp05-a' (got: []) |
| RP-06 | retrieve_precision | FAIL | missing required eval_id 'mem-rp06-target-directive' (got: []) |
| RP-07 | retrieve_precision | FAIL | missing required eval_id 'mem-rp07-target' (got: []) |
| RP-08 | retrieve_precision | PASS | all checks passed |
| DD-01 | dedup_accuracy | PASS | all checks passed |
| DD-02 | dedup_accuracy | PASS | all checks passed |
| DD-03 | dedup_accuracy | PASS | anchor score not measured (dry-run); skipping range check |
| DD-04 | dedup_accuracy | FAIL | count did not increase: 0->0 |
| DD-05 | dedup_accuracy | FAIL | count did not increase: 0->0 |
| DD-06 | dedup_accuracy | PASS | expected error raised (OK) |
| RC-01 | recency_bias | FAIL | rank_before: one of {'first_eval_id': 'mem-rc01-new', 'second_eval_id': 'mem-rc0 |
| RC-02 | recency_bias | FAIL | missing required eval_id 'mem-rc02-old-important' (got: []) |
| RC-03 | recency_bias | FAIL | missing required eval_id 'mem-rc03-inwindow' (got: []) |
| RC-04 | recency_bias | FAIL | missing required eval_id 'mem-rc04-valid' (got: []) |
| XC-01 | cross_collection | FAIL | missing required eval_id 'mem-xc01-target-reflection' (got: []) |
| XC-02 | cross_collection | PASS | all checks passed |
| XC-03 | cross_collection | FAIL | missing required eval_id 'mem-xc03-user' (got: []) |
| XC-04 | cross_collection | FAIL | missing required eval_id 'mem-xc04-user-raw' (got: []) |
| TR-01 | trajectory_replay | FAIL | missing required eval_id 'mem-tr01-seed' (got: []) |
| TR-02 | trajectory_replay | FAIL | missing required eval_id 'mem-tr02-seed' (got: []) |
| TR-03 | trajectory_replay | FAIL | missing required eval_id 'mem-tr03-seed' (got: []) |
| TR-04 | trajectory_replay | FAIL | missing required eval_id 'mem-tr04-seed' (got: []) |
