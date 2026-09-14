"""Consolidation 5a — adjudication-path repair tests.

Covers: two-knob budget independence (crit 3), malformed-does-not-consume-
success-budget (crit 4), missing-key abort (crit 2), and call_status coverage.
"""

import time

import pytest

from consolidation import main as main_mod
from consolidation.adjudicate import adjudicate_pair
from consolidation.config import load_config
from consolidation.metrics import Metrics
from consolidation.types import (
    Cluster,
    MemoryPoint,
    Pair,
    DS_LLM_ADJUDICATE,
    DECISION_MERGE,
    DECISION_KEEP_SEPARATE,
)

NOW = time.time()
OLD = NOW - 100 * 86400


def cfg(**overrides):
    return load_config(env={}, cli_overrides=overrides or None)


def mp(idx=0, mtype="insight"):
    return MemoryPoint(
        id=f"id-{idx}",
        collection="engram_user",
        vector=[1.0, 0.0],
        type=mtype,
        content=f"content {idx}",
        importance=5.0,
        tags=[],
        source_type="reflection",
        created_at=OLD,
        metadata={},
    )


def _merge_verdict(*args, **kwargs):
    return {
        "decision": DECISION_MERGE,
        "confidence": 0.9,
        "reasoning": "same fact",
        "merged_content": "merged",
        "keep_id": args[0].id if args else None,
        "delete_ids": [],
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "call_status": "ok",
    }


# ---------------------------------------------------------------------------
# crit 3: two-knob independence — merges gated separately from LLM calls.
# ---------------------------------------------------------------------------
def test_two_knob_independence(tmp_path, monkeypatch):
    n = 50
    points = [mp(i) for i in range(2 * n)]
    clusters = [
        Cluster(
            members_idx=[2 * i, 2 * i + 1],
            pairs=[Pair(2 * i, 2 * i + 1, 0.85)],
            max_similarity=0.85,
            zone="llm_adjudicate",
            decision_source=DS_LLM_ADJUDICATE,
        )
        for i in range(n)
    ]

    monkeypatch.setattr(main_mod.scanner, "qdrant_healthy", lambda url: True)
    monkeypatch.setattr(main_mod.scanner, "scan", lambda url, cols: (points, None))
    monkeypatch.setattr(main_mod, "build_clusters", lambda p, s, c: (clusters, []))
    monkeypatch.setattr(main_mod, "classify_all", lambda cl, p, c, now: None)
    monkeypatch.setattr(main_mod.adj, "adjudicate_pair", _merge_verdict)

    c = cfg(max_merges_per_run=1, max_llm_calls_per_run=50, min_memory_count=1)
    c.llm_api_key = "k"
    c.report_json_dir = str(tmp_path / "json")
    c.report_md_dir = str(tmp_path / "md")
    c.metrics_path = str(tmp_path / "metrics.jsonl")

    result = main_mod.run_pipeline(c, "dry-run")
    m = result["metrics"]
    assert m["adjudication_attempted"] == 50
    assert m["merges_proposed"] == 1


# ---------------------------------------------------------------------------
# crit 4: malformed response does not consume the success budget.
# ---------------------------------------------------------------------------
def test_malformed_does_not_consume_success_budget():
    def bad_call(prompt, cfg_, key):
        return {"text": "not json at all", "usage": {"input_tokens": 7, "output_tokens": 3}}

    verdict = adjudicate_pair(mp(0), mp(1), 0.85, cfg(), call_fn=bad_call, api_key="k")
    assert verdict["call_status"] == "malformed"

    m = Metrics()
    m.add_llm_usage(verdict["usage"], "malformed")
    assert m.llm_calls == 0
    assert m.llm_calls_malformed == 1
    assert m.adjudication_attempted == 1


# ---------------------------------------------------------------------------
# crit 2: missing key => abort with exit 3, no report written.
# ---------------------------------------------------------------------------
def test_missing_key_aborts_before_run(monkeypatch):
    c = cfg()
    c.enabled = True
    c.llm_api_key = ""
    monkeypatch.setattr(main_mod, "load_config", lambda **kw: c)

    def _boom(*a, **k):
        raise AssertionError("run_pipeline must not be reached on missing key")

    monkeypatch.setattr(main_mod, "run_pipeline", _boom)

    rc = main_mod.main(["--dry-run"])
    assert rc == 3


# ---------------------------------------------------------------------------
# call_status coverage.
# ---------------------------------------------------------------------------
def test_call_status_ok_merge():
    def good(prompt, cfg_, key):
        return {
            "text": '{"decision":"MERGE","confidence":0.9,"reasoning":"r","merged_content":"m"}',
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    v = adjudicate_pair(mp(0), mp(1), 0.85, cfg(), call_fn=good, api_key="k")
    assert v["decision"] == DECISION_MERGE
    assert v["call_status"] == "ok"


def test_call_status_ok_keep_separate():
    def good(prompt, cfg_, key):
        return {
            "text": '{"decision":"KEEP_SEPARATE","confidence":0.9,"reasoning":"r"}',
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    v = adjudicate_pair(mp(0), mp(1), 0.85, cfg(), call_fn=good, api_key="k")
    assert v["decision"] == DECISION_KEEP_SEPARATE
    assert v["call_status"] == "ok"


def test_call_status_no_key():
    v = adjudicate_pair(mp(0), mp(1), 0.85, cfg(), api_key="")
    assert v["call_status"] == "no_key"


def test_call_status_error():
    def raising(prompt, cfg_, key):
        raise RuntimeError("boom")

    v = adjudicate_pair(mp(0), mp(1), 0.85, cfg(), call_fn=raising, api_key="k")
    assert v["call_status"] == "error"
