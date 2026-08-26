import os
import time

import pytest

from consolidation import cluster as clu
from consolidation import scanner
from consolidation.adjudicate import adjudicate_pair
from consolidation.config import load_config
from consolidation.merge import resolve_fields, pick_superset
from consolidation.types import (
    Cluster,
    MemoryPoint,
    Pair,
    DS_AUTO_MERGE,
    DS_DEDUP_ANOMALY,
    DS_LLM_ADJUDICATE,
    DS_SKIP,
    DECISION_MERGE,
    DECISION_KEEP_SEPARATE,
    highest_trust_source,
)

NOW = time.time()
OLD = NOW - 100 * 86400  # 100 days ago (clears the 24h recency guard)


def cfg(**overrides):
    return load_config(env={}, cli_overrides=overrides or None)


def mp(idx=0, mtype="insight", importance=5.0, tags=None, source_type="reflection",
       created_at=OLD, collection="engram_user", content=None, metadata=None):
    return MemoryPoint(
        id=f"id-{idx}",
        collection=collection,
        vector=[1.0, 0.0],
        type=mtype,
        content=content if content is not None else f"content {idx}",
        importance=importance,
        tags=list(tags or []),
        source_type=source_type,
        created_at=created_at,
        metadata=metadata or {},
    )


def make_cluster(points, sim):
    return Cluster(members_idx=list(range(len(points))),
                   pairs=[Pair(0, 1, sim)], max_similarity=sim)


# ---------------------------------------------------------------------------
# R2: zone classification boundaries
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sim,expected_ds", [
    (0.80, DS_SKIP),             # < 0.82 -> distinct
    (0.819, DS_SKIP),
    (0.82, DS_LLM_ADJUDICATE),   # lower bound of LLM band
    (0.87, DS_LLM_ADJUDICATE),
    (0.88, DS_AUTO_MERGE),       # lower bound of auto-merge band
    (0.919, DS_AUTO_MERGE),
    (0.92, DS_DEDUP_ANOMALY),    # lower bound of dedup-anomaly band
    (0.99, DS_DEDUP_ANOMALY),
])
def test_zone_classification(sim, expected_ds):
    c = cfg()
    points = [mp(0), mp(1)]
    cl = make_cluster(points, sim)
    clu.classify_cluster(cl, points, c, NOW)
    assert cl.decision_source == expected_ds


def test_zone_band_labels():
    c = cfg()
    assert clu.zone_band(0.80, c) == "distinct"
    assert clu.zone_band(0.82, c) == "llm_adjudicate"
    assert clu.zone_band(0.88, c) == "auto_merge"
    assert clu.zone_band(0.92, c) == "dedup_anomaly"


# ---------------------------------------------------------------------------
# R2: type guards
# ---------------------------------------------------------------------------
def test_type_guard_directive():
    c = cfg()
    # directive in auto-merge band but below 0.95 -> never auto-merge (skip)
    points = [mp(0, "directive"), mp(1, "directive")]
    cl = make_cluster(points, 0.90)
    clu.classify_cluster(cl, points, c, NOW)
    assert cl.decision_source != DS_AUTO_MERGE
    assert cl.decision_source == DS_SKIP
    assert cl.skip_reason == "directive_below_threshold"

    # directive at >= 0.95 -> LLM-adjudicated (still never auto-merge)
    cl2 = make_cluster(points, 0.96)
    clu.classify_cluster(cl2, points, c, NOW)
    assert cl2.decision_source == DS_LLM_ADJUDICATE


def test_type_guard_identity():
    c = cfg()
    # identity in auto-merge band -> protected (skip)
    points = [mp(0, "identity"), mp(1, "identity")]
    cl = make_cluster(points, 0.90)
    clu.classify_cluster(cl, points, c, NOW)
    assert cl.decision_source == DS_SKIP
    assert cl.skip_reason == "identity_protected"

    # identity in dedup-anomaly band -> routed to LLM (full-dup check), never auto
    cl2 = make_cluster(points, 0.95)
    clu.classify_cluster(cl2, points, c, NOW)
    assert cl2.decision_source == DS_LLM_ADJUDICATE
    assert cl2.decision_source != DS_AUTO_MERGE


def test_cross_type_guard():
    c = cfg()
    # different types in auto-merge band -> never auto-merge, force LLM
    points = [mp(0, "insight"), mp(1, "event")]
    cl = make_cluster(points, 0.90)
    clu.classify_cluster(cl, points, c, NOW)
    assert cl.decision_source == DS_LLM_ADJUDICATE
    assert cl.guard == "cross_type"


# ---------------------------------------------------------------------------
# R2: recency guard
# ---------------------------------------------------------------------------
def test_recency_guard():
    c = cfg()
    points = [mp(0, created_at=NOW - 3600), mp(1, created_at=OLD)]  # one < 24h old
    cl = make_cluster(points, 0.95)
    clu.classify_cluster(cl, points, c, NOW)
    assert cl.decision_source == DS_SKIP
    assert cl.skip_reason == "recency_guard"


# ---------------------------------------------------------------------------
# R2: temporal evolution
# ---------------------------------------------------------------------------
def test_temporal_evolution():
    c = cfg()
    base = NOW - 100 * 86400
    points = [
        mp(0, created_at=base),
        mp(1, created_at=base + 8 * 86400),   # spans > 7 days
        mp(2, created_at=base + 9 * 86400),
    ]
    cl = Cluster(members_idx=[0, 1, 2],
                 pairs=[Pair(0, 1, 0.90), Pair(1, 2, 0.90)], max_similarity=0.90)
    clu.classify_cluster(cl, points, c, NOW)
    assert cl.guard == "temporal"
    assert cl.decision_source == DS_LLM_ADJUDICATE

    # 2 members never count as evolution
    assert clu.is_temporal_evolution([points[0], points[1]], c) is False


# ---------------------------------------------------------------------------
# R4: field resolution + provenance inheritance
# ---------------------------------------------------------------------------
def test_merge_field_resolution():
    members = [
        mp(0, importance=6.0, tags=["frank", "pref"], source_type="reflection", created_at=200),
        mp(1, importance=5.0, tags=["pref", "note"], source_type="user_input", created_at=100),
    ]
    r = resolve_fields(members, DS_AUTO_MERGE, now=NOW)
    assert r["importance"] == 6.0
    assert "consolidated" in r["tags"]
    assert set(r["tags"]) == {"frank", "pref", "note", "consolidated"}
    assert r["source_type"] == "user_input"   # highest trust
    assert r["created_at"] == 100             # earliest
    assert r["metadata"]["merge_reason"] == DS_AUTO_MERGE
    assert r["metadata"]["caller"] == "consolidation"


def test_provenance_inheritance():
    members = [
        mp(0, source_type="reflection",
           metadata={"source_type": "reflection",
                     "provenance_history": [{"source_type": "reflection", "merged_at": 1}]}),
        mp(1, source_type="user_input", metadata={"source_type": "user_input"}),
    ]
    r = resolve_fields(members, DS_LLM_ADJUDICATE, now=NOW)
    ph = r["metadata"]["provenance_history"]
    # inherits the prior entry + appends a consolidation record
    assert any(e.get("source_type") == "reflection" for e in ph)
    assert ph[-1]["source_type"] == "consolidation"
    assert ph[-1]["merged_from"] == ["id-0", "id-1"]
    assert r["source_type"] == "user_input"


def test_highest_trust_source():
    assert highest_trust_source(["reflection", "user_input", "unknown"]) == "user_input"
    assert highest_trust_source(["reflection", "web_search"]) == "web_search"
    assert highest_trust_source([None, None]) == "unknown"


def test_pick_superset():
    members = [mp(0, importance=3.0, content="short"),
               mp(1, importance=7.0, content="the longer more important content")]
    assert pick_superset(members).id == "id-1"


# ---------------------------------------------------------------------------
# R7: cluster max size split
# ---------------------------------------------------------------------------
def test_cluster_max_size():
    c = cfg()
    # 25 identical vectors -> one giant component that must be split to <= 20
    points = [MemoryPoint(id=f"id-{i}", collection="engram_user", vector=[1.0, 0.0, 0.0],
                          type="insight", content=f"c{i}", importance=5.0, tags=[],
                          source_type="reflection", created_at=OLD) for i in range(25)]
    sim = scanner.similarity_matrix(points)
    clusters, fringe = clu.build_clusters(points, sim, c)
    assert clusters, "expected at least one cluster"
    for cl in clusters:
        assert len(cl.members_idx) <= c.max_cluster_size
    total = sum(len(cl.members_idx) for cl in clusters)
    assert total == 25  # no member lost in the split


def test_within_collection_only():
    c = cfg()
    # two identical vectors in DIFFERENT collections must not cluster together
    points = [
        MemoryPoint(id="a", collection="engram_user", vector=[1.0, 0.0], type="insight",
                    content="x", importance=5.0, tags=[], source_type="reflection", created_at=OLD),
        MemoryPoint(id="b", collection="engram_reflection", vector=[1.0, 0.0], type="insight",
                    content="x", importance=5.0, tags=[], source_type="reflection", created_at=OLD),
    ]
    sim = scanner.similarity_matrix(points)
    clusters, fringe = clu.build_clusters(points, sim, c)
    assert clusters == []
    assert fringe == []


# ---------------------------------------------------------------------------
# R8: config priority
# ---------------------------------------------------------------------------
def test_config_priority(tmp_path):
    # defaults
    assert load_config(env={}).auto_merge_threshold == 0.88

    # file overrides defaults
    cfgfile = tmp_path / "c.yaml"
    cfgfile.write_text(
        "consolidation:\n  thresholds:\n    auto_merge: 0.89\n"
    )
    assert load_config(config_path=str(cfgfile), env={}).auto_merge_threshold == 0.89

    # env overrides file
    env = {"ENGRAM_CONSOLIDATION_AUTO_MERGE_THRESHOLD": "0.90"}
    assert load_config(config_path=str(cfgfile), env=env).auto_merge_threshold == 0.90

    # CLI overrides env
    got = load_config(config_path=str(cfgfile), env=env,
                      cli_overrides={"auto_merge_threshold": 0.915})
    assert got.auto_merge_threshold == 0.915


def test_config_enabled_env():
    assert load_config(env={"ENGRAM_CONSOLIDATION_ENABLED": "true"}).enabled is True
    assert load_config(env={"ENGRAM_CONSOLIDATION_ENABLED": "false"}).enabled is False
    assert load_config(env={}).enabled is False


def test_config_validation_rejects_bad_threshold(tmp_path):
    cfgfile = tmp_path / "bad.yaml"
    cfgfile.write_text("consolidation:\n  thresholds:\n    auto_merge: 1.5\n")
    with pytest.raises(ValueError):
        load_config(config_path=str(cfgfile), env={})


def test_config_validation_rejects_bad_ordering():
    with pytest.raises(ValueError):
        # auto_merge < llm_adjudicate violates ordering
        load_config(env={}, cli_overrides={"auto_merge_threshold": 0.70})


# ---------------------------------------------------------------------------
# R3: LLM confidence threshold + fallbacks
# ---------------------------------------------------------------------------
def _fake_call(decision, confidence):
    def call_fn(prompt, cfg_, api_key):
        text = (
            '{"decision": "%s", "confidence": %s, "reasoning": "r", '
            '"merged_content": "m", "keep_id": "id-0", "delete_ids": ["id-1"]}'
            % (decision, confidence)
        )
        return {"text": text, "usage": {"input_tokens": 10, "output_tokens": 5}}
    return call_fn


def test_llm_confidence_threshold():
    c = cfg()
    a, b = mp(0), mp(1)
    # low confidence merge -> downgraded to KEEP_SEPARATE
    low = adjudicate_pair(a, b, 0.85, c, call_fn=_fake_call("MERGE", 0.5), api_key="x")
    assert low["decision"] == DECISION_KEEP_SEPARATE

    # high confidence merge -> honored
    high = adjudicate_pair(a, b, 0.85, c, call_fn=_fake_call("MERGE", 0.9), api_key="x")
    assert high["decision"] == DECISION_MERGE
    assert high["merged_content"] == "m"
    assert high["usage"]["input_tokens"] == 10


def test_llm_missing_key_keeps_separate():
    c = cfg()
    a, b = mp(0), mp(1)
    r = adjudicate_pair(a, b, 0.85, c, api_key="")
    assert r["decision"] == DECISION_KEEP_SEPARATE


def test_llm_malformed_response_keeps_separate():
    c = cfg()
    a, b = mp(0), mp(1)

    def bad_call(prompt, cfg_, api_key):
        return {"text": "not json at all", "usage": {"input_tokens": 1, "output_tokens": 1}}

    r = adjudicate_pair(a, b, 0.85, c, call_fn=bad_call, api_key="x")
    assert r["decision"] == DECISION_KEEP_SEPARATE


def test_llm_api_error_keeps_separate():
    c = cfg()
    a, b = mp(0), mp(1)

    def boom(prompt, cfg_, api_key):
        raise RuntimeError("network down")

    r = adjudicate_pair(a, b, 0.85, c, call_fn=boom, api_key="x")
    assert r["decision"] == DECISION_KEEP_SEPARATE
