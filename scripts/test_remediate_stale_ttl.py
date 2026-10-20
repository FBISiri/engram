"""Dry-run tests for remediate-stale-ttl.py's permanent/far-future skip rule.

Network-free: exercises the pure partition_candidates / is_permanent_vu logic
with in-memory fixtures. No qdrant, no urllib.
"""
import importlib.util
import os

_SPEC = importlib.util.spec_from_file_location(
    "remediate_stale_ttl",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "remediate-stale-ttl.py"),
)
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)

NOW = 1_700_000_000.0  # fixed "now" well before the 2100 threshold
THRESHOLD = mod.PERMANENT_VU_THRESHOLD  # 4102444800.0 == 2100-01-01T00:00:00Z


def _pt(pid, valid_until, created_at=None, importance=9):
    if created_at is None:
        created_at = NOW - 100 * 86400
    return (pid, {
        "valid_until": valid_until,
        "created_at": created_at,
        "importance": importance,
        "content": f"insight {pid}",
        "_collection": "engram_user",
    })


def test_in_window_stale_is_patched():
    # A finite TTL expiring ~10 days out: stale snapshot -> would be patched.
    points = [_pt("stale", NOW + 10 * 86400)]
    rows, skipped = mod.partition_candidates(points, NOW)
    assert [r["id"] for r in rows] == ["stale"]
    assert skipped == []
    assert rows[0]["collection"] == "engram_user"


def test_far_future_2100_is_skipped():
    # Hand-set permanent sentinel 2100-01-01 -> skipped, never patched.
    points = [_pt("perm", THRESHOLD)]
    rows, skipped = mod.partition_candidates(points, NOW)
    assert rows == []
    assert [r["id"] for r in skipped] == ["perm"]


def test_boundary_just_below_threshold_is_patched():
    points = [_pt("below", THRESHOLD - 1)]
    rows, skipped = mod.partition_candidates(points, NOW)
    assert [r["id"] for r in rows] == ["below"]
    assert skipped == []


def test_boundary_at_threshold_is_skipped():
    points = [_pt("at", THRESHOLD)]
    rows, skipped = mod.partition_candidates(points, NOW)
    assert rows == []
    assert [r["id"] for r in skipped] == ["at"]


def test_is_permanent_vu_predicate():
    assert mod.is_permanent_vu(THRESHOLD) is True
    assert mod.is_permanent_vu(THRESHOLD + 86400) is True
    assert mod.is_permanent_vu(THRESHOLD - 1) is False
    assert mod.is_permanent_vu(0) is False
    assert mod.is_permanent_vu(None) is False


def test_mixed_batch_partitions_correctly():
    points = [
        _pt("a", NOW + 5 * 86400),    # stale -> patch
        _pt("b", THRESHOLD),          # permanent -> skip
        _pt("c", NOW + 365 * 86400),  # ~1y out, still finite stale -> patch
        _pt("d", THRESHOLD + 10 * 86400),  # far future -> skip
    ]
    rows, skipped = mod.partition_candidates(points, NOW)
    assert sorted(r["id"] for r in rows) == ["a", "c"]
    assert sorted(r["id"] for r in skipped) == ["b", "d"]
