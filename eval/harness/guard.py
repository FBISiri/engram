"""
guard.py — Prefix guard for eval collection safety.

HARD CONSTRAINT: ALL collection operations in the harness must pass through
guardCollection(). This is the single line of defense protecting production
collections from accidental writes/deletes.

Production collections (MUST NOT touch):
  engram_user, engram_agent_self, engram_reflection, siri, bmo, engram

Eval collections (allowed):
  engram_eval_* prefix ONLY
"""

EVAL_PREFIX = "engram_eval_"

PRODUCTION_COLLECTIONS = frozenset([
    "engram_user",
    "engram_agent_self",
    "engram_reflection",
    "engram",
    "siri",
    "bmo",
])


class GuardViolation(Exception):
    """Raised when a collection name violates the engram_eval_* prefix guard."""
    pass


def guard_collection(name: str) -> str:
    """
    Validates that name has the engram_eval_* prefix.

    This is the SINGLE guard function. It must be called before:
      - Creating a collection
      - Deleting a collection
      - Writing/reading memories in an eval context
      - Any Qdrant operation targeting a collection by name

    Returns the name unchanged on success.
    Raises GuardViolation on failure.
    """
    if not name:
        raise GuardViolation(
            f"SAFETY VIOLATION: empty collection name rejected"
        )
    if not name.startswith(EVAL_PREFIX):
        raise GuardViolation(
            f"SAFETY VIOLATION: collection {name!r} does not have required "
            f"{EVAL_PREFIX!r} prefix. Refusing to operate on non-eval collection. "
            f"Production collections must never be touched by the eval harness."
        )
    if name in PRODUCTION_COLLECTIONS:
        raise GuardViolation(
            f"SAFETY VIOLATION: {name!r} is a known production collection. "
            f"This should never happen given the prefix check, but defense in depth."
        )
    return name


def guard_collections(names: list) -> list:
    """Apply guard_collection to a list of names. Returns the list unchanged."""
    for name in names:
        guard_collection(name)
    return names


# ---------------------------------------------------------------------------
# Unit tests (run with: python -m pytest guard.py or python guard.py)
# ---------------------------------------------------------------------------

def _run_tests():
    import traceback

    tests = [
        # (name, expect_error)
        ("engram_eval_user", False),
        ("engram_eval_reflection", False),
        ("engram_eval_test_abc_123", False),
        ("engram_eval_", False),          # just the prefix — allowed
        ("engram_user", True),            # production collection
        ("engram_reflection", True),       # production collection
        ("engram_agent_self", True),       # production collection
        ("siri", True),                    # production collection
        ("bmo", True),                     # production collection
        ("engram", True),                  # legacy production collection
        ("production", True),              # no prefix
        ("", True),                        # empty name
        ("ENGRAM_EVAL_user", True),        # wrong case
        ("eval_user", True),               # missing engram_ prefix
    ]

    passed = 0
    failed = 0
    for name, expect_err in tests:
        try:
            guard_collection(name)
            got_err = False
        except GuardViolation:
            got_err = True
        except Exception as e:
            got_err = True

        if got_err == expect_err:
            passed += 1
        else:
            failed += 1
            print(f"FAIL: guard_collection({name!r}) got_err={got_err}, want_err={expect_err}")

    print(f"guard.py tests: {passed}/{passed+failed} passed", "✓" if failed == 0 else "✗")
    return failed == 0


if __name__ == "__main__":
    import sys
    ok = _run_tests()
    sys.exit(0 if ok else 1)
