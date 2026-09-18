#!/usr/bin/env python3
"""Migrate legacy dotted-literal provenance payload keys into the nested metadata dict.

Background
----------
A bug in provenanceMerge / the §5.3 legacy-backfill branch wrote payload fields
with DOTTED LITERAL keys ("metadata.source_type", "metadata.provenance_history")
via Qdrant SetPayload. Qdrant stored them verbatim as top-level keys instead of
inside the nested "metadata" object, so the Go read path (which reads the nested
"metadata" dict) never sees them. This script folds those dotted values back
into the nested metadata dict and removes the dotted keys.

Conflict rule
-------------
The DOTTED key holds the NEWER value (it was written by the most recent merge),
so on disagreement the dotted value WINS. In dry-run both values are printed so
a human can eyeball before applying.

Safety
------
DRY-RUN by default: prints what WOULD change and mutates nothing. Mutation only
happens with the explicit --apply flag. Re-running after --apply is a no-op
(idempotent) because the dotted keys are removed.

Usage
-----
    python3 scripts/migrate-provenance-dotted-keys.py            # dry-run
    python3 scripts/migrate-provenance-dotted-keys.py --apply    # mutate

REST base URL overridable via ENGRAM_QDRANT_REST_URL (default http://127.0.0.1:6333).
"""

import argparse
import json
import os
import sys
import urllib.request
import urllib.error

REST_URL = os.environ.get("ENGRAM_QDRANT_REST_URL", "http://127.0.0.1:6333").rstrip("/")

COLLECTIONS = [
    "engram_agent_self",
    "engram_user",
    "engram_reflection",
    "engram_pigo",
]

DOTTED_SOURCE_TYPE = "metadata.source_type"
DOTTED_PROVENANCE = "metadata.provenance_history"
DOTTED_KEYS = [DOTTED_SOURCE_TYPE, DOTTED_PROVENANCE]


def _request(method, path, body=None):
    url = f"{REST_URL}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise SystemExit(f"HTTP {e.code} on {method} {path}: {detail}")
    except urllib.error.URLError as e:
        raise SystemExit(f"cannot reach Qdrant at {url}: {e}")


def scroll_all(collection):
    """Yield (id, payload) for every point in a collection."""
    offset = None
    while True:
        body = {"limit": 256, "with_payload": True, "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        result = _request("POST", f"/collections/{collection}/points/scroll", body)["result"]
        for pt in result.get("points", []):
            yield pt["id"], (pt.get("payload") or {})
        offset = result.get("next_page_offset")
        if offset is None:
            break


def migrate_point(collection, point_id, payload, apply):
    """Fold dotted keys into nested metadata for one point. Returns True if dirty."""
    dirty = [k for k in DOTTED_KEYS if k in payload]
    if not dirty:
        return False

    nested = dict(payload.get("metadata") or {})

    print(f"  point {point_id} (collection {collection}):")
    for dkey in dirty:
        nested_key = dkey.split(".", 1)[1]  # "source_type" / "provenance_history"
        dotted_val = payload[dkey]
        nested_val = nested.get(nested_key, "<absent>")
        conflict = nested_key in nested and nested[nested_key] != dotted_val
        marker = "  CONFLICT (dotted wins)" if conflict else ""
        print(f"    {dkey}: dotted={json.dumps(dotted_val)}  nested={json.dumps(nested_val)}{marker}")
        # Dotted is the newer value and wins.
        nested[nested_key] = dotted_val

    if apply:
        _request(
            "POST",
            f"/collections/{collection}/points/payload?wait=true",
            {"payload": {"metadata": nested}, "points": [point_id]},
        )
        _request(
            "POST",
            f"/collections/{collection}/points/payload/delete?wait=true",
            {"keys": DOTTED_KEYS, "points": [point_id]},
        )
        print(f"    action: APPLIED (folded {len(dirty)} key(s), deleted dotted keys)")
    else:
        print(f"    action: WOULD fold {len(dirty)} key(s) into metadata and delete dotted keys")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="perform mutations (default: dry-run)")
    args = parser.parse_args()

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"=== migrate-provenance-dotted-keys ({mode}) ===")
    print(f"Qdrant REST: {REST_URL}")

    total_dirty = 0
    for collection in COLLECTIONS:
        print(f"\n[collection: {collection}]")
        count = 0
        for point_id, payload in scroll_all(collection):
            if migrate_point(collection, point_id, payload, args.apply):
                count += 1
        print(f"  {count} dirty point(s) in {collection}")
        total_dirty += count

    print(f"\n=== total dirty points: {total_dirty} ({mode}) ===")
    if not args.apply and total_dirty:
        print("Re-run with --apply to migrate.")


if __name__ == "__main__":
    main()
