#!/usr/bin/env python3
"""Remediate stale finite TTLs on high-importance insights (P17 R4 / P2 backstop).

Background
----------
P17 (investigation 2026-10-20-p17-hard-delete-investigation.md) found that a
large stock of `type=insight && importance>=8` memories carry a FINITE
`valid_until` that was snapshotted at write time and never recomputed after the
memory's importance rose. The TTL matrix (pkg/memory/ttl.go) says insight +
importance>=8 is PERMANENT (valid_until=0), so these finite stamps are stale and
make the records eligible for the unattended expiry tick's hard delete.

The Go fix self-heals these lazily as they come up for expiry. This script is
the one-off bulk remediation: it finds the whole stock and (in patch mode)
clears `valid_until` to 0 so they become permanent immediately.

Safety
------
DRY-RUN by default: READ-ONLY scroll + report, mutates nothing. Patch mode
requires BOTH --patch AND --confirm. The report is written to /tmp (never the
vault or repo). Secrets (API key) are never printed.

Usage
-----
    python3 scripts/remediate-stale-ttl.py                      # dry-run + report
    python3 scripts/remediate-stale-ttl.py --patch --confirm    # mutate (clears valid_until)

Qdrant REST base URL: ENGRAM_QDRANT_REST_URL, else ENGRAM_QDRANT_URL host from
.env rewritten to REST port, else http://127.0.0.1:6333.
"""

import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_env_file(path):
    """Parse KEY=VALUE lines from .env (ignoring comments). Returns a dict."""
    out = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return out


def _resolve_rest_url():
    if os.environ.get("ENGRAM_QDRANT_REST_URL"):
        return os.environ["ENGRAM_QDRANT_REST_URL"].rstrip("/")
    env = _load_env_file(os.path.join(REPO_ROOT, ".env"))
    rest = os.environ.get("ENGRAM_QDRANT_REST_URL") or env.get("ENGRAM_QDRANT_REST_URL")
    if rest:
        return rest.rstrip("/")
    # Fall back to default local REST port. We do NOT derive from the gRPC URL to
    # avoid guessing the port mapping; override via ENGRAM_QDRANT_REST_URL.
    return "http://127.0.0.1:6333"


REST_URL = _resolve_rest_url()
API_KEY = os.environ.get("ENGRAM_QDRANT_API_KEY") or _load_env_file(
    os.path.join(REPO_ROOT, ".env")
).get("ENGRAM_QDRANT_API_KEY", "")

FILTER = {
    "must": [
        {"key": "type", "match": {"value": "insight"}},
        {"key": "importance", "range": {"gte": 8}},
        {"key": "valid_until", "range": {"gt": 0}},
    ]
}


def _request(method, path, body=None):
    url = f"{REST_URL}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if API_KEY:
        req.add_header("api-key", API_KEY)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise SystemExit(f"HTTP {e.code} on {method} {path}: {detail}")
    except urllib.error.URLError as e:
        raise SystemExit(f"cannot reach Qdrant at {url}: {e}")


def list_engram_collections():
    result = _request("GET", "/collections")["result"]
    names = [c["name"] for c in result.get("collections", [])]
    return sorted(n for n in names if n.startswith("engram"))


def scroll_candidates(collection):
    offset = None
    while True:
        body = {"limit": 256, "with_payload": True, "with_vector": False, "filter": FILTER}
        if offset is not None:
            body["offset"] = offset
        result = _request("POST", f"/collections/{collection}/points/scroll", body)["result"]
        for pt in result.get("points", []):
            yield pt["id"], (pt.get("payload") or {})
        offset = result.get("next_page_offset")
        if offset is None:
            break


def _content_prefix(payload, n=100):
    c = (payload.get("content") or "").replace("\n", " ").replace("\r", " ")
    return c[:n].strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch", action="store_true", help="enable mutation (requires --confirm)")
    parser.add_argument("--confirm", action="store_true", help="second confirm flag for mutation")
    args = parser.parse_args()

    do_patch = args.patch and args.confirm
    if args.patch and not args.confirm:
        raise SystemExit("--patch requires --confirm; refusing to mutate")

    mode = "PATCH" if do_patch else "DRY-RUN"
    now = time.time()
    print(f"=== remediate-stale-ttl ({mode}) ===")
    print(f"Qdrant REST: {REST_URL}")

    rows = []
    for coll in list_engram_collections():
        for pid, payload in scroll_candidates(coll):
            vu = float(payload.get("valid_until") or 0)
            ca = float(payload.get("created_at") or 0)
            imp = float(payload.get("importance") or 0)
            days = int(round((vu - ca) / 86400)) if ca > 0 else -1
            rows.append(
                {
                    "id": pid,
                    "collection": coll,
                    "importance": imp,
                    "valid_until": vu,
                    "created_at": ca,
                    "ttl_days": days,
                    "expires_in_days": (vu - now) / 86400,
                    "content_prefix": _content_prefix(payload),
                }
            )

    total = len(rows)
    hist = {}
    for r in rows:
        hist[r["ttl_days"]] = hist.get(r["ttl_days"], 0) + 1
    soon = sorted((r for r in rows if r["expires_in_days"] <= 14), key=lambda r: r["valid_until"])

    report_path = f"/tmp/p17-stale-ttl-report-{dt.datetime.now():%Y%m%d-%H%M%S}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# P17 stale-TTL remediation report ({mode})\n\n")
        f.write(f"- generated: {dt.datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"- Qdrant REST: {REST_URL}\n")
        f.write("- target set: type=insight && importance>=8 && valid_until>0\n")
        f.write(f"- total matching records: **{total}**\n")
        f.write(f"- expiring within 14 days: **{len(soon)}**\n\n")
        f.write("## TTL histogram (valid_until - created_at, whole days)\n\n")
        f.write("| days | count |\n|---|---|\n")
        for d in sorted(hist):
            f.write(f"| {d} | {hist[d]} |\n")
        f.write("\n## Records expiring within 14 days\n\n")
        f.write("| id | collection | imp | valid_until | content prefix |\n|---|---|---|---|---|\n")
        for r in soon:
            vud = dt.datetime.fromtimestamp(r["valid_until"]).strftime("%Y-%m-%d")
            f.write(
                f"| {r['id']} | {r['collection']} | {r['importance']:.0f} | {vud} | "
                f"{r['content_prefix'].replace('|', '/')} |\n"
            )

    print(f"total matching: {total}")
    print(f"expiring within 14 days: {len(soon)}")
    print(f"histogram (ttl_days -> count): {dict(sorted(hist.items()))}")
    print(f"report written: {report_path}")

    if not do_patch:
        print("DRY-RUN: no mutations. Re-run with --patch --confirm to clear valid_until.")
        return

    patched = 0
    for r in rows:
        _request(
            "POST",
            f"/collections/{r['collection']}/points/payload?wait=true",
            {"payload": {"valid_until": 0, "expiry_eligible_at": 0}, "points": [r["id"]]},
        )
        patched += 1
    print(f"PATCHED: cleared valid_until on {patched} records")


if __name__ == "__main__":
    sys.exit(main())
