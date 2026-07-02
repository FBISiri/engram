"""Bootstrap an Engram project memory collection for Claude Code MCP integration.

Seeds 10 example memories for a hypothetical "payments-service" project via the
Engram REST API (POST /memories). Real embeddings are generated server-side so
memories are immediately retrievable via memory_search.

Idempotent: Engram's ~0.92 semantic auto-dedup prevents re-seeding exact
duplicates. Re-running is safe.

Usage:
    pip install requests python-dotenv
    cp .env.example .env          # fill in ENGRAM_OPENAI_API_KEY
    python bootstrap.py

    # Point at a non-default Engram server
    python bootstrap.py --base-url http://myserver:8080

    # Preview payloads without seeding
    python bootstrap.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_BASE_URL = (
    f"http://{os.getenv('ENGRAM_HOST', 'localhost')}"
    f":{os.getenv('ENGRAM_HTTP_PORT', '8080')}"
)
COLLECTION = os.getenv("ENGRAM_COLLECTION_NAME", "my_project")

# ── Example memories (payments-service project) ────────────────────────────────
#
# These are realistic examples for a backend service. Replace with your own
# project's directives, decisions, and conventions.

MEMORIES: list[dict[str, Any]] = [
    # ── Directives (importance 7-10): coding standards the agent always follows ──
    {
        "content": (
            "RULE: All database queries must use parameterised statements. "
            "Never interpolate user input into SQL strings. No exceptions."
        ),
        "type": "directive",
        "importance": 10,
        "tags": ["security", "sql", "must-follow"],
    },
    {
        "content": (
            "RULE: API responses must follow the standard envelope: "
            '{"data": ..., "error": null, "meta": {"request_id": "..."}}. '
            "Never return bare payloads or non-standard error shapes."
        ),
        "type": "directive",
        "importance": 9,
        "tags": ["api", "response-format", "must-follow"],
    },
    {
        "content": (
            "RULE: All Python must pass ruff (E,W,F,I) and mypy (strict). "
            "Type hints on every function signature. "
            "Never use mutable default arguments. Always pathlib over os.path."
        ),
        "type": "directive",
        "importance": 8,
        "tags": ["python", "linting", "standards"],
    },
    # ── Insights (importance 6-8): architectural decisions from past sessions ──
    {
        "content": (
            "The payments service uses cursor-based pagination (not offset). "
            "ADR-014 (2026-03-15): offset pagination caused drift under concurrent inserts. "
            "Cursor field: created_at DESC + id. Never use OFFSET."
        ),
        "type": "insight",
        "importance": 8,
        "tags": ["pagination", "adr", "payments"],
    },
    {
        "content": (
            "Stripe webhook verification: always validate X-Stripe-Signature before "
            "processing. Timing-safe compare only (hmac.compare_digest). "
            "Returning 200 to unknown events prevents Stripe retry loops — confirmed 2026-04."
        ),
        "type": "insight",
        "importance": 7,
        "tags": ["stripe", "webhooks", "security"],
    },
    {
        "content": (
            "The payments DB has a write-follower replica lag of ~50ms under peak load. "
            "Read-after-write queries must go to primary (use db.primary() context manager). "
            "Discovered during 2026-05-10 incident (P1 #738)."
        ),
        "type": "insight",
        "importance": 7,
        "tags": ["database", "replication", "gotcha", "incident"],
    },
    # ── Identity (importance 7-9): project context and team preferences ──
    {
        "content": (
            "Project: payments-service (Python 3.12, FastAPI, PostgreSQL 16, Stripe v3). "
            "Monorepo: pi-mono/packages/payments-service. "
            "Main branch: main. CI: GitHub Actions (runs ruff + mypy + pytest on every PR)."
        ),
        "type": "identity",
        "importance": 9,
        "tags": ["project", "stack", "ci"],
    },
    {
        "content": (
            "Team lead: Alex Chen (alex@example.com). "
            "Prefers concise responses, bullet points over paragraphs. "
            "Always show a summary of changes before writing code. "
            "Peak focus 10am-12pm HKT — don't schedule interruptions."
        ),
        "type": "identity",
        "importance": 7,
        "tags": ["team", "preferences", "communication"],
    },
    # ── Events (importance 4-6): recent task history ──
    {
        "content": (
            "2026-06-28: Upgraded Stripe SDK v2 → v3. "
            "Kept legacy v1 webhook handlers active for 90-day overlap (until 2026-09-28). "
            "PR #441 merged. Key risk: v3 uses different idempotency key format."
        ),
        "type": "event",
        "importance": 6,
        "tags": ["stripe", "upgrade", "pr-441"],
    },
    {
        "content": (
            "2026-07-01: Added /payments/refund endpoint. "
            "Partial refunds supported (amount_cents param). "
            "Full refunds trigger cancellation of related subscription line items. "
            "PR #442, all tests green."
        ),
        "type": "event",
        "importance": 5,
        "tags": ["refund", "endpoint", "pr-442"],
    },
]


# ── Seeding ────────────────────────────────────────────────────────────────────

def seed_memories(base_url: str, collection: str, dry_run: bool) -> None:
    url = f"{base_url}/memories"

    if dry_run:
        print(f"[dry-run] Would seed {len(MEMORIES)} memories to {url}")
        print(f"[dry-run] Collection: {collection}")
        print()
        payloads = [{**m, "collection": collection} for m in MEMORIES]
        print(json.dumps(payloads, indent=2, ensure_ascii=False))
        print(f"\n[dry-run] {len(payloads)} payloads. No network calls made.")
        return

    print(f"Seeding {len(MEMORIES)} memories → {url} (collection: {collection})")
    ok = 0
    for mem in MEMORIES:
        payload = {**mem, "collection": collection}
        try:
            r = requests.post(url, json=payload, timeout=30)
            if r.status_code in (200, 201):
                ok += 1
                label = f"[{mem['type']}] {mem['content'][:65]}..."
                print(f"  ✅ {label}")
            else:
                print(
                    f"  ❌ HTTP {r.status_code}: {r.text[:120]}",
                    file=sys.stderr,
                )
        except Exception as exc:  # noqa: BLE001
            print(f"  ❌ error: {exc}", file=sys.stderr)
        time.sleep(0.05)  # avoid hammering the embedder

    print(f"\nDone: {ok}/{len(MEMORIES)} seeded.")
    if ok < len(MEMORIES):
        sys.exit(1)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed Engram project memory for Claude Code MCP integration."
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Engram REST API base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--collection",
        default=COLLECTION,
        help=f"Target collection name (default: {COLLECTION})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print payloads as JSON and make no network calls.",
    )
    args = parser.parse_args()
    seed_memories(args.base_url.rstrip("/"), args.collection, args.dry_run)


if __name__ == "__main__":
    main()
