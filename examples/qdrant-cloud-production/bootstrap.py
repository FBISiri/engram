#!/usr/bin/env python3
"""
bootstrap.py — Config 6: Qdrant Cloud Production
Seeds 10 example memories for a production AI assistant deployment.
Scenario: customer support agent with accumulated operational knowledge.

Usage:
    pip install requests python-dotenv
    python bootstrap.py [--dry-run] [--base-url http://localhost:8080]
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

try:
    import requests
except ImportError:
    print("ERROR: 'requests' not installed. Run: pip install requests", file=sys.stderr)
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Optional — environment may already be set


def main():
    parser = argparse.ArgumentParser(description="Seed production example memories into Engram")
    parser.add_argument("--dry-run", action="store_true", help="Print memories without sending")
    parser.add_argument("--base-url", default=None, help="Engram HTTP base URL (overrides env)")
    args = parser.parse_args()

    host = os.environ.get("ENGRAM_HOST", "localhost")
    port = os.environ.get("ENGRAM_HTTP_PORT", "8080")
    base_url = args.base_url or f"http://{host}:{port}"
    api_key = os.environ.get("ENGRAM_API_KEY", "")

    if not api_key:
        print("WARNING: ENGRAM_API_KEY not set — HTTP auth may be required", file=sys.stderr)

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Health check
    if not args.dry_run:
        try:
            resp = requests.get(f"{base_url}/health", timeout=5)
            resp.raise_for_status()
            print(f"✓ Engram reachable at {base_url}")
        except Exception as e:
            print(f"✗ Cannot reach Engram at {base_url}: {e}", file=sys.stderr)
            print("  Is engram running with ENGRAM_TRANSPORT=http or ENGRAM_TRANSPORT=both?")
            sys.exit(1)

    memories = [
        # ── Directives (permanent: operational rules that never expire) ─────────
        {
            "content": "Always confirm successful order cancellations by checking order status API before telling the customer the cancellation was processed.",
            "type": "directive",
            "importance": 9,
            "tags": ["order-management", "customer-trust", "verification"],
        },
        {
            "content": "Escalate refund requests above $500 to the senior support queue (tier-2). Do not approve these unilaterally.",
            "type": "directive",
            "importance": 9,
            "tags": ["refunds", "escalation", "policy"],
        },
        {
            "content": "When a customer reports 'payment declined', first check if their card on file is expired (>80% of cases) before suggesting bank contact.",
            "type": "directive",
            "importance": 8,
            "tags": ["payments", "troubleshooting", "triage"],
        },
        # ── Insights (long-lived: operational intelligence with ~90d TTL) ────────
        {
            "content": "Shipping delays spike every year in the first two weeks of November (pre-Black Friday carrier overload). Proactively set customer expectations during this window.",
            "type": "insight",
            "importance": 7,
            "tags": ["shipping", "seasonal", "customer-expectations"],
        },
        {
            "content": "The 'track my order' flow breaks for orders placed before the 2025-03 platform migration — those orders use the legacy tracking API (v1) which requires the 8-digit legacy order ID, not the new UUIDs.",
            "type": "insight",
            "importance": 8,
            "tags": ["tracking", "legacy-api", "migration", "incident-history"],
        },
        {
            "content": "Customers who use the iOS app tend to have billing address mismatches because the app's address autocomplete drops the apartment/unit field. Ask iOS users to double-check their billing address before blaming the card issuer.",
            "type": "insight",
            "importance": 6,
            "tags": ["ios", "payments", "address", "bug"],
        },
        # ── Events (short-lived: recent incidents and changes, 7-30d TTL) ────────
        {
            "content": "2025-06-28: Scheduled maintenance on the returns portal from 02:00–04:00 UTC. Return submissions during this window may silently fail — check portal status page before processing.",
            "type": "event",
            "importance": 7,
            "tags": ["maintenance", "returns", "time-sensitive"],
        },
        {
            "content": "2025-07-01: Promo code SUMMER25 is live (20% off orders >$75, expires 2025-07-15). Applies to all product categories except gift cards.",
            "type": "event",
            "importance": 5,
            "tags": ["promotions", "discount", "time-sensitive"],
        },
        # ── Identity (permanent: agent self-knowledge) ────────────────────────────
        {
            "content": "This support agent handles Tier-1 customer inquiries for an e-commerce platform: orders, payments, shipping, returns, and account issues. Complex billing disputes and fraud escalate to human Tier-2.",
            "type": "identity",
            "importance": 8,
            "tags": ["scope", "tier-1", "agent-role"],
        },
        {
            "content": "Response tone policy: empathetic but efficient. Acknowledge the customer's frustration in the first sentence; resolve or escalate by the third. Avoid corporate filler phrases ('I apologize for the inconvenience').",
            "type": "identity",
            "importance": 7,
            "tags": ["tone", "communication-policy", "agent-role"],
        },
    ]

    print(f"\nSeeding {len(memories)} memories to {base_url}...\n")

    ok = 0
    for i, mem in enumerate(memories, 1):
        label = mem["content"][:60] + ("..." if len(mem["content"]) > 60 else "")
        if args.dry_run:
            print(f"[DRY RUN {i:02d}] [{mem['type']}] {label}")
            ok += 1
            continue

        try:
            resp = requests.post(
                f"{base_url}/memories",
                headers=headers,
                json=mem,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            mem_id = data.get("id", "?")
            print(f"  [{i:02d}] ✓ {mem_id[:8]}...  [{mem['type']}] {label}")
            ok += 1
        except requests.HTTPError as e:
            body = ""
            try:
                body = e.response.json()
            except Exception:
                body = e.response.text[:200]
            print(f"  [{i:02d}] ✗ HTTP {e.response.status_code}: {body}", file=sys.stderr)
        except Exception as e:
            print(f"  [{i:02d}] ✗ {e}", file=sys.stderr)

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Done: {ok}/{len(memories)} memories seeded.")
    if not args.dry_run:
        print("\nNext steps:")
        print(f"  curl -H 'Authorization: Bearer $ENGRAM_API_KEY' '{base_url}/memories/search?q=refund+policy&limit=3'")


if __name__ == "__main__":
    main()
