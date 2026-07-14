#!/usr/bin/env python3
"""
bootstrap.py — Config 7: Chatbot Session Memory
Seeds 12 example session memories across 3 simulated users.
Demonstrates per-user tagging, short-TTL event memories, and dedup behavior.

Usage:
    pip install requests python-dotenv
    python bootstrap.py [--dry-run] [--base-url http://localhost:8080]
"""

import argparse
import os
import sys

try:
    import requests
except ImportError:
    print("ERROR: 'requests' not installed. Run: pip install requests", file=sys.stderr)
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def main():
    parser = argparse.ArgumentParser(description="Seed chatbot session memory examples")
    parser.add_argument("--dry-run", action="store_true", help="Print memories without sending")
    parser.add_argument("--base-url", default=None)
    args = parser.parse_args()

    host = os.environ.get("ENGRAM_HOST", "localhost")
    port = os.environ.get("ENGRAM_HTTP_PORT", "8080")
    base_url = args.base_url or f"http://{host}:{port}"
    api_key = os.environ.get("ENGRAM_API_KEY", "")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if not args.dry_run:
        try:
            resp = requests.get(f"{base_url}/health", timeout=5)
            resp.raise_for_status()
            print(f"✓ Engram reachable at {base_url}")
        except Exception as e:
            print(f"✗ Cannot reach Engram: {e}", file=sys.stderr)
            sys.exit(1)

    # Three simulated users: alice, bob, carol
    # Each has 4 session memories seeded.
    memories = [
        # ── Alice: food ordering chatbot session ──────────────────────────────
        {
            "content": "Alice is vegetarian — no meat in any form, including broths and sauces.",
            "type": "event",
            "importance": 6,
            "tags": ["user:alice", "dietary", "session-context"],
        },
        {
            "content": "Alice prefers thin-crust pizza. She ordered margherita last time and liked it.",
            "type": "event",
            "importance": 4,
            "tags": ["user:alice", "food-preference", "session-context"],
        },
        {
            "content": "Alice's delivery address: 12 Sakura Street, Shibuya, Tokyo. Confirmed in this session.",
            "type": "event",
            "importance": 5,
            "tags": ["user:alice", "address", "session-context", "time-sensitive"],
        },
        {
            "content": "Alice mentioned she has a nut allergy. Flag any dishes that may contain almonds, walnuts, or peanut oil.",
            "type": "event",
            "importance": 8,
            "tags": ["user:alice", "allergy", "safety", "session-context"],
        },

        # ── Bob: travel booking chatbot session ───────────────────────────────
        {
            "content": "Bob is traveling Tokyo → Osaka on July 10. He prefers aisle seats.",
            "type": "event",
            "importance": 6,
            "tags": ["user:bob", "travel", "session-context"],
        },
        {
            "content": "Bob's budget cap for hotels is ¥15,000 per night. Business hotels preferred over hostels.",
            "type": "event",
            "importance": 5,
            "tags": ["user:bob", "budget", "hotel-preference", "session-context"],
        },
        {
            "content": "Bob asked to avoid Shinkansen during rush hour (07:00–09:00). He's flexible on departure time otherwise.",
            "type": "event",
            "importance": 4,
            "tags": ["user:bob", "transport", "schedule-preference", "session-context"],
        },
        {
            "content": "Bob mentioned he is traveling with his dog (shiba inu). Must filter for pet-friendly accommodations.",
            "type": "event",
            "importance": 7,
            "tags": ["user:bob", "pet", "filter-constraint", "session-context"],
        },

        # ── Carol: tech support chatbot session ───────────────────────────────
        {
            "content": "Carol is running macOS Sequoia 15.4 with an M3 MacBook Pro. Software version context for troubleshooting.",
            "type": "event",
            "importance": 5,
            "tags": ["user:carol", "system-info", "session-context"],
        },
        {
            "content": "Carol's issue: Bluetooth headphones disconnect randomly after ~20 minutes. Reported starting 2025-06-28 after OS update.",
            "type": "event",
            "importance": 7,
            "tags": ["user:carol", "bluetooth", "bug-report", "session-context"],
        },
        {
            "content": "Carol already tried: restarting Bluetooth, re-pairing device, disabling Wi-Fi (still disconnects). Next step: reset NVRAM.",
            "type": "event",
            "importance": 6,
            "tags": ["user:carol", "troubleshooting-history", "session-context"],
        },
        {
            "content": "Carol's headphone model: Sony WH-1000XM5. Serial: 4823XXXX (from warranty lookup). Covered under warranty until 2026-11.",
            "type": "event",
            "importance": 5,
            "tags": ["user:carol", "device-info", "warranty", "session-context"],
        },
    ]

    print(f"\nSeeding {len(memories)} session memories for 3 users (alice, bob, carol)...\n")

    ok = 0
    for i, mem in enumerate(memories, 1):
        user = mem["tags"][0]  # first tag is always user:xxx
        label = mem["content"][:55] + ("..." if len(mem["content"]) > 55 else "")
        if args.dry_run:
            print(f"  [DRY RUN {i:02d}] [{user}] [{mem['type']}] {label}")
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
            mem_id = resp.json().get("id", "?")
            print(f"  [{i:02d}] ✓ {mem_id[:8]}...  [{user}] {label}")
            ok += 1
        except requests.HTTPError as e:
            try:
                body = e.response.json()
            except Exception:
                body = e.response.text[:200]
            print(f"  [{i:02d}] ✗ HTTP {e.response.status_code}: {body}", file=sys.stderr)
        except Exception as e:
            print(f"  [{i:02d}] ✗ {e}", file=sys.stderr)

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Done: {ok}/{len(memories)} memories seeded.\n")

    if not args.dry_run:
        base = base_url
        api = api_key or "YOUR_KEY"
        print("Try searching per-user context:")
        print(f"  # Alice's dietary restrictions:")
        print(f"  curl -H 'Authorization: Bearer {api}' '{base}/memories/search?q=dietary+restriction&tags=user:alice&limit=3'")
        print(f"")
        print(f"  # Bob's travel preferences:")
        print(f"  curl -H 'Authorization: Bearer {api}' '{base}/memories/search?q=hotel+preference&tags=user:bob&limit=3'")
        print(f"")
        print(f"  # Carol's troubleshooting history:")
        print(f"  curl -H 'Authorization: Bearer {api}' '{base}/memories/search?q=bluetooth+troubleshooting&tags=user:carol&limit=5'")


if __name__ == "__main__":
    main()
