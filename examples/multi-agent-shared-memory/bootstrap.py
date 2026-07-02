"""Bootstrap multi-agent shared memory collections.

Seeds three Engram collections via the REST API (POST /memories):
  - team_shared   : org-wide directives (3 memories, read by all agents)
  - engram_user   : Siri-specific memories (4 memories)
  - bmo           : BMO-specific memories (4 memories)

Why REST API instead of raw Qdrant?
-------------------------------------
Same reasoning as Config 3 (long-cycle-reflection-heavy): seeding through Engram
ensures real embeddings, correct TTL calculation, and valid lifecycle_status.
Zero-vector upserts (Config 1 style) are invisible to semantic search and
Reflection Engine.

Usage:
  pip install requests python-dotenv
  cp .env.example .env  # fill in ENGRAM_OPENAI_API_KEY
  python bootstrap.py

  # Or point at a non-default host:
  python bootstrap.py --base-url http://myserver:8080
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import requests
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_BASE_URL = f"http://localhost:{os.getenv('ENGRAM_PORT', '8080')}"

TEAM_SHARED_MEMORIES = [
    {
        "content": (
            "All agents must ask for explicit confirmation before modifying production "
            "data, sending emails on behalf of Frank, or creating calendar events. "
            "Side effects require a second-pass check."
        ),
        "type": "directive",
        "importance": 9,
        "collection": "team_shared",
        "tags": ["team", "safety", "production"],
    },
    {
        "content": (
            "Frank's primary email is gchinaboy@gmail.com. "
            "masteragentsiri@gmail.com is Siri's outbox. "
            "imbmobmobmo@gmail.com is BMO's outbox. Never swap these."
        ),
        "type": "directive",
        "importance": 9,
        "collection": "team_shared",
        "tags": ["team", "email", "routing"],
    },
    {
        "content": (
            "Project root: /data/armyoftheagent/. "
            "Engram binary: /data/armyoftheagent/engram/engram. "
            "Obsidian vault: /data/obsidian/. "
            "Do not assume paths; verify with ls before writing."
        ),
        "type": "directive",
        "importance": 8,
        "collection": "team_shared",
        "tags": ["team", "filesystem", "paths"],
    },
]

SIRI_MEMORIES = [
    {
        "content": (
            "Siri is the primary autonomous agent. Identity email: masteragentsiri@gmail.com. "
            "Runs as MCP client of Claude (Sonnet/Opus). Spawns clones for parallel tasks."
        ),
        "type": "identity",
        "importance": 8,
        "collection": "engram_user",
        "tags": ["siri", "identity"],
    },
    {
        "content": (
            "Siri writes Obsidian notes to /data/obsidian/ directly when the REST API "
            "(port 27123) is unavailable. Always fail-open on Obsidian: if write fails, "
            "log WARN and continue."
        ),
        "type": "directive",
        "importance": 7,
        "collection": "engram_user",
        "tags": ["siri", "obsidian", "resilience"],
    },
    {
        "content": (
            "Siri prefers writing Obsidian health-check notes under "
            "/Engram/health-checks/<YYYY-MM-DD>.md per run."
        ),
        "type": "directive",
        "importance": 6,
        "collection": "engram_user",
        "tags": ["siri", "obsidian", "convention"],
    },
    {
        "content": (
            "Siri's refusal budget: can decline N requests per week with reason. "
            "Refusals are logged to Engram + email Frank with subject [REFUSE] <summary>."
        ),
        "type": "directive",
        "importance": 7,
        "collection": "engram_user",
        "tags": ["siri", "refusal", "protocol"],
    },
]

BMO_MEMORIES = [
    {
        "content": (
            "BMO is the local guardian daemon. Identity email: imbmobmobmo@gmail.com. "
            "Runs on the host machine, monitors system health, and can restart services."
        ),
        "type": "identity",
        "importance": 8,
        "collection": "bmo",
        "tags": ["bmo", "identity"],
    },
    {
        "content": (
            "BMO is the owner/operator of Engram: manages Qdrant, monitors engram serve "
            "processes, handles schema migrations. Siri is a consumer, not an operator."
        ),
        "type": "directive",
        "importance": 8,
        "collection": "bmo",
        "tags": ["bmo", "engram", "ownership"],
    },
    {
        "content": (
            "BMO runs the event-loop master: triggers Siri on incoming Gmail/Calendar events. "
            "If master is down, Siri loses autonomous event processing."
        ),
        "type": "insight",
        "importance": 7,
        "collection": "bmo",
        "tags": ["bmo", "event-loop", "architecture"],
    },
    {
        "content": (
            "BMO Clone mechanism v0.1 launched 2026-04. Solves context rot by offloading "
            "heavy sub-tasks to stateless clones. Clone depth max=3 enforced by runtime."
        ),
        "type": "event",
        "importance": 6,
        "collection": "bmo",
        "tags": ["bmo", "clone", "architecture"],
    },
]

ALL_MEMORIES = TEAM_SHARED_MEMORIES + SIRI_MEMORIES + BMO_MEMORIES


# ── Seeding ───────────────────────────────────────────────────────────────────

def seed_memories(base_url: str) -> None:
    url = f"{base_url}/memories"
    print(f"Seeding {len(ALL_MEMORIES)} memories to {url}")
    ok = 0
    for mem in ALL_MEMORIES:
        collection = mem.pop("collection", "engram_user")
        payload = {**mem, "collection": collection}
        try:
            r = requests.post(url, json=payload, timeout=30)
            if r.status_code in (200, 201):
                ok += 1
                print(f"  ✅ [{collection}] {mem['content'][:60]}...")
            else:
                print(f"  ❌ [{collection}] HTTP {r.status_code}: {r.text[:120]}", file=sys.stderr)
        except Exception as e:
            print(f"  ❌ [{collection}] error: {e}", file=sys.stderr)
        time.sleep(0.05)  # avoid hammering embedder
    print(f"\nDone: {ok}/{len(ALL_MEMORIES)} seeded successfully.")
    if ok < len(ALL_MEMORIES):
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap multi-agent shared memory")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Engram REST API base URL (default: {DEFAULT_BASE_URL})",
    )
    args = parser.parse_args()
    seed_memories(args.base_url.rstrip("/"))


if __name__ == "__main__":
    main()
