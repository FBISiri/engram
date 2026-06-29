"""Bootstrap an Engram single-agent personal memory collection in Qdrant.

Seeds 9 example memory points into a Qdrant collection using the REST API.
Idempotent: uses upsert, so running multiple times will not error.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv
import os

VECTOR_SIZE: int = 1536

MEMORIES: List[Dict[str, Any]] = [
    {
        "content": (
            "User is Alex Chen, software engineer at a fintech startup (Series B). "
            "Primary languages: Python, TypeScript. Prefers functional style over OOP."
        ),
        "type": "identity",
        "importance": 8,
        "tags": ["user-profile", "engineering"],
    },
    {
        "content": (
            "Alex prefers concise responses: bullet points over paragraphs, no "
            "preamble ('Sure!', 'Great question!'). Gets annoyed by filler text."
        ),
        "type": "identity",
        "importance": 9,
        "tags": ["user-profile", "communication-style"],
    },
    {
        "content": (
            "Alex works 9am–7pm HKT (Asia/Hong_Kong). Peak focus hours: 10am–12pm. "
            "Do not schedule interruptions during that window."
        ),
        "type": "identity",
        "importance": 7,
        "tags": ["user-profile", "schedule"],
    },
    {
        "content": (
            "RULE: Never send an email or calendar invite on Alex's behalf without "
            "showing a draft first and receiving explicit 'send it' confirmation."
        ),
        "type": "directive",
        "importance": 10,
        "tags": ["safety", "email", "calendar"],
    },
    {
        "content": (
            "RULE: All Python code must pass ruff linting and have type hints. Never "
            "suggest mutable default arguments. Always use pathlib over os.path."
        ),
        "type": "directive",
        "importance": 8,
        "tags": ["coding", "python", "standards"],
    },
    {
        "content": (
            "Alex tends to ask for 'quick checks' that turn into multi-step refactors. "
            "When Alex says 'just a quick look', scope-check before starting work."
        ),
        "type": "insight",
        "importance": 6,
        "tags": ["user-behavior", "scope-management"],
    },
    {
        "content": (
            "Alex's monorepo uses pnpm workspaces. Running 'npm install' at root "
            "breaks the lockfile. Always use 'pnpm install'."
        ),
        "type": "insight",
        "importance": 7,
        "tags": ["engineering", "monorepo", "gotcha"],
    },
    {
        "content": (
            "2026-06-15: Migrated Alex's payment service from Stripe v1 to v2 API. "
            "Took 3 sessions. Key decision: kept legacy webhook handlers for 90-day overlap."
        ),
        "type": "event",
        "importance": 6,
        "tags": ["project", "stripe", "payment-service"],
    },
    {
        "content": (
            "2026-06-20: Alex explicitly said 'stop adding comments to every function, "
            "only comment non-obvious logic.' Updated coding directive."
        ),
        "type": "event",
        "importance": 5,
        "tags": ["feedback", "coding", "comments"],
    },
]


def build_points() -> List[Dict[str, Any]]:
    """Construct Qdrant point dicts from the memory definitions."""
    now: int = int(time.time())
    points: List[Dict[str, Any]] = []
    for mem in MEMORIES:
        points.append(
            {
                "id": str(uuid.uuid4()),
                "vector": [0.0] * VECTOR_SIZE,
                "payload": {
                    "content": mem["content"],
                    "type": mem["type"],
                    "importance": mem["importance"],
                    "tags": mem["tags"],
                    "created_at": now,
                    "source": "bootstrap_example",
                },
            }
        )
    return points


def ensure_collection(base_url: str, name: str) -> None:
    """Create the collection if it does not already exist."""
    resp = requests.get(f"{base_url}/collections/{name}")
    if resp.status_code == 200:
        print(f"Collection '{name}' already exists.")
        return
    if resp.status_code != 404:
        resp.raise_for_status()

    create = requests.put(
        f"{base_url}/collections/{name}",
        json={"vectors": {"size": VECTOR_SIZE, "distance": "Cosine"}},
    )
    create.raise_for_status()
    print(f"Created collection '{name}'.")


def upsert_points(base_url: str, name: str, points: List[Dict[str, Any]]) -> None:
    """Upsert points into the collection (idempotent)."""
    resp = requests.put(
        f"{base_url}/collections/{name}/points",
        params={"wait": "true"},
        json={"points": points},
    )
    resp.raise_for_status()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed an Engram personal memory collection in Qdrant."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print payloads as JSON and make no network calls.",
    )
    args = parser.parse_args()

    load_dotenv()
    host: str = os.getenv("QDRANT_HOST", "localhost")
    port: str = os.getenv("QDRANT_REST_PORT", "6333")
    name: str = os.getenv("ENGRAM_COLLECTION_NAME", "my_assistant")
    base_url: str = f"http://{host}:{port}"

    points = build_points()

    if args.dry_run:
        payloads = [p["payload"] for p in points]
        print(json.dumps(payloads, indent=2, ensure_ascii=False))
        print(f"\n[dry-run] {len(payloads)} payloads. No network calls made.")
        return

    ensure_collection(base_url, name)
    upsert_points(base_url, name, points)

    confirm_url: str = f"{base_url}/collections/{name}"
    print("\nSeed complete.")
    print(f"  Collection:        {name}")
    print(f"  Points upserted:   {len(points)}")
    print(f"  Confirmation URL:  {confirm_url}")


if __name__ == "__main__":
    main()
