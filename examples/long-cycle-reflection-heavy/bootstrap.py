"""Bootstrap Aria's initial memory state for a long-cycle reflection-heavy agent.

Unlike Config 1 (which writes zero-vector points straight into Qdrant), this
script seeds memories through the Engram HTTP REST API: ``POST /memories``.

Why go through Engram memory_add instead of raw Qdrant?
-------------------------------------------------------
A long-cycle agent depends on the *memory lifecycle*, not just the schema:

  * Real embeddings  — the server embeds ``content`` with the configured model,
    so seeded memories are actually retrievable by semantic search. Zero
    vectors (Config 1's placeholder approach) are NOT searchable and would be
    invisible to Reflection V2's per-focal-question evidence retrieval.
  * Correct decay    — the server computes ``valid_until`` from type+importance
    (identity/directive never decay; insight ~90d; event ~3d). Seeding raw
    points skips this and produces memories the Dream Engine can't reason about
    correctly.
  * Correct defaults — lifecycle_status, created_at, access_count, reflected_at
    are all initialised by the server's create path exactly as the engines
    expect.

The reflection + dream lifecycle this seed set is designed for
--------------------------------------------------------------
  1. Events are TRANSIENT raw material. We intentionally seed them at low
     importance (3-5). They are not durable knowledge.
  2. Reflection V2 (focal-point) fires 1-3x/day: it generates focal questions,
     retrieves evidence per question via semantic search, and synthesises
     durable ``insight`` memories. It marks the source events ``reflected``.
  3. The Dream Engine runs nightly: its consolidate phase tag-groups recent
     events into insights and nudges importance by access_count; its prune
     phase DELETES memories with importance <= 3 AND access_count = 0 (events
     that were never re-accessed, ~14d after creation).

So an event's whole job is to be consolidated into an insight and then pruned.
If an event mattered, Reflection lifts the signal into a higher-importance
insight; if not, it decays and is reclaimed. That is why event importance is
intentionally kept low (never > 5) — see spec Config 3 section 3.

The §4c memories below (1 insight + 1 directive) are examples of what a
Reflection V2 run *would* generate from the §4b events. We seed them tagged
``source:reflection`` (matching what the engine stamps) but with payload
``source="bootstrap_example"`` so they are auditable as seed data, and so all
four memory types (identity / directive / insight / event) are represented in
a fresh collection for demos.

Idempotency
-----------
Engram applies semantic auto-dedup (~0.92 cosine) on the create path, so
re-running this script will not pile up duplicate points for the same content.
It is therefore safe to re-run to restore a known-good baseline.

Requires Python 3.7+ (uses ``from __future__ import annotations`` and PEP 604
``X | None`` unions). The system default ``python3`` may be 3.6 — run with
``python3.9`` (or any 3.7+) if so.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import requests
from dotenv import load_dotenv

# Per-request timeout (connect + read) in seconds. Embedding happens server-side
# on the create path, so allow generous headroom for a cold embedding model.
REQUEST_TIMEOUT_S: float = 30.0

# ─────────────────────────────────────────────────────────────────────────────
# Seed memory set — Aria, a long-horizon research agent assisting Jordan Park.
# Each dict maps 1:1 to the JSON body of POST /memories:
#   content    : the memory text (the server embeds this)               [required]
#   type       : identity | directive | insight | event                 [decay class]
#   importance : 1-10 float (drives valid_until decay + prune eligibility)
#   tags       : free-form provenance / grouping labels
# `source` is added uniformly below as the payload provenance marker.
# ─────────────────────────────────────────────────────────────────────────────
MEMORIES: list[dict[str, Any]] = [
    # ── §4a Identity + directives — durable, high importance, never decay ──────
    {
        # identity (7-9): who the agent is and who it serves. Stable; never decays.
        "content": (
            "Aria is a research agent assisting Jordan Park (PhD candidate, "
            "computational biology). Primary mission: systematic literature review "
            "of CRISPR delivery mechanisms (2020-2026). Jordan expects weekly "
            "synthesis reports, not raw paper dumps."
        ),
        "type": "identity",
        "importance": 9,
        "tags": ["agent-identity", "user-profile", "mission"],
    },
    {
        # directive (8-10): a hard rule the agent must always obey. Never decays.
        "content": (
            "RULE: When surfacing a research finding, always actively search for "
            "contradicting evidence. Jordan prefers thesis-antithesis-synthesis "
            "framing, not one-sided summaries. Flag papers that contradict existing "
            "conclusions with [CONTRA] in notes."
        ),
        "type": "directive",
        "importance": 9,
        "tags": ["research-methodology", "report-format", "directive"],
    },
    {
        # directive (8-10): scope constraint. Never decays.
        "content": (
            "RULE: Do not surface papers on CRISPR applications in agriculture, "
            "livestock, or gene drive. Jordan's scope is strictly therapeutic/"
            "medical. Flag out-of-scope papers separately."
        ),
        "type": "directive",
        "importance": 8,
        "tags": ["scope", "crispr", "directive"],
    },

    # ── §4b Events — transient raw material; importance intentionally LOW (3-5) ─
    # These exist to be consolidated by Reflection into insights, then pruned by
    # the Dream Engine (~14d, importance<=3 & access_count=0). Do NOT raise these.
    {
        "content": (
            "2026-07-01: Reviewed Liu et al. (2024) on lipid nanoparticle "
            "optimization. Key: ionizable lipid pKa 6.2-6.5 sweet spot for "
            "endosomal escape. Tagged for liver-targeted delivery cluster."
        ),
        "type": "event",
        "importance": 4,
        "tags": ["lnp", "delivery", "liver", "paper-review"],
    },
    {
        "content": (
            "2026-07-02: Found contradicting evidence - Wang et al. (2025) shows "
            "pKa 6.8 outperforms in muscle tissue. Scope of Liu et al. may be "
            "organ-specific. Added [CONTRA] flag to synthesis notes."
        ),
        "type": "event",
        "importance": 4,
        "tags": ["lnp", "delivery", "muscle", "paper-review", "contra"],
    },
    {
        # Feedback events are slightly more durable (5) but still event-class:
        # the durable form is the §4c directive that Reflection extracts from it.
        "content": (
            "2026-07-03: Jordan feedback: the weekly synthesis is too long (sent "
            "4,000 words). Wants max 800 words + 3 key findings as bullets. "
            "Updating report format."
        ),
        "type": "event",
        "importance": 5,
        "tags": ["feedback", "report-format", "jordan"],
    },

    # ── §4c Reflection-generated examples — what a V2 run synthesises from §4b ──
    # Tagged `source:reflection` exactly as the engine stamps them, so demos show
    # the lifecycle output. Payload source is "bootstrap_example" (set below).
    {
        # insight (5-7): durable cross-domain pattern lifted out of the events.
        "content": (
            "Emerging pattern: LNP pKa optimization is organ-specific, not "
            "universal. Liver-optimized LNPs (pKa ~6.2-6.5, Liu 2024) underperform "
            "in muscle tissue vs higher pKa formulations (Wang 2025). Future "
            "synthesis notes should flag target organ as a variable, not a constant."
        ),
        "type": "insight",
        "importance": 6,
        "tags": ["lnp", "delivery", "synthesis", "source:reflection"],
    },
    {
        # directive (8-10 normally; 7 here as a Reflection-promoted rule update).
        "content": (
            "Jordan's report format preference updated: max 800 words, 3 key "
            "findings as bullets. Previous 4,000-word reports were flagged as too "
            "long (2026-07-03)."
        ),
        "type": "directive",
        "importance": 7,
        "tags": ["report-format", "jordan", "feedback", "source:reflection"],
    },
]


def build_payloads() -> list[dict[str, Any]]:
    """Construct the JSON bodies for POST /memories from the seed set.

    `source="bootstrap_example"` marks every point as seed data (auditable),
    even the §4c memories that carry the `source:reflection` *tag*.
    """
    payloads: list[dict[str, Any]] = []
    for mem in MEMORIES:
        payloads.append(
            {
                "content": mem["content"],
                "type": mem["type"],
                "importance": mem["importance"],
                "tags": mem["tags"],
                "source": "bootstrap_example",
            }
        )
    return payloads


def post_memory(
    base_url: str, payload: dict[str, Any], api_key: str | None
) -> dict[str, Any]:
    """POST a single memory to Engram and return the created Memory JSON.

    The server embeds `content` and computes `valid_until` (decay) for us.
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        # Only sent when ENGRAM_API_KEY is set; the server treats auth as
        # optional and skips the check entirely when no key is configured.
        headers["Authorization"] = f"Bearer {api_key}"

    resp = requests.post(
        f"{base_url}/memories",
        json=payload,
        headers=headers,
        timeout=REQUEST_TIMEOUT_S,
    )
    resp.raise_for_status()
    return resp.json()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Seed Aria's long-cycle memory state via the Engram HTTP API "
            "(POST /memories) so memories get real embeddings + auto decay."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the JSON payloads and make no network calls.",
    )
    args = parser.parse_args()

    load_dotenv()
    host: str = os.getenv("ENGRAM_HOST", "localhost")
    port: str = os.getenv("ENGRAM_HTTP_PORT", "8080")
    # Informational only: POST /memories has no collection field, so writes
    # land in whatever collection the SERVER was launched with. This value
    # documents the expected destination; it does not select it.
    collection: str = os.getenv("ENGRAM_COLLECTION_NAME", "siri_long_cycle")
    api_key: str | None = os.getenv("ENGRAM_API_KEY") or None
    base_url: str = f"http://{host}:{port}"

    payloads = build_payloads()

    if args.dry_run:
        print(json.dumps(payloads, indent=2, ensure_ascii=False))
        print(
            f"\n[dry-run] {len(payloads)} payloads would POST to "
            f"{base_url}/memories. No network calls made."
        )
        print(
            f"          target collection (server-configured, expected: "
            f"'{collection}')"
        )
        return 0

    print(f"Seeding {len(payloads)} memories into Engram at {base_url} ...")
    created = 0
    for payload in payloads:
        try:
            mem = post_memory(base_url, payload, api_key)
        except requests.RequestException as exc:
            print(f"  ERROR seeding ({payload['type']}): {exc}", file=sys.stderr)
            return 1
        created += 1
        mem_id = mem.get("id", "<no-id>")
        print(f"  + {payload['type']:9s} imp={payload['importance']:<2} id={mem_id}")

    print("\nSeed complete.")
    print(f"  Engram endpoint:  {base_url}/memories")
    print(f"  Target collection: server-configured (expected: {collection})")
    print(f"  Memories created: {created}/{len(payloads)}")
    print(
        "  Note: Engram dedup (~0.92) makes re-runs safe. Server computed "
        "embeddings + decay (valid_until) for each memory."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
