"""Metrics collection + longitudinal JSONL append (spec §8)."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

# Haiku pricing (spec §5.4): $0.25 / M input, $1.25 / M output tokens.
COST_PER_INPUT_TOKEN = 0.25 / 1_000_000
COST_PER_OUTPUT_TOKEN = 1.25 / 1_000_000


@dataclass
class Metrics:
    total_memories_scanned: int = 0
    total_clusters_found: int = 0
    clusters_by_zone: Dict[str, int] = field(default_factory=dict)
    clusters_skipped_by_guard: Dict[str, int] = field(default_factory=dict)
    merges_proposed: int = 0
    merges_executed: int = 0
    merges_failed: int = 0
    memories_removed: int = 0
    memories_created: int = 0
    net_reduction: int = 0
    net_reduction_pct: float = 0.0
    llm_calls: int = 0
    llm_calls_malformed: int = 0
    llm_calls_failed: int = 0
    adjudication_attempted: int = 0
    adjudication_skipped_reason: Dict[str, int] = field(default_factory=dict)
    llm_tokens_input: int = 0
    llm_tokens_output: int = 0
    llm_cost_usd: float = 0.0
    run_duration_seconds: float = 0.0
    scan_duration_seconds: float = 0.0
    cluster_duration_seconds: float = 0.0

    def add_llm_usage(self, usage: Dict[str, int], status: str = "ok") -> None:
        self.adjudication_attempted += 1
        # Tokens count always (real cost), regardless of parse/success status.
        self.llm_tokens_input += int(usage.get("input_tokens", 0))
        self.llm_tokens_output += int(usage.get("output_tokens", 0))
        if status == "ok":
            self.llm_calls += 1  # SUCCESS budget only
        elif status == "malformed":
            self.llm_calls_malformed += 1
        else:
            self.llm_calls_failed += 1

    def note_skip(self, reason: str) -> None:
        self.adjudication_skipped_reason[reason] = (
            self.adjudication_skipped_reason.get(reason, 0) + 1
        )

    def finalize(self) -> None:
        self.llm_cost_usd = round(
            self.llm_tokens_input * COST_PER_INPUT_TOKEN
            + self.llm_tokens_output * COST_PER_OUTPUT_TOKEN,
            6,
        )
        self.net_reduction = self.memories_removed - self.memories_created
        if self.total_memories_scanned:
            self.net_reduction_pct = round(
                self.net_reduction / self.total_memories_scanned * 100, 3
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def append_jsonl(path: str, summary: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")


def last_run_timestamp(path, mode="live"):
    """Return run_timestamp_unix of the last JSONL entry matching `mode`.

    Used by the min-interval gate, which must only consider prior LIVE runs
    (a read-only dry-run/scan must never block the next live run).
    Returns 0.0 when no matching entry exists.
    """
    p = Path(path)
    if not p.exists():
        return 0.0
    last_ts = 0.0
    with open(p, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except (ValueError, TypeError):
                continue
            if mode is not None and entry.get("mode") != mode:
                continue
            try:
                last_ts = float(entry.get("run_timestamp_unix", 0.0))
            except (ValueError, TypeError):
                continue
    return last_ts
