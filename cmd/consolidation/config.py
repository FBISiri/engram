"""Configuration system (R8).

Three-level priority: CLI args > env vars > config file > defaults.
Validates threshold ranges and required fields.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - dependency guaranteed by task
    yaml = None


# ---------------------------------------------------------------------------
# Defaults (spec §5.2)
# ---------------------------------------------------------------------------
DEFAULTS: Dict[str, Any] = {
    "enabled": False,

    # Zone thresholds
    "auto_merge_threshold": 0.88,
    "llm_adjudicate_threshold": 0.82,
    "dedup_anomaly_threshold": 0.92,
    "cluster_edge_threshold": 0.88,

    # Type-specific overrides
    "directive_merge_threshold": 0.95,
    "identity_merge_enabled": False,
    "event_merge_threshold": 0.80,

    # Safety
    "recency_guard_hours": 24,
    "max_merges_per_run": 20,
    "llm_confidence_threshold": 0.7,
    "dry_run_default": True,
    "max_cluster_size": 20,

    # Gating
    "min_memory_count": 100,
    "min_interval_hours": 24,

    # LLM
    "llm_provider": "anthropic",
    "llm_model": "claude-haiku-4-20250414",
    "llm_max_tokens": 1024,
    "llm_temperature": 0.0,
    "llm_api_key": "",
    "llm_base_url": "https://api.anthropic.com/v1",

    # Collections
    "collections": ["engram_user", "engram_reflection", "engram_pigo"],

    # Undo / reports / metrics
    "undo_dir": "/data/armyoftheagent/engram/consolidation/undo",
    "undo_retention_days": 90,
    "report_json_dir": "/data/armyoftheagent/engram/consolidation/reports",
    "report_md_dir": "/data/obsidian-vault/Engram/consolidation-reports",
    "metrics_path": "/data/armyoftheagent/engram/consolidation/metrics.jsonl",

    # Endpoints
    "qdrant_url": "http://localhost:6333",
    "engram_url": "http://localhost:8081",
    "engram_api_key": "",

    # Per-type rules (mirrors spec §5.3 type_rules)
    "type_rules": {
        "identity": {"merge_enabled": False},
        "directive": {"merge_threshold": 0.95, "auto_merge_enabled": False},
        "insight": {"merge_threshold": 0.82},
        "event": {"merge_threshold": 0.80},
    },
}


@dataclass
class Config:
    enabled: bool
    auto_merge_threshold: float
    llm_adjudicate_threshold: float
    dedup_anomaly_threshold: float
    cluster_edge_threshold: float
    directive_merge_threshold: float
    identity_merge_enabled: bool
    event_merge_threshold: float
    recency_guard_hours: int
    max_merges_per_run: int
    llm_confidence_threshold: float
    dry_run_default: bool
    max_cluster_size: int
    min_memory_count: int
    min_interval_hours: int
    llm_provider: str
    llm_model: str
    llm_max_tokens: int
    llm_temperature: float
    llm_api_key: str
    llm_base_url: str
    collections: list
    undo_dir: str
    undo_retention_days: int
    report_json_dir: str
    report_md_dir: str
    metrics_path: str
    qdrant_url: str
    engram_url: str
    engram_api_key: str
    type_rules: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# YAML flattening: nested `consolidation.*` doc -> flat DEFAULTS keys
# ---------------------------------------------------------------------------
def _flatten_yaml(doc: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not doc:
        return {}
    root = doc.get("consolidation", doc)
    out: Dict[str, Any] = {}

    if "enabled" in root:
        out["enabled"] = bool(root["enabled"])

    th = root.get("thresholds", {}) or {}
    _map = {
        "auto_merge": "auto_merge_threshold",
        "llm_adjudicate": "llm_adjudicate_threshold",
        "dedup_anomaly": "dedup_anomaly_threshold",
        "cluster_edge": "cluster_edge_threshold",
    }
    for k, v in _map.items():
        if k in th:
            out[v] = float(th[k])

    tr = root.get("type_rules")
    if isinstance(tr, dict):
        out["type_rules"] = tr
        if "directive" in tr and "merge_threshold" in tr["directive"]:
            out["directive_merge_threshold"] = float(tr["directive"]["merge_threshold"])
        if "identity" in tr and "merge_enabled" in tr["identity"]:
            out["identity_merge_enabled"] = bool(tr["identity"]["merge_enabled"])
        if "event" in tr and "merge_threshold" in tr["event"]:
            out["event_merge_threshold"] = float(tr["event"]["merge_threshold"])

    safety = root.get("safety", {}) or {}
    _smap = {
        "recency_guard_hours": "recency_guard_hours",
        "max_merges_per_run": "max_merges_per_run",
        "llm_confidence_threshold": "llm_confidence_threshold",
        "dry_run_default": "dry_run_default",
        "max_cluster_size": "max_cluster_size",
        "min_memory_count": "min_memory_count",
        "min_interval_hours": "min_interval_hours",
    }
    for k, v in _smap.items():
        if k in safety:
            out[v] = safety[k]

    llm = root.get("llm", {}) or {}
    if "provider" in llm:
        out["llm_provider"] = llm["provider"]
    if "model" in llm:
        out["llm_model"] = llm["model"]
    if "max_tokens" in llm:
        out["llm_max_tokens"] = int(llm["max_tokens"])
    if "temperature" in llm:
        out["llm_temperature"] = float(llm["temperature"])
    if "api_key" in llm:
        out["llm_api_key"] = llm["api_key"]
    if "base_url" in llm:
        out["llm_base_url"] = llm["base_url"]

    if "collections" in root and root["collections"]:
        out["collections"] = list(root["collections"])

    undo = root.get("undo", {}) or {}
    if "dir" in undo:
        out["undo_dir"] = undo["dir"]
    if "retention_days" in undo:
        out["undo_retention_days"] = int(undo["retention_days"])

    report = root.get("report", {}) or {}
    if "dir" in report:
        out["report_md_dir"] = report["dir"]
    if "json_dir" in report:
        out["report_json_dir"] = report["json_dir"]

    endpoints = root.get("endpoints", {}) or {}
    for k in ("qdrant_url", "engram_url", "engram_api_key"):
        if k in endpoints:
            out[k] = endpoints[k]

    return out


def _from_env(env: Dict[str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if "ENGRAM_CONSOLIDATION_ENABLED" in env:
        out["enabled"] = env["ENGRAM_CONSOLIDATION_ENABLED"].strip().lower() in ("1", "true", "yes", "on")
    if env.get("ENGRAM_CONSOLIDATION_LLM_PROVIDER"):
        out["llm_provider"] = env["ENGRAM_CONSOLIDATION_LLM_PROVIDER"]
    if env.get("ENGRAM_CONSOLIDATION_LLM_MODEL"):
        out["llm_model"] = env["ENGRAM_CONSOLIDATION_LLM_MODEL"]
    elif env.get("ENGRAM_LLM_MODEL"):
        out["llm_model"] = env["ENGRAM_LLM_MODEL"]
    # LLM API key: consolidation-specific key has highest priority (backward
    # compat), then fall back to the shared engram LLM key.
    if env.get("ANTHROPIC_API_KEY"):
        out["llm_api_key"] = env["ANTHROPIC_API_KEY"]
    elif env.get("ENGRAM_LLM_API_KEY"):
        out["llm_api_key"] = env["ENGRAM_LLM_API_KEY"]
    if env.get("ENGRAM_LLM_BASE_URL"):
        out["llm_base_url"] = env["ENGRAM_LLM_BASE_URL"]
    if env.get("QDRANT_URL"):
        out["qdrant_url"] = env["QDRANT_URL"]
    if env.get("ENGRAM_URL"):
        out["engram_url"] = env["ENGRAM_URL"]
    if env.get("ENGRAM_API_KEY"):
        out["engram_api_key"] = env["ENGRAM_API_KEY"]
    # numeric threshold overrides via env
    for env_key, cfg_key, caster in (
        ("ENGRAM_CONSOLIDATION_AUTO_MERGE_THRESHOLD", "auto_merge_threshold", float),
        ("ENGRAM_CONSOLIDATION_LLM_THRESHOLD", "llm_adjudicate_threshold", float),
        ("ENGRAM_CONSOLIDATION_MAX_MERGES", "max_merges_per_run", int),
    ):
        if env.get(env_key):
            try:
                out[cfg_key] = caster(env[env_key])
            except ValueError:
                pass
    return out


def _validate(cfg: Dict[str, Any]) -> None:
    errors = []
    thr_keys = [
        "auto_merge_threshold", "llm_adjudicate_threshold", "dedup_anomaly_threshold",
        "cluster_edge_threshold", "directive_merge_threshold", "event_merge_threshold",
        "llm_confidence_threshold",
    ]
    for k in thr_keys:
        v = cfg.get(k)
        if not isinstance(v, (int, float)) or not (0.0 <= float(v) <= 1.0):
            errors.append(f"{k} must be in [0.0, 1.0], got {v!r}")

    # Required ordering: llm <= auto_merge <= dedup_anomaly
    if not errors:
        if not (cfg["llm_adjudicate_threshold"] <= cfg["auto_merge_threshold"] <= cfg["dedup_anomaly_threshold"]):
            errors.append(
                "threshold ordering violated: require "
                "llm_adjudicate <= auto_merge <= dedup_anomaly "
                f"(got {cfg['llm_adjudicate_threshold']}, {cfg['auto_merge_threshold']}, "
                f"{cfg['dedup_anomaly_threshold']})"
            )

    for k in ("recency_guard_hours", "max_merges_per_run", "max_cluster_size",
              "min_memory_count", "min_interval_hours", "llm_max_tokens"):
        v = cfg.get(k)
        if not isinstance(v, int) or v < 0:
            errors.append(f"{k} must be a non-negative int, got {v!r}")

    if not cfg.get("collections"):
        errors.append("collections must be a non-empty list")

    if not cfg.get("llm_model"):
        errors.append("llm_model is required")

    if errors:
        raise ValueError("Invalid consolidation config:\n  - " + "\n  - ".join(errors))


def load_config(
    config_path: Optional[str] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
    env: Optional[Dict[str, str]] = None,
) -> Config:
    """Resolve config with priority: CLI > env > file > defaults."""
    env = os.environ if env is None else env

    cfg: Dict[str, Any] = {}
    for k, v in DEFAULTS.items():
        cfg[k] = dict(v) if isinstance(v, dict) else (list(v) if isinstance(v, list) else v)

    # File (lowest override above defaults)
    if config_path:
        if yaml is None:
            raise RuntimeError("pyyaml is required to load a config file")
        with open(config_path, "r") as f:
            doc = yaml.safe_load(f)
        cfg.update(_flatten_yaml(doc))

    # Env
    cfg.update(_from_env(env))

    # CLI (highest priority) — only non-None keys override
    if cli_overrides:
        for k, v in cli_overrides.items():
            if v is not None and k in cfg:
                cfg[k] = v

    _validate(cfg)

    known = {f for f in Config.__dataclass_fields__}  # type: ignore[attr-defined]
    return Config(**{k: cfg[k] for k in known})
