"""Structured report generator (R6): JSON + Markdown."""

import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .config import Config


def _date_stamp(ts: datetime.datetime) -> str:
    return ts.strftime("%Y-%m-%d")


def build_report(
    run_meta: Dict[str, Any],
    metrics: Dict[str, Any],
    cluster_entries: List[Dict[str, Any]],
    execution: Dict[str, Any],
    effective_config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    return {
        "run_id": run_meta.get("run_id"),
        "run_timestamp": run_meta.get("run_timestamp"),
        "mode": run_meta.get("mode"),
        "collections": run_meta.get("collections"),
        "effective_config": effective_config or {},
        "adjudication": {
            "attempted": metrics.get("adjudication_attempted", 0),
            "succeeded": metrics.get("llm_calls", 0),
            "malformed": metrics.get("llm_calls_malformed", 0),
            "failed": metrics.get("llm_calls_failed", 0),
            "skipped_reason": metrics.get("adjudication_skipped_reason", {}),
        },
        "scan_summary": {
            "total_memories_scanned": metrics.get("total_memories_scanned"),
            "total_clusters_found": metrics.get("total_clusters_found"),
            "scan_duration_seconds": metrics.get("scan_duration_seconds"),
            "cluster_duration_seconds": metrics.get("cluster_duration_seconds"),
        },
        "zone_breakdown": metrics.get("clusters_by_zone", {}),
        "guard_breakdown": metrics.get("clusters_skipped_by_guard", {}),
        "clusters": cluster_entries,
        "execution": execution,
        "metrics": metrics,
    }


def render_markdown(report: Dict[str, Any]) -> str:
    L: List[str] = []
    L.append("# Engram Consolidation Report")
    L.append("")
    L.append(f"- **Run ID:** `{report.get('run_id')}`")
    L.append(f"- **Timestamp:** {report.get('run_timestamp')}")
    L.append(f"- **Mode:** `{report.get('mode')}`")
    L.append(f"- **Collections:** {', '.join(report.get('collections') or [])}")
    L.append("")

    ec = report.get("effective_config") or {}
    L.append("## Effective Config")
    L.append("")
    L.append("| Key | Value |")
    L.append("|-----|-------|")
    for k, v in ec.items():
        L.append(f"| {k} | {v} |")
    L.append("")

    adj = report.get("adjudication") or {}
    L.append("## Adjudication")
    L.append("")
    L.append(f"- Attempted: **{adj.get('attempted', 0)}**")
    L.append(f"- Succeeded: **{adj.get('succeeded', 0)}**")
    L.append(f"- Malformed: **{adj.get('malformed', 0)}**")
    L.append(f"- Failed: **{adj.get('failed', 0)}**")
    L.append("")
    L.append("### Skipped (by reason)")
    L.append("")
    L.append("| Reason | Count |")
    L.append("|--------|-------|")
    for reason, n in (adj.get("skipped_reason") or {}).items():
        L.append(f"| {reason} | {n} |")
    L.append("")

    ss = report.get("scan_summary", {})
    L.append("## Scan Summary")
    L.append("")
    L.append(f"- Total memories scanned: **{ss.get('total_memories_scanned')}**")
    L.append(f"- Total clusters found: **{ss.get('total_clusters_found')}**")
    L.append(f"- Scan duration: {ss.get('scan_duration_seconds')}s")
    L.append(f"- Cluster duration: {ss.get('cluster_duration_seconds')}s")
    L.append("")

    L.append("## Zone Breakdown")
    L.append("")
    L.append("| Zone | Clusters |")
    L.append("|------|----------|")
    for zone, n in (report.get("zone_breakdown") or {}).items():
        L.append(f"| {zone} | {n} |")
    L.append("")

    L.append("## Guard / Skip Breakdown")
    L.append("")
    L.append("| Guard | Clusters |")
    L.append("|-------|----------|")
    for g, n in (report.get("guard_breakdown") or {}).items():
        L.append(f"| {g} | {n} |")
    L.append("")

    m = report.get("metrics", {})
    L.append("## Merge Decisions")
    L.append("")
    L.append(f"- Proposed: **{m.get('merges_proposed')}**")
    L.append(f"- Executed: **{m.get('merges_executed')}**")
    L.append(f"- Failed: **{m.get('merges_failed')}**")
    L.append(f"- Memories removed: {m.get('memories_removed')} · created: {m.get('memories_created')}")
    L.append(f"- Net reduction: **{m.get('net_reduction')}** ({m.get('net_reduction_pct')}%)")
    L.append(f"- LLM calls: {m.get('llm_calls')} · tokens in/out: "
             f"{m.get('llm_tokens_input')}/{m.get('llm_tokens_output')} · "
             f"cost: ${m.get('llm_cost_usd')}")
    L.append("")

    L.append("## Clusters")
    L.append("")
    if not report.get("clusters"):
        L.append("_No clusters found._")
    for c in report.get("clusters", []):
        decision = c.get("decision") or c.get("decision_source")
        L.append(f"### Cluster {c.get('cluster_id')} — zone `{c.get('zone')}` — "
                 f"max sim {c.get('max_similarity')}")
        L.append("")
        L.append(f"- Decision source: `{c.get('decision_source')}`")
        if c.get("guard"):
            L.append(f"- Guard: `{c.get('guard')}`")
        if c.get("skip_reason"):
            L.append(f"- Skip reason: `{c.get('skip_reason')}`")
        L.append(f"- Decision: **{decision}**")
        if c.get("reasoning"):
            L.append(f"- Reasoning: {c.get('reasoning')}")
        L.append(f"- Members ({len(c.get('members', []))}):")
        for mem in c.get("members", []):
            preview = (mem.get("content_preview") or "").replace("\n", " ").strip()
            L.append(f"  - `{mem.get('id')}` type=`{mem.get('type')}` "
                     f"imp={mem.get('importance')} — {preview}")
        if c.get("proposed_merged_content"):
            pm = c["proposed_merged_content"].replace("\n", " ").strip()
            L.append(f"- Proposed merged content: {pm[:400]}")
        if c.get("would_reduce_by"):
            L.append(f"- Would reduce by: {c.get('would_reduce_by')}")
        L.append("")

    return "\n".join(L) + "\n"


def write_reports(cfg: Config, report: Dict[str, Any],
                  ts: datetime.datetime) -> Tuple[str, str]:
    stamp = _date_stamp(ts)
    json_path = Path(cfg.report_json_dir) / f"{stamp}.json"
    md_path = Path(cfg.report_md_dir) / f"{stamp}.md"

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w") as f:
        f.write(render_markdown(report))

    return str(json_path), str(md_path)
