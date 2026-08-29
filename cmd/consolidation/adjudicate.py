"""LLM adjudication module (R3).

Calls the Anthropic Messages API (Claude Haiku) with the exact prompt from
spec §2.2, parses the JSON verdict, and enforces the confidence floor
(< 0.7 -> KEEP_SEPARATE). All error paths fall back to KEEP_SEPARATE.
"""

import json
import re
from typing import Any, Callable, Dict, Optional

import requests

from .config import Config
from .types import (
    MemoryPoint,
    DECISION_KEEP_SEPARATE,
    DECISION_MERGE,
    DECISION_PARTIAL_MERGE,
)

ANTHROPIC_VERSION = "2023-06-01"

PROMPT_TEMPLATE = """You are a memory consolidation assistant. Given two memories from an AI agent's
long-term memory store, determine whether they express the same core insight/fact
or are genuinely different.

## Memory A
- ID: {id_a}
- Type: {type_a}
- Content: {content_a}
- Tags: {tags_a}
- Importance: {importance_a}
- Created: {created_a}
- Source: {source_type_a}

## Memory B
- ID: {id_b}
- Type: {type_b}
- Content: {content_b}
- Tags: {tags_b}
- Importance: {importance_b}
- Created: {created_b}
- Source: {source_type_b}

## Cosine Similarity: {similarity_score}

## Decision Criteria
Consider these signals:

| Signal             | MERGE indicator              | KEEP SEPARATE indicator         |
|--------------------|------------------------------|---------------------------------|
| Core claim         | Same conclusion/fact         | Different conclusions           |
| Evidence base      | Overlapping evidence         | Different evidence              |
| Temporal relation  | Evolution of same concept    | Distinct time-bound events      |
| Type               | Same type                    | Different types                 |
| Tags               | High tag overlap             | Low tag overlap                 |
| Contradiction      | No contradiction             | Contradictory claims            |
| Specificity        | One is strict subset of other| Both add unique information     |

## Output (JSON only)
{{
  "decision": "MERGE" | "KEEP_SEPARATE" | "PARTIAL_MERGE",
  "confidence": 0.0-1.0,
  "reasoning": "one sentence explanation",
  "merged_content": "if MERGE: the merged content that preserves all information from both",
  "keep_id": "if MERGE: ID of the memory to keep (will be updated)",
  "delete_ids": ["if MERGE: IDs of memories to delete"]
}}

IMPORTANT:
- When in doubt, KEEP_SEPARATE. False negatives (missing a merge) are far less
  costly than false positives (incorrectly merging distinct memories).
- If one memory is a strict SUBSET of the other, always MERGE (keep the superset).
- NEVER merge memories with contradictory claims — they record decision evolution.
- PARTIAL_MERGE: only if a cluster has 3+ members and some should merge but others shouldn't.
"""

DIRECTIVE_EXTRA = (
    "\n\nADDITIONAL RULE (directive memories): Directives encode operational rules. "
    "Only merge if one is a strict subset of the other. If they differ in any specific "
    "detail (scope, timing, exception), KEEP_SEPARATE."
)


def build_prompt(mem_a: MemoryPoint, mem_b: MemoryPoint, similarity: float) -> str:
    prompt = PROMPT_TEMPLATE.format(
        id_a=mem_a.id, type_a=mem_a.type, content_a=mem_a.content,
        tags_a=mem_a.tags, importance_a=mem_a.importance,
        created_a=mem_a.created_at, source_type_a=mem_a.source_type,
        id_b=mem_b.id, type_b=mem_b.type, content_b=mem_b.content,
        tags_b=mem_b.tags, importance_b=mem_b.importance,
        created_b=mem_b.created_at, source_type_b=mem_b.source_type,
        similarity_score=round(similarity, 4),
    )
    if mem_a.type == "directive" or mem_b.type == "directive":
        prompt += DIRECTIVE_EXTRA
    return prompt


def _keep_separate(reason: str, usage: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    return {
        "decision": DECISION_KEEP_SEPARATE,
        "confidence": 0.0,
        "reasoning": reason,
        "merged_content": None,
        "keep_id": None,
        "delete_ids": [],
        "usage": usage or {"input_tokens": 0, "output_tokens": 0},
    }


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    # Direct parse, then greedy brace fallback for models that wrap in prose.
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except (ValueError, TypeError):
        return None


def anthropic_call(prompt: str, cfg: Config, api_key: str, base_url: str, timeout: int = 60) -> Dict[str, Any]:
    """Raw Anthropic Messages API call. Returns {'text': str, 'usage': {...}}.

    `base_url` is the Anthropic-compatible API base (e.g. OpenRouter's
    https://openrouter.ai/api/v1); the endpoint is `base_url/messages`.
    """
    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    body = {
        "model": cfg.llm_model,
        "max_tokens": cfg.llm_max_tokens,
        "temperature": cfg.llm_temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    url = base_url.rstrip("/") + "/messages"
    r = requests.post(url, headers=headers, json=body, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    parts = data.get("content", [])
    text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
    usage = data.get("usage", {}) or {}
    return {
        "text": text,
        "usage": {
            "input_tokens": int(usage.get("input_tokens", 0)),
            "output_tokens": int(usage.get("output_tokens", 0)),
        },
    }


def adjudicate_pair(
    mem_a: MemoryPoint,
    mem_b: MemoryPoint,
    similarity: float,
    cfg: Config,
    call_fn: Optional[Callable[[str, Config, str], Dict[str, Any]]] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Adjudicate a single pair. Always returns a normalized verdict dict.

    `call_fn` is injectable for testing (defaults to anthropic_call).
    """
    api_key = api_key if api_key is not None else cfg.llm_api_key
    if not api_key:
        return _keep_separate("no LLM API key configured")

    if call_fn is None:
        def call_fn(prompt_: str, cfg_: Config, key_: str) -> Dict[str, Any]:
            return anthropic_call(prompt_, cfg_, key_, cfg.llm_base_url)
    prompt = build_prompt(mem_a, mem_b, similarity)

    try:
        raw = call_fn(prompt, cfg, api_key)
    except requests.RequestException as e:
        return _keep_separate(f"llm api error: {e}")
    except Exception as e:  # defensive: never let adjudication crash a run
        return _keep_separate(f"llm call failed: {e}")

    usage = raw.get("usage", {"input_tokens": 0, "output_tokens": 0})
    parsed = _extract_json(raw.get("text", ""))
    if not parsed or "decision" not in parsed:
        return _keep_separate("malformed llm response", usage)

    decision = parsed.get("decision")
    if decision not in (DECISION_MERGE, DECISION_KEEP_SEPARATE, DECISION_PARTIAL_MERGE):
        return _keep_separate(f"unknown decision: {decision!r}", usage)

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    # Confidence floor: below threshold -> KEEP_SEPARATE (spec §2.2).
    if decision in (DECISION_MERGE, DECISION_PARTIAL_MERGE) and confidence < cfg.llm_confidence_threshold:
        result = _keep_separate(
            f"confidence {confidence:.2f} < {cfg.llm_confidence_threshold}", usage
        )
        result["reasoning"] = parsed.get("reasoning", result["reasoning"])
        return result

    delete_ids = parsed.get("delete_ids") or []
    if not isinstance(delete_ids, list):
        delete_ids = []
    return {
        "decision": decision,
        "confidence": confidence,
        "reasoning": parsed.get("reasoning", ""),
        "merged_content": parsed.get("merged_content"),
        "keep_id": parsed.get("keep_id"),
        "delete_ids": [str(x) for x in delete_ids],
        "usage": usage,
    }
