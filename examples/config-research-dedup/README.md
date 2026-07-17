# Config Example 3 — Research Note Deduplication

**Shape:** dense technical notes, single or few authors, but *very* high semantic overlap —
successive notes on the same topic (e.g. "Engram dedup band analysis v1, v2, v3") pile up
as fragments. This config is deliberately the most aggressive on dedup, tuned to force
consolidation.

**Main risk:** **fragment pile-up** — dozens of overlapping insights that should have been
one evolving entry, with early wrong conclusions out-ranking their corrections.

---

## 关键取舍说明

研究笔记的敌人是**碎片堆积**：同一主题的 v1/v2/v3 各存一条，未来检索时早期错误结论因 access_count 更高反而排在修正版前面，形成“越旧越错越容易被召回”的反向优先级。对策是把 `dedupe_threshold` 压到最激进的 0.80——落进 0.80 以上相似度的新笔记直接走 update/supersede，而不是新增。`merge_strategy: replace_or_supersede` 让新笔记接管旧笔记的位置并在 tag 里链上 `supersedes:<old_id>`（零成本软引用约定，不改 schema），检索侧配合 `supersedes_aware: true` 优先返回被 supersede 链指向的最新版。

因为写时 dedup 只能防“相似内容重复写入”、防不了“后期 session 对前期结论的逻辑撤销”，这里额外开 `consolidation_pass: enabled`：一个周期性离线合并 pass，把碎片化 insight 折叠成一条演进条目、主动 prune 被反驳的旧结论。检索侧 `min_score` 提到 0.70、`importance_weight` 降到 0.20，因为在高密度同主题语料里，relevance 和 recency 已足够区分，importance 反而区分度低。`insight` 类不设 TTL——研究是累积的，正确做法是**去重合并而非时间过期**。

---

## Usage

Apply this configuration at runtime via the `memory_apply_config` MCP tool. The field
values below are identical to [`config.yaml`](./config.yaml):

```python
memory_apply_config(config=json.dumps({
    "retrieve_config": {
        "relevance_weight": 1.0,
        "recency_weight": 0.30,
        "importance_weight": 0.20,
        "limit": 8,
        "min_score": 0.70,
        "rerank_enabled": True,
        "supersedes_aware": True
    },
    "update_config": {
        "dedupe_threshold": 0.80,
        "default_importance": 5,
        "merge_strategy": "replace_or_supersede",
        "consolidation_pass": "enabled",
        "per_type_dedupe": {
            "insight": 0.80
        },
        "ttl": {
            "insight": "never"
        }
    }
}))
```

---

## ⚠️ Proposed fields

These fields are **not yet wired server-side — roadmap**. They express the intended
behavior but require BMO review before they map to existing `MemoryConfig`:

- `retrieve_config.supersedes_aware` — prefer the memory tagged `supersedes:<old_id>` (`# proposed`)
- `update_config.merge_strategy: replace_or_supersede` — new note supersedes old, chain the ref (`# proposed`)
- `update_config.consolidation_pass: enabled` — periodic offline merge of fragments (`# proposed`)
