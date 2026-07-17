# Config Example 1 — Personal AI Agent Memory

**Shape:** high-value, low-volume. A single agent accumulating identity facts, owner
preferences, relationships, and hard-won insights over months. Every memory is expensive
to lose; recall of a dormant-but-critical fact matters more than raw precision.

**Main risk:** *silent loss of a rare important memory.* Volume is low enough that
fragmentation is not the dominant failure mode — a key fact sinking into the long tail is.

---

## 关键取舍说明

个人 agent 记忆的核心矛盾是 **召回率 > 精确率**，但又不能靠“多存”来补召回——那样会污染每一次检索的 context。所以这里 `limit` 压到 6、`min_score` 放低到 0.55：宁可召回一条弱相关的真事实，也不要漏掉一条埋在长尾里的关键记忆（这正是 access-frequency 排序会造成“越少用越沉底”的静默退化点，因此 `recency_weight` 只给 0.20，避免旧身份事实被时间衰减压下去）。`importance_weight` 反而给到 0.35，让人工标注的重要性成为对抗遗忘的主力杠杆。

写入侧走**保守 dedup**（0.92）保留细微差异，但对 `identity`（0.95）和 `directive`（0.90）用更严的 per-type 阈值——身份记忆改动代价最高，宁可显式 `memory_update` 一条也不要新增相似条目造成自我认知分裂；directive 类的“用户偏好 X”正反馈飞轮最危险，更严的 dedup 让相似偏好走 update 而非碎片化沉淀。`per_type_min_importance` 是一道 admission 门：低于门槛的写入直接拒绝，防止执行细节噪声进入永久层。

---

## Usage

Apply this configuration at runtime via the `memory_apply_config` MCP tool. The field
values below are identical to [`config.yaml`](./config.yaml):

```python
memory_apply_config(config=json.dumps({
    "retrieve_config": {
        "relevance_weight": 1.0,
        "recency_weight": 0.20,
        "importance_weight": 0.35,
        "limit": 6,
        "min_score": 0.55,
        "rerank_enabled": False
    },
    "update_config": {
        "dedupe_threshold": 0.92,
        "default_importance": 6,
        "per_type_dedupe": {
            "identity": 0.95,
            "directive": 0.90,
            "insight": 0.92,
            "event": 0.92
        },
        "per_type_min_importance": {
            "identity": 7,
            "directive": 6,
            "insight": 5,
            "event": 3
        },
        "ttl": {
            "event": "90d",
            "insight": "90d",
            "identity": "never",
            "directive": "never"
        }
    }
}))
```

---

## ⚠️ Proposed fields

These fields are **not yet wired server-side — roadmap**. They express the intended
behavior but require BMO review before they map to existing `MemoryConfig`:

- `update_config.per_type_dedupe` — stricter per-type dedup thresholds (`# proposed`)
- `update_config.per_type_min_importance` — per-type admission gate (`# proposed`)
