# Config Example 2 — Team Knowledge Management

**Shape:** shared knowledge base, many contributors, higher write volume, mixed quality.
Precision and dedup matter more than exhaustive recall — a slightly-missed memory can be
re-added, but a knowledge base full of 5 phrasings of the same policy is unusable.

**Main risks:** duplicate/near-duplicate entries from different people describing the same
fact, and loss of *provenance* (who added what, from which source).

---

## 关键取舍说明

团队场景的第一威胁是 **同一事实被多人用不同措辞重复写入**，所以 `dedupe_threshold` 从个人场景的 0.92 收紧到 0.85——落进 0.85–0.92 灰区的近似条目会被合并而非并存。合并时用 `merge_strategy: append_provenance` 保留所有来源引用（配合强制的 `[source:]` / contributor tag，即 `require_source: true`），这样“谁在什么依据下加的”不会在合并中丢失——provenance 是共享知识库能被信任的前提。

检索侧 `recency_weight` 抬到 0.35，因为团队知识里新条目通常是对旧条目的修订；`importance_weight` 反而降到 0.25，因为跨贡献者的重要性标注噪声更大、不可尽信。`min_score` 提到 0.65 + 开启 `rerank_enabled` 和 `dedupe_on_read`：宁可少召回也要保证返回集干净，这是 precision-over-recall 的取舍。漏掉的记忆可以重加，但充斥重复的知识库无法使用。

---

## Usage

Apply this configuration at runtime via the `memory_apply_config` MCP tool. The field
values below are identical to [`config.yaml`](./config.yaml):

```python
memory_apply_config(config=json.dumps({
    "retrieve_config": {
        "relevance_weight": 1.0,
        "recency_weight": 0.35,
        "importance_weight": 0.25,
        "limit": 10,
        "min_score": 0.65,
        "rerank_enabled": True,
        "dedupe_on_read": True
    },
    "update_config": {
        "dedupe_threshold": 0.85,
        "default_importance": 5,
        "merge_strategy": "append_provenance",
        "require_source": True,
        "per_type_dedupe": {
            "directive": 0.85,
            "insight": 0.85,
            "event": 0.90
        },
        "ttl": {
            "event": "180d",
            "insight": "never"
        }
    }
}))
```

---

## ⚠️ Proposed fields

These fields are **not yet wired server-side — roadmap**. They express the intended
behavior but require BMO review before they map to existing `MemoryConfig`:

- `retrieve_config.dedupe_on_read` — collapse near-identical hits in the result set (`# proposed`)
- `update_config.merge_strategy: append_provenance` — keep both source refs on merge (`# proposed`)
- `update_config.require_source` — reject writes with no `[source:]` / contributor tag (`# proposed`)
