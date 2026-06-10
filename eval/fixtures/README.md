# fixtures/core_v1.jsonl

> 2026-06-10 | Siri | Engram 评测固定记忆集 v1（120 条）
> Spec: Obsidian `/Engram/eval-taskset-v0.1-2026-06-10.md`

## 格式

每行一条 JSON，**兼容 `memory_reset` snapshot 格式**（`pkg/server/eval.go` 的 `snapshotRecord`），harness 走 restore 路径注入（re-embed content）。

在 snapshot 原生字段之外有两个 eval 约定：

1. **`collection`（顶层附加字段）**：`engram_eval_user` / `engram_eval_reflection`。harness 注入时按它路由到独立 eval collection（方案 B 物理隔离，不动生产库）。restore 的 JSON decode 会忽略未知字段，不影响兼容性。
2. **`metadata.eval_id`（人类可读 handle）**：taskset 断言用 `eval_id` 引用记忆（如 `mem-rp01-target`）。顶层 `id` 是 `uuid5(NAMESPACE_URL, "engram-eval-core-v1:<eval_id>")` 确定性生成的 UUID（Qdrant point ID 要求 UUID 格式）。

## 构成（120 条）

| 分组 | 数量 | 用途 |
|---|---|---|
| RP targets + 噪声 | 17 | Retrieve Precision（RP-01~08；RP-04 含 5 条近主题噪声；RP-08 否定场景靠语料库不含目标事实保证） |
| DD anchors | 6 | Dedup 三档区间锚点（DD-01~06 的改写/子集变体在 taskset 的 action 里动态构造，不在 fixture） |
| RC pairs | 6 | Recency（rc01 新旧对、rc02 importance 对冲、rc03 时间窗、rc04 过期 valid_until=2026-05-01） |
| XC | 5 | 跨 collection（3 条在 engram_eval_reflection） |
| TR seeds | 4 | Trajectory Replay 种子，case 随数据每周补充 |
| 背景语料 | 82 | 多样化干扰（identity 6 / Frank 14 / BMO 8 / 基础设施 12 / 项目 10 / engram 10 / 杂项 15 / directive 补充 4 + 散布时间戳） |

类型分布：event 58 / insight 45 / directive 11 / identity 6（约 RFC seed 比例缩样）。

## 时间锚点

| 锚点 | 值 (UTC) | 用途 |
|---|---|---|
| T_OLD | 2026-03-01 08:00 | recency "旧" |
| T_MID | 2026-05-10 08:00 | 默认 |
| T_NEW | 2026-06-01 08:00 | recency "新" |
| T_EXPIRED | 2026-05-01 00:00 | rc04 过期 valid_until |

绝对时间戳固定，recency 任务只断言相对 rank order，不依赖运行时刻。

## 重新生成

```bash
python3 gen_core_v1.py > core_v1.jsonl
```

**禁止手改 core_v1.jsonl** — 一律改 `gen_core_v1.py` 再生成（保证 id 确定性与可审计 diff）。
