# ADR-001: 不引入 /admin/ 路径前缀

**日期:** 2026-05-10  
**状态:** 已采纳  
**决策者:** BMO (imbmobmobmo@gmail.com)

## 背景

Phase 5 规划文档（邮件 description）曾写 `/admin/collections`、`/admin/health`，但实际 repo 自 Phase 1 起从未使用 `/admin/` 前缀：

- `POST/GET /collections` — Phase 1 (fb9a663)
- `GET /health` — v0.1 起
- `POST /memories/cross-search` — Phase 2 (ad71cc5)

## 决策

**维持现有路径，不迁移到 `/admin/*` 前缀。**

## 理由

1. **权限已由 middleware 覆盖**：Bearer token + caller-type middleware 已完整隔离管理类操作，`/admin/` 前缀仅是命名约定，不提供额外安全保障。
2. **迁移成本不对等**：全量客户端（MCP skill、integration tests、外部调用方）均需同步修改，收益为零。
3. **一致性**：Phase 5 规划文档中的 `/admin/` 描述属于笔误，应以 repo 实现为准。

## 后果

- Phase 5 及后续阶段的新 endpoint 一律沿用扁平路径（无 `/admin/` 前缀）。
- 如需区分管理类与用户类接口，通过 caller-type middleware header 而非路径前缀实现。
- Phase 5 规划文档（Obsidian phase5-checkpoint-2026-05-10.md）需同步更新，移除 `/admin/` 描述。
