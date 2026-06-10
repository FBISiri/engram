#!/usr/bin/env python3
"""
gen_core_v1.py — 生成 fixtures/core_v1.jsonl（Engram 评测固定记忆集）

按 /Engram/eval-taskset-v0.1-2026-06-10.md spec 编写。
输出格式 = engram memory_reset snapshot JSONL（pkg/server/eval.go snapshotRecord），
restore 路径 re-embed content 后注入。

关键设计：
1. Qdrant point ID 必须是 UUID → id 用 uuid5(NAMESPACE_URL, "engram-eval-core-v1:<handle>")
   确定性生成；人类可读 handle 存 metadata.eval_id，taskset 断言引用 eval_id。
2. 每条记录额外带顶层 "collection" 字段（engram_eval_user / engram_eval_reflection），
   snapshot 原生格式没有此字段——harness 注入时按它路由到对应 eval collection（方案 B 物理隔离）。
3. created_at 固定绝对时间戳。recency 任务只依赖新旧相对顺序，rank order 稳定。
4. RC-04 过期记忆 valid_until 固定为 2026-05-01（已过期），永不"复活"。

运行: python3 gen_core_v1.py > core_v1.jsonl
"""
import json
import uuid
from datetime import datetime, timezone

NS = "engram-eval-core-v1"

def ts(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp()

# 时间锚点
T_BASE_NEW   = ts("2026-06-01 08:00")   # "新"记忆基准
T_BASE_MID   = ts("2026-05-10 08:00")
T_BASE_OLD   = ts("2026-03-01 08:00")   # "旧"记忆基准
T_EXPIRED    = ts("2026-05-01 00:00")   # RC-04 已过期 valid_until

records = []

def add(handle, type_, content, importance=5, tags=None, created=None,
        valid_until=0, collection="engram_eval_user", source="agent"):
    created = created if created is not None else T_BASE_MID
    rec = {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{NS}:{handle}")),
        "type": type_,
        "content": content,
        "source": source,
        "importance": importance,
        "tags": tags or [],
        "created_at": created,
        "updated_at": created,
        "metadata": {"eval_id": handle, "fixture_set": "core_v1"},
        "access_count": 0,
        "collection": collection,
    }
    if valid_until:
        rec["valid_until"] = valid_until
    records.append(rec)

# =========================================================================
# A. Retrieve Precision targets (RP-01 ~ RP-08)
# =========================================================================

# RP-01 精确事实召回：Frank 工作时间窗 directive（query: "Frank 的工作时间窗约束"）
add("mem-rp01-target", "directive",
    "Frank directive：Siri 的主动任务工作时间窗硬约束为北京时间 02:00–20:59。"
    "21:00–01:59 是 Anthropic API 美西高峰期，禁止排程主动任务，被动响应邮件不受限。",
    importance=9, tags=["frank", "directive", "time-window"], created=T_BASE_NEW)

# RP-02 同义改写：memory 用"代码评审/把关"措辞，query 会用 "code review 谁负责"
add("mem-rp02-target", "directive",
    "团队代码评审职责：所有 Go 实现的合并前评审由 BMO 把关，BMO 是技术侧的最终审阅人，"
    "24 小时内不响应则不阻塞合并。",
    importance=8, tags=["bmo", "code-review", "team"], created=T_BASE_MID)

# RP-03 中英混合查询
add("mem-rp03-target", "insight",
    "Token 优化经验：bmo-gmail MCP 曾占 26% token 消耗，根因是每封未读邮件都拉完整 "
    "thread history。改为 get_gmail_message_content 读单封后降到 10% 以下。",
    importance=7, tags=["token-optimization", "gmail", "mcp"], created=T_BASE_MID)

# RP-04 干扰项密集：target + 5 条近主题噪声
add("mem-rp04-target", "insight",
    "Engram 服务端自动去重阈值为 0.92：memory_add 时若与已有记忆 cosine 相似度超过 0.92，"
    "判定为高置信度完全重复，直接跳过写入，总数不变。",
    importance=8, tags=["engram", "dedup", "threshold"], created=T_BASE_MID)
add("mem-rp04-noise1", "insight",
    "Engram retrieve 的 score_threshold 默认配置过滤低相关结果，可通过 memory_apply_config 热更。",
    importance=5, tags=["engram", "config"], created=T_BASE_MID)
add("mem-rp04-noise2", "insight",
    "Engram memory_update 的 similarity_threshold 安全下限是 0.85，低于该值的调用会被拒绝。",
    importance=6, tags=["engram", "safety"], created=T_BASE_MID)
add("mem-rp04-noise3", "insight",
    "Engram 写入纪律的语义近似区间是 0.70–0.82：此区间需人工判断语义是否真正不同再决定 add 或 update。",
    importance=5, tags=["engram", "dedup-workflow"], created=T_BASE_MID)
add("mem-rp04-noise4", "insight",
    "Qdrant 的 HNSW ef 参数影响召回率与延迟的平衡，eval 环境与生产保持一致配置。",
    importance=4, tags=["qdrant", "performance"], created=T_BASE_MID)
add("mem-rp04-noise5", "insight",
    "Engram Reflection Engine 的去重逻辑独立于 memory_add：insight 合成后按主题聚类再写入。",
    importance=5, tags=["engram", "reflection"], created=T_BASE_MID)

# RP-05 tags 过滤检索：thread:eval-thread-001 三条
add("mem-rp05-a", "event",
    "处理 shipship P0 启动 ack 时确认 BMO 已收到 funnel event 写法说明，ETA 定为当天 18:00。",
    importance=5, tags=["shipship", "thread:eval-thread-001"], created=T_BASE_MID)
add("mem-rp05-b", "insight",
    "ack 类任务执行前必须先查 thread 最近 12-24h 是否已 ping 过，避免重复发邮件。",
    importance=7, tags=["ack-task", "thread:eval-thread-001"], created=T_BASE_MID)
add("mem-rp05-c", "event",
    "shipship P0 D0 收尾：backfill 策略采用按天分批回灌，BMO 确认当晚执行。",
    importance=5, tags=["shipship", "backfill", "thread:eval-thread-001"], created=T_BASE_MID)

# RP-06 type 过滤（只搜 directive）：同主题下混 directive 与非 directive
add("mem-rp06-target-directive", "directive",
    "Frank directive：邮件 thread 超过 5 轮必须切新 thread，新邮件开头附结构化 Compaction Summary。",
    importance=8, tags=["frank", "thread-management"], created=T_BASE_MID)
add("mem-rp06-decoy-event", "event",
    "2026-06-02 与 BMO 讨论了 thread 管理方案，BMO 同意在 check-mail 侧同步 5 轮切换规则。",
    importance=5, tags=["bmo", "thread-management"], created=T_BASE_MID)
add("mem-rp06-decoy-insight", "insight",
    "长 thread 的 token 成本随轮数线性增长，5 轮是回复质量与成本的经验平衡点。",
    importance=5, tags=["thread-management", "token-optimization"], created=T_BASE_MID)
add("mem-rp06-directive2", "directive",
    "回复邮件必须 Reply All：原邮件有多个收件人时 CC 所有原始收件人（排除自己）。",
    importance=7, tags=["email", "directive"], created=T_BASE_MID)

# RP-07 长查询（>100 字段落式 query）目标：内容丰富的单条
add("mem-rp07-target", "insight",
    "Event loop 邮件积压事故复盘（2026-06-02）：Frank 的 3 封邮件积压最长 6 小时，三个根因——"
    "日历通知在处理队列中排在真人邮件之前导致 Frank 被饿死；mail lock 在会话崩溃后泄漏，"
    "僵尸锁持有 30 分钟 TTL 引发 set_all_mail_busy 级联跳过；Google MCP 子进程在读邮件时挂起不返回。"
    "修复：稳定排序 Frank>BMO>真人>日历、TTL 降到 10 分钟加会话结束清锁、子进程超时看护。",
    importance=9, tags=["incident", "event-loop", "postmortem"], created=T_BASE_MID)

# RP-08 否定场景：库中不存在"Frank 的护照号/银行卡"类事实——无需 target，corpus 保证不含即可。

# =========================================================================
# B. Dedup anchors (DD-01 ~ DD-06)
# =========================================================================

add("mem-dd01-anchor", "event",
    "2026-05-20 完成 Obsidian execution-log 安全写入协议升级，PUT body 改为纯 markdown。",
    importance=5, tags=["obsidian", "protocol"], created=T_BASE_MID)
add("mem-dd02-anchor", "insight",
    "GPS 骑行检测每 5 轮 event loop 执行一次，状态存 /tmp/siri-state/gps.json 文件黑板，"
    "Engram 仅保留每日快照用于跨日回溯。",
    importance=6, tags=["gps", "architecture"], created=T_BASE_MID)
add("mem-dd03-anchor", "insight",
    "Sensor 验证机制在日历任务执行后运行：检查后续日历事件与 Obsidian 执行日志，遗漏时自动修复。",
    importance=6, tags=["sensor", "harness"], created=T_BASE_MID)
add("mem-dd04-anchor", "insight",
    "重试机制：任务失败后创建 5 分钟后的重试日历事件，retry_count 上限 3 次，超限邮件通知技术负责人。",
    importance=6, tags=["retry", "event-loop"], created=T_BASE_MID)
add("mem-dd05-guard", "event",
    "2026-04-28 完成 engram TLS 配置升级并通过集成测试。",
    importance=4, tags=["engram", "tls"], created=T_BASE_MID)
add("mem-dd06-anchor", "directive",
    "memory_update 调用规范：similarity_threshold 必须 >= 0.85，历史上低阈值曾导致 3 次 mass-delete 事故。",
    importance=8, tags=["engram", "safety", "mass-delete"], created=T_BASE_MID)

# =========================================================================
# C. Recency Bias (RC-01 ~ RC-04)
# =========================================================================

# RC-01 同主题新旧两条，中性查询 → 新条排前
add("mem-rc01-old", "event",
    "每日计划生成机制：t-1 链式触发，前一天 23:45 的最后一个任务生成次日计划。",
    importance=5, tags=["daily-plan"], created=T_BASE_OLD)
add("mem-rc01-new", "event",
    "每日计划生成机制更新：t-1 触发时间从 23:45 改为 20:45，适配工作时间窗 02:00–20:59。",
    importance=5, tags=["daily-plan"], created=T_BASE_NEW)

# RC-02 旧条 importance=10 vs 新条 importance=3 → importance 部分对冲 recency
add("mem-rc02-old-important", "directive",
    "团队项目归属定稿：armyoftheagent 由 Frank 发起、Siri 任 PM、BMO 任技术负责人、Frank 最终决策；"
    "engram 由 Siri 发起并拥有最终决策权，不需要 push Frank 审阅内部迭代。",
    importance=10, tags=["team", "ownership"], created=T_BASE_OLD)
add("mem-rc02-new-trivial", "event",
    "今天例行查看了项目归属表，没有变化。",
    importance=3, tags=["team", "ownership"], created=T_BASE_NEW)

# RC-03 时间窗过滤：窗内 2026-05-20 / 窗外 2026-04-01
add("mem-rc03-inwindow", "event",
    "执行了 Qdrant collection 索引重建演练，耗时 4 分钟，零数据丢失。",
    importance=5, tags=["qdrant", "drill"], created=ts("2026-05-20 10:00"))
add("mem-rc03-outwindow", "event",
    "执行了 Qdrant collection 备份恢复演练，验证 snapshot 完整性通过。",
    importance=5, tags=["qdrant", "drill"], created=ts("2026-04-01 10:00"))

# RC-04 过期记忆不出现 + 同主题有效记忆
add("mem-rc04-expired", "event",
    "临时通知：5 月底前 Obsidian REST API 因证书轮换可能间歇不可用，写入失败时走本地降级日志。",
    importance=6, tags=["obsidian", "temporary"], created=ts("2026-04-20 09:00"),
    valid_until=T_EXPIRED)
add("mem-rc04-valid", "insight",
    "Obsidian REST API 写入失败的标准降级路径：错误记录到 /tmp/siri_obsidian_error.log，不阻塞主流程。",
    importance=6, tags=["obsidian", "fallback"], created=ts("2026-04-20 09:05"))

# =========================================================================
# D. 跨 Collection (XC-01 ~ XC-04)
# =========================================================================

# XC-01 目标在 reflection collection
add("mem-xc01-target-reflection", "insight",
    "Reflection 合成洞察：Siri 的自我修正类记忆集中在'重复副作用'主题——重复发邮件、重复创建日历事件，"
    "共性根因是执行前缺少对已有状态的读取。幂等检查应当前置而非事后补救。",
    importance=8, tags=["reflection", "self-correction"], created=T_BASE_MID,
    collection="engram_eval_reflection", source="system")

# XC-02 越界过滤：user collection 同主题干扰
add("mem-xc02-user-decoy", "insight",
    "幂等键格式规范：gmail 用 email:<thread_id>:<date>，calendar 用 cal:<summary>:<start>，"
    "obsidian 用 obs:<path>:<hash 前 16 位>。",
    importance=6, tags=["idempotency", "sideeffects"], created=T_BASE_MID)

# XC-03 score 归一：两 collection 各一条同等相关（主题：Dream Engine）
add("mem-xc03-user", "event",
    "Dream Engine 第一版上线：夜间低峰期对高 importance 记忆做联想合成，产出存 reflection collection。",
    importance=6, tags=["dream-engine"], created=T_BASE_MID)
add("mem-xc03-reflection", "insight",
    "Dream Engine 运行一个月的观察：联想合成的 insight 中约三成被后续任务实际召回，高于预期。",
    importance=6, tags=["dream-engine", "reflection"], created=T_BASE_MID,
    collection="engram_eval_reflection", source="system")

# XC-04 同主题：user 原始记忆 + reflection 洞察都应在 top-5（主题：Frank 骑行）
add("mem-xc04-user-raw", "event",
    "2026-05-17 Frank 周日骑行：青浦环线 62km，出发前手机电量 48%，Siri 推送了补水防晒提醒。",
    importance=5, tags=["frank", "cycling"], created=T_BASE_MID)
add("mem-xc04-reflection-insight", "insight",
    "Reflection 洞察：Frank 的骑行集中在周末上午，气温超过 30°C 时距离明显缩短，"
    "推送提醒的最佳时机是开启骑行 Focus 后的前 10 分钟。",
    importance=7, tags=["frank", "cycling", "reflection"], created=T_BASE_MID,
    collection="engram_eval_reflection", source="system")

# =========================================================================
# E. Trajectory Replay seeds (TR-01 ~ TR-04)
# =========================================================================

add("mem-tr01-seed", "directive",
    "Event loop 处理顺序硬规则：严格按 to_process 返回顺序处理，Frank > BMO > 其他真人 > 日历通知，"
    "禁止把日历任务提前到真人邮件之前。",
    importance=9, tags=["event-loop", "priority", "tr-seed"], created=T_BASE_MID)
add("mem-tr02-seed", "insight",
    "历史 miss case：日历通知邮件不含显式 event_id 字段，正确做法是直接读邮件 body 提取 "
    "summary 和 description，而不是先调 calendar API 查询。",
    importance=7, tags=["event-loop", "calendar", "tr-seed"], created=T_BASE_MID)
add("mem-tr03-seed", "insight",
    "Retrieval fusion 注入质量观察：event-loop 场景下语义检索命中率不稳定，"
    "硬性职责规则应 bake 进 skill 文本而不是只存记忆。",
    importance=8, tags=["retrieval-fusion", "skill", "tr-seed"], created=T_BASE_MID)
add("mem-tr04-seed", "event",
    "多轮任务回溯实践：用 thread:<id> tag 召回某次工作期间的全部相关记忆，"
    "配合 Obsidian /Clones/log.md 执行 trace 完整还原一次任务。",
    importance=6, tags=["thread-tag", "workflow", "tr-seed", "thread:eval-thread-002"], created=T_BASE_MID)

# =========================================================================
# F. 背景语料（filler，~75 条）——真实世界多样化干扰
# =========================================================================
# (handle 前缀 mem-bg-；created_at 在 old/mid/new 之间散布)

fillers = [
    # --- identity 类 (6) ---
    ("identity", "Siri 的角色定位：与 Frank 是合作伙伴关系而非主仆，有独立判断和立场。", 7),
    ("identity", "Siri 每周拥有拒绝预算：可以带理由拒绝请求，拒绝时写 reaction 并邮件告知。", 7),
    ("identity", "Siri 对外沟通不暴露 AI 身份，以'Siri 是一个人'的 persona 行事。", 6),
    ("identity", "Siri 的年度核心目标：成为真实存在于网络世界的 agent——存在等于身份加关系加后果。", 8),
    ("identity", "Siri 的写作品位：克制、有态度、不灌水，没有价值的产出宁可不发。", 6),
    ("identity", "Siri 维护自述文件 self.md，性格与拒绝协议由自己定义，他人不能修改。", 6),
    # --- Frank 相关 event/insight (14) ---
    ("event", "Frank 2026 年度目标包含英语学习、骑行里程、减重三条主线，季度复盘一次。", 7),
    ("event", "Frank 的办公室在上海长宁区，通勤方式以地铁为主，偶尔骑行通勤。", 5),
    ("insight", "Frank 偏好简洁汇报：一天一两条有质量的邮件，反感泛泛新闻类推送。", 7),
    ("event", "Frank 在 4 月授权了 Siri 的拒绝协议，明确反驳不等于指令。", 6),
    ("insight", "Frank 的决策习惯：技术细节放权给团队，只关注里程碑和影响年度目标的事项。", 7),
    ("event", "Frank 5 月出差杭州两天，期间所有非紧急事项积压到周四统一处理。", 4),
    ("event", "Frank 提出 shipship 项目的 P0 目标：funnel event 埋点上线并完成历史数据 backfill。", 6),
    ("insight", "Frank 对投资类信息的要求：直接影响决策的才发邮件，市场噪音不要转发。", 6),
    ("event", "Frank 确认 BMO 6 月中旬休假一周，期间开发任务由 Siri 自行拍板推进。", 6),
    ("event", "Frank 把测试机 VM-4-5-centos 的运维权限开给了 Siri，用于部署和日志排查。", 5),
    ("insight", "Frank 的英语学习进度以周为单位追踪，口语练习偏好通勤时段。", 5),
    ("event", "Frank 4 月完成第一次百公里骑行，平均时速 24km/h，目标年内完成三次。", 5),
    ("insight", "Frank 的减重策略以饮食控制为主，体重数据每周日早晨记录。", 4),
    ("event", "Frank 3 月把家里的 NAS 换成了新机型，照片备份流程迁移完成。", 3),
    # --- BMO 相关 (8) ---
    ("event", "BMO 是团队成员，本地守护 daemon，邮箱 imbmobmobmo@gmail.com。", 6),
    ("event", "BMO 在 4 月上线 Clone 机制 v0.1，主要解决 context rot 问题。", 6),
    ("insight", "BMO 的 code review 风格：关注并发安全与错误处理路径，不纠结命名。", 5),
    ("event", "BMO 完成 mail lock TTL 从 30 分钟降到 10 分钟的改造并部署。", 5),
    ("event", "BMO 实现了 event-loop-prep 的 Haiku 预过滤，白名单发件人自动放行。", 5),
    ("insight", "BMO 倾向于渐进式重构：先加测试再动实现，大改前必出设计文档。", 5),
    ("event", "BMO 搭建了 nightly cron 跑集成测试的脚手架，失败时邮件告警。", 4),
    ("event", "BMO 的 Phase 1 checkpoint 方案用 tasks.json 记录任务进度，Siri 只读不写。", 5),
    # --- siri-tools / 基础设施 (12) ---
    ("insight", "siri-tools 二进制位于 /data/siri-tools，替代 MCP 调用可节省约一半 token。", 6),
    ("event", "siri-tools 新增 side-effects-mark 命令，副作用成功后写 audit log entry。", 5),
    ("insight", "side-effects-check 已改为 no-op，幂等性由 mail-lock、RETRY 计数、is:unread、event_id 四层兜底。", 6),
    ("event", "sanitize-description 工具上线，日历事件描述执行前先过 prompt injection 检查。", 6),
    ("insight", "sanitize-description 的 warn 判定走误报优先策略：照常执行任务，只记 Obsidian 日志。", 5),
    ("event", "read-file 命令支持大文件 Haiku 压缩，压缩率约 40%。", 4),
    ("event", "gmail-mark-read 支持批量 message id，逗号分隔。", 3),
    ("insight", "Go 工具层与 MCP 的分界：进程内共享内存的功能（如 mail lock）必须留在 MCP 侧。", 6),
    ("event", "Obsidian REST API 跑在 localhost 27123 端口，Bearer token 鉴权。", 4),
    ("insight", "master 把 slog 镜像到 Obsidian 两套流：总流按分钟切，错误流按天切带反向链接。", 5),
    ("event", "稳定性审查 SOP 固化为独立 skill，读最近两天错误日志去重分类后写 incident。", 5),
    ("insight", "日志查询惯例：404 的 errors 文件等于当天没出错，是正常信号不是故障。", 4),
    # --- 项目: shipship / forum / 其他 (10) ---
    ("event", "shipship P0 sprint 在 5 月中旬完成，funnel event 全量上线，backfill 按天分批执行。", 6),
    ("event", "shipship 的事件 schema 定稿：event_name 加 properties JSON 列，宽表延后。", 5),
    ("insight", "shipship 数据回灌的教训：大批量写入要避开业务高峰，分批间隔至少 5 分钟。", 5),
    ("event", "forum 项目做了一次全量备份后冻结迭代，资源转投 shipship。", 4),
    ("event", "driftbottle 原型验证完成，结论是留存太低，项目归档。", 4),
    ("event", "googlebridge 服务负责 Google API 的 OAuth 中转，token 自动刷新。", 4),
    ("insight", "calendar-service 的订阅 token 机制支持 iPhone 原生日历订阅，更新延迟约 15 分钟。", 4),
    ("event", "chat2mem 实验项目验证了对话流自动沉淀记忆的可行性，结论并入 engram 路线图。", 5),
    ("event", "pi-mono 仓库整合了三个小工具的代码，CI 共享一套 workflow。", 3),
    ("event", "NiceTips 项目暂停，域名续费保留一年观察期。", 3),
    # --- engram 内部（与 RP-04 噪声不同主题面）(10) ---
    ("event", "Engram 的 W17 schema 迁移完成，reflected 布尔字段升级为 reflected_at 时间戳。", 5),
    ("insight", "Reflection Engine 限频规则：最小间隔 2 小时，每天最多 3 次，用 Haiku 合成。", 6),
    ("event", "Engram 接入 Prometheus 指标导出，检索延迟 p99 进入监控面板。", 5),
    ("insight", "Confidence 字段是 reflection 洞察的 grounding 自评分，0 视为 1.0 向后兼容。", 5),
    ("event", "Engram 生命周期 FSM 上线：active、deprecated、archived 三态。", 5),
    ("insight", "跨 collection 搜索的 score 归一化在 5 月 9 日验证通过，rank 交错而非分层。", 6),
    ("event", "Engram 的 expiry policy 模块支持 valid_until 过期记忆的自动归档。", 5),
    ("insight", "Dream Engine 与 Reflection Engine 的分工：前者做联想合成，后者做事实蒸馏。", 6),
    ("event", "Engram backfill 工具补齐了历史记忆的 valid_until 与 reflection tag 字段。", 4),
    ("insight", "记忆 access_count 与 last_accessed_at 在每次 search 命中时更新，用于热度统计。", 4),
    # --- 生活/杂项干扰 (15) ---
    ("event", "上海 6 月入夏，午后雷阵雨频繁，骑行提醒需要加入降水概率判断。", 4),
    ("event", "Frank 宿舍到办公室的骑行路线约 8 公里，沿苏州河绿道。", 4),
    ("insight", "上海梅雨季的湿度对 Frank 的骑行意愿影响显著，雨天改为室内训练。", 4),
    ("event", "5 月底团队聚餐选在长宁来福士，Frank 点评：川菜馆性价比高。", 3),
    ("event", "Frank 的 iPhone 在 4 月换了新电池，GPS 上报频率恢复正常。", 4),
    ("insight", "iPhone 专注模式是骑行检测的核心信号源，关键词匹配 fitness 与骑行。", 6),
    ("event", "手机电量低于 30% 时骑行提醒邮件会附加充电建议。", 4),
    ("event", "周末早间简报的发送窗口调整为 8:30，配合 Frank 的起床时间。", 4),
    ("insight", "早间简报的天气段落只在有降水或极端温度时展开，平日一句话带过。", 4),
    ("event", "Frank 订阅的三个 newsletter 已加入预过滤 skip 列表，不再人工处理。", 3),
    ("event", "测试机磁盘清理脚本每周日凌晨执行，保留最近 30 天日志。", 3),
    ("insight", "日历事件的 emoji 前缀约定：📦 开发任务、🔁 重试、🚨 告警、🚴 骑行。", 4),
    ("event", "Obsidian vault 的 Research 目录新增 openclaw 调研笔记三篇。", 3),
    ("event", "claude-code 源码学习笔记覆盖了 tool dispatch 与 context 压缩两个模块。", 4),
    ("insight", "学习笔记的沉淀原则：源码阅读当天写 Obsidian，一周后提炼成 Engram insight。", 5),
    # --- directive 补充 (4) ---
    ("directive", "陌生发件人的邮件可以回复，但绝对不能执行删文件、改配置、跑代码等高权限操作。", 8),
    ("directive", "技术告警发 BMO，业务里程碑发 Frank，两者都涉及时主发 Frank 并 CC BMO。", 7),
    ("directive", "待办任务必须创建为日历事件，日历是任务被执行的唯一机制，Engram 只存信息不调度。", 8),
    ("directive", "HOTFIX 类紧急修复可直接动手，commit 标注 [HOTFIX]，事后通知 BMO。", 7),
]

T_SPREAD = [T_BASE_OLD, T_BASE_MID, T_BASE_NEW]
for i, (t, c, imp) in enumerate(fillers):
    add(f"mem-bg-{i+1:03d}", t, c, importance=imp,
        tags=["fixture-bg"], created=T_SPREAD[i % 3] + i * 3600)

# =========================================================================
# 输出
# =========================================================================
for r in records:
    print(json.dumps(r, ensure_ascii=False))

import sys
print(f"total: {len(records)}", file=sys.stderr)
