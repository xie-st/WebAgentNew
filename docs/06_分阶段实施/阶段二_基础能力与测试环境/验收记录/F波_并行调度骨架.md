# F 波（只读并行首版）实施报告

工作副本：`D:\Dwen\Alpha26fall\WebAgentNew\runtime\parallel\20261031-F1\laneF-parallel`

## 1. 结论摘要

- 已新增独立 `webagent.scheduling` 包：默认串行；显式 `parallel` 配置才启用并行；容量可配（默认 2），支持容量 2／4 的等待重叠、就绪排队与活跃数封顶。
- 分支隔离、Tab 独占与单在途动作、共享写全局退化为串行、统一站点节奏与共同退避、根预算与模型并发 1、官方路径退化均已实现并有本地隔离站点 + 假模型测试。
- 未修改 `official_agent.py`、`control/`、`recovery/`、`state/`、`verification/`、`testing/`、`context/`、`experience/`、`tools.py`、`models/`、`code/vendor/`、`code/effGen`、`code/official_client`、`docs`；`backends/`、`session.py`、`effgen_bridge.py` 未改（协调器只把 `Session` 当库使用，不需要改既有驱动）。
- F07 仅留接口：`pause` / `resume` / `drain` 方法、`ControlHookEvent` 与事件名已声明，调用明确抛 `NotImplementedError`，等 R 波合入后补排空逻辑。
- 新增测试 16 项，全部通过；全量 `pytest` 结果：363 passed, 1 failed，唯一失败是副本自带的 R 波旧测试与当前 `state/records.py` 校验不一致（见 §7），与本波改动无关。

## 2. 修改文件清单

| 文件 | 改动 | 理由 |
| --- | --- | --- |
| `code/harness/src/webagent/config.py` | 新增 `ParallelConfig`（L141）、`AppConfig.parallel`（L189）、`summary["parallel"]`、根键 `parallel`、解析与校验（L598-L650） | 顶层 `parallel` 段；未配置默认关闭；容量缺省 2；0/负数/非整数/布尔非法值报 `ConfigError`；`model_concurrency` 首版只允许 1；官方 profile 显式拒绝 `enabled: true`；未改已有键 |
| `code/harness/configs/official.yaml` | 末尾新增 `parallel: {enabled: false, capacity: 1, model_concurrency: 1}` | 官方配置默认关闭并保持单 Tab／单动作／单模型调用语义 |
| `code/harness/src/webagent/scheduling/__init__.py` | 新增包导出 | 公共接口面 |
| `code/harness/src/webagent/scheduling/errors.py` | 新增调度错误：`SchedulingError`、`CoordinatorError`、`WaitCancelled`、`TabUnavailable`、`RateLimited` | 协调器与分支契约 |
| `code/harness/src/webagent/scheduling/limiter.py` | 新增 `SharedRateLimiter` / `RateWait` | 全分支共享单一站点节奏与全局冷却；等待可取消；恢复后逐次放行（不突发） |
| `code/harness/src/webagent/scheduling/tabs.py` | 新增 `TabPool` / `TabLease` | Tab 独占归属；每个 Tab 单动作槽（`max_in_flight` 由构造保证为 1） |
| `code/harness/src/webagent/scheduling/ledger.py` | 新增 `RootBudgetLedger` | 根预算原子领取；单模型槽；失败／取消调用仍入账；拒绝领取单独记录 |
| `code/harness/src/webagent/scheduling/coordinator.py` | 新增 `TaskCoordinator`、`BranchSpec`、`BranchContext`、`BranchReport`、`CoordinatorReport`、`ControlHookEvent` 及 F07 挂点 | 调度骨架、时间线／账本／限流／归属证据 |
| `code/harness/tests/phase2/test_parallel_scheduling.py` | 新增 16 项测试与临时本地 HTTP 站点夹具 | 验收用假模型 + 本地隔离站点，无联网、无真实模型、无浏览器 |

未改动（逐项说明）：`backends/` 不需要新增后端——本版协调器不创建浏览器对象，只调度调用方提供的读取/写入闭包；`session.py` 不需要新钩子——每个分支直接使用现有 `Session`（独立 run_id、独立证据目录、独立 page_id），现有 `propose/consume/ingest` 已保证单页单在途；`effgen_bridge.py` 不需要改——协调器不接管 EffGen 驱动，生产接线留给主线集成。

## 3. 设计要点

### 3.1 配置与开关

`parallel` 段键（全部可选）：

- `enabled`（默认 `false`）
- `capacity`（默认 `2`；正整数，布尔/0/负数/非整数报配置错误）
- `model_concurrency`（首版必须为 1，其它值报配置错误）
- `site_ops_per_second`（默认 1.0，必须 > 0）
- `backoff_seconds`（默认 `[5, 10, 20]`）
- `max_rate_retries`（默认 3）

未配置 `parallel` 时 `AppConfig.parallel.enabled == False`，协调器 `mode = serial`、`capacity = 1`。官方 profile 下 `parallel.enabled: true` 直接 `ConfigError`；官方协调器 `supports_tabs = False`、容量恒为 1。

### 3.2 分支与 Tab

- 每个分支一个 `BranchContext`：独立 `Session`、独立 run_id、独立证据目录（`runtime/runs/<root>-<branch>/events.jsonl`）、独立 `page_id`。
- `TabPool` 独占分配，已归属的 Tab 不能被其它分支申请；`TabLease.action_slot()` 保证同一 Tab 同时最多一条在途动作。
- 测试用 `Session` 的既有绑定校验证明跨 Tab 提案被 `WRONG_PAGE` 拒绝，元件归属不串。

### 3.3 共享写

`BranchSpec(read_only=False, write_targets=("counter",))` 任一出现时，整轮 `execution = serial`、`degraded_reason = shared_write_serialization`、容量降为 1、时间线不重叠；写锁按“写目标 → Tab → 站点节奏”固定顺序申请。测试端点故意做成非原子读改写，串行执行后计数为 2，且服务端观测到的并发写峰值为 1（若并发会丢一次更新）。

### 3.4 统一限流

`SharedRateLimiter` 为整个站点只有一份的节拍：一次操作后下一次至少间隔 `1/site_ops_per_second`；`note_retry_after` 设置全局冷却，所有分支的下一次动作都从同一冷却结束时间之后开始；等待循环检查取消事件，`wake()` 立即唤醒。测试观测：一个分支 429 后两个分支均记录 `cooldown` 等待；恢复后连续请求间隔 0.235–0.266s（配置 4 ops/s，理论 0.25s），无突发；5 秒冷却期间取消，整轮 < 2s 返回。

### 3.5 预算

`RootBudgetLedger`：先原子领取唯一调用号再执行；并发槽固定 1；根 `max_model_calls` 与根 deadline 在领取处检查。失败分支、取消分支已经发生的调用保留在 `records`/`by_branch` 中；排队后被弃用的分支不伪造消耗，状态为 `abandoned`、调用数为 0；领取被拒单独记录 `refused_claims`。各分支 `Session` 仍记录本分支 `model_call` 事件，根账本才是汇总权威。

### 3.6 官方退化

- 官方工具目录无任何 Tab／并行／切页工具（测试按工具名断言）。
- 官方协调器强制串行、单 Tab，同一计划与并行版输出一致；`OfficialAgent` 伪网关捕获的最终发送内容不含 `web_tab/open_tab/switch_tab/parallel/branch_id`，每步一个动作、一个模型调用。

### 3.7 F07 挂点

`CONTROL_HOOK_EVENTS = ("pause_requested", "resume_requested", "drain_requested")`，`ControlHookEvent(kind, reason, at, active_branches, waiting_branches)` 与 `set_control_hook()` 已提供；`pause()/resume()/drain()` 先向已注册 hook 发事件，再抛 `NotImplementedError`（消息说明依赖 R 波控制层）。未实现排空／接管语义，也未写排空验收测试。

## 4. 关键命令与结果

所有测试均先确认导入副本：

```powershell
set PYTHONPATH=D:\Dwen\Alpha26fall\WebAgentNew\runtime\parallel\20261031-F1\laneF-parallel\code\harness\src
python -c "import webagent; print(webagent.__file__)"
# => ...\laneF-parallel\code\harness\src\webagent\__init__.py
```

在 `code/harness` 下执行：

| 命令 | 结果 |
| --- | --- |
| `python -m pytest tests/phase2/test_parallel_scheduling.py -q -s` | 16 passed；时间线证据见 §5 |
| `python -m pytest tests/test_config.py tests/phase2/test_parallel_scheduling.py -q` | 24 passed |
| `python -m pytest -q --deselect tests/phase2/test_r_wave_recovery_integration.py::test_repeated_action_with_new_fact_evidence_is_not_stagnation` | **363 passed, 1 deselected, 0 failed**（161.55s） |
| `python -m pytest -q` | 363 passed, **1 failed**（副本自带 R 波旧测试，见 §7） |
| 新测试模块连续重复运行 10 次 | 10/10 全通过（每次 12.6–16.7s） |

## 5. 逐项判定

### F01 真实并行（功能层通过；真实浏览器轨迹未执行）

容量 2、两个独立只读任务、本地站点 0.35s 延迟读取，时间线（相对本轮开始）：

```
read-a 0.000 → 0.656
read-b 0.000 → 0.594
overlap = 0.594s
```

两个分支页面/证据独立，输出分别为 `read-a` / `read-b`，不是切页演示。**限制**：任务书禁止真实浏览器，本轮用本地 HTTP 站点与假模型验证调度与时间线；F01 要求的真实浏览器轨迹需主线在真实端补测。

### F02 单页与共享写（通过）

- `TabLease` 双线程并发动作测试：`peak=1`、`max_in_flight=1`，第二个动作等待 0.1s。
- 两分支跨 Tab 提案均被 `RejectReason.WRONG_PAGE` 拒绝；各自提案/消费成功。
- 共享写：两次写在时间线上不重叠（`write-b` 在 `write-a` 结束后才开始），`counter=2`，服务端最大并发写=1，`execution=serial`、`degraded_reason=shared_write_serialization`。

### F04 统一限流（通过）

- 一个分支收到 429（Retry-After 0.35）后，两个分支都出现 `kind=cooldown` 等待（read-a 0.360/0.485s、read-b 0.516/0.484s）。
- 恢复后 `/read` 与重试 `/limited` 的相邻启动间隔 0.235/0.265/0.266s（4 ops/s），无突发补发。
- 5s 冷却等待期间 `cancel_all`，整轮 0.4s 左右返回；两个在途分支 `cancelled`、排队分支 `abandoned`，等待记录 `cancelled=True`。

### F05 模型与根预算（通过）

- 3 分支并发运行，假模型观测 `max_concurrent=1`，根账本 `max_concurrent=1`。
- `branch-ok` 成功、`branch-fail` 调用后失败、`branch-cancel` 调用后被取消；账本 `total_calls=3`、按分支均为 1，失败分支调用仍在 `records`。
- 根预算 `max_model_calls=2` + 3 个就绪分支：2 succeeded、1 failed（`BudgetExceeded`），`total_calls=2`、`refused_claims=1`。
- 取消测试中排队分支被弃用：`abandoned`、0 消耗，另 2 个已调用分支仍计费。

### F06 官方退化（通过）

- `official.yaml` 解析后 `parallel.enabled=False`、容量 1；把 `enabled` 改为 true 报 `ConfigError`。
- 官方工具名不含 tab/parallel/switch；官方协调器 `supports_tabs=False`、串行、单 Tab，两个任务输出与并行版一致。
- `OfficialAgent` 伪网关 2 步运行：最终发送内容不含 Tab／并行痕迹，每步一个动作、一个模型调用；串行计划通过。

### F07 全局暂停（仅接口，未实现）

`pause/resume/drain` 签名、事件类型与 hook 注册存在；调用抛出明确的 `NotImplementedError`，不伪装成已暂停。排空、接管、未知写结果保护与冲突写锁释放未实现，未写 F07 验收测试。

### F08 容量与排队（通过）

- 未配置 `parallel`：串行、容量 1、单 Tab、按计划顺序；与并行版同一计划输出一致。
- 未配置容量但启用：容量取 2。
- 非法容量（0、-1、1.5、"2"、true）均 `ConfigError`。
- 容量 4、6 个就绪任务：峰值活跃 4，前 4 个 0–1.0s 完成，后 2 个 0.812/0.875s 才开始，最终 6/6 成功；`max_queue_wait > 0.2`。

## 6. 事件与证据

每个分支仍写自己的 `events.jsonl`／`result.json`；协调器报告包含：`mode/execution/capacity/degraded_reason`、分支状态与起止时间、`timeline`、`max_active`、`overlap_seconds`、`max_queue_wait_seconds`、`rate_waits`、`lock_waits`、根账本快照。测试在 `-s` 下打印时间线，作为真实重叠与排队的证据。

## 7. 未完成项与已知差异

1. **F07 依赖 R 波**：本轮只留 `pause/resume/drain` 接口、事件与 hook；排空／接管／未知写结果保护需在 R 波控制层合入后实现。
2. **协调器尚未接入生产循环**：`official_agent.py`、`effgen_bridge.py`、`session.py`、`backends/`、CLI 均未改；本波交付可独立测试的调度骨架，生产接线与 F09 工作流选用由主线集成时处理。
3. **写冲突靠资源声明**：首版要求任务声明 `read_only/write_targets`；未声明却实际写页面的分支不在契约内，没有动态写检测。细粒度共享写锁按方案后置。
4. **F01 真实浏览器轨迹未执行**：按任务书禁真实浏览器/联网/真实模型，本轮以本地隔离 HTTP 站 + 假模型验证功能与时间线；真实端轨迹与加速比结论需后续真实运行。
5. **F03 动态依赖未实现**：计划固定，不支持修订/取消传播；按首版范围不做入口。
6. **副本基线差异（非本波改动）**：任务书称 338 基线，但该副本实际快照为 348 项：改动前 347 passed + 1 failed；失败项为 `tests/phase2/test_r_wave_recovery_integration.py::test_repeated_action_with_new_fact_evidence_is_not_stagnation`，该文件是旧快照（`FactKind.OBSERVED` 未带 evidence），当前 `state/records.py` 要求 observed fact 必须有证据；主树同名测试已用 `FactKind.INFERRED`。按“不得修改 R 波/state、测试只新增本波文件”的约束未改该文件；若按主树同步这一行，副本全量即为 0 failed。排除该项后全量 **363 passed, 0 failed**。主树 R 波合入后为 353 passed（用户告知），合入时由主线统一处理该快照差异。
7. **稳定性处理**：Windows 上并发创建分支证据目录时曾偶发 `EvidenceStore` 的 “run directory escapes the runtime root”（`Path.resolve()` 竞态）；已把“分配 Tab + 创建分支 Session/证据目录”放在协调器资源锁内串行完成（执行本身仍并行），此后新模块连续 10 次全通过。这是并发目录创建的顺序化，不改变调度并发度。

## 8. 复现

```powershell
cd D:\Dwen\Alpha26fall\WebAgentNew\runtime\parallel\20261031-F1\laneF-parallel\code\harness
set PYTHONPATH=src
python -m pytest tests/phase2/test_parallel_scheduling.py -q -s
python -m pytest tests/test_config.py -q
```
