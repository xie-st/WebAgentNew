# 第二阶段 P 波 B 线报告：状态失效入口与恢复反馈摄入解耦

- 副本工作目录：`D:\Dwen\Alpha26fall\WebAgentNew\runtime\parallel\20261031-P1\laneB-state`
- Python：`D:\Dwen\Alpha26fall\WebAgentNew\.venv\Scripts\python.exe`（3.12.9）
- 范围：仅修改 `code/harness/src/webagent/state/`、`code/harness/src/webagent/recovery/`、`code/harness/tests/phase2/` 下文件。
- 未修改 `official_agent.py`、`effgen_bridge.py`、`tools.py`、`config.py`、`session.py`、`verification/`、`backends/`、`testing/`、`control/`，也未修改主树。

## 修改文件清单

新增：

- `code/harness/src/webagent/state/invalidation.py`
  - `GoalInvalidation`：一次程序化撤回的结果与审计记录（from/to、原因、前后 revision、plan_version、来源分支、旧/新增/全部证据 ID、时间）。
  - `InvalidationLog`：`runtime/runs/<run_id>/state/invalidations.jsonl` 追加写与读取；`read_invalidations()` 供重载后查询。
  - 目标状态解析与输入校验：`verified`/`awaiting_verification` 才可撤回；目标仅 `invalidated`/`awaiting_verification`。
- `code/harness/src/webagent/recovery/intake.py`
  - `FeedbackIntake`：新观察 + 真实已执行反馈 + 控制层已丢弃提案记录。
  - `RecoveryIntake`：每个新观察最多形成一次恢复决策；丢弃提案不作为已执行动作，不花恢复尝试或规划轮。
  - `IntakeResult`：下一步、恢复计划/结果、失败证据、丢弃提案、重放动作列表（恒为空）。

修改：

- `code/harness/src/webagent/state/store.py`
  - 新增 `StateStore.invalidate_goal(goal_id, reason, evidence_refs=None, *, to_status=None, base_revision=None, plan_version=None, source_branch=None, event_log=None)`。
  - 新增 `invalidation_path`、`read_invalidations()`、`attach_event_log()`；构造器新增可选 `event_log`。
  - 撤回提交后追加审计记录；有事件日志时写一条 `recovery` 事件（失败不影响账本提交）。
- `code/harness/src/webagent/state/__init__.py`
  - 导出 `GoalInvalidation`、`InvalidationLog`、`WITHDRAWABLE_FROM`、`WITHDRAWAL_TARGETS`、`read_invalidations`。
- `code/harness/src/webagent/recovery/__init__.py`
  - 导出 `FeedbackIntake`、`IntakeResult`、`RecoveryIntake`。

新增测试：

- `code/harness/tests/phase2/test_state_invalidation.py`（11 项）
- `code/harness/tests/phase2/test_recovery_fresh_observation.py`（4 项）

## 关键命令与结果

先确认导入本副本：

```powershell
cd D:\Dwen\Alpha26fall\WebAgentNew\runtime\parallel\20261031-P1\laneB-state\code\harness
$env:PYTHONPATH="D:\Dwen\Alpha26fall\WebAgentNew\runtime\parallel\20261031-P1\laneB-state\code\harness\src"
& "D:\Dwen\Alpha26fall\WebAgentNew\.venv\Scripts\python.exe" -c "import webagent; print(webagent.__file__)"
```

结果：

```text
D:\Dwen\Alpha26fall\WebAgentNew\runtime\parallel\20261031-P1\laneB-state\code\harness\src\webagent\__init__.py
```

新增定向测试：

```powershell
$env:PYTHONPATH="D:\Dwen\Alpha26fall\WebAgentNew\runtime\parallel\20261031-P1\laneB-state\code\harness\src"
& "D:\Dwen\Alpha26fall\WebAgentNew\.venv\Scripts\python.exe" -m pytest tests\phase2\test_state_invalidation.py tests\phase2\test_recovery_fresh_observation.py -q
```

结果：`15 passed in 1.95s`。

现有 state/recovery 语义回归：

```powershell
cd D:\Dwen\Alpha26fall\WebAgentNew\runtime\parallel\20261031-P1\laneB-state\code\harness
$env:PYTHONPATH="D:\Dwen\Alpha26fall\WebAgentNew\runtime\parallel\20261031-P1\laneB-state\code\harness\src"
& "D:\Dwen\Alpha26fall\WebAgentNew\.venv\Scripts\python.exe" -m pytest tests\phase2\test_state_facts.py tests\phase2\test_state_persistence.py tests\phase2\test_state_plan.py tests\phase2\test_state_tools.py tests\phase2\test_recovery_closure.py tests\phase2\test_recovery_stagnation.py tests\phase2\test_recovery_unknown_effect.py tests\phase2\test_recovery_restart.py tests\phase2\test_integration_e1_l04_l05.py -q
```

结果：`47 passed in 15.95s`。

全量：

```powershell
cd D:\Dwen\Alpha26fall\WebAgentNew\runtime\parallel\20261031-P1\laneB-state\code\harness
$env:PYTHONPATH="D:\Dwen\Alpha26fall\WebAgentNew\runtime\parallel\20261031-P1\laneB-state\code\harness\src"
& "D:\Dwen\Alpha26fall\WebAgentNew\.venv\Scripts\python.exe" -m pytest -q
```

结果：`313 passed in 150.15s`；基线为 `298 passed`，新增 15 项，`0 failed`。

## 验收项逐项判定

### 1. 失效入口：通过

- verified 目标失效后状态正确：
  - 带新冲突证据：`verified → invalidated`，state revision +1，`completion_status() == incomplete`，`verified_goal_ids` 移除该目标。
  - 不带新证据：`verified → awaiting_verification`，不编造因果；也可显式 `to_status="awaiting_verification"`。
  - `awaiting_verification → invalidated`（新证据到达）通过。
- 来源保留：
  - 目标记录不删除；requirement、acceptance_criteria、object_scope、completed_object_ids 原样保留。
  - 旧证据保留并追加新证据，审计记录同时保存 `previous_evidence_ids`、`appended_evidence_ids`、`evidence_ids`。
  - 撤回后重新 `verify_goal` 可再次 verified，旧/冲突/复核三份证据同时保留。
- 事件可查：
  - 每次真实撤回写一条 `state/invalidations.jsonl`，`store.read_invalidations()` / `read_invalidations(state_dir)` 重载后可读。
  - 注入 `EventLogger` 时另写一条 `recovery` 事件（含 `goal_id`、`state_revision`、`result.invalidation`），测试从 `events.jsonl` 读回。
- 幂等：
  - 同目标、同目标状态重复调用返回原记录 `changed=False`，revision 不变，不追加第二条审计、不重复发事件。
- 无效输入报错：
  - 未知目标、空/纯空白/非字符串理由、pending/running 等非完成声明、非法目标状态、错误证据类型、跨 run 证据均抛 `StateValidationError` / `EvidenceReferenceError`，账本与审计均不变。
- 模型不可达：
  - `STATE_TOOL_NAMES` 仍为 `{state_read, state_update, plan_update}`；`StateToolRuntime` 无撤回方法。

### 2. 解耦回归：通过

`test_recovery_fresh_observation.py` 用真实 `ControlCoordinator.discard_proposal` 构造“旧提案已取消、新观察已到”：

- 无已执行动作时：`RecoveryIntake` 不调用 planner、不授权恢复尝试、不改账本；`next_step` 引用新观察 ID，旧 action/旧 observation 均不出现，`replayed_action_ids == ()`。
- 有真实已执行动作时：`FailureEvidence.observation_id` 与新观察一致，计划锚点为真实执行的动作，awaiting 目标被撤销，尝试计数为 1。
- 同一新观察重复回调：返回 `reused_observation=True`，不再增加恢复尝试或规划轮。
- 把已丢弃提案伪报为 `executed_action_id` 会抛 `RecoveryError`，阻止接线错误重放旧提案。

未强行政接口：现有 `RecoveryPlanner` / `RecoveryController` 语义未改，新增的只是摄入边界的窄接口。

### 3. pytest 全量 0 failed：通过

- 导入确认指向本副本 `src`。
- 全量 `313 passed`，无 failed/error。

### 4. 不破坏现有 state/、recovery/ 测试语义：通过

- 现有 state/recovery 定向 47 项通过。
- 全量 313 项通过（含 E1 恢复、E2 控制、D3 集成与第一阶段回归）。
- 特别保留 `test_verified_affected_goal_is_reported_as_missing_ledger_support` 的现有语义：`RecoveryController` 仍拒绝直接撤回 verified 目标，不在本线改动。

## 未完成项与边界

- **生产控制器未接线**：`RecoveryController.apply` 仍按原语义拒绝 verified 目标；新增 `invalidate_goal` 已可供 R 波/G3 接线使用，本线未改 controller，以免破坏现有测试语义。
- **控制循环未改**：`ControlledLoopRunner` 的 deferred-feedback 仍会先推进生成器产生旧观察提案再丢弃（`effgen_bridge` 缺 `ingest_feedback_without_advance`，该文件不在本线所有权）。恢复侧摄入接口已就绪，生产循环替换属共享集成文件接线。
- **事件字段**：未改 `events.py`；撤回事件使用既有 `recovery` 事件类型，审计另存 `invalidations.jsonl`。
- **未运行**：真实模型、官方平台、公网真实网站、CLI `suite`；本线只做离线单元与本地隔离范围。
- **不在首版范围**：多分支并发撤回、失效传播闭包自动化（闭包仍由 `RecoveryPlanner` 计算）、细粒度共享写锁。
