# P 波集成接线报告：P03 同循环纠错与 P04 行为开关

日期：2026-10-31。工作根目录：`D:\Dwen\Alpha26fall\WebAgentNew`。范围：官方生产入口 `OfficialAgent`、EffGen 交互驱动、显式行为开关及对应测试。未派发 worker，未联网，未调用真实模型，未启动浏览器，未使用凭据。

## 结论

| 项 | 判定 | 依据 |
| --- | --- | --- |
| P03 同循环纠错 | 通过（假模型生产入口） | `request_finish` 拒绝经同一生成器回传并继续：错误追加→拒绝→纠正→再次请求→通过；拒绝上限与空文本 type 回传均有测试 |
| P04 行为开关 | 通过 | 关闭时工具不注册、最终发送内容无对应 schema、gate 不安装；启用情况写入事件与 `result.json` |
| 全量回归 | 通过 | `338 passed, 0 failed`（基线 328 + 新增 10） |

## 修改文件清单

| 文件 | 类型 | 内容 |
| --- | --- | --- |
| `code/harness/src/webagent/official_agent.py` | 修改 | 官方入口接线：创建 `StateStore` 并注册状态工具；新增 `GoalDeclarationRuntime`、`goal_declare`、程序绑定 revision/plan_version/branch 的 `state_read`/`state_update`/`plan_update`；注册 `request_finish` 并传入完成核验 gate；`ActionRejected` 转入 `driver.reject_action`；行为开关、运行摘要与 `behaviors` 事件；属性 `store`/`behaviors` |
| `code/harness/src/webagent/effgen_bridge.py` | 修改 | `LoopSettings` 增加 `finish_tool_name`、`max_finish_rejections`；驱动将 `request_finish` 作为同一循环内的核验调用处理：拒绝经 `Generator.send` 回传，接受按 completed 收尾，第三次拒绝转 incomplete；新增 `reject_action`（沿用 `_corrections`/`max_format_corrections`）；`request_finish` 与 finish 兜底 gate 均写 `verification` 事件 |
| `code/harness/src/webagent/config.py` | 修改 | 新增 `OfficialBehaviors`（verification/state_tools/recovery），解析 `official.behaviors` 并写入 `summary()`；默认 true/true/false |
| `code/harness/configs/official.yaml` | 修改 | 显式 `official.behaviors` 开关组；`deadline_margin_seconds` 保持 0 |
| `code/harness/src/webagent/session.py` | 修改 | `set_runtime_summary()`；`result_summary()` 追加 `runtime` 段，`result.json` 记录实际启用项 |
| `code/harness/src/webagent/events.py` | 修改 | 新增事件类型 `behaviors` |
| `code/harness/tests/phase2/test_p03_p04_integration.py` | 新增 | 10 项生产入口测试：同循环纠错、无截图证据、拒绝上限、空文本回传、格式错误上限、三组开关组合、程序绑定字段与旧 revision 拒绝、配置解析 |

未修改 `tools.py`、`models/official.py`、`code/vendor/`、`state/`、`verification/`、`recovery/`、`testing/`；未改 `code/effGen`、`code/official_client` 或 docs 其他文件。

## 关键实现

### P03 同循环纠错

- `request_finish` 通过既有 `extra_tools` 注册，同时加入 `LoopSettings.external_tools`，EffGen 在生成器内**让出**该调用而不是进程内直接执行；驱动在生成器仍存活时运行 `CompletionChecker` gate。
- 拒绝：把 `remaining_goal_ids`、`remaining_conditions`、原因与“不要以部分进度收尾”的指示作为该工具结果 `send` 回同一生成器，循环继续，无旧动作重放。
- 接受：驱动以 `COMPLETED` 收尾并调用 `Session.finish`；claim 为程序文本“every declared requirement is verified”，不声称官方判分。
- 拒绝上限 2：第 3 次拒绝转 `INCOMPLETE`，claim 明确“partial progress is not complete”；每次拒绝/接受均写 `verification` 事件（`stage=request_finish`、`accepted`、`continued`、`attempt`、remaining 条件）。
- 模型不调用 `request_finish` 而直接给最终答案时，保留 `_finish_response` 的 gate 兜底：拒绝即诚实 `INCOMPLETE`，并写 `stage=finish_gate` 事件。
- 空文本 `type`：`OfficialBackend.serialize_action` 抛出的 `ActionRejected` 在 `_consume_step` 被捕获，进入 `driver.reject_action`；未消费提案先 `drop_proposal`，correction 文本经同一 `_corrections` 通道送入下一轮模型输入；沿用 `max_format_corrections=2`，超限按 `FORMAT_ERROR` 终局。

### 目标来源与状态工具

- 模型用 `goal_declare` 声明业务要求与确定性条件（`visible_value_equals`、`visible_value_contains`、`page_text_contains`、`page_url_contains`、`screenshot_saved`、`evidence_saved`、`fact_value_equals`、`goal_objects_complete`）；`goal_declare` 将 requirement、可读验收条件、object scope 写入 `state/` 的 `Goal`，执行条件保存在集成层。
- `state_read`/`state_update`/`plan_update` 接到已合入 `state/` 包；revision、plan_version、branch 由程序注入。模型若显式提交旧 revision，仍按原 `StateConflict` 拒绝并返回当前状态（不悄悄改写）。
- 证据归属由程序绑定：优先使用当前观察已下载的 artifact（screenshot/html/dom）；没有二进制 artifact 时引用 Session 已保存的该观察快照（`kind=None`），确定性条件通过后由 `StateStore.verify_goal` 提交 verified。
- 未声明任何 goal 时，`request_finish` 的请求集为空，`CompletionDecision.accepted=True`，行为与未接 gate 时一致。

### P04 行为开关

- `official.behaviors`：`verification=true`、`state_tools=true`、`recovery=false`（占位，R 波接线）。
- `verification=off`：不注册 `request_finish`，不传 `completion_gate`；`state_tools=off`：不注册 `state_read`/`state_update`/`plan_update`/`goal_declare`。工具不在 `Agent` 中，EffGen 最终提示词自然不含其 schema。
- 启用项写入 `behaviors` 事件与 `result.json` 的 `summary.runtime`（含实际工具名、`completion_gate`、`deadline_margin_seconds`）。
- `deadline_margin_seconds=0` 及既有默认值未改。

## 关键命令与结果

```powershell
& "D:\Dwen\Alpha26fall\WebAgentNew\.venv\Scripts\python.exe" -c "import webagent; print(webagent.__file__)"
# D:\Dwen\Alpha26fall\WebAgentNew\code\harness\src\webagent\__init__.py
```

```powershell
cd D:\Dwen\Alpha26fall\WebAgentNew\code\harness
& "D:\Dwen\Alpha26fall\WebAgentNew\.venv\Scripts\python.exe" -m pytest tests\phase2\test_p03_p04_integration.py -q
# 10 passed
```

```powershell
& "D:\Dwen\Alpha26fall\WebAgentNew\.venv\Scripts\python.exe" -m pytest tests\test_official_agent.py tests\test_official_backend.py tests\test_official_model.py tests\test_official_probe.py tests\test_effgen_interactive.py tests\test_tools.py tests\test_session_lifecycle.py tests\test_events.py -q
# 99 passed
```

```powershell
& "D:\Dwen\Alpha26fall\WebAgentNew\.venv\Scripts\python.exe" -m pytest tests -q
# 338 passed in 137.59s (0:02:17)
```

## P03 逐项判定

| 要求 | 判定 | 证据 |
| --- | --- | --- |
| 完成核验拒绝在生成器结束前反馈，同一循环继续 | 通过 | `test_request_finish_refusal_is_returned_to_the_same_loop_and_then_passes`：同一步内 `request_finish` 拒绝后继续产出纠正 `web_type`；`driver` 与 `driver.agent` 身份不变；拒绝文本进入下一轮 prompt |
| 事件记录 gate 拒绝与继续 | 通过 | 同一测试断言 `stage=request_finish` 事件序列 `(accepted, continued)=[(False,True),(True,False)]`，含 `remaining_goal_ids=["g1"]`、`remaining_conditions=["g1:value"]` |
| 拒绝-继续上限 2，超限诚实 incomplete | 通过 | `test_finish_refusals_are_bounded_and_end_honestly_as_incomplete`：3 次拒绝，`driver.finish_rejections=3`、终态 `incomplete`、goal 仍为 `running`、claim 含 “partial progress is not complete” |
| 空文本 `type` 拒绝进入下一轮模型输入且运行继续 | 通过 | `test_empty_type_refusal_enters_the_next_model_turn_and_the_run_continues`：拒绝后同一 step 返回下一动作 `click [2]`，prompt 含 “web_type needs non-empty text”；仅 1 条 `dispatched` |
| 格式纠正超限按 FORMAT_ERROR | 通过 | `test_empty_type_corrections_reach_the_format_error_cap`：3 次空文本，`driver.terminal.status=format_error`，无 dispatched |
| 目标来自状态工具声明，条件可为 `visible_value_equals` | 通过 | P03 主测试：`goal_declare` 写入 goal，检查当前官方 AX 子 `StaticText` 值；goal 最终 `verified`，证据 `ev-obs-2-screenshot` |
| 工程字段程序绑定、旧版本拒绝 | 通过 | `test_state_tools_bind_engineering_fields_and_reject_a_stale_revision`：未给 revision 的更新被程序绑定并成功；显式 revision=0 返回 `stale_base_revision` 与当前 revision=1 |
| 未声明 goal 时 `request_finish` 直接通过 | 通过 | `test_verification_on_without_state_tools_keeps_the_gate_and_passes_with_no_goals`：无 goal，`request_finish` 直接 completed |
| 不重放旧动作、不把部分进度记为完成 | 通过 | 主测试仅 2 条 dispatched（错误追加、纠正），无重放；上限测试目标保持 `running` 而非 verified |
| 无截图时完成核验仍有真实证据 | 通过 | `test_completion_uses_the_saved_observation_when_no_screenshot_exists`：引用已保存观察快照（`ev-obs-0-observation`，`kind=None`），goal verified |

## P04 逐项判定

| 要求 | 判定 | 证据 |
| --- | --- | --- |
| 配置显式开关组与默认值 | 通过 | `test_official_config_exposes_the_explicit_behavior_group`：`official.yaml` 解析为 true/true/false；错误布尔与未知键报 `ConfigError` |
| 全关闭时工具不注册、最终发送内容无 schema、gate 不传 | 通过 | `test_all_behavior_switches_off_remove_tools_schemas_and_the_gate`：捕获 gateway 最终 messages，`state_read`/`state_update`/`plan_update`/`goal_declare`/`request_finish` 均不出现；`agent.store is None`、`driver.completion_gate is None`；行为与现状一致（`stop [done]`） |
| verification 关闭但 state_tools 开启 | 通过 | 同上测试的反向组合 `test_state_tools_on_with_verification_off_keeps_ledger_tools_and_no_gate`：ledger 工具在提示词中，`request_finish` 不在，gate 为 None |
| state_tools 关闭但 verification 开启 | 通过 | `test_verification_on_without_state_tools_keeps_the_gate_and_passes_with_no_goals`：无 state schema，`request_finish` 与 gate 保留，零 goal 直接通过 |
| 运行摘要/事件记录实际启用情况 | 通过 | `behaviors` 事件含 `behaviors`、`tools`、`completion_gate`、`deadline_margin_seconds`；`result.json` 的 `summary.runtime` 同值 |
| 提前截止保持关闭 | 通过 | `official.yaml` 与 `BudgetConfig` 的 `deadline_margin_seconds=0` 未改；行为摘要记录 0.0 |
| recovery 占位默认 false | 通过 | `OfficialBehaviors.recovery=False`，无恢复/接管接线，事件与摘要均为 false |

## 未完成项与边界

1. **未运行真实模型与官方平台**：P03/P04 仅以假模型走真实 `OfficialAgent` + 假 `model_api`/`browser_api` 验证；P05 与本波真实端验收由主线程执行。
2. **恢复/接管未接线**：`official.behaviors.recovery=false` 为 R 波占位，未实现恢复循环、`RecoveryIntake` 生产替换或 `deadline` 相关改动。
3. **条件注册不持久化**：`GoalDeclarationRuntime` 的可执行条件保存在进程内；快照只保存 Goal 的 requirement/可读验收条件。进程重启后重载状态不会重建可执行条件，`request_finish` 会以验收条件文本拒绝而非恢复原核验计划。
4. **语义核验未开放给 `goal_declare`**：声明工具只接受确定性检查；`semantic` 条件仍需 E01 的统一模型路径，未在 P03 接线。
5. **对象范围完成度依赖模型状态更新**：`object_scope` 的 `completed_object_ids` 仍由模型经 `state_update` 记录，官方入口未接观察记录器；带对象范围的 goal 可能保持未核验。
6. **接受路径不产生最终自然语言答案**：`request_finish` 通过即终止，claim 为程序文本；模型最后的自然语言总结不进入 `result.json`。
7. **空文本拒绝产生两条 `format_correction` 事件**：一条来自 `backends/official.py` 的本地拒绝（`dispatched=False`），一条来自驱动纠正通道（`corrections_so_far`）；两者阶段不同，未合并事件。
8. **P04 仅覆盖官方入口**：其他入口（本地 fake/真实站点）未加开关；官方配置下 deadline 与单动作约束不变。
