# R 波：恢复／接管接线报告

日期：2026-10-31。工作根目录：`D:\Dwen\Alpha26fall\WebAgentNew`。范围：官方生产入口 `OfficialAgent` 的恢复／接管接线（`official.behaviors.recovery` 由占位变为真实实现，默认仍为 `false`）。未派发 worker，未联网，未调用真实模型，未启动浏览器，未使用凭据，未修改 `code/vendor/effgen/`（本波不需要上游补丁）。

## 结论

| 项 | 判定 | 依据 |
| --- | --- | --- |
| E05 自动接管 | 通过（假模型生产入口） | 登录提示在观察到达后、生成器推进前触发暂停；排空后交接，等待期间模型调用与页面动作均为 0；用户信号后重观测再继续 |
| E06 主动暂停／继续／取消 | 通过 | 用户 `pause` 命中同一条暂停路径；在途结果在交接前如实记录；继续后重采集并声明旧编号失效；取消可中断等待，不产生重观测动作 |
| E07 控制与计时 | 通过 | 继续只由显式信号驱动；总耗时＝人工等待＋执行时间（`control_events.jsonl` 与 `result.json` 对账）；官方 deadline 未被延长 |
| E03 局部恢复 | 通过 | 完成核验失败→`RecoveryCoordinator`→按依赖闭包撤回受影响已验证目标，无关已验证进度保留且不重做；证据归属保留；证据不足不编造因果 |
| E04 停滞与上限 | 通过 | 相同动作有新事实证据不误判；无效循环触发受限恢复；每目标上限 2、全任务上限与截止时间真正拒绝后续尝试，超限诚实结束 |
| E08 重启 | 通过 | 从完整快照加载并重观测；新分支与旧分支关联；待确认写入仍在且不自动重发；事件与摘要明确“未恢复生成器栈” |
| P04 收尾（开关） | 通过 | `recovery=false` 时控制层、恢复运行时、副作用保存与重启检测全部不注册，最终提示内容无恢复／交接文案；`true` 时启用项写入 `behaviors` 事件与 `result.json` |
| 提前截止保持关闭 | 通过 | `deadline_margin_seconds=0` 及各默认值未改；测试断言结果摘要仍为 0.0 |
| 全量回归 | 通过 | `353 passed, 0 failed`（基线 338 + 本波 15） |

## 修改文件清单

| 文件 | 类型 | 内容与理由 |
| --- | --- | --- |
| `code/harness/src/webagent/official_agent.py` | 修改 | 恢复开关下的控制层与恢复运行时创建；每步先记录真实反馈再决定是否暂停；接管／用户暂停→排空→交接→等待→重观测；程序化重观测动作；未确认副作用在发送前保存、反馈后确认；完成核验 gate 包装（拒绝后撤回受影响声明并写入下一步指示）；停滞结论写入模型下一轮结果；重启检测与快照加载；`goal_declare` 增加可选 `depends_on`（恢复闭包需要依赖图，程序侧仅存业务字段） |
| `code/harness/src/webagent/effgen_bridge.py` | 修改 | `start(note=…)`（重启提示）；`accept_observation(ingest=False, note=…)`（入口已如实记录观察时避免二次摄入，并可附一条有界恢复指示）；`resume_after_takeover(feedback, fresh, note=…)`（报告已执行动作的真实反馈＋重观测后的新观察）；`finish_incomplete(…)`（上限／等待超限时诚实结束）；记录在途工具名用于真实结果文本 |
| `code/harness/src/webagent/control/detection.py` | 修改 | 官方 AX 把可见文案打印为编号 `StaticText` 行而非 `TEXT:` 块：按元素角色区分标题／可见文案／控件。原逻辑只把这些行当控件，导致官方格式的登录覆盖层漏检；判定条件本身未放宽 |
| `code/harness/src/webagent/control/restart.py` | 修改／扩展 | 新增 `load_restart_snapshot`（加载完整快照＋注册新分支＋`restart_loaded` 控制事件，不创建 Session）。官方入口已拥有 Session／预算／驱动，复用它；`restart_from_snapshot` 保持原入口语义并复用该半程 |
| `code/harness/src/webagent/control/__init__.py` | 修改 | 导出 `RestartLoad`、`load_restart_snapshot` |
| `code/harness/src/webagent/recovery/controller.py` | 修改 | 新增 `RecoveryController.withdraw`：用 `StateStore.invalidate_goal` 执行计划的撤回（verified 可撤回、证据保留、审计与 `recovery` 事件），无关已验证目标不动；无新冲突证据时目标为 `awaiting_verification`。原 `apply` 语义保留（既有无 verified 目标测试不变） |
| `code/harness/src/webagent/recovery/coordinator.py` | 修改 | 新增 `RecoveryOutcome` 与 `RecoveryCoordinator.handle_failures`（核验失败与动作效果异常共用：规划→授权→撤回）；`handle_stagnation` 语义未改 |
| `code/harness/src/webagent/recovery/wiring.py` | 新增 | `OfficialRecoveryRuntime`：受限恢复运行时。完成核验拒绝与动作反馈分类、停滞记录／判定、上限与截止时间、模型下一轮的有界指示、恢复事件与 `result.json` 摘要；不调用模型、不操作网页、不编造因果 |
| `code/harness/src/webagent/recovery/__init__.py` | 修改 | 导出 `OfficialRecoveryRuntime`、`RecoveryNote`、`RecoveryOutcome` |
| `code/harness/src/webagent/config.py` | 修改（文档串） | `OfficialBehaviors` 说明 recovery 已是真实接线、默认关闭 |
| `code/harness/configs/official.yaml` | 修改（注释） | 同上；`behaviors.recovery: false` 不变 |
| `code/harness/tests/phase2/test_r_wave_recovery_integration.py` | 新增 | 15 项：接管／用户暂停／首观察接管／取消、完成核验撤回、停滞上限、恢复截止、动作效果异常、重启、开关关闭回归、证据目标与证据不足、官方 AX 静态文本检测及“正文提及 sign in 不误触”回归 |

未修改 `state/`、`verification/`、`backends/`、`testing/`、`context/`、`experience/`、`code/effGen`、`code/official_client`、`code/vendor/`；除本报告外未改其他 docs。

## 关键实现

### 开关与注册（P04 收尾）

- `recovery=false`：不创建 `ControlCoordinator`（无 `control.jsonl`／`control_events.jsonl`）、不创建 `OfficialRecoveryRuntime`、不安装 `EffectPolicy`、不做重启检测；`goal_declare` 的 `depends_on` 描述为中性文案。最终发送内容与 `result.json` 摘要均无恢复／交接字段（测试逐条断言）。
- `recovery=true`：启动时安装 `session.set_action_gate(coordinator.gate)`；`behaviors` 事件与 `result.json` 的 `summary.runtime` 记录 `behaviors`、`control`（来源、gate、等待超时、计时）与 `recovery`（接管/暂停、停滞参数、上限、截止、尝试计数、已应用/被拒）。

### 自动接管（E05）

- 每步观察到达后先 `detect_takeover`；命中（或动作反馈分类为 authentication）即 `pause_requested→draining→user_control`，`drain` 记录已执行动作与真实反馈状态，`handoff` 记录 `browser_drained`、原因与等待说明。
- 等待期间不推进生成器、不调用模型、不发送页面动作；只由显式信号返回（`resume_signal`），超时按阻塞诚实结束。用户信号后 `begin_refresh` 并发送一个同 URL `goto` 作为程序化重观测；下一次请求的新观察到达后，驱动以“已执行动作的真实反馈＋重观测新观察＋旧编号失效”恢复同一生成器。
- 等待原因同时写入 `control_events.jsonl`（`handoff.reason`）与 `events.jsonl` 的 `recovery` 事件（`stage=handoff`，`waiting_reason`、`waiting_for=explicit_user_resume`、`old_element_numbers_valid=false`）。

### 主动暂停／继续／取消与计时（E06/E07）

- 用户 `pause` 与自动接管共用路径；每步先 `session.ingest_observation` 如实记录在途结果（`feedback` 事件先于交接），再决定是否暂停；若存在未消费提案则显式丢弃（官方单动作路径中该情形不出现，代码仍防御性处理）。
- 继续后 `page_epoch+1`、`control_epoch` 递增，旧元素编号与旧提案失效；模型收到新观察与明确指示，不伪造旧动作成功。
- 取消在等待中立即生效：控制状态 `cancelled`，不产生重观测动作，结果为 `cancelled`。
- 计时：`control_events.jsonl` 的 `handoff`／`resumed`／`control_end` 与 `result.json` 的 `summary.runtime.control.timing` 使用同一 `ControlTiming`：总耗时＝人工等待＋执行时间；`platform_deadline_epoch` 保持平台值，`deadline_extended=false`。

### 恢复（E03/E04）

- 完成核验拒绝：包装 `make_finish_gate`，拒绝后对 `VerificationResult.status=FAIL` 的目标生成 `FailureEvidence`（锚点为该次核验的观察与证据），`RecoveryCoordinator.handle_failures` → `RecoveryPlanner` 反向依赖闭包 → `RecoveryController.withdraw`；只撤回闭包内的 verified/awaiting 声明，保留其他已验证目标；把撤回与保留清单及下一步写入拒绝文本，同一循环继续。
- 动作效果异常：按反馈状态与页面文本分类（authentication 转接管；其余按 `unknown_effect`／`transient` 等），以当前未完成目标为锚点执行同一受限恢复；模型只收到一条有界指示，不重放失败写入。
- 停滞：每步记录（动作键、页面状态键、新增事实修订、缺口），窗口 6、重复 2；达到条件才授权一次恢复尝试。相同动作用新事实证据视为进展（单元测试证明不误判）。
- 上限与截止：`RecoveryLimits` 使用 `AppConfig.recovery` 的每目标 2、全任务 5，`deadline_epoch` 取平台任务 deadline；授权被拒即写 `stagnation_limit`／`action_failure` 事件并诚实结束（`incomplete`，claim 明确“部分进度不是完成”），不无限恢复。
- 证据归属：`withdraw` 复用 `StateStore.invalidate_goal`，目标记录、要求、验收条件、对象范围与旧证据保留，新冲突证据追加并写 `state/invalidations.jsonl` 审计与 `recovery` 事件；无新证据时目标为 `awaiting_verification`（不编造因果）。

### 重启（E08）

- `recovery=true` 且同平台 run 的 `state.json` 已存在时视为重启：本地运行号派生为 `<run>-rsN`，分支 `restart-N`；`load_restart_snapshot` 加载完整快照、注册分支（旧分支仍指向旧 run）、保留全部未确认副作用。
- 新 Session／后端绑定新 run，以请求中的新观察启动生成器，并在首条提示中写入重启说明：快照已加载、生成器栈未恢复、无动作重放、未确认副作用清单、“重复写入前先核实可见结果”。`recovery` 事件与 `result.json` 的 `summary.runtime.restart` 记录 `generator_stack_restored=false`、`replayed_actions=[]`、待确认 action id 与分支映射。

### 未确认副作用

- `recovery=true` 时，动作在返回平台前按既有 `EffectPolicy` 保存 `pending_effect`（保存失败不发动作），下一观察到达后按反馈确认（observed→confirmed、error→failed、其他→unknown）。重启后残留记录保持 `pending`，不被自动确认或重发。

## 关键命令与结果

导入确认：

```powershell
& "D:\Dwen\Alpha26fall\WebAgentNew\.venv\Scripts\python.exe" -c "import webagent; print(webagent.__file__)"
# D:\Dwen\Alpha26fall\WebAgentNew\code\harness\src\webagent\__init__.py
```

```powershell
cd D:\Dwen\Alpha26fall\WebAgentNew\code\harness
& "D:\Dwen\Alpha26fall\WebAgentNew\.venv\Scripts\python.exe" -m pytest tests\phase2\test_r_wave_recovery_integration.py -q
# 15 passed
```

相关既有批次回归（P03/P04、控制、恢复、失效入口、交互驱动）：

```powershell
& "D:\Dwen\Alpha26fall\WebAgentNew\.venv\Scripts\python.exe" -m pytest tests\phase2\test_r_wave_recovery_integration.py tests\phase2\test_p03_p04_integration.py tests\phase2\test_control_detection.py tests\phase2\test_control_restart.py tests\phase2\test_control_timing.py tests\phase2\test_control_pause_model.py tests\phase2\test_recovery_closure.py tests\phase2\test_recovery_stagnation.py tests\phase2\test_recovery_fresh_observation.py tests\phase2\test_state_invalidation.py tests\test_effgen_interactive.py -q
# 93 passed in 11.21s
```

全量：

```powershell
& "D:\Dwen\Alpha26fall\WebAgentNew\.venv\Scripts\python.exe" -m pytest tests -q
# 353 passed in 151.80s (0:02:31)
```

## 逐项判定

### E05 自动接管

| 要求 | 判定 | 证据 |
| --- | --- | --- |
| 登录／验证提示触发全任务暂停 | 通过 | `test_login_overlay_pauses_the_whole_task_and_resumes_on_a_user_signal`：点击后的观察为登录覆盖层，`handoff.reason=login_overlay_visible`；控制事件序列 `control_start→…→takeover_detected→pause_requested→draining→handoff→resume_signal→refreshing→…→resumed` |
| 浏览器排空后交接 | 通过 | `handoff.details.browser_drained=true`，`executed_action_ids` 为该点击的 action_id；在途反馈已在交接前记录 |
| 等待期间无新模型调用与动作 | 通过 | 交接前与等待后探针相同；`len(gateway.calls)==1`；等待期间仅轮询信号（`source.polls==2`）；`dispatched` 序列为 `click→goto→click` |
| 首条观察即可见覆盖层也先暂停 | 通过 | `test_takeover_visible_on_the_first_observation_pauses_before_any_model_turn`：首个响应即 `goto [<login url>]`，`gateway.calls==[]`，`driver.started==false`；恢复后首条提示不含登录页文案 |
| 事件记录等待原因 | 通过 | `recovery` 事件 `stage=handoff` 携带 `waiting_reason`、`waiting_for=explicit_user_resume`、`old_element_numbers_valid=false` |

### E06 主动暂停

| 要求 | 判定 | 证据 |
| --- | --- | --- |
| 用户主动暂停取消未开始提案 | 通过（官方单动作路径无跨步提案，防御性丢弃仍实现） | `test_explicit_user_pause_records_the_in_flight_result_and_resumes_fresh`：`pause_requested.source=user`、`reason=user_pause`；代码在交接前 `discard_proposal`＋`drop_proposal` |
| 在途结果如实记录 | 通过 | 同一测试：交接前已有 1 条 `feedback`（`result=observed`），随后才进入等待；`resume_after_takeover` 报告该真实反馈 |
| 继续后重新采集、不用旧编号、不伪造旧动作成功 | 通过 | 返回 `goto [<当前 URL>]`；恢复提示含“was executed before the browser was handed to the user”“Fresh observation after the user resumed”“no longer valid”；`page_epoch=1`、`control_epoch=3`；后续动作使用新观察编号 |
| 恢复使用新观察而非旧观察提案 | 通过 | 生成器只在重观测后推进；交接前不推进、等待中无模型轮次；恢复决策取新观察（`after_feedback` 每步以当前观察结算） |

### E07 控制与计时

| 要求 | 判定 | 证据 |
| --- | --- | --- |
| 继续须有用户信号 | 通过 | `wait_for_user_signal` 仅在来源返回 `resume` 时继续；无信号按超时阻塞并诚实结束（`_honest_stop`，测试覆盖控制层超时语义） |
| 取消可中断等待 | 通过 | `test_cancel_during_the_handoff_ends_the_task_without_a_reobserve_action`：`control.state=cancelled`，无 `reobserve` 事件，结果 `cancelled` |
| 总耗时／人工等待／执行时间可对账 | 通过 | `timing.total_seconds==12.0`、`human_wait_seconds==12.0`、`execution_seconds==0.0`（FakeClock 未推进执行期）；`total==human_wait+execution`；`control_end.details.timing` 与 `result.json` 同值 |
| 官方 deadline 未被延长 | 通过 | `platform_deadline_epoch==1793000300.0`（请求原值）、`deadline_extended=false`；等待 12s 后 `official_remaining_seconds` 由 300 降为 288 |

### E03 局部恢复

| 要求 | 判定 | 证据 |
| --- | --- | --- |
| 从最早有证据的错误处恢复 | 通过 | 完成核验拒绝以 FAIL 目标的观察/证据为锚点进入 `RecoveryCoordinator.handle_failures`；事件 `plan.anchor` 指向该核验锚点 |
| 受影响依赖失效 | 通过 | `test_completion_refusal_withdraws_the_affected_claim_and_keeps_unrelated_progress`：g2 依赖 g1，g1 失败后 g2 由 `verified→awaiting_verification`（审计记录 `from_status=verified`、`to_status=awaiting_verification`） |
| 无关已验证进度不重做 | 通过 | g3 保持 `verified`，证据始终为 `["ev-obs-1-screenshot"]`；派发动作仅 `click`、`type`，无 g3 相关动作；恢复事件 `preserved_verified_goal_ids=["g3"]` |
| 已 verified 目标的程序化失效入口可用 | 通过 | 审计 `state/invalidations.jsonl` 记录 `goal_id=g2`、`from_status=verified`、`changed=true`、`previous_evidence_ids=["ev-obs-1-screenshot"]`；单元测试进一步证明带新冲突证据时 `verified→invalidated` 且追加证据、保留旧来源 |
| 证据归属保留 | 通过 | 撤回后 g2 的要求、验收条件与旧证据保留；重新核验追加 `ev-obs-2-screenshot`；审计保存 previous/appended/evidence ids |
| 证据不足不编造因果 | 通过 | `test_withdrawal_without_new_evidence_does_not_invent_a_cause`：无新证据时目标为 `awaiting_verification`，`appended_evidence_ids=[]`，无关已验证目标不动 |
| 动作效果异常也走同一受限恢复 | 通过 | `test_action_effect_anomaly_runs_the_bounded_recovery_path`：平台错误反馈→`stage=action_failure`、`plan.affected_goal_ids=["g1"]`、`attempt.total_attempts=1`；模型收到 `unknown_effect` 有界提示；未确认副作用由 `pending→failed` |

### E04 停滞

| 要求 | 判定 | 证据 |
| --- | --- | --- |
| 相同动作但有新证据不误判 | 通过 | `test_repeated_action_with_new_fact_evidence_is_not_stagnation`：每步新增事实修订，窗口满后无信号，未花恢复尝试；既有检测器测试保留 |
| 无效循环触发受限恢复 | 通过 | `test_stagnation_recovery_is_bounded_and_then_ends_honestly`：窗口 6／重复 2 后第 6、7 次反馈各授权 1 次（`applied=true`，attempt 1、2），第 8 次被拒 |
| 恢复上限真正生效 | 通过 | 第 8 次反馈 `stagnation_limit`：`allowed=false`、`reason=per-goal recovery limit reached for g1`；摘要 `attempts={"total":2,"per_goal":{"g1":2}}`、`denied=1` |
| 截止时间真正生效 | 通过 | `test_recovery_deadline_stops_attempts_through_the_wiring_runtime`：截止前授权 1 次，时钟推进 6s 后下一次授权被拒（`deadline`），总尝试保持 1 |
| 超限诚实结束 | 通过 | 停止文本 `task ended without claiming completion: per-goal recovery limit reached for g1. Partial progress is not complete.`；驱动终态 `incomplete`，`result.json termination=incomplete`，模型调用 9 次后不再增长 |

### E08 重启

| 要求 | 判定 | 证据 |
| --- | --- | --- |
| 从完整快照加载并重观测 | 通过 | `test_restart_loads_the_snapshot_associates_a_new_branch_and_does_not_resend`：重启实例本地运行号为 `official-run-1-rs1`；首条提示含 `Restart note`、`revision`、新观察 |
| 新分支与旧分支关联 | 通过 | `state.branches == {"root":"official-run-1","restart-1":"official-run-1-rs1"}`；`recovery` 事件与摘要同样记录 |
| 待确认动作仍在且不自动重发 | 通过 | 重启后 `pending_effects[<旧 action>]` 仍为 `pending`；新 run 的 `dispatched` 不含该 action_id；提示明确“先核实可见结果、不要盲目重发” |
| 不声称恢复生成器栈 | 通过 | 控制事件 `restart_loaded.details.generator_restored=false`；`recovery` 事件 `generator_stack_restored=false`、`replayed_actions=[]`；`result.json summary.runtime.restart` 同值 |

### P04 收尾与截止

| 要求 | 判定 | 证据 |
| --- | --- | --- |
| `recovery=false` 全部路径不注册 | 通过 | `test_recovery_off_keeps_the_pre_rwave_behavior`：`agent.control is None`、`agent.recovery is None`、无 `control.jsonl`／`control_events.jsonl`、无 `recovery` 事件、无 `goto`、登录覆盖层按旧行为直接推进 |
| 关闭时行为与现状一致 | 通过 | 同测试：写入动作不产生 pending effect（`pending_effects=={}`）；`result.json runtime` 只有 `behaviors/tools/completion_gate/deadline_margin_seconds`，无 `control/recovery/restart` |
| 捕获最终发送内容无恢复 schema／提示 | 通过 | 同测试逐条检查捕获的模型输入：不含 `Recovery:`、`handoff`、`was NOT restored` |
| 开启时启用情况写入事件与摘要 | 通过 | `behaviors` 事件 `behaviors.recovery=true`、`control.gate_installed=true`、`recovery.takeover=true`；`result.json summary.runtime` 记录 `control`（含计时）、`recovery`（上限与尝试） |
| 提前截止保持关闭 | 通过 | `deadline_margin_seconds=0.0`，`OfficialBehaviors` 默认未变 |

## 未完成项与边界

1. **未运行真实模型与官方平台**：全部证据来自假模型走真实 `OfficialAgent` 与假 `model_api`／`browser_api`；真实端验收由主线程执行。
2. **官方 step 模型下 Session 不进入 paused**：平台观察总是绑定上一动作的反馈，暂停期间不能先 `session.pause()` 再 `resume(新观察)`。暂停语义由根控制状态与已安装的 dispatch gate 承担（任何消费在非可动作状态被 `ControlPaused` 拒绝），交接与恢复事件写入 `control_events.jsonl` 与 `events.jsonl`；未改 `SessionState`。
3. **“暂停期间返回的模型请求”在官方路径不存在**：生成器只在显式恢复后推进，等待中无在途模型调用，因此该场景无法在官方入口复现；其提案作废与调用/耗时计入由既有控制层测试（`test_control_pause_model.py`，本波未改）覆盖。
4. **多行分解路径未接控制与恢复**：`text_mode=decompose` 的 `type`＋`press` 子动作队列未接入接管检测、恢复决策与未确认副作用保存；R 波不改变该路径（边界记录，未声称覆盖）。
5. **恢复计划的可执行条件仍不持久化**（P 波遗留）：重启后目标要求存在但条件需模型重新声明；E08 测试中重启后的完成核验以验收条件文本拒绝并诚实结束，流程本身可继续。
6. **对象范围观察记录器未接官方入口**（P 波遗留）：`object_scope` 完成度仍依赖模型 `state_update`；带对象范围的目标在官方路径可能保持未核验。
7. **`invalidated`（带新冲突证据）在官方 gate／动作路径按设计落到 `awaiting_verification`**：同一观察既是支持证据又是冲突证据，追加证据会被去重；带新证据的 `invalidated` 路径由同一 `RecoveryController.withdraw` 入口单元验证（`test_withdrawal_with_new_conflicting_evidence_invalidates_and_keeps_sources`）。
8. **`RecoveryIntake` 未用于官方入口**：官方路径不存在“丢弃提案后仍用旧观察推进一轮再丢弃”的缝隙；解耦通过“先如实记录反馈、重观测后再推进”实现并有测试。既有 `RecoveryIntake` 语义与测试未改。
9. **瞬时失败重试退避未实现生产调度**：`RecoveryLimits.retry_backoff_seconds` 仍可配置但官方路径不自动重试只读动作；是否重试由模型在下一轮决定（不新增双重退避链）。
10. **完成核验拒绝时的上限耗尽**不是立即终止：仍由驱动既有拒绝上限（2 次）收束为 `incomplete`；停滞路径则立即诚实结束。
11. **计时的执行时间为 0**：FakeClock 在动作执行期未推进，只验证对账关系；真实运行时间由主线程真实时钟产生。
12. **未覆盖** F／G 波内容、原生 WorkflowDAG 适配、CLI `suite` 与真实端并行；未运行官方真实任务。
