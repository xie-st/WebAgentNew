# G 波 G1 线报告：基础上下文保护 + 独立开关

日期：2026-10-31
工作副本：`D:\Dwen\Alpha26fall\WebAgentNew\runtime\parallel\20261031-G1\laneG1-context`
范围：G 首版收窄范围中的「输入预算最终校验与必要区保护」。摘要、历史差分、历史折叠、观察筛选、经验加载均未进入本线（见未完成项）。

## 1. 结论

- 新增独立顶层 `context` 配置段（默认 `enabled: false`），只新增键，未改动任何既有配置键。
- 新增 `webagent/context/` 包：四类必要区（任务原文、当前 AX 观察、账本短视图、未确认副作用记录）的确定性装配、最终发送前一次预算校验、必要区字节级完整性校验。
- 官方入口在开关开启时：把账本短视图与未确认副作用块追加到实际提示词；每次模型调用前对「即将发出的那份 prompt」做一次校验；超限或必要区缺失时明确报错并拒绝发送，不静默截断必要区。
- 开关关闭（默认）时：不创建保护器、不安装任何钩子、不追加任何文本；捕获到的最终发送与「显式写出 `enabled: false`」完全逐字节一致，模型调用次数不变。
- 全量 pytest：365 passed / 0 failed（该副本既有基线 346 = 主树 338 基线 + 已并入本副本的 8 个 R 波测试；本线新增 19 个测试）。

## 2. 修改文件清单

新增（同目录）：
| 文件 | 内容 |
| --- | --- |
| `code/harness/src/webagent/context/__init__.py` | 包导出面 |
| `code/harness/src/webagent/context/errors.py` | `ContextError` / `ContextProtectionError` / `ContextBudgetExceeded`（携带报告） |
| `code/harness/src/webagent/context/sections.py` | 四类必要区模型、账本短视图与未确认副作用块（纯文本、确定性、可截断的仅可选事实明细） |
| `code/harness/src/webagent/context/guard.py` | 单次最终发送的测量与判定（字节级包含、token/字节/消息上限、`over_limit`/`missing` 判定、报告 `to_dict`） |
| `code/harness/src/webagent/context/protector.py` | 面向单个官方任务：`turn_note()` 装配注入块；`check_final_send()` 做发送前校验并落盘 `context_checks.jsonl` |
| `code/harness/tests/phase2/test_context_budget.py` | 配置、视图、校验器、保护器纯程序规则测试（11 项） |
| `code/harness/tests/phase2/test_context_budget_official.py` | 官方入口集成测试（开关开/关、溢出拒绝、行为组独立、P 波回归）（8 项） |

修改（最小、纯新增为主）：
| 文件 | 改动 | 合入理由与冲突面 |
| --- | --- | --- |
| `code/harness/src/webagent/config.py` | 新增 `ContextSettings` 数据类；`AppConfig` 新增 `context` 字段；`parse_config` 允许根键 `context` 并新增独立解析块；`summary()` 新增 `context` 键；开关开启时校验不变量 `context.max_input_tokens <= budget.input_tokens`（与原生压缩预算共用同一有效上限；关闭时不施加该约束，避免只改预算的既有配置回归） | 只新增顶层段与相关代码块，既有键与其默认值均未改动；与 R 波对 config 的改动不重叠（R 波在 recovery/official 段）。新增的构造参数只加一行 `context=context,`，未重排既有参数。 |
| `code/harness/configs/official.yaml` | 新增 `context:` 段（enabled/max_input_tokens/max_request_bytes/ledger_max_chars/effects_max_chars） | 纯追加到文件末尾；既有段未触碰。 |
| `code/harness/src/webagent/official_agent.py` | 5 处纯新增：① `from .context import ContextProtector` 一行；② `context_protector` 只读属性（从 driver 已安装钩子反查，不新增实例字段，故 `reset()` 无需改动）；③ `_start_task` 中 `self._model = model` 之后新增「仅在 `context.enabled` 时创建保护器」块；④ `self._driver = driver` 之后新增「把 `turn_note` / `check_final_send` 装到 driver 上」块；⑤ `_runtime_summary_payload` 末尾新增 `summary["context"]`（记录开关、上限、原生 context_budget） | 未移动/未重排任何既有代码；没有改 `create_driver(...)` 调用参数，避免与 R 波在同一调用点冲突。 |
| `code/harness/src/webagent/effgen_bridge.py` | ① `EffGenLoopDriver.__init__` 末尾新增两个默认 `None` 属性 `context_note_provider` / `final_send_hook`；② 新增 `_with_context_note()` 私有方法；③ `start()`、`accept_observation()`、`accept_split_observation()`、`resume()`、`resume_after_takeover()` 的文本装配各加一行 `text = self._with_context_note(text)`；④ `_before_model_call()` 在 `_model_turn += 1` 之后新增三行：若已装钩子则用即将发送的 prompt 调用它 | 全部为插入行；钩子为 `None` 时与原路径逐字节一致（`_with_context_note` 直接返回原文，`_before_model_call` 跳过）。R 波在同文件 note 处理附近改动时，冲突面仅为 note 追加语句后的一行插入。 |

未修改（按禁令核对）：`state/`、`verification/`、`recovery/`、`control/`、`backends/`、`testing/`、`experience/`、`tools.py`、`models/`、`code/vendor/`、`code/effGen`、`code/official_client`、`docs`、`events.py`、主树任何文件。本线未派发 worker，未联网、未调用真实模型、未启动真实浏览器、未触碰凭据。

## 3. 关键设计点

- 校验位置：`EffGenLoopDriver._before_model_call(prompt)`。该 prompt 就是随后交给 `OfficialModel.generate` 的那份内容；官方入口未设置 system prompt，因此最终消息就是 `[{role: user, content: prompt}]`，与 `OfficialModel.build_messages` 的输出一致。校验发生在 SDK 调用之前，且晚于 EffGen 原生 context_budget/compaction，二者处于同一链路。
- 必要区识别：四类必要区由程序侧对象确定，不按分隔符猜测 ReAct 文本：`session.task`、`render_observation(session.observation)`、由 `StateStore.state` 生成的账本短视图、由 `StateStore.state.pending_effects` 生成的未确认副作用块（`pending`/`unknown`）。校验方式是「这些确切字节必须出现在最终 prompt 中」，缺失即 `ContextProtectionError`。
- 账本进入实际提示词：通过 driver 已有的「程序侧 note」通道，在 `start()` 与每步 `accept_observation()` 装配时追加。`context.enabled=false` 时该通道不启用。
- 超限处理：不裁剪消息。必要区本身放不下或整体超预算时抛 `ContextBudgetExceeded`，运行以 `error` 终态结束并返回明确 stop 文本；报告同时给出四类必要区的 `expected_bytes/matched_bytes/byte_complete`，证明拒绝时必要区仍完整。可选区缩短只发生在账本块装配阶段（仅事实明细逐行取舍），并在报告里以 `trimmed_optional_chars` 明示。
- 确定性：`context` 包无模型调用、无时钟、无随机数；视图按 id 排序、文本压缩按固定字符数截断。同一状态两次渲染、两个相同 prompt 两次校验，结果字节一致（有专门测试）。
- 证据：每次校验追加一行到 `<run_dir>/context_checks.jsonl`（报告全字段，`sort_keys`）；运行摘要 `summary.runtime.context` 记录开关与实际上限及原生 `context_budget`。没有新增事件类型，避免修改跨波共用的 `events.py`。

## 4. 关键命令与结果

前置（每次测试前执行）：

```powershell
$env:PYTHONPATH="D:\Dwen\Alpha26fall\WebAgentNew\runtime\parallel\20261031-G1\laneG1-context\code\harness\src"
& "D:\Dwen\Alpha26fall\WebAgentNew\.venv\Scripts\python.exe" -c "import webagent; print(webagent.__file__)"
```

输出（确认解析到本副本）：

```
D:\Dwen\Alpha26fall\WebAgentNew\runtime\parallel\20261031-G1\laneG1-context\code\harness\src\webagent\__init__.py
```

| 命令 | 结果 |
| --- | --- |
| `python -m pytest tests\phase2\test_context_budget.py -q` | 11 passed |
| `python -m pytest tests\phase2\test_context_budget_official.py -q` | 8 passed |
| `python -m pytest tests\test_config.py tests\test_official_agent.py tests\phase2\test_p03_p04_integration.py -q` | 30 passed |
| `python -m pytest tests -q`（PYTHONPATH 指向本副本） | **365 passed in 154.17s，0 failed** |
| `python -m pytest tests -q --collect-only` | 365 tests collected；其中既有 `test_r_wave_recovery_integration.py` 8 项已在本副本内 |

基线说明：本副本在我改动前实际为 346 项（365 − 19 新增），即主树 338 基线加上已并入本副本的 8 个 R 波测试；全量运行 0 failed。主树 R 波已合入（主树全量 353 passed）后，本副本仍按 338 快照口径交付；`config.py` / `official_agent.py` / `effgen_bridge.py` 的合入冲突由主线程处理，本报告第 2 节已逐文件说明改动边界与最小插入位置。

## 5. 逐项判定

### 验收 1：开关开启时发送前校验，四类必要区完整保留，超限明确报错/只截断可选区

通过。

- `test_enabled_switch_appends_and_checks_every_final_send`：3 次模型发送对应 3 次校验（`protector.checks == len(gateway.calls)`）；每次发送都含任务原文、`CURRENT OBSERVATION`（当前 AX）、`LEDGER VIEW (harness-maintained…`、`PENDING SIDE EFFECTS (unconfirmed…`；`context_checks.jsonl` 每条记录四类 `byte_complete=True` 且 `matched_bytes == expected_bytes`。同时断言账本短视图内容（`revision=1 plan_version=0`、`- g1 [running]`）确实进入 prompt。
- `test_over_budget_final_send_is_refused_before_the_model_is_called`：`max_input_tokens=200` 时首发送即被拒绝；`gateway.calls == []`（没有真的发出去）；stop 文本含「project context budget」「every necessary region is intact」；终态 `error`；记录中 `verdict=over_limit`、`mandatory_intact=True`、四类均字节完整。
- `test_shortened_optional_detail_is_reported_not_hidden`：账本可选事实明细按 `ledger_max_chars` 取舍，`trimmed_optional_chars>0` 进入报告，必要区仍字节完整。
- 无「先丢弃再补」路径：校验器不修改消息；拒绝时不发送。

### 验收 2：保护逻辑确定性，纯程序规则，超长用例断言四类必要区字节完整

通过。

- `test_ledger_view_is_deterministic_and_carries_the_mandatory_core`、`test_protector_turn_note_is_stable_for_the_same_state`、`test_complete_final_send_passes_and_every_region_is_byte_complete`（两次 `evaluate` 的 `to_dict` 完全相同）。
- `test_overlong_input_still_reports_all_four_regions_byte_complete`：构造约 5 万字符超长 AX，断言 `verdict=over_limit`、`mandatory_intact`、四类 `matched_bytes == expected_bytes`，其中 AX 必要区字节数等于 `render_observation(observation)` 的 UTF-8 字节数。`context` 包不引用任何模型 API。

### 验收 3：开关关闭（默认）回归一致，无 context 校验痕迹，不新增模型调用

通过。

- `test_disabled_switch_sends_the_identical_prompt_and_adds_no_call`：分别以「显式 `enabled: false`」与「不写该字段（默认）」跑同一脚本，两次捕获的所有最终消息逐字节相等；`context_protector is None`；driver 的 `context_note_provider` / `final_send_hook` 均为 `None`；prompt 内无 `LEDGER VIEW` / `PENDING SIDE EFFECTS` / `harness-maintained`；无 `context_checks.jsonl`。
- `test_disabled_switch_keeps_the_p_wave_finish_gate_behavior`：默认开关下 P 波 `request_finish` 拒绝后同循环继续、`finish_rejections == 1`、最终 `stop [completion verified…]`，且无 context 记录。
- 结构保证：关闭时 `official_agent` 不创建保护器、不装钩子；`effgen_bridge` 两个钩子默认 `None`，两条插入路径直接返回原文/跳过，因此不存在新增模型调用或提示词变化的路径。

### 验收 4：全量 pytest 0 failed（338 基线，PYTHONPATH 指向本副本）

通过：365 passed / 0 failed；PYTHONPATH 与 `webagent.__file__` 已确认指向本副本 `src`（见第 4 节）。

### 验收 5：不破坏 P 波行为

通过：`tests/phase2/test_p03_p04_integration.py` 全通过（含行为开关组、`request_finish` 同循环纠错、状态工具绑定、`runtime.behaviors` 断言）；全量 365 项 0 failed，R 波既有 8 项亦通过。

## 6. 未完成项与边界

1. 摘要、历史差分、历史折叠、观察筛选、经验加载：按 G 首版收窄范围不在本线实现（对应 G02–G05、G06 另一线）。
2. 真实模型/真实页面证据：按要求未联网、未调用真实模型、未启动浏览器；本报告结论仅覆盖假网关/假 SDK 回调与单元级确定性。
3. 账本视图刷新时机：块在「装配下一次发送文本」时刷新（`start()`、每步 `accept_observation()` 等）。同一 pump 内若先执行内部 `state_update` 后立即进入下一次模型调用（未经过 action 反馈），该次 prompt 携带的是上一次装配的账本块；下一步装配即刷新。这满足「发送内容受保护且账本进入提示词」，但不是逐步实时刷新；如需每轮实时刷新需在 EffGen 原生 prompt 构造处增加可改写钩子，属于跨波公共文件改动，未做。
4. 事件类型：`docs` 拟新增的 `context_pack` 事件需修改 `events.py` 的封闭 `EVENT_TYPES`，该文件不在本线独占范围，故改用运行目录下的 `context_checks.jsonl` 记录同一份报告；后续由公共集成负责人决定是否注册该事件类型。
5. 组合路径测试：`context.enabled` 与 R 波 `recovery` 同时开启的端到端用例未单独编写；`resume()` / `resume_after_takeover()` / 拆分动作路径均已插入同一注意块，但未覆盖到组合测试。
6. CLI 未新增 context 开关入口；配置仅落在 `official.yaml`（其他 profile 配置未加该段，默认值等价于关闭）。
7. 本副本已有 `.pytest_cache` 保持原状；测试生成的 `__pycache__` 已清理。
