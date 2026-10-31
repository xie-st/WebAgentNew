# C批接入修复

日期：2026-10-31。依据：本目录`C_线上结果复核.md`“先补的接入项”三节。基线：`4e5c1c6`（代码干净）。全程离线：假模型、假时钟；未启动浏览器、未连接官方平台、未调用真实模型，未读取任何凭据文件。

范围：`code/harness/src/webagent/`下`backends/official.py`、`models/official.py`、`official_agent.py`、`effgen_bridge.py`、`budget.py`、`config.py`、`tools.py`、`events.py`、`session.py`及测试；另改`code/harness/configs/official.yaml`（配置项落地所需，见“范围说明”）。未改`code/vendor/effgen`、`code/official_client`、`code/effGen`、docs其他文件、根配置。

## 范围说明与配置偏差

1. 任务文字要求在`config.py`的`budget`组内新增`max_messages`、`max_request_bytes`。核对当前源码后，`config.py`已有独立的`limits`组显式声明这两项（默认256与1 MiB），并被全部四份配置样例和`BudgetLimits`使用。再在`budget`组放一份会造成两个事实源和漂移风险，因此沿用`limits`组，并通过`AppConfig.budget_limits()`接入预算对象；默认值依据为平台README：“总输入JSON最多1 MiB，最多256条消息”。若后续统一配置分组，只需在`parse_config`增加迁移，不在本次扩大。
2. `official.yaml`是配置声明文件而非生产源码；新增`official.text_mode`、`budget.deadline_margin_seconds`的注释和值，是“配置项命名与文档要支持real-page验证”的必要落点，因此纳入改动。`code/harness/README.md`未改（不在允许文件内）；本报告与`official.yaml`注释承担文档职责。

## 修复1：官方文本动作的换行与特殊字符

### 改动文件

- `src/webagent/backends/official.py`
- `src/webagent/official_agent.py`
- `src/webagent/tools.py`
- `src/webagent/session.py`
- `src/webagent/effgen_bridge.py`
- `src/webagent/config.py`、`configs/official.yaml`
- `src/webagent/events.py`（新增`expanded`、`subaction`事件类型）
- 测试：`tests/test_official_backend.py`、`tests/test_official_agent.py`、`tests/test_config.py`

### 关键设计

**文本上线策略做成显式配置。** `official.text_mode`，仅official profile有效，默认`decompose`。`AppConfig.summary()`输出当前模式，便于run摘要审计。

- `raw`：文本按原样进入动作，不转义。换行、回车、制表符原样保留，用于在官方真实页面观察服务端对换行、方括号、反斜杠的实际语义。仍拒绝：代理项（保留原检查）、除`\n\r\t`外的C0控制字符、`U+007F`。
- `decompose`（默认）：`type`文本中的换行不在一条`type`动作里传输；执行层拆成`type(首段)` + `press [Enter]` + `type(后续段)`子动作序列。

两种模式都**不再转义反斜杠和方括号**。线上task 731已证明服务端不实现反转义：原`_escape_text`把换行写成字面量`\n`，页面正文与截图都出现字面转义字符。因此保留转义等于静默污染文本。字符确实无法上线时（上述控制字符、代理项）显式抛`OfficialActionFormatError`，不做任何替换。方括号暂按可承载字符原样发送：当前证据只证明换行未反转义，未证明方括号语义，需由real-page验证裁定（见下）；发送后若平台拒绝，是平台可见错误，不是本地静默改写。

`serialize_stop`不再转义；`stop [说明]`是终止控制行、无法拆分，换行/回车归一为空格（文档化行为），方括号与反斜杠原样保留。

**拆分位置与队列。** `official_agent.py`新增`plan_type_text`与step消费路径队列：

- 一次模型提案（`web_type`文本含换行）先整体校验（`Session.expand_proposal`），随后逐平台step交出：首段`type`、中间每处换行一个`press [Enter]`、末段`type`；`submit`只落在最后一个非空段；末尾换行后还要submit时追加一个`press [Enter]`。CRLF/CR按一个换行处理。
- 中间子动作的观察照常`ingest`并写入`observation`/`feedback`事件，另写`subaction`事件（`index/total/remaining/delivered_to_model=false`）；不送入模型。最后一个子动作的观察作为该次工具调用的结果送回同一生成器（`delivered_to_model=true`）。
- 后续子动作绑定**当时当前观察**的元素编号重新生成`WebAction`；若元素已不存在，序列化/提案阶段显式失败，不再盲目沿用旧编号。
- 中途截止或出错：只按真实已执行前缀收尾。`subaction`事件记`aborted=true`与`executed/total/remaining`；`result.json`的`finish.claim`写明“executed N of total”，`summary.unfinished`写明“N of total sub-actions executed, M not sent”，termination为`deadline`或`error`；不写completed。若出错时仍有未应答动作（平台观察无法ingest），`Session.finish_incomplete`先记录该动作未完成再写终态。

**模型可见说明。** `web_type`描述改为：多行文本可直接给出，含换行；执行层会拆分为输入与Enter子动作，无需模型自己拆行。参数`text`说明同步。

**real-page验证入口。** 配置项即入口：将`official.text_mode`设为`raw`即可在同一官方流程中原样发送待验证字符，无需改代码；`official.yaml`注释列出两种模式及用途。建议验证步骤：raw模式跑一个输入含换行、`[`、`]`、`\`、制表符的任务；读该run的`observations/obs-*.json`正文与截图确认实际值；再以decompose跑多行任务核对分段与子动作事件；据结果决定最终默认值并另行提交。本次未执行真实页面验证（离线约束）。

### 测试与结果

命令：全量`python -m pytest code/harness/tests -q`已通过；下列用例位于`test_official_backend.py`、`test_official_agent.py`、`test_config.py`。

覆盖点：

- `test_action_commands_are_exact_and_text_is_not_escaped`：decompose下含换行文本序列化显式报错；raw下同一文本原样上线，命令内出现真实换行且不含`\n`/`\[`/`\]`/`\\`字面转义；单行方括号/反斜杠按原样进入命令。
- `test_stop_keeps_one_line_without_escaping_the_bracket`：stop换行归一空格、方括号不转义、单个终止括号。
- `test_unrepresentable_action_content_is_an_explicit_error`：代理项与`U+0007`在两种模式下均显式拒绝。
- `test_reset_step_close_with_one_generator_and_one_action_per_step`：5平台step依次为`click`→`type [1] [价格（含税）] [0]`→`press [Enter]`→`type [1] [第二行 [备注]] [0]`→`stop`；仅最后一个观察进入第3次模型输入（`obs-4`在prompt，`obs-2/3`不在）；`expanded`事件父动作未被dispatch；`subaction`的`delivered_to_model`为`[false,false,true]`。
- `test_raw_text_mode_sends_the_line_break_in_one_type_command`：raw下一条命令携带真实换行。
- `test_split_stops_at_the_deadline_recording_only_the_executed_prefix`：过期后停止，stop与结果均含“1 of 3”；`executed=1,total=3,remaining=2`；termination=deadline。
- `test_split_stops_on_an_error_and_keeps_the_executed_prefix`：目标元素在中间页消失；已执行首段与Enter后停止，`2 of 3`、`remaining=1`、termination=error，未伪造整条成功。
- `test_official_text_mode_and_deadline_margin_are_explicit_configuration`：默认decompose、raw可切换、非法值与非official profile的`official`组均报ConfigError。

### 验收证据

- 证据目录：每个测试的`tmp_path/runtime/runs/<run_id>/`下`events.jsonl`、`result.json`、`run_status.json`、`observations/`；断言直接读取这些文件。
- 关键断言：raw命令包含真实换行且不含字面转义；decompose序列化拒绝换行；子动作事件与模型输入分离；aborted前缀计数与termination。

### 未完成

- 方括号、反斜杠、制表符、回车在官方页面的最终语义未验证（离线约束），默认值待real-page结果确认；若平台证明方括号不可承载，应把该字符加入显式拒绝清单，而不是恢复转义。
- 若中间`press [Enter]`导致页面跳转且目标元素编号变化，当前按显式失败收尾；是否需要在第二阶段做“重新定位元素”的恢复，未在本次范围内。

## 修复2：A09项目输入限额接入实际请求

### 改动文件

- `src/webagent/budget.py`
- `src/webagent/models/official.py`
- `src/webagent/official_agent.py`
- `src/webagent/effgen_bridge.py`
- `src/webagent/config.py`（沿用既有`limits`组并接入预算对象）
- 测试：`tests/test_official_model.py`、`tests/test_official_agent.py`、`tests/test_budget.py`、`tests/test_effgen_interactive.py`

### 关键设计

**项目预算映射到EffGen原生入口。** `build_effgen_config`设置`EffGenAgentConfig.context_budget=int(app.budget.input_tokens)`，取项目值原样传入，不做钳制。低于vendor最小预算（`MIN_CONTEXT_BUDGET_TOKENS=512`）时抛`BridgeError`说明，不静默抬高。

与原生`auto`的关系：`auto`在同一32768窗口上按“预留8192输出、再乘0.85”得到20889；项目值24000现在是显式整数预算，`resolve_context_budget`返回`source="config"`、`budget_tokens=24000`，`auto`推导不再生效。即EffGen按项目值压缩自身会话，而不再用推导值。项目值比该窗口的auto略高，这是配置选择而非回归；报告已记录。

**最终SDK出口校验。** `models/official.py`在`build_messages`完成、调用`third_party_agent.model.generate`之前：

- 统计消息条数；按`json.dumps(messages, ensure_ascii=False, separators=(",",":"))`的UTF-8字节数统计“序列化后请求字节数”。
- 与`BudgetLimits`的`max_messages`、`max_request_bytes`比较；超限抛`RequestLimitExceeded`并带`limit_name/limit/actual`，错误消息写明是哪一项。
- 不裁剪任何消息。拒绝发生在SDK调用前，因此平台不会收到该请求；本地只留一条`status="rejected"`的调用记录。
- 每次成功请求在`GenerationResult.metadata`记录`request_message_count`、`request_bytes`、`request_mandatory_bytes`，供证据复核。

**必须保留项判定。** `budget.build_prompt_messages`给出与`ContextPacker`同形的“任务原文+当前观察+待确认”视图；`OfficialAgent`把它作为`mandatory_provider`交给模型。出口先测必须保留项：若其本身超限（条数或字节），抛`MandatoryContentOverflow`（`RequestLimitExceeded`子类，`scope="mandatory"`），明确表示“即使去掉历史也放不下”，绝不裁掉它们；若必须保留项能放下、只是整包超限，则是普通`RequestLimitExceeded`。当前`Session`没有“待确认信息”字段，故实际只含任务与当前观察；packer/限额函数本身支持`pending`，将来接入即可。

**拒绝的可见性。** 出口拒绝通过`limit_sink`写入`error`事件（stage `official_request_limit`，带异常类型与含限额名的消息），并把具体限额写进stop说明；同时给异常附上EffGen文档化的`error_context`（category `invalid_request`），让确定性拒绝快速失败而不是被vendor按unknown重试3次。

**默认值依据。** `limits.max_messages=256`、`max_request_bytes=1048576`直接来自官方客户端README的请求约束（最多256条消息、总输入JSON最多1 MiB）。官方适配器实际只发system+user两条扁平消息，故字节限制是主约束，消息数限制是防御性校验。

### 测试与结果

- `test_final_sdk_exit_enforces_project_message_and_byte_limits`：正常请求通过并记录字节/条数；2条消息对`max_messages=1`抛`max_messages`；500字节prompt对64字节限制抛`max_request_bytes`；超限时fake gateway零调用。
- `test_mandatory_task_and_observation_overflow_is_reported_separately`：必须保留项自身超限→`MandatoryContentOverflow`（`scope=mandatory`）且零SDK调用；必须保留项能放下、整包超限→普通`RequestLimitExceeded`。
- `test_official_agent_refuses_an_oversized_request_before_the_sdk`（端到端）：小字节限额下首个step即拒绝；stop含`max_request_bytes`；SDK零调用；`error`事件含限额名；termination=error；计数`platform_calls=0, local_attempts=1, failures=1`。
- `test_project_input_tokens_become_the_effgen_context_budget`：`context_budget=24000`；`resolve_context_budget`返回`source=config, budget_tokens=24000`；同窗口`auto≠24000`；自定义12000原样传入。
- `test_mapped_context_budget_is_active_in_the_run_metadata`：以12000项目预算跑完一个假模型交互任务，最终`AgentResponse.metadata["context_budget"]`为`budget_tokens=12000, source=config`，证明映射在真实循环中生效而不只是配置对象值。
- `test_request_gate_names_the_limit_and_the_mandatory_case`（纯函数）：必须保留项与整包两类报错、字段与measurement。

### 验收证据

- 端到端拒绝run的`result.json`：`termination=error`，`summary.model_calls`三分口径；`events.jsonl`含`model_call`（failure_type=request_limit）与`error`（`MandatoryContentOverflow`，消息含`max_request_bytes`）；fake gateway的`calls==[]`证明未到SDK。
- 映射证据：EffGen配置对象与原生解析结果在测试中直接断言，无需真实模型。

### 未完成

- 尚未在真实官方请求上核对实际HTTP请求体字节与我们序列化口径的差异；当前口径是本地JSON序列化，保守性需线上样本确认。
- `Session`没有“待确认信息”载体；限额函数支持但生产流程尚无该字段。
- 未做压缩循环（按复核结论不扩大范围）；EffGen自身按`context_budget`压缩其会话，与本出口校验是两层。

## 修复3：deadline收尾与调用对账

### 改动文件

- `src/webagent/budget.py`、`config.py`、`session.py`、`events.py`、`effgen_bridge.py`、`official_agent.py`
- `configs/official.yaml`
- 测试：`tests/test_official_agent.py`、`tests/test_budget.py`、`tests/test_effgen_interactive.py`

### 关键设计

**安全边际。** `budget.deadline_margin_seconds`，默认30.0，写入`BudgetLimits`并由`config.budget_limits()`传入。`Budget.require_model_call_margin`在剩余时间≤边际时抛`DeadlineMarginReached`（`DeadlineExceeded`子类）。门只设在新模型调用之前（`EffGenLoopDriver._before_model_call`），不影响`Budget.begin_model_call`的调用记账；因此“不再发起新调用”和“已发生调用的计数”互不干扰。剩余时间≤0仍先按原deadline路径记录。

默认值依据：本批6个官方run共74次本地尝试，官方调用的中位延迟为12.6–19.2秒、逐调用最大值20.7–21.9秒（`e47b…`19次中位12.6s、最大21.9s；`c79d…`13次中位18.5s、最大21.1s等）。30秒覆盖观测到的最大延迟并留出请求收尾余量，同时不超过300秒窗口的10%。该值可配置，real-page批次可据新分布调整。

**可控收尾终态。** 边际触发后由驱动走既有`_finish_deadline`：`session.finish(Finish(reason=deadline))`写`result.json`，`OfficialAgent`在终态调用`_finalize_run()`关闭session，保证`run_end`事件落地，即使SDK从不调用`close`。`result.json`新增`summary`：`steps_completed`、`observations`、`actions_proposed`、`actions_dispatched`、`last_action`（动作id/op/target/keys/文本长度与预览/submit）、`unfinished`（终止说明）、`model_calls`三分口径。`termination`统一由session记录，`run_end`与`result.json`一致；不伪造completed。

**调用对账三口径。** `result.json`与`run_status.json`的`model_calls`分别为：

- `platform_calls`：本进程见到平台受理的调用数（拿到不同平台`request_id`的数量）；
- `local_attempts`：适配器`generate`出口的尝试数，含失败与出口拒绝；
- `failures`：带失败类型的尝试数。

三者不合并。平台批次结果仍是权威口径；本地值用于对账，差异应逐项解释而不是推定为漏记。

**轻量run状态文件。** `Session`在`runtime/runs/<run_id>/run_status.json`维护状态，`EventLogger`每写一条事件后由`RunStatusWriter`原子重写（临时文件+fsync+replace）。内容：`state`、`termination`、`steps.completed/proposed/dispatched`、`last_action`、`last_event`（type/time/action/observation/model_calls_total）、`model_calls`三口径、`model_calls_total`、`updated_at`。SDK硬杀时该文件保留最后一步的事实与计数，供事后与平台结果对账；状态写失败不影响run与事件日志。

**终态消息不再被固定文案覆盖。** `OfficialAgent._terminal_response`使用驱动step的真实消息（deadline/分段前缀/限额详情），因此stop说明、`result.json`、`run_end`保留实际原因。另将EffGen的`generation_failed`返回映射为`error`终态（此前会被误记为`incomplete`）。

### 测试与结果

- `test_deadline_margin_closes_out_without_a_new_model_call`：剩15秒（<30）时不再调用模型（fake gateway仅1次）；stop含margin；`result.json` termination=deadline、`steps_completed=1`、`unfinished`含deadline margin；`run_end` termination=deadline；`run_status.json` termination=deadline、计数`1/1/0`。
- `test_hard_kill_leaves_a_status_file_for_reconciliation`：第二次模型调用内抛`KeyboardInterrupt`模拟SDK硬杀，未走收尾；`result.json`不存在；`run_status.json`为`state=running`、`termination=null`、`steps={completed:1,proposed:1,dispatched:1}`、`last_action.op=click`、`last_event`为observation/feedback、计数`1/1/0`；被中断的调用没有记录，故未计入attempts（不臆造）。
- `test_deadline_margin_refuses_only_new_model_calls`（Budget单元）：边际外允许、边际内拒绝且带remaining/margin；会计路径不受影响；margin=0关闭。
- 前文分段截止/出错用例同时验证“截止附近不继续无效调用”和前缀终态。

### 验收证据

- 边际用例的`events.jsonl`含`deadline_margin`（stage/remaining/margin/action=controlled_closeout）、`agent_claim`、`run_end`；三个终态文件相互一致。
- 硬杀用例直接读取`run_status.json`；无`result.json`，证明不伪造终态。

### 未完成

- EffGen在单次生成内部的重试（vendor `max_retries=3`）仍可能在一次已开始的turn内越过边际；门控的是“新turn的新调用”。出口拒绝已用vendor错误上下文标记为不可重试，普通模型失败仍按vendor策略。若要求turn内也严格停止，需要vendor或适配器层改动，本次未扩大。
- 硬杀瞬间正在进行的调用没有`model_call`记录，因此不会进入三口径；这是“无记录不计数”的诚实限制，报告不把它推定为漏记。
- 默认边际30秒基于本批6个任务的观测；样本有限，real-page批次应复核延迟分布后再定最终值。

## 回归与命令

执行环境：`D:\Dwen\Alpha26fall\WebAgentNew\.venv\Scripts\python.exe`。

```powershell
.\.venv\Scripts\python.exe -m pytest code/harness/tests -q
.\.venv\Scripts\python.exe -m compileall -q code\harness\src code\harness\tests
```

结果：`162 passed in 40.13s`；`compileall OK`。第一阶段原有测试全部通过（含为语义变化更新的旧断言）。

改动文件清单：

```text
code/harness/configs/official.yaml
code/harness/src/webagent/backends/official.py
code/harness/src/webagent/budget.py
code/harness/src/webagent/config.py
code/harness/src/webagent/effgen_bridge.py
code/harness/src/webagent/events.py
code/harness/src/webagent/models/official.py
code/harness/src/webagent/official_agent.py
code/harness/src/webagent/session.py
code/harness/src/webagent/tools.py
code/harness/tests/test_budget.py
code/harness/tests/test_config.py
code/harness/tests/test_effgen_interactive.py
code/harness/tests/test_official_agent.py
code/harness/tests/test_official_backend.py
code/harness/tests/test_official_model.py
```

未改vendor：`context_budget`是既有字段，`MIN_CONTEXT_BUDGET_TOKENS`、`error_context`均按vendor既有接口使用，未发现必须修改vendor的缺口。

## 未完成项汇总

1. 文本字符的官方真实语义未验证，`official.text_mode`默认值待real-page结果确认；验证入口已就绪。
2. 官方请求体字节口径为本地JSON序列化，未与真实SDK请求逐字节核对。
3. `Session`无“待确认信息”字段；限额函数已支持。
4. turn内vendor重试不受边际门控；硬杀瞬间的在途调用无本地记录（不计数）。
5. 分段输入中页面跳转导致目标元素变化时，当前显式失败收尾，恢复策略留给第二阶段。

## 附：本次执行环境说明

本会话开始时命令行无法获得工作区写权限（Windows文件权限），按诊断工具检查后被拒绝修复；用户将本会话切到完全权限后命令恢复。该诊断在仓库外`D:\Dwen\Alpha26fall\acl-recovery\`留下一个只读报告JSONL，未修改仓库文件权限或内容；本次所有代码与报告改动均在上述清单内。

## 探测Agent

日期：2026-10-31。范围：仅新增`code/harness/src/webagent/official_probe.py`与`code/harness/tests/test_official_probe.py`；未修改任何现有生产文件、配置或测试。全程离线：假SDK请求、假浏览器读取器、假时钟；未连接官方平台、未读取连接文件、未启动真实浏览器、未调用真实模型。

### 用途与入口

`OfficialProbeAgent`继承`OfficialAgent`。SDK仍走`third_party_agent connect`；探测模式把`--agent`替换为`webagent.official_probe:OfficialProbeAgent`，并设置`WEBAGENT_PROBE=1`。子类构造参数`probe=True/False`优先于环境变量；未启用时所有SDK钩子委托父类，普通任务路径与`OfficialAgent`完全一致。

主线程将执行的命令：

```powershell
cd D:\Dwen\Alpha26fall\WebAgentNew\code\official_client
$env:WEBAGENT_PROBE='1'
..\..\.venv\Scripts\python.exe -m third_party_agent connect `
  --connection-file "<连接文件绝对路径>" `
  --agent webagent.official_probe:OfficialProbeAgent
```

`WEBAGENT_PROBE=1`须在启动connect的同一进程环境中（SDK以spawn子进程继承环境）。未设置该变量时，同一入口会按普通`OfficialAgent`执行任务本身，因此主线程应先确认变量已生效。探测结束后看平台stop说明与`runtime/runs/<local_run_id>/result.json`。

### 探测串明文

| 串 | 字符构成 | Python字面量 |
| --- | --- | --- |
| A | `P[1]\a`：`P`、左方括号、`1`、右方括号、反斜杠、`a`；单行 | `"P[1]\\a"` |
| B | `L2` + 一个制表符（U+0009真实字符）+ `L3` | `"L2\tL3"` |

两串均经`validate_action_text`校验，不含换行、不可承载的控制字符或代理项。默认脚本不含换行字符：换行/Enter语义由独立的`press [Enter]`覆盖；另提供构造参数`probe_text_a/probe_text_b`，测试用含真实换行的A验证raw模式不转义。

### 上线行为

reset后第一个step取当前观察，从AX元素中选编号最小的可输入元素（`textbox`、`searchbox`、`contenteditable`、`textarea`等，按数字比较），随后固定脚本：

1. `type [N] [P[1]\a] [0]`
2. `press [Enter]`
3. `type [M] [L2<TAB>L3] [0]`（命令内为真实制表符）
4. `stop [character probe completed; task not executed]`

- N/M每次从当时当前观察重新选择；press后页面编号变化不会沿用旧target。任一步找不到可输入元素时直接`stop [probe: no typeable element]`，本地`finish.message`附加`task not executed`。
- 探测模式强制`text_mode=raw`，由子类自行构造`OfficialBackend`；`configs/official.yaml`即使为`decompose`也忽略，原值保留在`configured_text_mode`供审计。未改`OfficialBackend`。
- 每个动作都经正常路径：`Session.propose`→`backend.serialize_action`→`Session.consume`；每个观察经`backend.from_request`→`Session.ingest_observation`。`proposed/dispatched/observation/feedback/artifact_saved/agent_claim/run_end`照常落盘，每个观察一条`observation`事件并保存`observations/obs-*.json`。
- 终止前下载当前观察截图：探测模式把`screenshot`并入backend的`download_artifacts`集合（每个带截图元数据的观察自动下载），并显式调用`download_artifact("screenshot")`。可用则写入`finish.evidence_refs`并在claim中写明引用；不可用则把原因写进claim；下载异常写`error`事件并计入stop汇总。
- 动作报错不重试：平台返回`feedback.status=error`时记录`op: error_type: message`，脚本继续下一动作。本地序列化/提案异常记录后跳过该动作继续；两类错误都汇总到stop说明。汇总中的方括号替换为圆括号以保持stop单行单括号结构，不修改type文本本身。
- 结束时走`Session.finish(Finish(reason=INCOMPLETE))`写`result.json`：`termination="incomplete"`；`finish.claim`含`character probe`与`task not executed`；`summary.unfinished`为完整stop说明（正常完成、无可输入元素或deadline版本）；证据引用指向截图。不写`completed`，不伪造任务完成。
- 不创建`OfficialModel`和EffGen驱动，`model_calls`三口径均为0；不执行任务本身，只执行上述探测动作。本地`termination=incomplete`用于避免把探测误记为任务完成；平台仍正常收到`stop`终止行。若后续要求本地completed语义，应另行评审，本次未改。

### 测试与结果

新增`tests/test_official_probe.py`5个用例：

- `test_probe_script_is_type_press_type_stop_and_uses_raw_wire_text`：decompose配置被忽略，agent与backend均为raw；命令序列为`type [1] [P[1]\a] [0]`→`press [Enter]`→`type [1] [L2<TAB>L3] [0]`→stop；命令含真实`[`、`]`、`\`、TAB，不含`\[`、`\]`、`\n`、`\t`、`\\`字面转义；4个观察各入`observation`事件；dispatched依次为type/press/type；截图文件与`artifact_saved`存在；result为termination=incomplete、claim含probe与task not executed、unfinished为完整stop说明、evidence_refs指向截图、model_calls全0。
- `test_missing_typeable_element_stops_honestly_without_actions`：页面只有button/link时首个step返回`stop [probe: no typeable element]`，零dispatch，result写明probe与task not executed。
- `test_platform_action_error_is_summarised_and_does_not_stop_the_script`：平台拒绝括号后feedback=error，脚本仍继续press/type；stop与unfinished含`action errors: type: invalid_action: brackets are not supported`。
- `test_probe_is_opt_in_via_environment_or_constructor`：未设环境变量且`probe=None`时行为如普通agent（decompose）；`WEBAGENT_PROBE=1`时raw探测；显式`probe=False`压过环境变量。
- `test_probe_text_with_a_real_line_break_is_not_escaped`：构造含真实换行的探测串时，type命令含真实换行且无`\n`字面量，覆盖换行语义；默认A/B为单行、不含换行。

命令与结果：

```text
.\.venv\Scripts\python.exe -m pytest code/harness/tests/test_official_probe.py -q
5 passed in 1.48s

.\.venv\Scripts\python.exe -m pytest code/harness/tests -q
167 passed in 40.54s
```

全量较修复批162条增加5条，原有用例无回归。`compileall`对新文件通过。

### 文件清单

```text
code/harness/src/webagent/official_probe.py
code/harness/tests/test_official_probe.py
```

未改`official_agent.py`、`backends/official.py`、配置或其他现有文件。

### 未完成与说明

- 真实页面探测未执行（离线约束）；方括号、反斜杠、制表符、Enter的真实落页语义仍待主线程按上述命令跑一次，读`observations/obs-*.json`正文、`screenshots/`与平台stop说明确认。
- 探测不处理任务语义；若选中的可输入元素属于搜索框，输入与Enter可能触发搜索或页面跳转，这是预期的页面交互，不是任务执行。
- 平台action error若无message，只记录error_type；stop汇总按已记录内容生成。

### 真实落页验证（主线程，2026-10-31 04:23–04:30）

平台批次 `f229a221`（forum-practice-webarena-8-v3-300s-managed，6 项）由探测 Agent 接收：4 项完整执行探测脚本（2 项被用户取消，无本地 run）。探测不执行任务，`model_calls=0`，平台计费 $0；0/6 得分是预期结果，不代表修复后 Agent 的任务能力。

本地 run（4 个，例 `runtime/runs/a9f86462…/`）与截图证据结论：

| 探测内容 | 落页结果 | 证据 |
| --- | --- | --- |
| `P[1]\a`（`[`、`]`、`\` 原样） | 搜索框为 `P[1]\a`，服务端渲染 "50 results for P[1]\a:"，URL 编码 `P%5B1%5D%5Ca` 为单反斜杠 | `obs-2.json` AX 正文、`screenshots/9d27023f….png` |
| `press [Enter]` | 真实提交搜索表单（搜索结果页加载），Enter 键语义成立 | 同上 |
| `L2<TAB>L3`（真实制表符） | 制表符作为真实字符插入文本（顶栏搜索框显示 `P[1]\aL2␉L3`），不移动焦点 | `obs-3.json`、同截图顶栏 |

结论：

1. 方括号与反斜杠在真实页面原样落页且经服务端往返无变形；修复批取消转义的决定被真实页面证实（此前 `\n` 字面量问题是转义路径污染，非服务端限制）。
2. `press [Enter]` 的按键语义真实生效，支撑 decompose 模式（多行文本拆 type→Enter→type）的正确性。
3. **`text_mode` 最终默认值定为 `decompose`**（维持修复批选择）：换行走拆分+按键；raw 模式对单行特殊字符亦安全，保留作探测与调试用途。raw 下制表符会作为字面字符插入输入框——属真实页面行为，记录备查。

三项接入修复至此全部关闭（工程实施、离线回归、真实落页三重验收）。
