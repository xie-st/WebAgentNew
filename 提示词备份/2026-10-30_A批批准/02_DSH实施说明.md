# 第一阶段 DSH 实施说明

状态：待审批，当前不派发。工作根目录：`D:\Dwen\Alpha26fall\WebAgentNew`。

本文中的包名、类型和命令是要实现的接口，不表示仓库已经提供。实施时同时遵守项目与上级 AGENTS.md、本阶段 [详细方案](01_详细方案.md) 和 [验收要求](03_验收要求.md)。

## 执行分工

主线程先准备固定源码、SDK 运行副本和安装环境；DSH 不接触平台密码、连接令牌与模型密钥，不操作评测台，不发布仓库。主线程不将凭据、完整 env 或原始登录状态传入派发提示词。

按上级《子智能体.md》使用 `ds-flash`、`fork_turns="none"`，不设置 effort，沿用 high。不让 worker 再次派发。每次提示词列出绝对路径、任务编号、允许修改文件与对应验收项。worker 不是独占工作区，不得撤销其他人的改动；发现契约变化先报告。

| 任务 | 所属批次 | 负责范围 | 先决条件 |
| --- | --- | --- | --- |
| A1 | A | `code/harness` 项目骨架、协议类型、预算、日志、配置、离线夹具 | 主线程完成版本冻结 |
| A2 | A | `code/vendor/effgen` 循环扩展、harness 中循环驱动与对应测试 | A1 协议冻结 |
| B1 | B | Playwright／browser-use 接入、真实模型适配、CLI 与网页验证夹具 | A 批通过，用户同意继续 |
| C1 | C | 官方 Agent 适配与模型网关、集成测试、运行说明 | B 批通过，用户同意继续 |

默认顺序完成以上任务。确需并行时，先冻结协议，再将真实端与官方端交给两个 worker，各自只改负责模块；集成由主线程完成。DSH 结束须返回修改文件、验证命令和结果、未完成项及证据路径。

A2 的上游修改限于 `effgen/core/agent_loop.py`、`agent.py`、`agent_react.py`、`agent_tool_execution.py`、`agent_config.py`、新增的 `agent_interactive.py` 及对应测试。模型适配代码写在 harness，不修改上游供应商适配器。确需超出该范围时先说明原因，由主线程确认后更新任务范围。

## 1. 主线程准备

1. 确认 `code/effGen` 工作树与提交仍为 `37de05a6b4ed01c90ee38b9c254a162927ca2ac2`；变化时先比较，不覆盖用户修改。
2. 将固定提交导出到 `code/vendor/effgen`，保留 Apache-2.0 许可证与安装文件。原检出保持不动，不复制内层 `.git`。快照清单记录上游 URL、commit、版本、复制范围和补丁说明。
3. 从 `docs/05_赛题与要求/原件/2026-10-30_新版连接器/webarena-outbound-client.zip` 解压到 `code/official_client`。校验 SHA-256 为 `ec1a6697656d5e1b6ba328ea4c69601a6c6e82ab9e63f428f23db71afa16feb4`，拒绝路径穿越，不覆盖旧包或连接文件。
4. 确认 Python 3.12，使用 uv 创建项目专用环境，安装固定快照与 harness。记录实际成功的安装命令、依赖版本与导入检查。优先复用上游 CPU 依赖约束；不下载模型权重或配置 GPU。不能用 `--no-deps` 加导入屏蔽伪装安装成功。
5. 根 Git 忽略 `code/effGen/`、虚拟环境、缓存、`runtime/`、凭据、连接文件和 env。保留必要的非敏感配置示例。新建 GitHub 私有仓库时使用已登录账号，不改已有仓库远端；名称优先采用 WebAgentNew，冲突时回报。

安装或真实服务配置失败时，先保留离线可验证产物；报告具体阻塞，不把阶段写成通过。GitHub 发布放在 C 批末，提交前检查文件与历史均不含秘密。

## 2. 模块结构

在 `code/harness/src/webagent` 实现以下模块，可按职责拆成子文件：

| 模块 | 职责 |
| --- | --- |
| `protocol.py` | 不可变观察、动作、反馈、能力与终止类型；不依赖具体浏览器 |
| `session.py` | 单任务生命周期、待返回动作、观察版本、暂停和关闭 |
| `effgen_bridge.py` | 装配 EffGen，推进可暂停循环；内部工具直接返回 |
| `tools.py` | 唯一网页工具定义与参数规则；两端不各造一份 schema |
| `models/official.py` | 官方文本模型适配 |
| `models/real.py` | 真实端统一模型适配与调用记录 |
| `backends/official.py` | 平台 request／action 转换、AX 编号校验和原件读取 |
| `backends/playwright.py` | 有界面浏览器、定位表、观察与截图 |
| `backends/browser_use.py` | browser-use 接入试验，仅复用浏览器层 |
| `official_agent.py` | SDK 所需 `reset/step/close` 入口 |
| `budget.py`、`events.py`、`evidence.py` | 截止时间、日志、原始观察与证据 |
| `cli.py`、`__main__.py` | 诊断、真实端执行、暂停／继续及证据导出 |

配置在 `configs/official.yaml`、`real.yaml`、`example.env`，只提交非敏感内容。依赖与开发命令在 `pyproject.toml` 声明。测试放 `code/harness/tests`，本地验证网页放 `tests/fixtures/site`。

## 3. 统一协议

使用有类型的 dataclass／Enum 或等价校验结构，JSON 序列化后不得丢失下面的含义：

### Observation

- `run_id: str`、`page_id: str`、`observation_id: str`：任务、页面和本次观察标识。
- `url: str`、`ax_text: str`：当前 URL 与完整当前 AX 文本。
- `elements: dict[str, ElementRef]`：本次可操作元素；ElementRef 绑定 page_id、observation_id、元素编号及本地后端定位信息。定位信息不发给官方平台。
- `previous_action_id: str | None`、`feedback: Feedback | None`：与前一动作的关联。
- `artifacts`：截图、HTML、DOM 的可选引用；记录可用、缺失或失败原因。
- `capabilities: Capabilities`：当前允许的观察与动作能力。

`observation_id` 可由适配器按任务内递增序号生成，不能假定官方已提供同名字段。真实端 page_id 绑定实际 Page；官方端使用该任务唯一页面。每次观察创建新定位表，旧表失效。

### WebAction

包含 `action_id`、`run_id`、`page_id`、`observation_id`、`op`、参数。`op` 仅为 `click/type/press/scroll/goto/go_back/go_forward`；DOM nodeId 与 backendNodeId 禁止作为 AX 元素编号。

`click` 参数为当前 target；`type` 为 target、text、submit（布尔）；`press` 为 keys；`scroll` 为 up/down；`goto` 为 url；前进后退无额外参数。输入按官方“可能追加”语义实现。覆盖字段须由 Agent 显式点击、全选、输入；本地后端不能暗中改成总是清空。

### Feedback 与 Finish

反馈至少区分 `observed/error/unknown`，含关联动作、新观察引用、错误类型和可读说明。`observed` 只表示收到页面反馈，不代表业务成功。没有明确执行证据时记 unknown，不编造点击成功或保存成功。

Finish 包含 Agent 的结束声明、说明和终止原因。正常结束、未完成、预算耗尽、错误、取消分别记录。官方评分不在 Finish 中臆造，由主线程下载后按 run 关联。

## 4. EffGen 循环扩展

目标接口：在 vendored EffGen 中提供生成器入口，能产生外部工具请求，并通过 `send(tool_result)` 返回执行结果；返回值为原框架 AgentResponse。名称可采用 `run_interactive()`，但实施后统一固定并写入使用文档。该入口第一阶段只接受 SINGLE、文本 ReAct、非流式配置。

实现要点：

1. 复用 `effgen/core/agent_loop.py` 的 `drive/step` 和现有 thread、解析、结果记录；不重写规划循环。网页工具调用产出请求，内部工具沿普通执行路径运行。
2. 在实际 dispatch 处增加可恢复的外部执行分支。现有单工具和批工具路径都要防止静默执行网页操作；本配置拒绝一次模型输出多个网页动作，并回给模型明确格式错误，不排队执行带旧编号的动作。
3. 保留调用前参数验证与工具白名单，以及结果返回后的 ActionStep／ObservationStep、计数和错误处理。可抽取公共 helper，不能复制一份以后会漂移的完整循环。原普通 `Agent.run()` 和非网页工具仍可运行。
4. 外部请求是“等待执行”，不是成功返回的工具结果。只有环境返回后才能记录对应 ObservationStep；close、取消或超时时不能虚构返回值。
5. 新入口正确建立、保存和清理 EffGen 的运行状态。生成器跨 SDK 回调存续，但每次推进的 contextvars 要在该回调内进入和退出；不得让旧 model.session 或 reset token 跨回调泄漏。不得启动常驻线程、子 Agent、异步模型调用或新的事件循环代替暂停。
6. 工具执行计时与环境等待分别记录；不要让跨 yield 的后台 span、账单上下文或锁阻塞下次 step。需要调整原 bookkeeping 时记录改动和回归依据。
7. 网页工具的去重和缓存必须绑定页面与观察版本。同一 `scroll` 或同一数字编号在新观察中可能是合法新动作，不能返回旧工具结果而不执行；同一待执行请求也不能被重复消费。保留普通内部工具原行为，网页无进展判断须依据返回观察。核对默认重复工具保护和 `max_turns_without_progress`，不能让它们把正常的多页浏览截断。

网页工具使用 `web_click/web_type/web_press/web_scroll/web_goto/web_back/web_forward`；`web_observe` 返回最近一次真实观察，不在后台重新采集。内部 `task_note` 保存任务内事实，`evidence_save` 读取／保存本次已有原件；二者不产生网页动作。HTML、截图的官方 SDK 下载仍须在有效回调中同步完成。

### 驱动顺序

```text
reset(request)       创建空任务容器；清理旧任务；不假定已有 task
首次 step           读取完整 task、观察、deadline；创建 Agent 和生成器
advance             内部工具继续运行；遇网页请求校验并返回一个动作
后续 step           建立新观察；关联上一动作；send(工具反馈) 后继续
生成器结束          记录结束原因；转换为 stop 或真实端结束结果
close               关闭生成器、释放文件与自有资源；重复调用安全
```

每次推进前及模型调用前检查剩余预算。模型格式错误最多追加2次纠正调用，仍共用原截止时间；不得在错误时自动发 scroll、click 或提交动作。没有步骤上限不等于无限内部循环，内部无进展也必须受 deadline 和取消约束。

配置显式设置 SINGLE、`tool_calling_mode="react"`、禁用 capability_probe、sub_agents、fallback、speculative_execution 与额外 models。任务内记忆由 harness 提供，本阶段禁用框架跨任务持久记忆。`max_iterations` 需支持在该入口以 deadline 为终止依据，不能保留默认10步；不以循环重建 Agent 绕过上限。新增配置必须有测试并说明默认行为不变。

## 5. 模型适配

实现 BaseModel 要求的 load、generate、generate_stream、count_tokens、get_context_length、unload 等契约，返回框架实际 GenerationResult。明确文本工具模式，不探测原生工具、视觉或其它模型。

官方端：所有模型请求最终只调用当前回调中的 `model.generate(messages)`。如框架使用平铺 ReAct 提示，可将完整提示作为字符串 user 消息；system 提示保留其角色；不得把 `tool` role、图像、temperature、模型名、max_tokens 或 tools 参数传给 SDK。保留文字中的工具调用与结果语义。token 用量未知填 null，记录 request_id 和实际调用数，估算另列。

真实端：主线程读取指定 env，确认字段与统一模型，再提供环境变量名映射。worker 只用假值和假服务测试。采用供应商实际兼容的文本接口；计入所有重试。没有供应商 usage 时仍填 null。无模型服务时可以进行假模型集成，但不能宣称真实模型验收通过。

输入预算默认约24K，配置规划上下文32K；估算明确标记 `token_count_source`。序列化请求还须满足官方1 MiB、256消息限制。超限只压缩旧过程；任务原文、当前 AX、待确认信息不能静默裁剪。若仍无法发送，给出明确超限结果。

## 6. 官方后端

实现 `OfficialAgent.reset/step/close`，入口为 `webagent.official_agent:OfficialAgent`。复用冻结的 `third_party_agent`，不修改其 model/session/outbound。连接文件通过绝对路径传参，主线程提供，程序不复制或输出令牌。

动作按 README 精确序列化：`click [id]`、`type [id] [text] [0或1]`、`press [keys]`、`scroll [down|up]`、`goto [url]`、`go_back/go_forward`、`stop [说明]`。用有括号、换行和中文的输入测试转义；对协议无法表示的内容显式报错，不能破坏 action 结构。

动作 ID、页面和观察版本均校验；goto 仅限本次任务观察到的同源网址，不从本机访问服务器内部 URL。无合法目标时把错误交回 Agent 修正，不能用 DOM 编号凑一个动作。

AX 是默认观察。DOM 直接读取 `observation.dom.root`；HTML 和 PNG 通过 browser SDK 按需下载。处理 unavailable、size_limit、capture_failed、缺字段与过期；无字段时不沿用上次原件。HTML 内容与下载元数据的 SHA-256／大小核对由 harness 执行，PNG 使用 SDK 核验保存方法。原件留存共用任务预算。

结束时可返回 `stop [说明]`；错误导致退出且仍允许回应时，说明任务未完成，日志记真实原因。deadline 已过则停止本地工作，由 SDK 结束；不能继续调用模型或下载。不得将框架 AgentResponse.success 直接转换为官方通过成绩。

## 7. 真实浏览器后端

使用独立、可见浏览器和专用 profile，不读取用户日常浏览器资料。通过页面真实渲染生成 AX 风格文本与当前元素定位表，使用浏览器正常点击、输入、滚动、导航；不通过目标网站业务 API 完成任务。

Playwright 后端负责等待页面稳定到可观察状态、生成当前元素索引与定位、执行后重新观察。控件状态、角色、名称和输入值进入观察；隐藏、不可交互或被遮挡目标不作为可直接执行的有效元素。不要用 JS 事件派发绕过点击失败。观察提取可以读取当前渲染 DOM，与业务操作分开。

browser-use 试验使用同一测试网页、同一模型与工具语义。先查选定版本公开浏览器 API，再固定依赖，验证是否能复用观察和执行而不启动另一规划循环。成功则实现共同后端接口；若必须替换 EffGen 决策循环才可接入，保存具体调用和阻塞证据，由主线程交用户决定，不自行扩大改造。

CLI 提供：

```text
python -m webagent doctor --config <配置>
python -m webagent run --config <配置> --backend playwright --task-file <任务文件>
python -m webagent run --config <配置> --backend browser-use --task-file <任务文件>
```

doctor 只检查配置、导入和目录，不调用真实模型或启动官方任务。执行入口支持暂停、继续和取消：暂停后不再发新动作，待在途动作结束或明确超时后提示可接管；继续先重采集；取消记录并清理自有资源。登录与验证由用户完成。

## 8. 日志与证据

每项任务创建 `runtime/runs/<run_id>/`，内含 `events.jsonl`、原始观察、按需截图与 `result.json`。run_id 不直接使用不可信任务文本构造路径。

每行事件包含：`schema_version,event_id,run_id,time,elapsed_ms,event_type,page_id,url,observation_id,action_id,action_type,result,model_calls_total,model_request_id`。不适用字段为 null。模型事件另含输入／输出 token、计数来源、耗时及失败类别；动作事件区分 proposed、dispatched、feedback、agent_claim；阶段终态另列 cancelled、deadline、error。不得将 proposed 记为执行结果。

time 使用带时区的 ISO 8601，持续时间使用单调时钟。记录操作时间、目标页面、动作类型、结果、模型调用量五个赛题必需字段；平台模型精确 token 与费用等待平台报告，保留关联 ID。

日志、配置摘要与异常统一脱敏；账号、密码、密钥、连接令牌不写入提示词、日志或仓库。页面原文按本地运行资料保存，不随工程提交。证据有唯一引用与哈希；保存失败可见，不能用旧图代替。

## 9. 运行交接

工程完成时将下列命令写成实际可执行的 Windows 说明，并在专用环境验证路径与导入。官方连接在 `code/official_client` 下执行：

```text
<项目Python> -m third_party_agent connect --connection-file <连接文件绝对路径> --agent webagent.official_agent:OfficialAgent
```

worker 不执行该真实连接命令。主线程确认同一令牌无旧客户端、同队无在途评测任务后启动，逐项运行并下载结果。连接成功不算任务通过。

交付报告按 [验收要求](03_验收要求.md) 的编号列结果，不写笼统的“全部完成”。未执行的真实模型、浏览器或官方检查标为待主线程验收。
