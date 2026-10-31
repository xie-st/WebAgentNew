# WebAgentNew

2026-10-31：工程已迁入[提交目录中的代码](docs/08_提交材料/代码/运行说明.md)。按用户最新要求，后续补丁、运行及交付验证以该目录为工作位置；本目录原代码保留为迁入来源。

基于 **EffGen** 的网页 Agent 工程，用同一套任务循环和网页工具连接官方评测平台与本机真实浏览器。

EffGen 负责规划和工具选择，harness 负责动作交接、观察绑定、状态保存、预算、核验与运行证据。项目同时维护可重置的本地任务，用于验证机制和开展单变量实验。

## 两种运行方式

| | 官方评测平台 | 真实浏览器 |
| --- | --- | --- |
| 入口 | `webagent.official_agent:OfficialAgent` | `python -m webagent run` |
| 浏览器 | 赛方托管，由官方 SDK 返回观察、执行动作 | 本机 Playwright Chromium，支持显示窗口 |
| 模型 | 官方 `model.generate(messages)`，模型与参数由平台固定 | 配置的 OpenAI 兼容文本接口 |
| 输入任务 | 平台下发 | 外部 JSON 任务文件 |
| 控制 | 每队单在途任务，每次交出一个动作 | 当前 CLI 支持文件驱动的暂停、继续、取消 |
| 结果 | 平台评分与本地日志分别保存 | 运行结果、观察、截图与模型调用记录 |

当前交互入口是命令行和浏览器。真实端的多 Tab 调度、自动接管与上下文增强以各模块的实际接线和验收记录为准。

## 系统结构

```text
官方任务 / JSON任务
        ↓
EffGen交互生成器 ↔ harness驱动 ↔ Session、网页工具、预算与证据
                                 ├─ OfficialBackend → 官方SDK → 托管浏览器
                                 └─ PlaywrightBackend → 本机Chromium
```

网页动作绑定当前观察和元素编号。人工接管后重新观察，旧提案与旧编号失效。任务状态、核验和恢复通过驱动扩展点接入，沿用同一条 EffGen 循环。

运行使用的是带本地补丁的 `code/vendor/effgen`，固定来源和补丁说明见 [UPSTREAM.json](code/vendor/effgen/UPSTREAM.json)。`code/effGen` 是本机保留的原始检出，已被 Git 忽略。

## 安装

以下命令使用 Windows PowerShell，在仓库根目录执行。需要 Python **3.12+** 和 `uv`；项目当前使用 Python 3.12。

首次创建环境并安装 vendored EffGen、harness 与浏览器依赖：

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe `
  -e code/vendor/effgen `
  -e "code/harness[real,dev]"

.\.venv\Scripts\python.exe -m playwright install chromium
uv pip check --python .venv/Scripts/python.exe
```

已有环境可以从安装依赖步骤开始。不要用 PyPI 原版 EffGen 替换本项目的 vendored 补丁。

默认真实后端是 Playwright。`browser-use` 为独立试验后端；其当前版本与 EffGen 的 `aiohttp` 依赖约束冲突，不纳入上述安装组合。

## 先跑一个离线示例

使用脚本模型和假后端驱动实际 EffGen 交互循环，不访问模型服务、浏览器或官方平台：

```powershell
.\.venv\Scripts\python.exe code\harness\examples\interactive_demo.py `
  runtime\interactive-demo
```

示例输出循环状态、回答与证据目录。它用于检查工程链路，策略效果需要真实模型实验验证。

也可以用本地网页演示实际浏览器。先在一个终端启动静态站点：

```powershell
.\.venv\Scripts\python.exe -m http.server 8765 `
  --bind 127.0.0.1 `
  --directory code\harness\tests\fixtures\site
```

在另一个终端、同一仓库根目录执行：

```powershell
.\.venv\Scripts\python.exe -m webagent run `
  --config code\harness\configs\real.yaml `
  --backend playwright `
  --task-file code\harness\tests\fixtures\tasks\local_chain.json `
  --model-script code\harness\tests\fixtures\fake_scripts\local_chain.json `
  --no-headless `
  --run-id "local-demo-$([guid]::NewGuid().ToString('N'))" `
  --max-seconds 120
```

这里的 `--model-script` 提供预设响应，不需要模型密钥。每次使用新的 run ID，日志和控制文件分别归档。

## 在真实网站运行

### 1. 配置模型

在启动任务的 PowerShell 窗口设置服务地址和密钥：

```powershell
$env:WEBAGENT_MODEL_BASE_URL = '你的模型服务地址，例如 https://服务域名/v1'
$env:WEBAGENT_MODEL_API_KEY = '你的模型密钥'
```

使用 [real.yaml](code/harness/configs/real.yaml) 配置真实端。将 `model.name` 改为服务实际接受的模型标识；当前模板值为 `unified-model`。适配器调用文本 `/chat/completions` 接口。比赛运行使用允许的统一模型。

[example.env](code/harness/configs/example.env) 仅列出变量名和假占位值。真实凭据由运行环境提供，不写进共享任务文件或文档。

检查配置、导入、浏览器安装和环境变量是否齐全：

```powershell
.\.venv\Scripts\python.exe -m webagent doctor `
  --config code\harness\configs\real.yaml
```

`doctor` 不调用模型，也不启动浏览器或官方评测。

### 2. 编写任务

将以下内容保存为 `runtime/task.json`，先创建 `runtime` 目录：

```json
{
  "task": "打开网页，读取页面标题并回答，不修改页面内容。",
  "start_url": "https://example.com",
  "headless": false,
  "max_seconds": 180
}
```

`task` 必填，起始地址通过 `start_url` 或命令行 `--start-url` 提供。任务文件还支持 `run_id`、`seed`、`draft_text` 和 `description`；任务写目标与约束，执行动作由 Agent 决定。

### 3. 启动

```powershell
.\.venv\Scripts\python.exe -m webagent run `
  --config code\harness\configs\real.yaml `
  --backend playwright `
  --task-file runtime\task.json `
  --no-headless
```

终端显示 run ID、控制文件路径、动作、反馈和结果目录。更多参数：

```powershell
.\.venv\Scripts\python.exe -m webagent run --help
```

`--profile-dir` 可指定专用浏览器配置目录，用于保留该目录中的登录状态；默认目录按 run ID 隔离。

## 人工接管：暂停、登录、继续

启动终端会打印 `control file:`。在另一个 PowerShell 窗口，将该完整路径赋给变量：

```powershell
$controlFile = '粘贴本次运行的 control.jsonl 完整路径'
```

需要接管时追加暂停命令：

```powershell
Add-Content -LiteralPath $controlFile -Value 'pause' -Encoding utf8
```

等待运行终端显示 **`PAUSED`**，再在 Agent 打开的浏览器中完成登录、验证或其他人工操作。处理完成后继续：

```powershell
Add-Content -LiteralPath $controlFile -Value 'resume' -Encoding utf8
```

程序重新观察页面并沿用原任务继续，不需要重写任务。每次接管都采用同样流程。当前真实 CLI 在动作之间处理控制命令，使用显式 `resume` 恢复；自动登录检测模块的实现不等于该 CLI 已自动启用。

结束任务：

```powershell
Add-Content -LiteralPath $controlFile -Value 'cancel' -Encoding utf8
```

`Ctrl+C` 也会触发取消。暂停等待上限由 `--pause-timeout` 指定，默认600秒；若设置了任务总时限，该时限在暂停期间仍继续计算。

## 连接官方平台

官方端使用仓库内冻结 SDK 和赛方连接文件，无需设置真实端模型 API key，也无需在本机运行官方网站。

从仓库根目录执行：

```powershell
Push-Location code\official_client
try {
  ..\..\.venv\Scripts\python.exe -m third_party_agent connect `
    --connection-file '你的官方连接文件绝对路径' `
    --agent webagent.official_agent:OfficialAgent
} finally {
  Pop-Location
}
```

看到 `Agent connected` 后，保持连接进程运行，在评测网页选择 Agent 并开始批次。同一连接文件只运行一个客户端；官方任务始终串行。

官方入口默认读取 [official.yaml](code/harness/configs/official.yaml)。当前已接入的行为开关如下，修改后在下一次受控启动中生效：

| 配置 | 默认值 | 作用 |
| --- | --- | --- |
| `official.behaviors.state_tools` | `true` | 注册状态与目标工具 |
| `official.behaviors.verification` | `true` | 完成核验及同循环纠错 |
| `official.behaviors.recovery` | `false` | R波接管、受限恢复与快照重启接线 |
| `context.enabled` | `false` | G波必要上下文保护与最终输入检查 |
| `budget.deadline_margin_seconds` | `0` | 不预留额外提前截止时间，遵守实际硬截止 |

官方截图、HTML／DOM能力及限制见 [SDK接口核查](docs/05_赛题与要求/08_HTML与DOM接口核查.md)；模型网关当前仍接收文本。平台分数来自官方判定，Agent自报完成与本地 `result.json` 不替代平台评分。

## 测试与运行证据

运行 harness 的离线回归，包括假模型契约与隔离本地浏览器用例：

```powershell
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe -m pytest code\harness\tests -q
```

需要已安装 Chromium。完整回归、真实模型效果、官方成绩分别记录；当前源码版本以对应验收记录为准。

默认单次运行产物位于 `runtime/runs/<run_id>/`：

| 文件／目录 | 内容 |
| --- | --- |
| `events.jsonl` | 观察、动作、反馈、调用、终态等事件 |
| `observations/` | 每次观察及对应元素信息 |
| `screenshots/` | 与观察关联的截图 |
| `result.json` | 本地结果、结束原因与实际启用项 |
| `model_calls.json` | 真实 CLI 模型调用记录，包含失败尝试 |
| `control.jsonl` | 暂停、继续、取消指令 |

具体字段和额外状态文件随入口与开关产生。平台硬截止可能终止任务进程，缺失本地终态时保留原日志并以平台结果另行对账。

## 目录与开发状态

```text
code/
  harness/          主工程：双环境适配、状态、核验、恢复、测试与CLI
  vendor/effgen/    固定来源、带交互生成器补丁的EffGen
  official_client/ 冻结官方连接器
docs/
  01_路线与框架/   RoadMap与任务进度
  03_策略与论文方法/ 方法来源与核查
  05_赛题与要求/   比赛规则、答疑及接口原件
  06_分阶段实施/   各阶段方案、实验设计与验收记录
  07_汇报材料/     报告、演示与现场执行材料
runtime/           本机日志、浏览器配置、冻结副本及实验产物（Git忽略）
```

当前已具备双环境基础链路、状态与核验工具、官方入口恢复接线及必要上下文保护代码。真实 CLI 的手动接管入口可独立使用。F的多分支调度、G的经验与历史增强按独立批次继续交付；已实现、默认启用、真实验证通过是不同状态，详细进度见下列记录。

- [总路线](docs/06_分阶段实施/00_总路线.md)
- [任务进度](docs/01_路线与框架/提示词&任务进度.md)
- [阶段二：基础能力与测试环境](docs/06_分阶段实施/阶段二_基础能力与测试环境/README.md)
- [阶段三：实验验证](docs/06_分阶段实施/阶段三_实验验证/README.md)
- [双环境测试、F/G开发与实验并发方案](docs/06_分阶段实施/07_双环境测试_FG开发与实验_并发方案.md)
- [最终验收准备与现场执行](docs/07_汇报材料/最终验收准备与现场执行.md)
- [比赛要求](docs/05_赛题与要求/WebAgent比赛要求.md)

开发和离线分析可在独立副本并行，公共入口由一人集成；正式运行冻结版本，实验一次只改变声明变量。官方与真实测试的版本和结果分别保存。

## 来源与本地文件

EffGen来源与补丁见 [UPSTREAM.json](code/vendor/effgen/UPSTREAM.json)，其许可证见 [EffGen LICENSE](code/vendor/effgen/LICENSE)；官方连接器许可证见 [SDK LICENSE](code/official_client/LICENSE)。保留第三方原有许可与归属。

连接文件、真实密钥、账号文件和浏览器状态仅在本机使用。[.gitignore](.gitignore) 已忽略相应凭据及 `runtime/`，`example.env` 保留为假值模板。
