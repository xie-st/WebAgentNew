# WebAgent 本地演示面板

本地运行的中文演示工作台：输入任务与起始网址，点击开始后通过子进程调用现有
`python -m webagent run` CLI，EffGen + Playwright 在真实浏览器中执行；面板持续
显示 run_id、执行状态、动作日志、最终回答与真实截图，并支持暂停 / 继续 / 取消、
证据下载与强制结束。技术栈为 Python 标准库 http.server + 原生 HTML/CSS/JS +
短轮询，无前端构建链。

## 环境前提

- Windows + 仓库根目录 `D:\Dwen\Alpha26fall\WebAgentNew`（其他根目录亦可，路径自动推导）。
- 仓库自带虚拟环境 `.venv\Scripts\python.exe`（已可编辑安装 `webagent`，已装 Playwright）。
- Playwright Chromium 已安装（`python -m playwright install chromium` 可补装）。

## 两种模式

| 模式 | 说明 | 前提 |
| --- | --- | --- |
| 真实模型（默认） | 实际调用已配置模型，任务与网址自由编辑 | 环境变量 `WEBAGENT_MODEL_BASE_URL`、`WEBAGENT_MODEL_API_KEY` 已设置 |
| 离线脚本演示 | 使用固定响应脚本（`code/harness/tests/fixtures/fake_scripts/local_chain.json`），任务与网址锁定为本地预设，页面始终标注"未调用真实模型" | 无需密钥 |

密钥只从环境变量读取，不落盘、不进入日志与下载。

## 一次启动

```powershell
Set-Location D:\Dwen\Alpha26fall\WebAgentNew
.\.venv\Scripts\python.exe code\harness\demo\app.py --port 8787
```

启动后自动打开 `http://127.0.0.1:8787`；自动打开失败时终端保留可点击地址。
端口被占用时程序明确报告并退出（不影响占用者），换 `--port` 重启即可，
预设网址会随端口自动变化。也可运行 `start_demo.ps1`（等价包装）。

## 使用流程

1. 页面填入任务描述、起始网址、最长运行秒数（离线模式下任务与网址自动锁定为预设）。
2. 点击"开始运行"，可见 Chromium 打开，面板同步显示动作、反馈与截图。
3. 需要人工处理时点"暂停"，面板显示"已暂停，可以在浏览器中操作"后接管页面；
   点"继续"后 Agent 重新观察页面（旧元素编号失效），沿原任务继续。
4. 结束后查看最终回答、终止原因、截图；在"运行记录与证据"区下载任务输入、
   结果 JSON、事件日志、stdout/stderr 与截图。回答按"Agent 返回完成"口径展示，
   当前 CLI 未提供核验证据，本地核验一栏固定显示"未提供本地核验结果"。

## 数据位置

- 面板侧元数据与 stdout/stderr：`runtime/demo/<run_id>/`
- 执行原生产物（events.jsonl、result.json、screenshots/）：`runtime/runs/<run_id>/`
- 预设演示站点由本服务以 `/fixture/` 提供：
  `http://127.0.0.1:<端口>/fixture/index.html?seed=alpha-17`

## 控制协议

暂停 / 继续 / 取消通过现有 `runtime/runs/<run_id>/control.jsonl` 行协议实现
（每行 `pause` / `resume` / `cancel`，由面板写入）。暂停以 CLI 的结构化
`paused` 事件为准；取消以进程退出与 result.json 终态为准；子进程无响应时
可在"高级配置"中强制结束（只清理本面板启动的进程树，记为中断并保留证据）。
关闭面板（Ctrl+C）会清理其启动的执行进程与浏览器。

## 测试

```powershell
Set-Location D:\Dwen\Alpha26fall\WebAgentNew
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe -m pytest code\harness\tests\demo -q
```

组件级测试（校验、进程锁、控制状态机、结果映射、证据白名单、脱敏）秒级完成；
端到端测试（`test_demo_e2e.py`）用离线脚本模式真实调用 CLI 与 Playwright
Chromium（headless），每个用例约 1–2 分钟。以上命令已被仓库回归命令
`python -m pytest code\harness\tests -q` 一并收集。
