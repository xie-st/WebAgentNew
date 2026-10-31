# WebArena 选手接入：300 秒、统一模型

## 十分钟开始

要求 Python 3.10+。下载并解压客户端，在在线平台「Agent 连接」创建连接，把 `agent-connection.json` 放到能直接看到 `third_party_agent/` 的目录。连接文件是平台令牌，不能提交或分享。

```sh
python3 -m third_party_agent connect --connection-file agent-connection.json --agent my_agent:MyAgent
```

看到 `Agent connected` 后保持终端运行，网页选择在线 Agent 开始整批评测。包内 `agent_logic.py` 是最简接入示例，每步向平台模型发起一次调用；无需模型 SDK、模型 API key 或本机浏览器。

## 只改算法入口

```python
from third_party_agent import model

def decide(*, task, observation, history, action_history, state):
    messages = build_prompt(task, observation, state)
    reply = model.generate(messages)
    return parse_action(reply.text)
```

`build_prompt` 和 `parse_action` 表示你的算法；可直接参考包内最简示例。`messages` 是非空列表，每项只含 `role` 和字符串 `content`，role 为 system/user/assistant。总输入 JSON 最多 1 MiB，最多 256 条消息。`generate` 只接受 messages，不接受模型名、temperature、输出上限、工具参数、服务地址或密钥。模型及推理参数由服务器固定；接口只返回 `.text` 和平台 `.request_id`，不返回供应商模型标识。

每任务最多一个模型调用同时执行；调用供应商的网络等待上限为 90 秒且不超过任务剩余时间。调用失败不自动重新请求供应商，避免重复计费。需要重试可由算法在剩余预算内再次调用。平台记录 token 用量、按管理员定价计算费用，usage 或定价缺失会显示费用待核对。

可在自己的虚拟环境中安装算法依赖。`model.generate` 在当前 Agent 调用线程中使用任务绑定；自建线程须显式传播 contextvars 上下文，且不能突破单任务并发限制。禁止通过 import 阶段启动常驻程序，multiprocessing 入口须有 `if __name__ == '__main__':` 保护。SDK 每任务使用独立 spawn 子进程；关闭/超时后结束进程，避免阻塞后续任务。

## 输入和动作

- task：当前任务文字，不含隐藏判分条件。
- observation：type=`accessibility_tree`、url、text。text 为当前视口带编号的页面树；截图通过可选 screenshot 元数据单独下载，不含完整 DOM。
- history：先前观察和动作；action_history：文字动作描述。
- state：仅在本项任务内复用的本地字典，新任务清空。

URL 的 `127.0.0.1:动态端口` 指服务器内部网站，不可从选手本机直接访问。通过返回动作让服务器操作。同一页面编号会变化，必须使用本次观察里的真实编号。

| 动作 | 格式 |
|---|---|
| 点击 | `click [编号]` |
| 输入 | `type [编号] [文字] [0或1]`；1 随后回车 |
| 按键 | `press [Ctrl+A]` |
| 滚动 | `scroll [down]` / `scroll [up]` |
| 导航 | `goto [当前实例内的网址]` |
| 返回/前进 | `go_back` / `go_forward` |
| 完成 | `stop [答案或说明]` |

一次只返回一个动作字符串或 `{"action":"..."}`。输入可能追加，覆盖前须点击并全选。不要自动重放可能已经生效的发帖或提交动作。`reset` 不一定含任务文字；第一份 step 才有完整 task。

## 时间和判分

浏览器与登录准备完成后，平台开始每项固定 300 秒执行计时；模型等待、网络往返、算法计算和浏览器动作均计入，不限制步数，不因重复动作或三次解析错误提前停止。主动 stop 或到期后撤销执行权限，再按网站最终状态及任务所需答案判分。截止之后的动作和模型结果不会继续驱动网页；已产生或迟到的供应商用量仍归属原任务。临近截止的在途网站请求可能完成，不能承诺撤销已经提交的业务操作。

环境准备有独立 600 秒故障保护；最终判分和清理有截止后 60 秒保护，不占执行预算。平台是 POSIX worker 部署；选手客户端支持 Windows/macOS/Linux。

运行记录显示 score 和 finish_reason（stop/deadline/error），以及模型调用、token 和费用统计。每组十项，成功率=通过项数/10。执行截止不是直接判零；最终判分出错仍为未评分。整批无额外总时限。

旧版 `WEBARENA_AGENT_WAIT`、本地模型 .env、24 步规则不再适用，请重新下载客户端。同一连接文件只运行一个客户端。停止可在网页取消整批或按 Ctrl+C。

本地执行模式无法从技术上阻止自带模型或外部网络；正式比赛应配套规则审计，需要强约束时使用平台受控容器。

## 当前视口截图

支持截图的服务器会在每次 `observation` 中附加 `screenshot` 元数据，包含 `image_id`、`width`、`height`、`mime_type`、`step`、`byte_size` 和 `sha256`。它对应本次观察已经采集的 PNG，不是执行到一半时重新拍摄的页面。旧服务器可能没有此字段。

```python
from third_party_agent import browser

# 在 Agent.step(request) 内调用；与 model.generate 使用相同的任务上下文。
metadata = request['observation'].get('screenshot')
if metadata:
    image_bytes = browser.get_screenshot(metadata['image_id'])
    # image_bytes 是 PNG 字节，可以交给自己安装的 OCR 或图像处理库。
```

下载接口为 `GET /api/client/screenshots/{image_id}?job_id=...&run_id=...`，SDK 自动附加当前任务信息和连接令牌。图片只对拥有该任务且仍在执行中的 Agent 开放；跨任务访问返回 404，会话失效或超过执行时间返回 409。SDK 不跟随重定向，不接受任意文件路径；PNG 最大 4 MiB，单边最大 4096 像素，总像素最大 4194304。实际尺寸以元数据为准，目前浏览器统一使用 1280×720、缩放比例 1。

截图发布、下载和本地处理都消耗同一 300 秒执行预算，不增加步数限制。历史观察中仅保存图片元数据，不嵌入图片二进制。图片保存在服务端对应运行目录，随该目录的运维保留及清理策略管理，不放在公开静态目录。

此接口不开放 CDP、JavaScript、任意 HTTP 请求或坐标动作。统一模型网关仍然只接收文本，不能直接将 PNG 字节或图片消息传入 `model.generate`。使用更新后的客户端下载包才能导入 `third_party_agent.browser`；原纯文本 Agent 可以忽略新增字段。

## 获取截图并接入算法

请从平台重新下载客户端 ZIP；已有连接文件可继续使用，但同一令牌先停止旧客户端，再启动新版。无需在本机安装浏览器、Docker 或图像处理库。

算法中的最小用法（在 `decide()` 内）：

```python
from third_party_agent import browser

metadata = observation.get('screenshot')
if metadata:
    png_bytes = browser.get_screenshot(metadata['image_id'])
    # 在这里交给自己的 OCR 或图像处理逻辑。
```

截图为当前观察对应的视口 PNG，实际尺寸以元数据为准，默认 1280×720。它与这一步的页面树一起产生；下载不会再次拍摄、滚动或执行动作。下载及本地处理都计入 300 秒预算。只能在所属任务执行期间下载，跨任务不可访问；旧服务可能不提供此字段。只能读取 PNG，当前不开放任意 HTTP、JS 注入、CDP 或坐标动作。统一模型接口目前只接受文本，不能把图片直接传给 `model.generate`。

### 在原 Demo 中保存截图

仍使用原来的 `my_agent:MyAgent`。默认不自动下载图片；需要每步保存时，在启动前设置本机输出目录：

Linux / macOS：

```sh
export WEBARENA_SCREENSHOT_DIR=./screenshots
python3 -m third_party_agent connect --connection-file agent-connection.json --agent my_agent:MyAgent
```

Windows PowerShell：

```powershell
$env:WEBARENA_SCREENSHOT_DIR=".\screenshots"
python -m third_party_agent connect --connection-file agent-connection.json --agent my_agent:MyAgent
```

在网页开始评测后，PNG 会保存到本机指定目录，文件名为 `<image_id>.png`。Demo 保存前核对 PNG 格式、宽高、字节数和 SHA-256，并在算法的 `state["_screenshot_path"]` 中提供本次图片路径；没有截图时不会沿用上一帧路径。保存图片不会替换原有 `decide()`、动作或模型调用，也不会自动提升任务成功率。

也可在自己的 `decide()` 中按需调用 `browser.save_screenshot(observation, "./screenshots")`，返回 `Path`，旧服务没有截图字段时返回 `None`。只想取得 PNG 字节进行 OCR 时继续使用 `browser.get_screenshot(image_id)`。不要同时开启自动保存又重复下载同一帧，除非算法确有需要。

同一连接文件先停止旧客户端再启动新版本。截图与本地计算都计入原有 300 秒预算；文件可能包含业务内容，请妥善保管，不要把截图、连接文件或令牌提交到 GitHub。不要在回调返回后另起后台任务下载，任务上下文和下载权限会随执行结束失效。

## 完整 HTML 快照

支持此能力的服务器在每步 `observation.html` 中提供 `html_id`、`step`、`byte_size`、`sha256` 和 `mime_type`。这是同次观察已经保存的 `page.content()` 原文，不按视口裁剪，不过滤脚本、注释、隐藏节点或属性。不是原始 HTTP 响应，也不包含 Cookie、localStorage、JS 堆或完整 iframe/Shadow DOM 内容；表单当前属性不一定反映在 HTML 中。HTML id 属性不是 WebArena 动作编号。

```python
from third_party_agent import browser
metadata = request['observation'].get('html')
if metadata and metadata.get('html_id'):
    html = browser.get_html(metadata['html_id'])  # UTF-8 字符串
    # 自行使用 BeautifulSoup / lxml 等解析
```

下载接口 `GET /api/client/html/{html_id}?job_id=...&run_id=...`，SDK 自动使用当前任务令牌。与截图相同，只允许当前任务所属 Agent 在执行会话和 300 秒时限内下载；不接受任意网址或文件路径，也不跟随重定向。响应为附件，不直接执行 HTML。每份最大 4 MiB，超限时元数据为 `available: false, reason: size_limit`，不返回截断内容。旧服务器可能没有 html 字段。HTML 原文不会自动加入模型提示；选手自行处理。需重新下载连接器以获得 `browser.get_html()`，旧 Agent 可继续运行。

## 原始 DOM Tree（直接随本步 observation 返回）

```python
tree = request['observation'].get('dom')
if tree and 'root' in tree:
    root = tree['root']  # CDP 原始节点字典，无需 DOM ID 或下载
```

服务器每步仅调用一次 Chromium 原始 `DOM.getDocument(depth=-1, pierce=True)`，原样返回 JSON 字典；保留脚本、注释、隐藏节点、属性、Shadow DOM（含 closed shadow root）以及该调用返回的 iframe 子文档。不额外适配跨进程 iframe，不保证所有 iframe 都包含。不重建树、不生成选择器、不开放任意 JS 或 CDP 远程调用。`nodeId`/`backendNodeId` 不是 a11y 动作编号；坐标、计算样式和表单运行时属性不由该接口全部提供。

只有当前观察包含 DOM 正文；后续请求的 history 中不重复携带旧 DOM。每份最大 4 MiB，超限返回 `available: false, reason: size_limit`，采集失败返回 `capture_failed`，不截断或复用上一帧。旧服务器可能没有 dom 字段。HTML 和截图继续按原方式下载；旧 Agent 无需修改。
