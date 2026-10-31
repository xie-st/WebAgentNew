# HTML 与 DOM 接口核查

核查日期：2026-10-30。已使用本队账号从评测台下载新版连接器，原件保存在[下载目录](原件/2026-10-30_新版连接器/)。本次仅检查文件，尚未运行评测。

ZIP SHA-256：`ec1a6697656d5e1b6ba328ea4c69601a6c6e82ab9e63f428f23db71afa16feb4`。下载来源与大小见同目录 `SOURCE.json`，接口依据为包内 README 与 `browser.py`。

| 能力 | 获取方式 | 关键含义 |
| --- | --- | --- |
| AX | `observation.text` | 当前视口带编号的树；动作使用本次 AX 编号 |
| 截图 | `observation.screenshot` 元数据；`browser.get_screenshot(image_id)` 或 `save_screenshot()` | 同次观察保存的视口 PNG，下载不会重新拍摄 |
| HTML | `observation.html` 元数据；`browser.get_html(html_id)` | 同次观察的 `page.content()`；完整页面内容，非原始 HTTP 响应 |
| DOM | `observation.dom.root` | 原始 `DOM.getDocument(depth=-1, pierce=True)` 返回的节点树，直接随当前观察提供 |

HTML 与 DOM 保留脚本、注释、隐藏节点和属性，需要本地筛选。HTML 不保证覆盖完整 iframe／Shadow DOM 或表单运行时值；DOM 保留该 CDP 调用返回的子文档与 Shadow DOM，但不保证所有 iframe，也不完整提供坐标、计算样式及运行时属性。两者都不是 AX 动作编号的替代品。

DOM 不在后续 history 中重复提供。超过大小限制或采集失败时返回不可用原因，不能沿用旧树充当本次观察。HTML、DOM、截图各有4 MiB上限；旧服务可能缺字段。截图和 HTML 下载只在所属任务的有效回调期间可用，处理时间计入原300秒预算。

新版 `browser.py` 还保留 `get_dom(dom_id)` 函数；当前 README 明确要求优先直接读取 `observation.dom.root`。不能仅凭函数存在就假定服务端还会提供 `dom_id` 下载协议。

模型与动作约束未变：模型仅收文本消息，不开放图片输入、原生工具调用参数或自选推理参数；每任务最多一个模型调用同时执行；一次 step 只返回一个动作。DOM 由服务端采集不等于开放任意 CDP、JS 注入或多 Tab 操作。

与本地旧 ZIP 比较：新增 `browser.py`，更新两份 README、`my_agent.py` 与 `agent_logic.py`；模型、会话与传输源码未变。新版算法示例已换成最简的一次模型调用，不能将旧 baseline 的记忆与动作校验逻辑视为新版示例自带能力。

原 `code/webarena-outbound-client.zip` 保留。阶段实施时使用此处冻结的新包，并在真实任务回调中确认实际字段；本轮未覆盖运行目录、创建连接或提交评测。
