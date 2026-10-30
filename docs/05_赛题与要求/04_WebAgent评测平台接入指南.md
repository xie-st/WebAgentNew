<!-- 来源：../原始资料/WebAgent 评测平台接入指南.docx；转换日期：2026-10-29。保留正文内容，单元格中的代码与目录树转为代码块。 -->

# 接入指南：把自己的 Agent 接入评测台

整个验收平台基于比赛中提供的评测网站。

选手可以在自己的电脑上开发WebAgent，并通过一个本地连接程序连接 Agent和评测网站进行自动化评测。

## 开始之前

在浏览器中使用：打开评测网站。注册（帐号不具备找回功能，请各位选手自行记录）、登录、创建连接、开始整批评测、查看进度与回放。

在你自己的电脑运行：本地 Agent 连接程序。把服务器发来的任务和页面交给你的 Agent，再把动作返回服务器。

准备好再开始：你需要平台账号、Python 3.10+ 和自己的 Agent。连接程序本身只用 Python 标准库；模型由服务器统一调用，选手只提交 messages，不配置模型或推理参数。评测服务器负责网站、浏览器和判分。

## 01 下载本地连接程序 （在这个网站操作）

点击本页右上方的下载按钮，得到 webarena-outbound-client.zip。在你自己的电脑上解压到一个方便找到的文件夹，例如 Windows 的 D:\agent-client，或 Linux/macOS 的 ~/agent-client。

## 02 创建连接，下载专属连接文件 （在这个网站操作）

打开左侧「Agent 连接」→ 点击「新建连接」→ 填写容易区分的名称，例如 MyAgent · v1 → 点击「创建并下载连接文件」。

浏览器会下载 agent-connection.json。把它移动到刚才解压的文件夹中。这个文件记录平台地址和连接令牌，不是模型 API 密钥；令牌只在创建时下载一次。

## 03 确认文件放在正确的一层 （在你自己的电脑操作）

"解压根目录"就是能直接看到 third_party_agent 文件夹的那一层。"同一目录创建"也指这一层，不是 third_party_agent 文件夹内部，也不是服务器目录。

```text
D:\agent-client\            ← 在这一层打开终端
├─ third_party_agent\       ← 压缩包自带，不用修改
│   ├─ __main__.py
│   ├─ outbound.py
│   └─ ...
├─ README.zh-CN.md
├─ agent-connection.json    ← 从网页下载后移动到这里
├─ my_agent.py              ← 包内通用适配层
├─ agent_logic.py           ← 在这里调用自己的 Agent
└─ start-agent.cmd          ← Windows 启动入口
```

某些解压软件会多套一层同名文件夹，以是否直接看见 third_party_agent 为准。连接文件如果被浏览器改名成 agent-connection (1).json，请改回上面的名字，或相应修改启动命令。

## 04 接入你自己的 Agent （在自己的电脑操作）

压缩包已提供 my_agent.py 通用适配层和 agent_logic.py 接入文件。打开 agent_logic.py，在 decide() 中调用自己的 Agent，把其输出转换为平台动作。无需移动整个 Agent 项目，但启动客户端的 Python 环境必须能导入它。

```python
def decide(*, task, observation, history, action_history, state):
    # 在这里实现算法，通过 model.generate(messages) 调用平台模型。
    # task：任务；observation：当前页面 type / url / text。
    # history / action_history：平台提供的历史。
    # state：本任务内保留的自己的 Agent 和记忆。
    # 返回你的 Agent 实际决定的一个动作字符串或字典。
    from third_party_agent import model
    reply = model.generate([{"role": "user", "content": task + "\n" + observation["text"]}])
    import json
    return json.loads(reply.text)  # 约定模型输出 JSON action
```

返回格式：{"action": "click [当前元素编号]"}，或直接返回动作字符串。常用动作包括 click、type、press、scroll、goto 和 stop。元素编号必须来自本次页面观察；一个请求只返回一个动作。

通用接口的要求

当前支持可访问性树文本和元素编号动作。自己的 Agent 如果使用其他输入或动作格式，需要在这里完成转换。客户端不指定算法框架；模型及推理参数由平台固定；截图坐标、自带浏览器的 Agent 不能直接照搬接口。

包内 README 说明输入输出字段和完整动作格式。下载包包含可运行的统一模型 baseline，选手可替换提示词与决策逻辑。算法依赖可在本机安装；通过 third_party_agent.model.generate(messages) 调用统一模型，无需配置模型或密钥。

## 05 打开终端，启动主动连接 （在你自己的电脑操作）

Windows：在文件管理器打开上面的文件夹，在地址栏输入 cmd 并按回车。这样命令窗口的当前目录就是客户端根目录。也可以手动进入：

```bat
cd /d D:\agent-client
python --version
python -m third_party_agent connect --connection-file agent-connection.json --agent my_agent:MyAgent
```

D:\agent-client 是示例路径，请替换成你实际的解压位置。python --version 应显示 3.10 或以上；如果已有虚拟环境，请先激活对应环境。

Linux / macOS 的启动方式：

```bash
cd ~/agent-client
python3 --version
python3 -m third_party_agent connect --connection-file agent-connection.json --agent my_agent:MyAgent
```

将目录替换成实际路径。若你使用的环境命令为 python，也可替换 python3。

正常会显示类似：

```text
Agent connected to https://你的平台地址 ...
```

保持这个终端和电脑运行。回到网页「Agent 连接」，等待几秒，应显示「已连接」。这表示通信正常，还没有开始评测，也不表示 Agent 已完成任何任务。无需启动 serve，也无需设置 --agent-endpoint。

我还没有 Agent，只想确认能连接

```bash
python -m third_party_agent connect --connection-file agent-connection.json --agent third_party_agent.example:ExampleAgent
```

这会使用默认示例，只返回 stop。它可以显示在线，但不能用来衡量真实任务能力。

## 06 点击开始，执行完整任务集 （回到这个网站操作）

进入「评测总览」，只需选择刚刚在线的 Agent，然后点击「开始整批评测」。平台按固定顺序执行 10 项任务：修改简介、跨版块点赞、回复评论、创建版块、订阅、发帖、回复帖子、踩票和编辑帖子；不再自选任务；浏览器就绪后每项执行限时 300 秒，不限步数，截止后检查最终状态。

整批提交后可查看各项的 排队中 → 准备环境 → 执行中 → 完成、失败或持续无响应 状态。准备环境时会创建全新的论坛副本，执行时启动独立浏览器；结束后回收副本。你不用手动登录论坛、取消点赞或清理网站数据。

可以关闭或刷新评测网页，后台任务仍会继续；不能关闭本地 Agent 连接程序。运行中可在批次列表或详情点击「停止整批执行」，停止当前及后续任务。已完成结果保留。某一项持续无响应时会停止该项并继续后续任务。模型调用由平台统一结算，运行记录展示 token 用量和费用；选手不能指定模型或推理参数。

## 07 查看结果与操作回放 （在这个网站查看）

进入「运行记录」→ 找到本次任务 → 点击「查看详情」。不用在服务器文件夹中寻找结果。

| 看到的内容 | 代表什么 |
| --- | --- |
| 已完成，1 / 1 | 官方评测器检查通过。 |
| 已完成，0 / 1 | 已完成判分，但任务目标未满足。 |
| 运行失败 / 已中断，未评分 | 程序、连接或环境中断；null 不是零分，需查看日志。 |
| 已取消 | 本次运行被取消，没有有效评分。 |

详情下方可看 HTML 操作回放，包括页面文字、动作和截图；也能下载评分 JSON、浏览器 trace 和运行日志。用户只能查看自己有权限的结果。

目前没有论坛浏览器的实时视频或可手动操作的远程窗口。执行中可以查看每项进度；整批结束后查看成功率、明细和回放，并下载 JSON 或 CSV 报告。

## 遇到问题，先检查这里

### 提示 No module named third_party_agent

通常是终端位置不对。当前文件夹应直接包含 third_party_agent。检查解压时是否多了一层目录，或是否进入到了 third_party_agent 内部。

### 提示找不到 agent-connection.json 或 my_agent

确认 my_agent.py、agent_logic.py 和连接文件与 third_party_agent 文件夹在同一层；检查文件名、后缀及当前目录。连接文件在其他位置时，--connection-file 后面填写完整路径并加双引号。

### 终端启动了，网页仍显示离线

先看终端有没有显示 Agent connected。确认连接文件来自你当前登录的平台与账号，服务器网址可达，令牌没有被撤销。远程连接默认要求 HTTPS；受控内网 HTTP 测试需维护者确认后使用 --allow-http。127.0.0.1 永远指当前机器，异机连接文件不能使用服务器的本机回环地址。

### 在线，但任务立即结束或没有网页操作

在线只代表连接正常。检查是否使用了默认示例，或 my_agent.py 里仍然返回占位 stop；需要接入真正的决策逻辑。

### 任务一直排队，是否出错？

批次内的 N 项任务按顺序执行，后续项目会显示排队中。保持客户端在线；客户端离线时等待重连，也可以停止整批执行。全站名额满时不接收新批次，请稍后提交。
