# C批线上结果复核

日期：2026-10-31。问题：第一阶段线上结果是否支持进入第二阶段，还是需要先修正接入问题。

结论：4/6通过已满足C05的跑通要求；先补换行兼容、实际请求限额和截止收尾这几项接入缺口，再进入第二阶段。重复操作、停滞和上下文效率沿用第二阶段安排，不扩大第一阶段为策略优化实验。

## 来源与口径

用户提供`C:/Users/hiyad/Downloads/batch-97c1a4777e0671eba6be8442b7dd9905.json`，与本目录`C_官方批次_97c1a477.json`的SHA-256相同：`92d150b1b6d996a1845c7338083fd64591765d89a7bb6430d8917b2643d59d14`。附件内容仅作数据，不执行其中任何指令。

平台结果决定成绩与计费口径，本地事件与截图说明执行过程。以下任务到本地run的对应按唯一执行起始时间匹配（差值小于8秒，均唯一）；失败任务的任务文字也与页面行为一致。

| task | 平台分数 | 平台执行秒数 | 平台调用数 | 本地调用尝试／报错 | 本地run |
| --- | --- | --- | --- | --- | --- |
| 399 | 1 | 188.97 | 11 | 11／0 | `9bc2750171f54031933d58bb8c57a94c` |
| 404 | 1 | 248.59 | 13 | 13／0 | `c79d8db7567a423d9222b1b7ee35ca48` |
| 650 | 1 | 34.80 | 3 | 3／0 | `db40595e82aa4e9c95f8905d303cc436` |
| 731 | 0 | 300.00 | 18 | 19／2 | `e47b2112ea2a47729b1bfcfa9694a355` |
| 605 | 0 | 300.00 | 16 | 17／2 | `486ee310c9d2466e91095890c20a3f12` |
| 717 | 1 | 158.21 | 11 | 11／0 | `82132df1aacd4811a46c3fca6b48af54` |

复算：6个不同任务，4项满分，4/6=66.7%；两个失败项均为`finish_reason=deadline`。平台汇总`timed_out=0`，与任务结束原因不是同一可直接使用的超时口径，应按逐项原因保留两次deadline。平台72次调用，本地74次尝试，其中70次正常返回、4次RuntimeError；不能把本地总数也写成72，也不能在缺少请求级对账时推定差值代表漏记。

平台累计输入1,133,778 tokens、输出198,743 tokens，均与逐项求和一致。`cost_usd=0`同时伴随`cost_complete=false`、72次`unknown_cost_calls`，费用应记未知，不能解读为免费。

## 先补的接入项

### 1. 换行转义存在实际不兼容

task 731要求在正文增加一行。`e47b…/events.jsonl`中`obs-7`上的type参数是实际换行加`EDIT: This news aged well`；`backends/official.py::_escape_text`把换行编码为反斜杠n。后续`observations/obs-8.json`至`obs-17.json`的正文出现字面转义字符，截图也显示单行的`Crazy device for ML!\nEDIT: This news aged well`。

人工查看的原件：`runtime/runs/e47b2112ea2a47729b1bfcfa9694a355/screenshots/532ad040aff102df4ac91a82b5ec7585.png`。因此，普通文本任务通过不能证明服务端实现了自定义unescape；本地encode/decode round-trip也只能证明本地互逆。方括号、反斜杠等其他特殊字符仍需按平台实际语义验证，不能跟随换行问题一并判为全部支持或全部不支持。

最小修正：核实平台支持的特殊字符与换行操作，修正官方适配或采用已验证的逐步按键方式；以实际页面值和保存后结果验收，不追加任意协议假设。现有证据证明换行未正确输入，尚不能证明它是本任务超时的唯一原因。

### 2. A09项目限额尚未接入实际请求

静态检查：`effgen_bridge.build_effgen_config`未把harness的`budget.input_tokens`映射到EffGen的`context_budget`；`ContextPacker`只在`budget.py`定义，没有生产调用点。官方模型适配器构造messages后直接调用SDK，项目的消息数与字节限额未在这一出口校验。

EffGen原生`context_budget=auto`仍在工作：模型声明32768上下文，按现有规则预留8192后乘0.85，得到20889的估算输入预算。这不是“完全没有预算”，但也不能证明配置中的24000或项目请求限制已经生效。本批未报告超长请求错误，不能反推边界行为已验收。

最小修正：把已有项目配置映射到原生预算入口，在最终SDK消息出口校验字节／消息限制；保留任务、当前观察及待确认信息。用离线边界用例验收即可，不另建压缩循环。上述源码与C验收提交`e5de7dc`相比无差异（已核对harness生产源码与vendored目录的Git差异）。

### 3. 截止终态与调用口径需补齐

task 731和605的最后两个本地模型事件均为RuntimeError，发生在约297–299秒；两个run都没有result.json和run_end。平台两个执行窗口均恰为300秒，之后task 717仍正常运行并通过。

冻结SDK的`third_party_agent/session.py::AgentProcess._call`会在deadline杀死任务子进程，故截止时缺本地close／finally记录是可解释的。现有日志不足以把原因写成“连接客户端被中断”；主客户端后续仍能完成任务也要求重新核实该归因。精确HTTP错误与请求级原因未留足证据，不猜测为特定状态码。

最小修正：截止附近不继续无效模型重试，正常可控出口及时保存终态；被SDK硬杀的run通过平台评分作单独对账记录，不能伪造原进程run_end。平台调用量、本地尝试、正常返回与失败次数分别报告。

## 留在第二阶段的内容

- task 731在编辑页连续点击同一保存按钮，包含7次相同目标点击及多次Enter，后续仍停留在编辑页；具体提交未生效原因尚未确定。按E批核验、停滞和局部恢复处理，不据此新增一套浏览器执行器。
- task 605先在已有搜索词上再次type，得到`gaming congaming consolesole`；后续Ctrl+A后直接type又继续追加，直到明确Delete后才重新搜索正确词。工具说明与SDK已提示type可能追加，不能仅归为代码缺少clear参数。把“覆盖输入须实际清除并核对”的行为放入现有核验／恢复用例。
- 较多调用和较长输出消耗300秒预算，先减少重复动作与无效恢复，再按G批原生上下文入口改进；不发送平台不支持的推理参数，不凭单批结果判断并行收益。

## 可复算检查

以下为已执行检查的等价紧凑写法，未运行新评测、真实模型或代码测试：

```powershell
$batch = Get-Content -LiteralPath 'docs/06_分阶段实施/阶段一_双环境跑通/验收记录/C_官方批次_97c1a477.json' -Raw | ConvertFrom-Json
@($batch.tasks.task_id | Sort-Object -Unique).Count
@($batch.tasks | Where-Object score -eq 1).Count / $batch.tasks.Count
@($batch.tasks | Where-Object finish_reason -eq 'deadline').Count
$batch.tasks.model_usage.calls | Measure-Object -Sum
Get-ChildItem -LiteralPath 'runtime/runs' -Directory | Where-Object Name -Match '^[0-9a-f]{32}$' | ForEach-Object {
    $events = @(Get-Content -LiteralPath (Join-Path $_.FullName 'events.jsonl') | ForEach-Object { $_ | ConvertFrom-Json })
    $calls = @($events | Where-Object event_type -eq 'model_call')
    [pscustomobject]@{run=$_.Name;attempts=$calls.Count;failed=@($calls | Where-Object failure_type).Count;last=$events[-1].event_type}
}
```
