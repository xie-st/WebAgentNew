# P 波 A 线实施报告：输入契约与官方 AX 适配

日期：2026-10-31
工作副本：`D:\Dwen\Alpha26fall\WebAgentNew\runtime\parallel\20261031-P1\laneA-input`
范围：仅修改本副本；未修改主树、未派发 worker。

## 修改文件清单

| 文件 | 类型 | 内容 |
| --- | --- | --- |
| `code/harness/src/webagent/verification/ax_official.py` | 新增 | 官方 AX 适配：引号内名称提取、按编号定位缩进元素行、读取缩进子 `StaticText` 值；值扫描只限匹配元素的后代，不取同名兄弟标签 |
| `code/harness/src/webagent/verification/checks.py` | 修改 | `visible_value` 接入名称后缀剥离与子 `StaticText` 回退；`_element_line` 支持带缩进的官方行；`visible_element` 同步使用名称匹配 |
| `code/harness/src/webagent/backends/official.py` | 修改 | 空文本 `type` 契约；提交后取证事件标记；新增两个公开常量 |
| `code/harness/tests/phase2/test_ax_official.py` | 新增 | P02 官方观察回归：399 obs-5 追加值 FAIL、obs-10 正确值 PASS；名称后缀、标签行不取值 |
| `code/harness/tests/phase2/test_official_input_contract.py` | 新增 | P01 官方输入契约：空文本 + submit、空文本无 submit、正常文本格式不变、提交后标记 |

未修改 `official_agent.py`、`effgen_bridge.py`、`tools.py`、`config.py`、`session.py`、`state/`、`recovery/`、`testing/`、`control/`、`events.py`、`protocol.py`。

## 关键实现

### 官方 AX 适配

- `extract_quoted_name("'Biography' required: False describedby: ...") -> "Biography"`。
- `name_matches` 用剥离后的名称比较，本地无引号名称和官方带属性名称均可匹配。
- `find_element_line` 接受行首缩进，按完整 `[编号]` 定位，避免把 `[1]` 误配到 `[10]`。
- `child_static_text_value` 只扫描编号行之后、缩进更深的行；遇到缩进不大于基准行即停止。因此输入框前面的同名 `StaticText 'Biography'` 标签、或同层兄弟标签都不会被当作值。
- `visible_value` 顺序：`localization.value` → 同一行 `value=` → 缩进子 `StaticText` → 空字符串（元素存在但无值）。未找到元素仍返回 `None`。

### 空文本 `type` 契约

- `text == "" and submit=True`：`serialize_action` 返回 `press [Enter]`，不再产生 `type [n] [] [1]`。
- `text == "" and submit=False`：`validate_action` 本地拒绝，抛出 `ActionRejected(INVALID_ARGUMENTS)`；`correction` 说明改用 `web_press`/`keys='Enter'`，并写一条 `format_correction` 事件（`reason=empty_type_text`, `dispatched=False`）。
- 非空文本的原有格式保持不变（含 RAW/DECOMPOSE、方括号和反斜杠不转义）。

### 提交后取证标记

- `OfficialBackend.from_request` 在 `previous_action.op == TYPE and previous_action.submit` 且新观察已生成时，发出 `artifact_saved` 事件，`result` 包含：
  - `kind="post_submit_observation"`、`post_submit_evidence=True`；
  - `submit_action_id`、`submit_target`、`submit_text_chars`、`submit_was_empty`；
  - 新观察 `observation_id` 及当前观察各 artifact 的可用性/引用。
- 未改 EvidenceStore 接口；观察仍由 Session 原路径 `save_observation` 保存。

## 关键命令与结果

```powershell
set PYTHONPATH=<副本>\code\harness\src
python -c "import webagent; print(webagent.__file__)"
```

结果：

```text
D:\Dwen\Alpha26fall\WebAgentNew\runtime\parallel\20261031-P1\laneA-input\code\harness\src\webagent\__init__.py
```

确认导入解析到本副本 `src`。

修改前基线：

```powershell
python -m pytest tests -q
```

```text
298 passed in 149.02s (0:02:29)
```

针对性回归：

```powershell
python -m pytest tests/phase2/test_ax_official.py tests/phase2/test_official_input_contract.py tests/test_official_backend.py tests/test_official_agent.py -q
```

```text
38 passed in 4.16s
```

修改后全量：

```powershell
python -m pytest tests -q
```

```text
308 passed in 173.69s (0:02:53)
```

（298 项既有测试 + 10 项新增测试，0 failed。）

## 验收项逐项判定

### 1. 官方夹具取值与名称匹配：PASS

- `Observation.from_dict` 加载两个夹具；`visible_value(role="textbox", name="Biography")`：
  - `official_399_obs5_appended.json` → `t2_5adwlxvnI am a robot`；
  - `official_399_obs10_correct.json` → `I am a robot`。
- `visible_value_equals(expected="I am a robot")`：obs-5 判 `FAIL`，obs-10 判 `PASS`。
- 两批页面都含 `The biography was updated.`，判定不是靠“已保存”提示，而是靠字段值。
- 名称 `'Biography' required: False describedby: ...` 匹配 `Biography`。
- 同名 `StaticText 'Biography'` 标签行不被取值；无值子节点时返回空值而非标签。
- 证据：`tests/phase2/test_ax_official.py` 全部通过。

### 2. 空文本 `type` 与提交：PASS（序列化契约）/ PARTIAL（生产回传）

- 空文本 + `submit=true`：固定返回 `press [Enter]`，两种 `text_mode` 均测试；断言不等于 `type [1] [] [1]`。
- 空文本 + `submit=false`：本地 `ActionRejected`，不在平台产生动作；`correction` 为模型可读文本并写 `format_correction` 事件。
- 证据：`tests/phase2/test_official_input_contract.py` 全部通过。
- **未完成边界**：本线只允许改 `official.py`，`official_agent.py`/`effgen_bridge.py`/`tools.py`/`session.py` 不可改。当前官方入口在 `serialize_action` 抛 `ActionRejected` 后由 `OfficialAgent.step` 的通用异常分支结束任务，**不会把 `correction` 注入下一轮模型提示词**。本次已把错误文本放在 `ActionRejected.correction` 并落 `format_correction` 事件，供 P03/G3 接线直接使用；生产路径的“回给模型层”仍需集成负责人接 `_record_rejection` 同类通道。

### 3. 提交后取证标记：PASS

- `submit=true` 的动作产生新观察时，事件含 `post_submit_evidence=True`、`kind="post_submit_observation"`、提交动作 ID、目标、文本长度及新观察 ID。
- 空文本提交（实际发送 `press [Enter]`）同样记录标记，`submit_was_empty=True`。
- 非提交动作不产生标记。
- 未改 EvidenceStore 接口。
- 证据：`test_submit_observation_records_a_post_submit_evidence_marker`、`test_empty_text_submit_still_marks_the_resulting_observation`、`test_non_submit_observation_has_no_post_submit_marker`。

### 4. pytest 全量：PASS

`python -m pytest tests -q` → `308 passed`，0 failed。

### 5. 第一阶段 `WebAction` 序列化与既有测试：PASS

- 既有 `test_official_backend.py`、`test_official_agent.py`、`test_official_probe.py` 等全部通过。
- 非空 `type` 命令格式未变；多行拆分为 `type`/`press [Enter]` 的既有路径未变。
- 全量 298 项既有测试保持通过。

## 未完成项与边界

1. **空文本无 submit 的模型层回传未在生产入口验证**：受独占文件限制，无法修改 `official_agent.py`/`effgen_bridge.py`/`tools.py`；当前只保证本地拒绝、结构化 `correction` 和 `format_correction` 事件。需要在 P03/G3 由集成负责人把该拒绝接到现有格式纠正通道，或把空文本校验前移到工具 guard。
2. **未运行真实官方平台/真实模型**：本线只做离线夹具、假时钟和本地输入契约测试；605 的平台端具体异常原因未复现，本改动只是移除已知会触发无反馈的空 `type` 线上形态。
3. **提交后标记复用 `artifact_saved` 事件类型**：未新增事件类型（`events.py` 不在本线范围）。下游若要求独立事件类型，需要集成负责人扩展 `EVENT_TYPES`，当前可按 `result.post_submit_evidence` 过滤。
4. **截图自动留存依赖既有配置**：标记记录观察中 screenshot 的可用性和引用，但不强制下载；官方默认 `download_artifacts` 含 screenshot 时沿用原逻辑。
5. **未改 `verification/__init__.py`**：新模块按直接导入使用，未加入包级导出，避免越出本线文件范围。
