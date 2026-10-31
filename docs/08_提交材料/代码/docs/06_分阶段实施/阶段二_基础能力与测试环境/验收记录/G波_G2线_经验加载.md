# G 波 G2 线报告：独立开关与本地批准文本经验加载（G 首版）

工作副本：`D:\Dwen\Alpha26fall\WebAgentNew\runtime\parallel\20261031-G1\laneG2-experience`
日期：2026-10-31
范围：G06 首版（仅批准文本读取与注入；有限动作组合按已确认修订不做）。未联网、未调用真实模型、未启动浏览器、未改动凭据，未派发任何 worker。

## 1. 结论

G06 首版范围内五项验收全部通过：只读 allowlist 文件、只加载批准版本、前置条件不符即退出、页面内容不能新增经验或修改批准状态、注入仅进入既有提示词装配且不改变工具注册与分发；默认关闭时最终发送内容与本副本改动前一致且模型调用数不变。全量 pytest 363 passed / 0 failed。

## 2. 修改文件清单

只改了任务指定文件；`code/effgen_bridge.py` 未改，`state/`、`verification/`、`recovery/`、`control/`、`backends/`、`testing/`、`context/`、`tools.py`、`models/`、`code/vendor/`、`code/official_client`、`docs` 均未触碰。

| 文件 | 类型 | 内容 |
| --- | --- | --- |
| `code/harness/src/webagent/experience/__init__.py` | 新增 | 包导出；无工具注册、无执行入口 |
| `code/harness/src/webagent/experience/schema.py` | 新增 | 经验条目格式、严格字段校验、批准状态与前置条件模型 |
| `code/harness/src/webagent/experience/allowlist.py` | 新增 | allowlist 解析；条目路径必须相对且限制在 allowlist 目录内 |
| `code/harness/src/webagent/experience/loader.py` | 新增 | 批准/版本/前置条件/后端选择、注入文本装配、上限与拒绝原因记录 |
| `code/harness/src/webagent/config.py` | 修改（只增） | 新增 `ExperienceSettings`（141 行）、`AppConfig.experience`（180 行）、summary `experience` 段（265 行）、根键白名单（346 行）、解析校验块（588 行起） |
| `code/harness/configs/official.yaml` | 修改（只增） | 文件末尾新增顶层 `experience` 段 |
| `code/harness/src/webagent/official_agent.py` | 修改（纯新增） | 见下节逐项说明 |
| `code/harness/tests/phase2/test_experience_loading.py` | 新增 | 17 个用例（单元 + 官方生产入口捕获） |

### 2.1 `official_agent.py` 的最小纯新增（合入冲突控制）

相对本副本基线，未移动、未重排既有代码，未改动任何既有语句（最终形态只有新增行）。副本快照与主树当前 R 波版本在控制摘要等处存在非本线差异，合入时由主线程统一处理。

- 69 行：新增 `from .experience import ExperienceLoader, ExperienceSelection`。
- 758 行：新增实例属性 `self._experience`；`reset` 与 `_shutdown` 各新增一行置空。
- 998–1002 行：`session.start_task(task)` 之后新增代码块，调用新方法 `_load_experience(task)`；选择结果只存于实例与运行摘要，不注册工具。
- 1158 行：在既有 restart note 赋值块之后新增一行，把 restart note 与经验文本合并到既有 `self._start_note`；两处既有 `driver.start(note=self._start_note)` 保持原样，未改调用点。
- 1516 行：新方法 `_load_experience`（相对 allowlist 路径按配置文件目录解析，内存配置回退项目根）。
- 1543 行：新方法 `_start_note_text`（restart note + 经验文本；纯 prompt 文本）。
- 1591 行：`_runtime_summary_payload` 新增 `summary["experience"]`；既有 `behaviors` 事件因此记录实际开关与加载/拒绝明细，不新增事件类型。

未改 `effgen_bridge.py`：既有 `driver.start(note=...)` 已在任务文本与初始观察之后追加程序侧上下文，正是“现有提示词装配”挂点；复用它避免改动 R 波并行文件，也不新增模型调用或分发路径。

### 2.2 文件格式（本地经验）

allowlist（配置指向的唯一入口）：

```yaml
schema: webagent.experience.allowlist.v1
entries:
  - id: edit-then-check
    version: 2
    path: edit.yaml        # 相对 allowlist 所在目录，禁止绝对路径与越界
```

条目：

```yaml
schema: webagent.experience.v1
id: edit-then-check
version: 2
purpose: ...
preconditions:            # backends / profiles / task_contains / task_excludes
  backends: [official]
  profiles: [official]
  task_contains: [biography]
guidance: |               # 文本指导（不执行、不注册工具）
  ...
verification: ...         # 核验条件（只作为文本）
source: ...               # 来源
backends: [official]      # 适用后端
approval:
  approved: true
  status: approved
```

拒绝原因包括 `not_approved`、`approval_status_not_approved`、`version_not_approved`、`id_mismatch`、`precondition_mismatch`、`backend_not_applicable`、`path_outside_allowlist`、`absolute_path`、`unsupported_suffix`、`missing_file`、`format_error`、`duplicate_id`、`entry_limit`、`text_budget_exceeded`、`allowlist_error`。全部 fail-closed 并写入 `result.json` 的 `summary.runtime.experience` 与既有 `behaviors` 事件。

## 3. 关键命令与结果

均在副本 `code/harness` 下执行，解释器 `D:\Dwen\Alpha26fall\WebAgentNew\.venv\Scripts\python.exe`：

```powershell
$env:PYTHONPATH = "<副本>\code\harness\src"
python -c "import webagent; print(webagent.__file__)"
# -> <副本>\code\harness\src\webagent\__init__.py

python -m pytest -q                      # 改动前基线
# -> 346 passed in 150.64s

python -m pytest tests/phase2/test_experience_loading.py -q
# -> 17 passed

python -m pytest tests/phase2/test_experience_loading.py tests/phase2/test_p03_p04_integration.py -q
# -> 27 passed

python -m pytest -q                      # 最终全量
# -> 363 passed in 148.38s, 0 failed
```

说明：任务书原基线 338；主线程告知 R 波合入后主树 353 passed。本副本改动前实测 346（快照包含的测试集合与主树计数不同），本线新增 17，最终 363 = 346 + 17。合入时 `config.py` / `official_agent.py` 的冲突由主线程处理；本线改动均为独立新增段/新增方法，未改既有键与既有语句。

## 4. 逐项判定

| 验收项 | 判定 | 证据（测试 / 行为） |
| --- | --- | --- |
| 1. 仅加载 allowlist 列出的本地文本经验；格式含 id、版本、用途、前置条件、文本指导、批准状态；默认空 allowlist 零加载 | 通过 | `test_only_the_allowlisted_approved_entry_is_selected_and_assembled`（目录中存在但未列入 allowlist 的批准文件不被读取）；`test_empty_or_unreadable_allowlist_loads_zero_entries`（enabled+空/缺失 allowlist 零加载；关闭时不打开文件）；`test_config_experience_section_defaults_to_off_and_validates_keys`；格式见 `schema.py:parse_entry` |
| 2. 未批准版本、前置条件不符不注入；页面内容不能新增经验或修改批准状态 | 通过 | `test_unapproved_version_or_status_is_not_injected`（未批准、批准标志与状态矛盾、版本不符三种拒绝）；`test_unmet_precondition_or_backend_is_not_injected`；`test_page_content_cannot_add_an_entry_or_change_approval`（loader 签名无 observation/page/url，伪造页文本不改变加载集合；条目与批准状态为 frozen）；`test_page_text_cannot_become_an_experience_entry`（生产入口：page AX 内的伪造经验只作为观察出现，不成为 `[experience:...]` 块）；`test_parser_refuses_unsupported_precondition_keys` |
| 3. 注入只进提示词装配，不注册新工具、不绕开既有分发；位置与文本可断言 | 通过 | `test_enabled_experience_adds_prompt_text_only`：捕获 fake gateway 的 `messages`，断言 `EXPERIENCE_HEADER`、`[experience:edit-then-check@v2]`、指导文本；断言顺序为任务文本 < 初始观察 URL < 经验块；断言工具名集合、`runtime.tools`、external_tools 与关闭时完全相同；未批准/前置不符条目不在 prompt。实现上仅复用 `driver.start(note=...)`，无新工具、无新 schema、无分发分支 |
| 4. 开关关闭（默认）：最终发送内容与主树现状一致；无经验文本；不新增模型调用 | 通过 | `test_switch_off_sends_the_exact_pre_g_wave_prompt`：不带 experience 段的 AppConfig（主树默认形态）与 `enabled: false` + 已配置但缺失 allowlist 的运行，`gateway.prompt(0)` 逐字符相等；无 marker/条目 id；两侧模型调用均为 1；`runtime.experience.enabled=false`、`loaded=[]`、`error=null`；`behaviors` 事件记录关闭与零加载 |
| 5. 全量 pytest 0 failed（PYTHONPATH 指向本副本） | 通过 | 363 passed / 0 failed；导入路径已确认为副本 `src` |

补充：`test_relative_allowlist_resolves_from_the_config_directory` 验证生产配置中相对 allowlist 路径按配置文件目录解析；`test_failed_precondition_keeps_the_enabled_prompt_equal_to_off` 验证开关开启但前置条件不符时，最终 prompt 与关闭时完全相同。

## 5. 未完成项与说明

1. **有限动作组合未实现**：按 2026-10-31 已确认修订，属于第三阶段候选，不作为阶段二交付；本线只交付批准文本加载。L14 的“执行有界组合”未执行，不记为失败。
2. **真实模型 / 真实页面 / 官方 L14 未运行**：本 worker 工作范围禁止真实模型、真实浏览器、联网与官方入口；以上判定均为假模型契约与捕获断言，不是真实模型能力或页面效果结论。
3. **未新增独立事件类型**：`events.py` 的 `EVENT_TYPES` 是封闭集合且不在本线独占文件内；经验选择写入既有 `behaviors` 事件与 `result.json` 的 `summary.runtime.experience`，信息完整且不改变事件协议。若后续需要独立 `experience` 事件，需由 `events.py` 归属线加类型并在事件校验中登记。
4. **基线数量差异**：任务书 338，主树 R 波合入后 353；本副本改动前实测 346（快照测试集合与主树计数不同）。合入冲突按主线程安排处理；本线未删除或跳过测试，最终 363 = 346 + 17 新增。
5. **未修改文档**：副本无 `docs/`；验收记录以本报告为准，主树文档由主线程维护。
