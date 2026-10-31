# P 波 C 线报告：本地任务与统一 judge 分派

工作副本：`D:\Dwen\Alpha26fall\WebAgentNew\runtime\parallel\20261031-P1\laneC-judge`
范围：仅本地 case 与外部 judge 分派；未派发任何 worker，未修改主树。

## 修改文件清单

| 文件 | 改动 |
| --- | --- |
| `code/harness/src/webagent/testing/cases/extended/L15.yaml` | 新增本地 case“替换已保存字段值并纠正错误追加”。任务先保存 baseline，再替换为指定新值；支持正确替换、错误追加、追加后纠正三种终态。 |
| `code/harness/src/webagent/testing/cases/loader.py` | 新增 `manifest_path()` 与 `extended/` 命名空间解析；新增 `extended_case_ids()`、`list_extended_cases()`。`case_ids()`/`list_cases()` 仍只列冻结的 L01–L07。 |
| `code/harness/src/webagent/testing/cases/__init__.py` | 导出上述新增入口。 |
| `code/harness/src/webagent/testing/evaluation/judge.py` | 新增统一分派表 `_CHECKER_REGISTRY`、`register_judge_checker()`、`judge_checker_name()`、`resolve_judge_checker()`、`JudgeDispatchError`；`LocalJudge.evaluate()` 改为按 case 声明的 `hidden.judge.checker` 分派。注册 `l01/l02/l03` 既有检查器；新增通用 `field_replacement` 检查器，只读站点可观察状态与外部截图证据。`CheckResult` 增加可选 `gap` 字段，失败时给出机器可读缺口。 |
| `code/harness/src/webagent/testing/evaluation/hidden.py` | 新增 `saved_draft_texts()` 与 `current_draft()`，供外部 judge 读取保存历史与当前持久字段值。 |
| `code/harness/src/webagent/testing/evaluation/__init__.py` | 导出分派入口与错误类型。 |
| `code/harness/tests/phase2/test_evaluation_unified_judge.py` | 新增 5 个测试：统一分派/未知 checker 失败闭合、正确替换通过、错误追加失败并给出具体缺口、追加后纠正通过、判定只依赖可观察状态（answer 声明无效、站点停止后仍可判定、源码不引入 Agent 内部模块）。 |

未修改 `site/`、`state/`、`recovery/`、`verification/`、`control/`、`official_agent.py`、`effgen_bridge.py`、`tools.py`、`config.py`、CLI，也未修改主树。

## 关键设计

- 统一入口：`prepared.judge()` → `webagent.testing.evaluation.judge()` → `LocalJudge.evaluate()` → `resolve_judge_checker(case)`。
- case 通过 `hidden.judge.checker: field_replacement` 声明判定器，不再在测试脚本里写 case 专用判定；`LocalJudge` 上不存在 `_judge_l15`。
- `field_replacement` 只读：hidden DB 中的 `view_item`、`draft_saved` 活动、`drafts` 当前值，以及调用方提供的截图 bundle。`answer` 参数不参与判定。
- 检查项：`expected_count`、`target_viewed`、`baseline_saved`、`field_saved`、`field_value_exact`、`field_replaced`、`screenshot`、`submissions_zero`。
- 错误追加的具体缺口写入 `CheckResult.gap` 与 `detail`，例如：
  `leftover baseline 'Baseline profile note: ...' kept before the requested value; the value was appended, not replaced`。

## 关键命令与结果

环境：`D:\Dwen\Alpha26fall\WebAgentNew\.venv\Scripts\python.exe`

1. 确认导入本副本 src：

```powershell
$env:PYTHONPATH='D:\Dwen\Alpha26fall\WebAgentNew\runtime\parallel\20261031-P1\laneC-judge\code\harness\src'
python -c "import webagent; print(webagent.__file__)"
```

输出：`...\laneC-judge\code\harness\src\webagent\__init__.py`（确认非主树）。

2. 新测试：

```powershell
python -m pytest tests/phase2/test_evaluation_unified_judge.py -q
```

结果：`5 passed in 6.16s`

3. 现有 evaluation/cases 回归 + 新测试：

```powershell
python -m pytest tests/phase2/test_cases_manifest.py tests/phase2/test_evaluation_l01_l02.py tests/phase2/test_evaluation_l03.py tests/phase2/test_site_reset.py tests/phase2/test_evaluation_unified_judge.py -q
```

结果：`37 passed in 34.80s`（其中原有 32 项全部通过）。

4. 全量 pytest：

```powershell
python -m pytest -q
```

结果：`303 passed in 180.85s (0:03:00)`，0 failed。日志：`runtime/pytest_full.log`。

## 验收项逐项判定

| 验收项 | 判定 | 依据 |
| --- | --- | --- |
| 1a. 新 case 经统一 judge 入口，正确替换判通过 | 通过 | `test_correct_replacement_passes_the_unified_judge`：先保存 baseline，再替换并保存，`result.passed=True`，`field_value_exact`/`field_replaced`/`baseline_saved`/`screenshot` 均通过。 |
| 1b. 错误追加判不通过并给出具体缺口 | 通过 | `test_wrong_append_fails_with_a_concrete_gap`：保存 `initial+expected` 后 `passed=False`，`field_replaced` 失败，`gap` 包含 `appended, not replaced` 与 baseline 原文；DB 当前值确为追加值。 |
| 1c. 判定不读 Agent 内部状态 | 通过 | `test_judge_reads_only_observable_site_state`：伪造 `answer`（声称通过、带 agent_state/goals）不改变结果；站点停止后仍能从 DB + 截图判定；`judge.py` 源码不含 `webagent.session/state/official_agent/effgen_bridge/verification` 引用。 |
| 2. 现有 evaluation/cases 测试继续通过 | 通过 | 上述 37 项组合运行全部通过；原有 32 项无失败。 |
| 3. 全量 pytest 0 failed（导入指向本副本 src） | 通过 | 导入路径已确认；`303 passed`，0 failed。 |

## 未完成项与边界

1. **case 命名空间**：L15 放在 `cases/extended/`。原因是本线只允许新增测试文件，而 `test_cases_manifest.py` 将 L01–L07 作为精确集合断言；若把 L15 放进顶层会破坏现有测试。`load_case("L15")`、`start("L15", seed)` 均可正常解析，`list_extended_cases()` 可列出。主线合入时若允许更新 manifest 断言，可将 L15 移回顶层并删除 extended 回退；当前实现保留该回退以兼容后续扩展 case。
2. **判定器覆盖**：统一分派表目前注册 `l01/l02/l03` 与 `field_replacement`。L04/L05 仍在 `e1_helpers.py` 用测试脚本专用判定；迁移它们需要改测试脚本（不在本线允许范围），本次未做。新 case 已完全不依赖脚本专用编排。
3. **真实运行接线**：本线只交付本地 case、统一 judge 与测试，未跑真实模型/Agent 端到端；P03“同一循环纠错”属于 B 线，本文只证明外部 judge 能对追加给出缺口并可对纠正后的终态判通过。
4. **初始 baseline**：任务文本要求 Agent 先通过站点正常 Save draft 保存 baseline，不依赖测试私有注库，因此 case 可由正常 runner 执行；判定中的 `baseline_saved` 检查保存活动中确实出现过 baseline 原文。
5. 截图仅校验存在、PNG 签名与可选哈希，不判断截图内文字（沿用现有 D1 judge 边界）。
