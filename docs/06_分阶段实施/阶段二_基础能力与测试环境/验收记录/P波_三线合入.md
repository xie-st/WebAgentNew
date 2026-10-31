# P 波三线合入记录（主线程）

日期：2026-10-31。三线在独立源码副本并行完成后，由主线程按文件归属审查合入。

## 派发与返回

| 线 | 作业 | 副本 | 结果 | 自报全量 |
| --- | --- | --- | --- | --- |
| A 输入／AX 适配 | `hub-8-mvvjm94q` | `runtime/parallel/20261031-P1/laneA-input` | done（77 步 / 89 次工具调用） | 308 passed |
| B 状态失效／恢复解耦 | `hub-9-mvvjm970` | `runtime/parallel/20261031-P1/laneB-state` | done（75 步 / 98 次） | 313 passed |
| C 统一判定 | `hub-10-mvvjm99b` | `runtime/parallel/20261031-P1/laneC-judge` | done（58 步 / 82 次） | 303 passed |

均使用 SiliconFlow DeepSeek-V4.1-Flash、effort high、fork_turns=none、worker 未再派发。测试经 `import webagent` 确认解析到各自副本 src。

## 归属核对（diff -rq 相对主树）

- A 线：`verification/ax_official.py`（新）、`verification/checks.py`、`backends/official.py`、测试×2 与官方观察夹具×2。无越界。
- B 线：`state/invalidation.py`（新）、`state/store.py`、`state/__init__.py`、`recovery/intake.py`（新）、`recovery/__init__.py`、测试×2。无越界。
- C 线：`testing/evaluation/judge.py`、`hidden.py`、`__init__.py`、`testing/cases/loader.py`、`__init__.py`、`cases/extended/L15.yaml`（新）、测试×1。无越界。
- 三线改动文件零重叠；主树抽查（checks.py、official.py、store.py）通过。

## 合入后主树验证

- 首次全量 3 failed：合入命令漏拷 `verification/checks.py`（多源单目标 cp 被吞），补拷后复跑。
- **最终：328 passed，0 failed（3分18秒）**；基线 298 + 新增 30。

## 遗留到集成作业（P03/P04）

1. A 线明示：空文本 type 无 submit 的 `ActionRejected.correction` 尚未接入生产回传通道（官方入口 `step` 的通用异常分支会直接终局），需由集成负责人接到既有格式纠正通道。
2. 官方入口仍未接 completion gate、request_finish、状态工具与行为开关（P03/P04 主体）。
