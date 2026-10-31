# 迁入工程验证

日期：2026-10-31。工作目录：`docs/08_提交材料/代码/`。后续补丁直接修改该工程。

| 验证 | 结果 | 证据 |
| --- | --- | --- |
| 初始文件复制 | 2134 个文件已核对 SHA-256 | 工程根目录 `delivery-manifest.json` |
| 独立环境 | Python 3.12.9；141 个安装包一致性检查通过 | [依赖检查](依赖检查.txt)、[版本清单](依赖版本.json) |
| 导入与配置 | harness、EffGen、官方 SDK 均从迁入目录导入；3 份 YAML 均可加载且无重复键 | [导入与配置](导入与配置.json) |
| demo 实际流程 | 离线脚本驱动实际 Chromium 无界面运行，通过面板启动 CLI 子进程，完成检索、详情、输入、草稿预览、取证及回答；run_id 为 `demo-20261031-094414-f9cc` | [结果与事件索引](demo_smoke.json)、[启动截图](01_迁入面板启动.png)、[完成截图](02_迁入面板完成.png)、[执行脚本](smoke_demo.py) |
| 完整 harness 回归 | **427 passed，1 skipped**，用时 223.15 秒；跳过项是未安装的 browser-use 试验后端 | [回归输出](回归输出.txt)、[JUnit XML](回归.xml) |
| 官方网络接入与真实模型调用 | 本次未启动；实际使用本队最终连接文件、服务地址与密钥后单独联调 | 启动方式见工程根目录 `运行说明.md` |

迁入后的配置修正：删除 `official.yaml` 重复的同值 `parallel` 段；`real.yaml` 模型标识合并原本机配置的 `qwen3.6-35b-a3b`；官方 `.cmd` 入口改为本工程虚拟环境及 `webagent.official_agent:OfficialAgent`。其余实际启用开关保持迁入时的值。

完整回归命令，从迁入工程根目录执行：

```powershell
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe -m pytest code\harness\tests -q --junitxml runtime\迁移验证\回归.xml
```

本次验证的是迁移后安装、导入、配置与本地执行链。正式评测结果另按官方平台记录。
