# A 批环境准备

日期：2026-10-30。用户已批准阶段一，本轮仅 A 批。

- Python：3.12.9，项目环境 `.venv`。
- EffGen：1.3.0，上游 `https://github.com/ctrl-gaurav/effGen.git`，固定提交 `37de05a6b4ed01c90ee38b9c254a162927ca2ac2`，原检出工作树干净。
- 快照：`git archive` 导出全部 1795 个版本控制文件至 `code/vendor/effgen`；保留许可证、安装文件及测试。未复制 `.git` 或本地配置。原始文件哈希见 `A_上游文件哈希.json`。
- SDK：已核验新版 ZIP SHA-256 `ec1a6697656d5e1b6ba328ea4c69601a6c6e82ab9e63f428f23db71afa16feb4`，安全解压至 `code/official_client`；基准见 `A_SDK文件哈希.json`。
- 根目录已初始化本地 Git；未提交或发布。

## 安装命令

在项目根目录执行：

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -c code/vendor/effgen/constraints-cpu.txt -e code/vendor/effgen pytest pytest-timeout --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
```

首次安装在 pytest wheel 写入时遇 Windows 拒绝访问；使用同一命令追加 `--link-mode copy` 后成功（137 个包解析完成）。

SDK 无独立安装元数据，项目环境中 `webagent_official_sdk.pth` 指向 `code/official_client`；这是冻结源码的实际导入路径。已实测导入 `effgen`、`effgen.Agent`、`torch`、`third_party_agent`；EffGen 来自 vendored 路径，SDK 来自冻结目录，torch 为 `2.6.0+cpu`。

依赖版本见 `A_依赖版本.txt`。本轮不加载模型权重，不读取项目凭据，不进行 B／C 的真实运行。
