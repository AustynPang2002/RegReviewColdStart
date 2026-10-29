# 最终运行代码汇总

本目录是可以上传到 GitHub 的独立代码目录。英文主说明见 [README.md](README.md)。论文正文仍单独保存在 Overleaf 包中，本包不包含论文 PDF 或作者联系邮箱。

## 包含内容

- 数据审计、时序划分、历史特征、FT-Transformer、文本融合及竞争风险损失。
- 六种模型配置、五个种子的主实验，以及原始基线和验证集调参基线。
- 药名组配对 bootstrap、删失敏感性、校准诊断和批准时间核验。
- 辅助损失敏感性、设备交叉实验的运行与分析脚本。
- 12 项逻辑测试及实验依赖说明；不含测试结果或研究数据文件。
- 实际使用的第三方 RTDL 包源码及其 Apache 2.0 许可证。

## 使用方式

1. 解压后，将该目录内的文件上传到新建的 GitHub 仓库。也可以按英文 README 中的 Git 命令推送。尚未创建或上传远程仓库。
2. 在 Python 虚拟环境中安装 `requirements.txt`，再运行 `python -m pytest tests -q`。
3. 将 `config.example.json` 复制为 `config.json`，填入有权使用的数据库 ZIP 路径和内部 Excel 文件名。
4. 依照英文 README 的顺序运行数据准备、主实验、基线和审计脚本。完整主实验是 30 次拟合，默认不会自动启动。
5. Ascend NPU 环境请遵循 `requirements-npu-analysis.txt` 的说明，先安装兼容的驱动、CANN、torch 和 torch_npu，不要用通用依赖覆盖 NPU 专用 torch。

无需原始数据即可运行使用内存合成样例的逻辑测试。绘图须先自行生成校准汇总，再运行 `python plot_calibration.py --input results/final_diagnostics/calibration_bins.csv`。重新训练完整模型需要原始数据；不能将演示或小样本运行作为论文结果。

## 已排除的内容

未打包原始数据库、下载记录、汇总结果、CSV 数据文件、逐条申请预测、训练权重、序列化拟合对象、服务器连接信息、密码和本机虚拟环境。数据来源见 [DATA_SOURCES.md](DATA_SOURCES.md)。`.gitignore` 已排除这些常见文件，但不会自动保护被强制添加或已被 Git 跟踪的文件。

本包保留现有研究的局限：观测截止日期和历史字段可用性未核实，资格排除依赖结局，原始敏感性实验存在设备混杂。换设备或软件版本后，数值不保证逐位一致。具体实验路径和四次设备交叉拟合的说明见 `docs/EXPERIMENTS.md`。

项目原创代码的开源许可证尚未由权利人指定，因此未擅自添加 MIT 或 Apache 授权；公开托管与开源授权不是同一件事。见 `LICENSE.md`。第三方 RTDL 代码继续适用其自己的许可证。
