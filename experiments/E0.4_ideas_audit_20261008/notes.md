# E0.4：ideas pipeline 严格审核

结论：端到端工程链路可运行，但严格公式和机制对齐未通过。
完整分析：[审核报告](/home/henry/ot_idea/CellTracker/docs/reports/ideas_pipeline_audit_20261008.md)。

本次只增审核报告与留痕：没有算法修复、训练、GPU 消耗或官方指标重测。
15 项 CPU 探针验证 FGW 解析最优、熵、原始代价保存、历史消息通路、GT 标签、
encoder 对齐、生死语义、tracklet 单位和谱系等。单节点 kNN 疑点未复现。
既有 89+16 项测试均通过，说明这些行为未被原回归套件的断言约束。

不得把本次数值差作为性能提升或损害量；性能归因须在口径确定和修复后独立重训、
双序列官方 DET/SEG/TRA 与新设置噪声地板复验。原实验结果保持不变。
下一步先决定 raw/cond、生死阈值、tracklet 口径，再逐 bug 补回归与修复。
