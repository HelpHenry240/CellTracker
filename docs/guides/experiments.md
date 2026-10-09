# 实验记录与历史恢复

任务入口统一为 [INDEX](../../experiments/INDEX.md)，阶段叙述统一为 [RECORDS](../../experiments/RECORDS.md)。原始官方结果的来源和散列登记在 `experiments/ledger.json`；它是证据清单，不能把重复拷贝计为独立实验。

新实验创建独立目录，保留config、command、env、git_commit、metrics、logs、figures、notes八件套；已有正式指标不覆盖。当前阶段以 `metrics_final.json` 为最终数字，历史执行日志中的绝对路径属于当时工作环境。

此次按用户要求一次性重建旧索引并合并重复文档；以后新增结果继续追加独立记录。已有正式指标、官方日志与小权重保持字节不变，校验见 `retention_manifest.json`；整理过程、迁移清单及测试见E1.1目录。

历史错误、偏离、中断和被替代实验完整压缩保存，没有抹去失败证据。压缩包逐文件校验通过才清理原件，包清单、SHA256、源commit和恢复命令见 [archives](../../experiments/archives/README.md)。

大型压缩包仅在本地保存，Git保留目录、散列和恢复说明。克隆后如需原始未入库文件，必须另行拷贝对应压缩包；旧提交只能恢复曾经入库的部分。原始数据与nnU-Net大权重仍使用既有数据备份。

中断恢复先查看git log/status、阶段目录和云端日志，再判断续跑；不要根据记忆重复训练。正式小权重是推理副本，不能恢复优化器；继续训练使用原云端完整断点。
