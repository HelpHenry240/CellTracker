# 完整历史归档

此次按用户要求压缩合并历史实验。每个压缩包包含逐文件SHA256、字节数、权限、原始路径和源commit的内嵌清单；打包并逐项校验成功后才清理源目录。已保留的正式文件校验见 [retention_manifest](../retention_manifest.json)。

| 包 | 内容 |
| --- | --- |
| 00_source_and_documents.tar.gz | 整理前src/paperpipe/scripts/configs/docs/tests、README和原INDEX |
| 01_foundations.tar.gz | 基础、数据统计、指标自校验、经典基线与nnU-Net训练记录 |
| 02_historical_tracking.tar.gz | 早期E2–E4、A/B、P追踪与消融 |
| 03_historical_frontend.tar.gz | C阶段前端、旧真实检测与旧Oracle |
| 04_rebuild_audit_and_failed_runs.tar.gz | E0.4–E0.6审计/重建/被替代与中断探索，含未入库中间产物 |
| 05_formal_evidence_complete.tar.gz | E0.7–E1.0正式阶段整理前的全部证据 |

每包散列、大小、文件数、来源目录与commit见 [catalog.json](catalog.json)。大压缩包保存在本地且不入Git；若要迁移机器，应额外复制该目录下的压缩包。源码旧提交能恢复曾入库文件，不能替代对未入库文件的压缩包备份。

```bash
python scripts/manage_archives.py verify experiments/archives/04_rebuild_audit_and_failed_runs.tar.gz
python scripts/manage_archives.py restore experiments/archives/04_rebuild_audit_and_failed_runs.tar.gz   --out /tmp/celltracker_history_restore
```

恢复目标必须是新目录，验证失败不会解包；不覆盖当前文件。恢复后维持原仓库相对路径。若本地压缩包不可用，可从整理前提交426e41b提取原先入库的文件：

```bash
mkdir -p /tmp/celltracker_git_history
git archive 426e41b experiments docs paperpipe scripts | tar -x -C /tmp/celltracker_git_history
```

归档与删除的工程证据见 [E1.1](../E1.1_repository_cleanup_20261009/notes.md)。
