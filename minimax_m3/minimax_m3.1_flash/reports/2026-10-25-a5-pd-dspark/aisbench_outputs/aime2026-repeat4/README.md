# AIME2026 repeat4 AISBench 原始输出

归档来源：177.11.26.1:/home/j00893003/minimax-dspark/cxy-pd-pr19723-20261025/runs。这是历史测试的补充归档。

| 轮次 | AISBench 目录时间戳 | 正确 / 总数 | 返回成功请求 | 原始输出 |
| --- | --- | --- | --- | --- |
| 1 | 20261024_212941 | 27 / 30 | 29 | [outputs](aime2026-0001/aisbench/20261024_212941/) |
| 2 | 20261024_214513 | 28 / 30 | 29 | [outputs](aime2026-0002/aisbench/20261024_214513/) |
| 3 | 20261024_220737 | 28 / 30 | 29 | [outputs](aime2026-0003/aisbench/20261024_220737/) |
| 4 | 20261024_223305 | 29 / 30 | 29 | [outputs](aime2026-0004/aisbench/20261024_223305/) |

共 **112/120（93.33%）**。每轮均按当时要求，在只剩 ID15 时停止该请求，因此共有 116 条实际返回，ID15 四次未返回均计入 120 的分母，未补造回答。原始推理进程退出码 143 对应上述取消操作，评估退出码均为 0。

每轮保留完整 aisbench/时间戳 目录树：configs、predictions（含临时预测文件）、results、logs、summary。另附原始运行配置、取消记录、逐题判定、得分和进程退出码，共 88 个原始文件；这些文件按字节保留。[manifest.json](manifest.json) 记录每个文件的大小、SHA256 和原始路径，可核验归档完整性。

完整回答位于 predictions/minimax-pd-dspark/aime2026.jsonl；逐题评分与原始回答位于 results/minimax-pd-dspark/aime2026.json。归档中的服务器路径属于当时环境。
