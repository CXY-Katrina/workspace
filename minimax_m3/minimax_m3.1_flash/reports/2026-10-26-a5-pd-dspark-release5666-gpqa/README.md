# GPQA Diamond：release 5666 双机 PD + DSpark 原始 outputs

本目录归档 177.11.26.1 Prefill、177.11.27.1 Decode 使用 release 5666e6616（已合入 PR22）和 l32 权重完成的一轮 GPQA Diamond。未额外修改运行源码。

| 项目 | 结果 |
| --- | --- |
| 官方 AISBench 精度 | **168/198（84.848485%）** |
| 成功返回 / 失败 / 取消 | 198 / 0 / 0 |
| 草稿采信率，accepted / proposed | **24.670347%**，1,692,840 / 6,861,841 |
| 每轮接受草稿数 / 含目标 token 的平均输出长度 | 1.726924305 / **2.726924305** |
| 采样 | temperature=1，top_p=0.95，seed=1042 |
| chat_template_kwargs | reasoning_effort=max，thinking_mode=enabled |
| 并发 / 最大输出 / 上下文 | 16 / 81920 / 133000 |
| 主模型 KV / 草稿 KV | FP8 / BF16，草稿权重保持原生 MXFP8 |

## 完整 AISBench outputs

[原始输出目录](runs/gpqa-0001/aisbench/20261025_145310/) 完整保留 configs、predictions、results、logs 和 summary；运行目录原名为 aisbench。所有原始文件按字节复制。

- [全部 198 条原始预测](runs/gpqa-0001/aisbench/20261025_145310/predictions/minimax-pd-dspark/GPQA_diamond.jsonl)
- [官方逐题评分和原始回答](runs/gpqa-0001/aisbench/20261025_145310/results/minimax-pd-dspark/GPQA_diamond.json)
- [30 道错题的完整提示词及原始回答](wrong-answer-full-responses.json)
- [完整测试报告与两台服务器配置](report.txt)
- [完整逐题结果](runs/gpqa-0001/per-question.json)
- [采信率分实例审计及跨机 KV 传输记录](final-runtime-metrics-audit.json)
- [逐文件 SHA256 清单](manifest.json)

记录来源：177.11.27.1:/home/j00893003/minimax-dspark/cxy-pd-release5666-dspark-gpqa-20261025。测量开始于北京时间 2026-10-25 22:53，结束于 2026-10-26 00:02；归档内 AISBench 时间戳沿用原始目录名称。

## 评分口径与输出质量

官方成绩保持 **168/198**。11 条返回缺少推理结束标记，重新编码长度为 81912–81920 tokens，接近输出上限。ID2、89、116 虽未完成最终回答，官方后处理仍从推理文本中的第一个 ANSWER: X 匹配到正确选项，将其计为正确。

若补充采用“未完成最终回答统一计错”的口径，结果为 **165/198（83.333333%）**；这不替代或修改官方成绩。见 [口径审计](closed-answer-audit.json)。

198 个返回共 8,244,008 个字符，检出 0 个 NUL、1 个 Unicode 替换字符（ID181 的推理中，最终答案仍正确）。另保留中文片段、未闭合推理及重复行检查的 [具体上下文](output-quality-details.json)。原始预测没有保存 finish_reason，重新编码长度可能与原始生成 token 数有少量差异。

JSONL 文件按 LF 分隔记录；部分回答含合法 Unicode 行分隔字符，程序读取时应按换行符 LF 拆分，再解析 JSON。

本目录仅补充归档已完成测试的输出，不启动、重启或停止服务；历史四轮 GPQA 仍保留在 [原报告](../2026-10-25-a5-pd-dspark/README.md)。
