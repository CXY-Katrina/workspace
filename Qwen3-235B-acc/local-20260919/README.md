# 本地源码安装轮次：精度下降与输出异常

本轮按用户要求提前结束，**没有完整 1,319 条的最终 AISBench 得分**。冻结并保存了 1,314 条原始回答，使用本轮相同的 AISBench 后处理重新评分：**1,244 正确、70 错误，临时精度 94.6728%**。

未保存的样本编号为 187、380、434、901、1005（编号从 0 开始）。最后监控曾收到 1,316 条返回，但只有 1,314 条进入冻结文件；不将未落盘数据计入审计。即使剩余 5 条全部答对，全量准确率最多为 **94.6929%**，低于用例允许下限 **95.74%**。

## 输出质量

| 检查项 | 样本数 | 其中答案正确 / 错误 |
|---|---:|---:|
| 含 U+FFFD 替换字符 `�` | 497 | 478 / 19 |
| 重复 `</think>` 至少 3 次 | 18 | 12 / 6 |
| 重复 16-word 片段至少 8 次 | 11 | 5 / 6 |
| 长行重复至少 6 次 | 1 | 0 / 1 |
| `<think>` 未闭合 | 7 | 3 / 4 |
| 未提取到数字答案 | 6 | 0 / 6 |
| 整段回答完全重复（规范化空白后） | 0 组 | — |
| prompt 重复 | 0 组 | — |

各类标记有重叠，不可相加。共有 517 条触发至少一项检查。原始 JSON 严格 UTF-8 解码成功；`�` 是回答内真实存在的替换字符，不是本地终端编码造成的显示异常。重复短语也可能来自合理的推理复核，需阅读原文；下列样本则有明确异常。

- **[122](examples/sample-122.txt)**：反复生成 `Wait, wait.`，推理中的人名出现 `Sad�t`，最终 `answer:48` 正确。
- **[429](examples/sample-429.txt)**：英文数学题中突兀出现“游戏副本”，重复 7 次 `</think>`，输出 `answer:14`，标准答案为 13。
- **[543](examples/sample-543.txt)**：同样出现“游戏副本”和 7 次 `</think>`，但最终 `answer:$6` 正确。
- **[330](examples/sample-330.txt)**：563 次 `</think>`，答案提取为 NULL，标准答案为 31。
- **[569](examples/sample-569.txt)**：1,240 次 `</think>`，后续大量 `nonenonenone…`，结尾 `Mitchell sold �t`，提取为 NULL，标准答案为 21。
- **[368](examples/sample-368.txt)**、**[652](examples/sample-652.txt)**：末尾重复生成无意义的混合字符，提取出的数字分别为 888、15，标准答案分别为 18、75。

这证明“答案数字被判正确”不能代表整段输出质量正常。此次不根据这些现象直接断言具体算子、版本或通信模块是根因。

## 采样和后处理

| 参数 | 模型 generation_config | 本次生效值 / 来源 |
|---|---:|---|
| temperature | 0.6 | 0.6，AISBench 显式传入 |
| top_p | 0.95 | 0.95，AISBench 显式传入 |
| top_k | 20 | 20，AISBench 未传，由服务端继承 |

全部 33 个 API rank 均在日志中确认加载这些默认值，见 [sampling-evidence.json](sampling-evidence.json) 和 [generation_config.json](generation_config.json)。

后处理顺序：模型级 `extract_non_reasoning_content`（默认 `<think>` / `</think>`）移除配对的思考段；数据集级 `gsm8k_postprocess` 截断首个 `Question:` 后内容，提取最后一个整数或小数，找不到则为 NULL；`Gsm8kEvaluator` 与 `#### ` 后的标准答案比较，数值容差为 1e-6。未闭合思考段不会被正则移除。评估细节中的 `origin_prediction` 保留完整原文，审计基于该原文。

## 与最近一次 CI 的已确认差异

最近 CI 为 [2026-09-19 01:57 开始的任务](https://github.com/vllm-project/vllm-ascend/actions/runs/35376647863/job/105704758372)，得分 **96.28506444275966%（1,270 / 1,319）**。

| 项目 | 最近 CI | 本次本地 |
|---|---|---|
| vLLM | v0.28.0，`2cf0a6915ce54` | 相同 commit |
| Ascend | `c8addbb24fe8` | `e139b7d573d3`，向前 3 个提交 |
| 基础镜像 | `nightly-ci-main-a3`，CI 镜像 | 缓存的 8 月 22 日 nightly，源码重新安装 |
| CANN | 日志显示 9.1.0 | 9.1.0 |
| AISBench | 3.1.20260609 | 相同 tag，增加 `--dump-eval-details` |
| 服务拓扑/关键启动参数 | Prefill TP16；Decode TP1/DP32/EP32 | 相同；地址与本地路径做适配 |
| 网络/设备 | CI Pod 网络 eth0 | 三台裸机容器 host 网络 data0.173；驱动存在不同版本 |
| 权重 | 相同模型仓库名，CI 缓存内容哈希未提供 | 官方文件全量 SHA-256 校验通过 |

新增的 3 个 Ascend 提交分别为 KV cache dtype 重构 #15514、MRV2 PCP+DP 适配 #16853、QLI k-scale 地址溢出修复 #16824。它们只是已确认的代码差异，尚未做逐提交回退实验，不能直接认定哪一个导致退化。完整清单见 [ci-local-compare.json](ci-local-compare.json)，命令差异见 [ci-local-commands.json](ci-local-commands.json)。

固定权重/数据/客户端，使用用户指定的原始 `quay.io/ascend/vllm-ascend:v0.23.0-a3` 进行下一轮对照；该对照同时更换多个服务端组件，因此即使恢复，也只能先定位到软件栈差异，不能单独归因于 vLLM 或 Ascend。

## 可复查证据

- [环境、方法与边界](methods.md)
- [审计汇总](output-audit.json)、[标记样本上下文](flagged-samples.json)、[全部错题上下文](wrong-samples.json)
- [完整冻结输出与服务日志压缩包](run01-partial-evidence.tar.gz)：含 `saved-predictions.jsonl`、部分结果重算文件、完整原始 outputs 和 rank logs。
- [审计脚本](audit_outputs.py)：可直接对压缩包内 `run01/partial-final/partial-eval-details.json` 重新运行。

```bash
python audit_outputs.py run01/partial-final/partial-eval-details.json --output-dir audit-recheck
```

旧服务为切换镜像已停止，旧客户端容器仅用于运行下一轮 AISBench，不启动旧版本模型服务。
