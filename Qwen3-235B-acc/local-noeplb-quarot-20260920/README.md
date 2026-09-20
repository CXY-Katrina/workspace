# QuaRot 关闭动态 EPLB：全量 GSM8K 结果

2026-09-20，按原 AISBench 方法完成 **1319/1319 条，1273 正确、46 错误，精度 96.512509%**（AISBench 显示 96.51）。全部请求成功，退出码 0；没有提前停止或跳过长尾题目。

关闭动态 EPLB 后未复现之前的连续感叹号，但仍有 **2 条持续短语循环，达到 32768 token 上限**。两题在 v0.23.0 保存的基线结果中均正常答对。因此，不能将所有重复问题都归因于 EPLB 开启，也不能把关闭 EPLB 当作完整修复。原先连续感叹号的具体原因仍未确认。

## 服务器、容器与版本

| 服务器 | 角色 | 本轮容器 |
| --- | --- | --- |
| 192.168.13.197 | Prefill TP16、代理端口 1999、AISBench | qwen235-acc-20260919 |
| 192.168.13.198 | Decode TP1，DP rank 0–15 | qwen235-acc-20260919 |
| 192.168.13.159 | Decode TP1，DP rank 16–31 | qwen235-acc-20260919 |

三机各 16 张 A3 NPU，Decode 合计 DP32/EP32。源码容器的基础镜像 ID 为 `sha256:49b1ebd1bb032910e01613bbb8f9115a087a00928550c852cc7a6d15224c7a97`。

- vLLM：`2cf0a6915ce544dc493a0990f2ea38d81601128a`（v0.28.0）。
- vLLM Ascend：`e139b7d573d3769fd1407d5027d7d4831f5469f0`。
- torch：2.10.0+cpu；torch-npu：2.10.0.post4.dev20260715；transformers：5.14.1。
- 权重：`/mnt/weight/Qwen3-235B-A22B-w8a8-QuaRot`。
- 对照基线镜像：`quay.io/ascend/vllm-ascend:v0.23.0-a3`，容器 `qwen235-acc-v023-20260919`，本轮未运行。

测试完成后，AISBench 已退出，三机源码推理服务保留。没有永久代码修复；此前 EPLB 临时打点已清除。

## 测试方法及唯一配置变更

相对原 QuaRot YAML，只将两个 Decode 配置的 `dynamic_eplb` 从 true 改为 false，其他内容字节一致。Prefill 原本即为 false。33 个 rank 启动日志均记录 `Dynamic EPLB is False`，本轮日志没有专家热度重排/完整专家更新记录。

配置 SHA256：`7adc22139bad1a41a1d4c8243fcdb71b9cf72f4faa5d4945552e69dacbcd769e`。

| 参数 | 值 |
| --- | --- |
| 数据集 | GSM8K 全量 1319 条，gsm8k_gen_0_shot_cot_chat_prompt |
| batch_size / max_out_len | 64 / 32768 |
| temperature / top_p | 0.6 / 0.95 |
| top_k | 继承模型 generation_config 的 20 |
| thinking / ignore_eos | true / false |
| 服务 seed | 1024 |
| 评测后处理 | extract_non_reasoning_content → gsm8k_postprocess → Gsm8kEvaluator |
| AISBench | 原固定客户端，保留 --dump-eval-details |

数据集 SHA256：`3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`。

```bash
ais_bench --models vllm_api_general_chat_custom \
  --datasets gsm8k_gen_0_shot_cot_chat_prompt --dump-eval-details --debug
```

最终文件验证了 ID 0–1318 完整且唯一。与基线共同保存的 1314 条，原始 prompt 和 gold 逐条一致。

## 精度对照

| 结果范围 | 正确 / 总数 | 精度 |
| --- | ---: | ---: |
| 本轮关闭 EPLB，全量 | 1273 / 1319 | 96.512509% |
| 本轮关闭 EPLB，仅双方共同样本 | 1269 / 1314 | 96.575342% |
| v0.23.0 基线实际保存样本 | 1271 / 1314 | 96.727549% |

共同样本中，本轮相对基线 11 题由错变对、13 题由对变错，净少 2 题，差 0.152207 个百分点。采样和软件依赖存在差异，这次对照不证明该小幅得分差异具有统计显著性。基线缺失 ID 为 1039、1187、1254、1288、1303，不能把基线部分得分当作全量分数。

此前源码环境出现连续字符退化的 60 道题，本轮全部返回，54 道答对，均无连续字符退化。

## 输出审计与异常原文

- U+FFFD `�`、可疑乱码、异常控制字符：0。
- 连续 50 个以上相同非空白字符、完整回答重复、重复 prompt：0。
- 短语重复启发式命中 13 条：11 条为题干引用/普通 Wait 推理重复，忽略；2 条为持续循环，未闭合思考且没有最终答案。
- 两条异常重新用相同模型 tokenizer 编码，均为 32768 token。Decode 指标记录 2 次 length 结束；指标是整个服务生命周期计数，含就绪探测，不能直接作为数据集请求总数。

样本 ID 为从 0 开始的原始数据集索引。

### ID 1052：算式循环

输出 47751 字符，其中 `90 + 100 = 190? No. Wait,` 出现 1746 次。末尾一直重复：

```text
90 + 100 = 190? No. Wait, 90 + 100 = 190? No. Wait,
90 + 100 = 190? No. Wait, 90 + 100 = 190? No. Wait, ...
```

无 `</think>` 或最终答案。原后处理从截断文本提取到 190，标准答案 110。基线同题正常结束于 `answer:110`。

### ID 1078：Hmm 循环

输出 82001 字符，`Hmm.` 出现 16309 次：

```text
So 10 toothpicks per week times 12 weeks. That's 10*12 = 120 toothpicks.
...
Hmm. Hmm. Hmm. Hmm. Hmm. Hmm. Hmm. Hmm. ...
```

无 `</think>` 或最终答案。原后处理提取到 120，标准答案 8。基线同题仅 981 字符，正常结束于 `answer:8`。这里是直到长度上限都未恢复的循环，不按普通 Wait 重复忽略。

## 证据与结论边界

- [完整证据包](evidence.tar.gz)：原 AISBench outputs、配置、版本清单、日志、指标、审计脚本和异常/基线原文。
- [输出审计](output-audit.json)、[人工复查汇总](review.json)、[共同样本比较](paired-comparison.json)。
- [ID 1052 完整异常回答](abnormal-1052.txt)、[ID 1078 完整异常回答](abnormal-1078.txt)。
- 服务端目录：`/mnt/share/qwen235-acc-20260919/run05-noeplb-quarot-20260920`；容器内对应 `/workspace/run05-noeplb-quarot-20260920`。
- 最终结果：`client/outputs/default/20260920_102723/results/vllm-api-general-chat/gsm8k.json`。
- 最终结果 SHA256：`b1d3077de4f69326b30bd787d37007f90d6c4fd0ac3485ba47825a2c12da43e1`。

本轮完成了关闭 EPLB 的原方法全量验证。得分回到 96.xx，但仍存在基线同题未出现的循环。后续应优先用 1052、1078 两题区分非 EPLB 路径的重复问题，并独立追踪原先连续感叹号现象。当前没有经验证的根因或代码修复。
