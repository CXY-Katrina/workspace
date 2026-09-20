# QuaRot 权重精度测试（2026-09-20）

状态：测试仍在运行。以下为已保存的 294 条结果快照，不是完整 GSM8K 精度。

## 环境与唯一配置变更

复用三台服务器的 `qwen235-acc-20260919` 容器，未重新安装软件。

- 197：Prefill TP16、代理和 AISBench 客户端。
- 198、159：Decode TP1 / DP32 / EP32，每台 16 个 rank。
- vLLM：v0.28.0，`2cf0a6915ce544dc493a0990f2ea38d81601128a`。
- vLLM Ascend：`e139b7d573d3769fd1407d5027d7d4831f5469f0`。
- 仅将 YAML 的 `model`、`benchmarks.acc.model_path` 改为 `/mnt/weight/Qwen3-235B-A22B-w8a8-QuaRot`。
- temperature=0.6、top_p=0.95、模型默认 top_k=20、batch_size=64、max_out_len=32768、thinking=true、ignore_eos=false。
- 实际 AISBench 命令包含 `--dump-eval-details --debug`。

三台机器的 68 个模型目录文件逐一 SHA256 一致。相对昨天官方模型，56 个权重分片全部不同；config、generation_config、tokenizer、量化描述等 8 个配置文件完全相同，详见 [文件对照](comparison-to-official.json)。QuaRot 目录的上游版本未独立确认，以本轮文件哈希为准。

## 阶段性结果

| 项目 | 结果 |
|---|---:|
| 数据集总量 | 1319 |
| 已落盘样本 | 294 |
| 正确 / 错误 | 241 / 53 |
| 局部精度 | 81.972789% |
| 长串重复字符 | 60 |
| 思考未闭合 | 56 |
| 重复短语规则命中 | 1 |
| U+FFFD（�） | 0 |
| 完整回答重复 / 题目重复 | 0 / 0 |

最早完成的 234 条是 232 正确、2 错误（99.145299%）；随后长回答返回，局部精度下降。先完成样本存在选择偏差，不能拿早期高分与 CI 完整精度直接比较。

60 条长串重复字符样本中有 9 条仍被原 AISBench 数字答案规则判对；得分正确不代表原始文本正常。重复短语样本 139 是思考中多次引用题干，最终输出 `answer:70`，应与长串感叹号退化区分。

## 具体异常与时间线

样本 12 在正常推理后接连续 **29,004 个感叹号**，没有闭合 `</think>`。关键衔接片段：

```text
Thus, answer is 13.

Therefore, the correct answer should be 13. However, depending on the interpretation, some might say 12. But according to strict calculation!!!!!!!!!!!!!!!!...
```

[完整原始回答](examples/sample-12.txt)、[原始样本 JSON](examples/sample-12.json)、[统计](examples/sample-12-analysis.json)。全文 42,040 字符；重编码为 7,390 tokens，**重编码长度不是服务实际生成 token 数**，重复字符可被 tokenizer 合并。

约完成 234 条后，64 条请求持续生成二十多分钟。服务并未退出，Decode 生成吞吐和 KV 缓存持续增长。Decode 指标快照记录 stop=235、length=64、abort=0、error=0；该快照与 294 条文件快照的时间不同，不应按同一集合逐条对应。`length` 证实已有请求达到长度上限。

服务日志同时出现动态 EPLB 专家重排：容器日志时间为 **06:58:29 UTC** 开始记录重排，**06:58:41 UTC** 记录完整专家权重更新周期完成。这里只记录时间关联，尚未做关闭 EPLB 的对照，不能认定根因。

## 证据与运行目录

- [阶段性完整证据归档](partial-evidence.tar.gz)：原始预测、按原后处理重算的明细、审计、样本和配置。
- [审计摘要](partial-snapshot/audit/output-audit.json)。
- [运行配置与版本](run-manifest.json)、[YAML](QWEN3_235B_PD.quarot.yaml)、[权重 SHA256](weights-host127.json)。
- [服务结束原因指标快照](finish-reasons-snapshot.json)。
- 宿主共享目录：`/mnt/share/qwen235-acc-20260919/run03-quarot-20260920`。
- 容器目录：`/workspace/run03-quarot-20260920`；原始评测输出在 `client/outputs/default/20260920_065412`。

测试仍在继续；当前数据不能当作完整 1319 条的最终精度结果。
