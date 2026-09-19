# QWEN3_235B_PD 精度配置历史核对

核对区间：2026-08-01 至 2026-09-19；时间以下均为北京时间。资料来自 GitHub 仓库文件与 commits API，未以运行结果推断配置历史。

## 统计口径

- 2026-09-07 10:12，该用例从 `tests/e2e/weekly/multi_node/external_dp/config/QWEN3_235B_PD.yaml` 原样移动到 `tests/e2e/nightly/multi_node/external_dp/config/QWEN3_235B_PD.yaml`，文件内容增删均为 0。8 月历史必须包含 weekly 工作流，不能仅搜索当前 nightly 路径。[迁移提交](https://github.com/vllm-project/vllm-ascend/commit/015c56ae577813fa7b99d920af65d2c108a21de9)
- 对新旧路径全部区间内提交逐一检查：`benchmarks.acc` 未发生实质变更。唯一精度数据集始终为 `vllm-ascend/gsm8k`，配置 `gsm8k_gen_0_shot_cot_chat_prompt`；基线 96.74，阈值 1，最大输出 32768，batch size 64，temperature 0.6，top_p 0.95，thinking true，ignore_eos false。[区间初配置](https://github.com/vllm-project/vllm-ascend/blob/9c624b98bf3882368a281488d79eea5f7a98e4c0%5E/tests/e2e/weekly/multi_node/external_dp/config/QWEN3_235B_PD.yaml)；[区间末配置](https://github.com/vllm-project/vllm-ascend/blob/65047dc4f814ddc106e2d4a75fa5cc3f04649b58/tests/e2e/nightly/multi_node/external_dp/config/QWEN3_235B_PD.yaml)
- `tools/aisbench.py::_accuracy_verify` 使用闭区间 `baseline - threshold <= acc_value <= baseline + threshold`，因此通过区间为 **95.74–97.74**。这是绝对百分点，非相对 1%。GSM8K 分数取 AISBench 输出 CSV 第一行最后一列。[校验源码](https://github.com/vllm-project/vllm-ascend/blob/65047dc4f814ddc106e2d4a75fa5cc3f04649b58/tools/aisbench.py)

## 服务配置变化

| 日期 | 变化 | 来源 |
|---|---|---|
| 08-04 19:47 | 增加 `VLLM_USE_MODELSCOPE=true` | [9c624b98](https://github.com/vllm-project/vllm-ascend/commit/9c624b98bf3882368a281488d79eea5f7a98e4c0) |
| 08-08 14:18 | `ASCEND_BUFFER_POOL` 替换成 `ASCEND_ENABLE_USE_FABRIC_MEM` | [f82d49c8](https://github.com/vllm-project/vllm-ascend/commit/f82d49c8f960ba83380a3966ee2eaf72f8ee4be9) |
| 08-09 10:50 | 删除上一项新增的 fabric memory 环境变量 | [e6022c0d](https://github.com/vllm-project/vllm-ascend/commit/e6022c0d982bae648eba0092ab419a8bd5cbcd03) |
| 08-25 20:09 | chunked prefill 改为 vLLM CLI；旧 EPLB 配置迁移到嵌套 `eplb_config` | [6fd6ea16](https://github.com/vllm-project/vllm-ascend/commit/6fd6ea161b904ee4b379f6a18bc29d3e076cf25a) |
| 08-29 09:50 | MLAPO 环境变量迁移到 `additional_config.enable_mlapo` | [895ca078](https://github.com/vllm-project/vllm-ascend/commit/895ca078b2ce52e4ec40287776233e693aa592d7) |
| 09-07 10:12 | weekly → nightly，内容不变 | [015c56ae](https://github.com/vllm-project/vllm-ascend/commit/015c56ae577813fa7b99d920af65d2c108a21de9) |
| 09-18 00:06 | Decode 显式启用 Model Runner V2，迁移 EPLB 启动配置 | [c7ca0b67](https://github.com/vllm-project/vllm-ascend/commit/c7ca0b676b9668535f467b78cff1274a4ddb63b2) |
| 09-18 17:49 | 回退上一项 MRV2 变化 | [65047dc4](https://github.com/vllm-project/vllm-ascend/commit/65047dc4f814ddc106e2d4a75fa5cc3f04649b58) |

## 评测代码变化的影响

区间内 `tools/aisbench.py` 的变动分别为性能阈值支持、LongBenchV2 分数分支、spec decode 测试支持、可选 reasoning_effort 以及后者回退；未改变该用例 GSM8K 的取分及阈值公式。[08-24](https://github.com/vllm-project/vllm-ascend/commit/9fdcd6e77204a74ce8c4ea1f4e6dd3c1f78242a9)、[08-28](https://github.com/vllm-project/vllm-ascend/commit/656b35ea4e48aa3c45097ec7a97c7716cb14eb3b)、[09-09](https://github.com/vllm-project/vllm-ascend/commit/ad86348b0cb2324d643df6a496f2d9c3879e4481)、[09-18 启用](https://github.com/vllm-project/vllm-ascend/commit/c7ca0b676b9668535f467b78cff1274a4ddb63b2)、[09-18 回退](https://github.com/vllm-project/vllm-ascend/commit/65047dc4f814ddc106e2d4a75fa5cc3f04649b58)。

因此可以以同一数据集/基线绘制历史折线，但服务配置、依赖和代码提交并非固定控制变量；不能单凭折线将波动归因到某一改动。未出分记录应记为缺失，不应画成零分。以上配置核对不能独立证明各次运行使用的模型文件和外部 AISBench 安装版本完全一致。
