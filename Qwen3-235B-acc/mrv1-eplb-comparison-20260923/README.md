# MRV1：QuaRot 权重 EPLB 对照结果

本目录汇总 Qwen3-235B-A22B-w8a8-QuaRot 权重在 MRV1（vLLM model runner V1）下，开启和关闭动态 EPLB 的测试结果、原始输出证据和复现脚本。

## 结果对照

| 配置 | 数据量 | 正确/错误 | 精度 | 输出质量 |
|---|---:|---:|---:|---|
| MRV1 + EPLB 开启 | 294 条阶段性快照 | 241/53 | 81.972789% | 替换字符：0；长串重复字符：60；思考未闭合：56；完整回答重复：0 |
| MRV1 + EPLB 关闭 | GSM8K 全量 1319 条 | 1273/46 | 96.512509% | 替换字符：0；长串重复字符：0；完整回答重复：0；重复短语命中 13 条，其中 11 条为正常题干/Wait 重复，2 条为持续循环 |

开启 EPLB 的 294 条是阶段性结果，存在完成时序偏差，不能与关闭 EPLB 的全量得分直接作严格精度比较。关闭 EPLB 的全量结果中，ID 1052 和 1078 分别出现算式 Wait 循环和 Hmm 循环，达到 32768 token 上限。

## 目录说明

- eplb-on/：开启 EPLB 的 294 条阶段性输出、审计、样例和证据包。
- eplb-off/：关闭 EPLB 的 1319 条全量输出、审计、异常回答和证据包。
- scripts/：两轮使用的配置、AISBench 启动脚本、服务启动脚本及评分脚本。

两轮均使用 AISBench 的 gsm8k_gen_0_shot_cot_chat_prompt 数据集，并带有 dump-eval-details 和 debug 参数。

公共参数：temperature=0.6、top_p=0.95、top_k=20（模型 generation_config）、thinking=true、max_out_len=32768、batch_size=64、seed=1024。两轮均使用三机 PD 拓扑：197 Prefill TP16，198/159 Decode DP32/EP32。

## 版本

- vLLM：2cf0a6915ce544dc493a0990f2ea38d81601128a（v0.28.0）
- vLLM Ascend：e139b7d573d3769fd1407d5027d7d4831f5469f0
- 基础镜像：sha256:49b1ebd1bb032910e01613bbb8f9115a087a00928550c852cc7a6d15224c7a97
- 数据集：GSM8K test，1319 条，SHA256 3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14

evidence.tar.gz 和 partial-evidence.tar.gz 内含 AISBench outputs、配置、审计明细及运行证据；解压后可复查逐条回答。
