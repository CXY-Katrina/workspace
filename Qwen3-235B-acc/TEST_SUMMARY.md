# Qwen3-235B PD 测试总览

更新日期：2026-09-20。用例：`tests/e2e/nightly/multi_node/external_dp/config/QWEN3_235B_PD.yaml`。

**当前状态：第三轮 QuaRot 精度测试运行中，复用昨天源码安装的三台容器。已保存 294 条快照：241 正确、53 错误（局部精度 81.97%），60 条长串重复字符。尚无完整 1,319 条结果；前两轮已停止。**

第三轮仅更换为 `/mnt/weight/Qwen3-235B-A22B-w8a8-QuaRot`，其他评测参数保持不变。详见 [本轮报告与异常原文](local-quarot-20260920/README.md)。

## 1. 服务器与分工

三台服务器均通过 root 公钥认证登录，使用 Ascend A3，每台 16 个可见 NPU，共 48 个。

| 管理地址 | 数据网地址 / 网卡 | 主机名 | 分工 | 驱动版本 |
|---|---|---|---|---|
| 192.168.13.197 | 178.27.4.197 / data0.173 | host127 | Prefill TP16 / DP1；代理；AISBench 客户端所在节点 | 25.5.0 |
| 192.168.13.198 | 178.27.4.198 / data0.173 | ipb21b04c6.dynamic.kabel-deutschland.de | Decode TP1，DP rank 0–15 | 26.0.rc1 |
| 192.168.13.159 | 178.27.4.159 / data0.173 | ipb21b049f.dynamic.kabel-deutschland.de | Decode TP1，DP rank 16–31 | 26.0.rc1 |

两台 Decode 合计 DP32 / EP32。容器使用 host 网络、host IPC 和 NPU 设备权限，共享挂载 `/mnt`；本任务共享目录挂载为容器内 `/workspace`。

## 2. 容器清单及当前状态

| 服务器 | 容器名 | 容器 ID（短） | 用途 | 当前状态 |
|---|---|---|---|---|
| 197 | qwen235-acc-20260919 | 34b5e77cda81 | 第一轮源码安装环境；后作为第二轮 AISBench 独立客户端 | 运行中：第三轮 |
| 198 | qwen235-acc-20260919 | cc188442d6f6 | 第一轮 Decode | 运行中：第三轮 |
| 159 | qwen235-acc-20260919 | ffc36594c961 | 第一轮 Decode | 运行中：第三轮 |
| 197 | qwen235-acc-v023-20260919 | 543625899cde | 第二轮 Prefill + 代理 | 已停止 |
| 198 | qwen235-acc-v023-20260919 | b347439d7874 | 第二轮 Decode rank 0–15 | 已停止 |
| 159 | qwen235-acc-v023-20260919 | 9b45424ead67 | 第二轮 Decode rank 16–31 | 已停止 |

前两轮已结束。9 月 20 日复用源码安装容器执行第三轮；v0.23.0 容器在本轮开始前已停止。

服务端点：

- 代理：`http://178.27.4.197:1999`，健康检查 `/healthcheck`，聊天接口 `/v1/chat/completions`。
- Prefill：`http://178.27.4.197:7100`。
- Decode：`http://178.27.4.198:7100` 至 `:7115`，以及 `http://178.27.4.159:7100` 至 `:7115`。
- 当前服务模型名：`/mnt/weight/Qwen3-235B-A22B-w8a8-QuaRot`；前两轮为 `/workspace/model-official`。

9 月 19 日汇总时已核查 33 个服务 `/health` 和代理 `/healthcheck`，**34 / 34 返回 HTTP 200**。这只是可用性检查，不代表精度或文本质量通过。

## 3. 两轮镜像与软件版本

| 项目 | 第一轮：源码安装 | 第二轮：指定原始镜像 |
|---|---|---|
| 基础镜像 | 缓存的 2026-08-22 nightly | quay.io/ascend/vllm-ascend:v0.23.0-a3 |
| vLLM | v0.28.0，2cf0a6915ce544dc493a0990f2ea38d81601128a | 0.23.0+empty，0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665 |
| vLLM Ascend | e139b7d573d3769fd1407d5027d7d4831f5469f0 | 0.23.0，5cb98caaadeff42b5b62b996e34bb2aaa29d20fd |
| Python | 3.12.13 | 3.12.13 |
| torch | 2.10.0+cpu | 2.10.0+cpu |
| torch_npu | 2.10.0.post4.dev20260715 | 2.10.0.post4 |
| transformers | 5.14.1 | 5.5.4 |
| CANN 主版本 | 9.1.0 | 9.1.0 |
| 安装方式 | 构建 Rust frontend；源码 editable 安装；Ascend setup.py 两处 ROOT_DIR 改为 abspath | 保持镜像自带 vLLM/Ascend，不重新安装 |

第一轮基础镜像 ID：

```text
sha256:49b1ebd1bb032910e01613bbb8f9115a087a00928550c852cc7a6d15224c7a97
```

第二轮三台镜像 ID：

```text
sha256:7b485a6e417b682a49c43b3bd902f2a5c9b85d18101fcaebbc11822318031cb1
```

第二轮仓库摘要：

```text
quay.io/ascend/vllm-ascend@sha256:8e931f2a1908f4213ec31a349d5263a19c480e2e1e2dda801fdc35dd6ab279a5
```

三台均发起了指定镜像的 docker pull。197 下载大层较慢，随后停止其慢速拉取，导入 198 已拉取完成的同一镜像；三台镜像 ID 一致。

AISBench 两轮均使用 `v3.1-20260609-master`，commit `0da56eadb2ac85c31c2540f4f5b69af3ec5717a5`。第二轮使用第一轮容器作为独立评测客户端，模型服务均在新 v0.23.0 容器中；旧客户端不启动模型服务。第二轮代理与服务管理脚本来自当前测试脚本版本 e139，而非重新安装 v0.23.0 包。

## 4. 模型、数据与评测配置

- 模型：`vllm-ascend/Qwen3-235B-A22B-W8A8`。
- Modelscope revision：`a19b5fffb5a09ee570acfa0473146a76a1910a05`。
- 官方模型文件 66 个，含 56 个权重分片；全部通过官方清单 SHA-256 校验。
- 模型路径：容器内 `/workspace/model-official`。
- 原服务器上的近似命名 QuaRot 权重与官方清单的 56 个分片均不相同，未用于这两轮精度测试。
- GSM8K test 共 1,319 条，0-shot CoT；数据路径 `/workspace/datasets/gsm8k`。
- test.jsonl SHA-256：`3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`。

| 参数 | 实际设置 |
|---|---|
| temperature | 0.6，AISBench 显式传入 |
| top_p | 0.95，AISBench 显式传入 |
| top_k | 20，服务继承模型 generation_config；两轮均核查全部 33 个 API rank |
| thinking | true |
| batch_size | 64 |
| max_out_len | 32768 |
| stream / retry | false / 2 |
| 服务 seed | 1024 |
| max-model-len | 36864 |
| Prefill | eager；max-num-batched-tokens=16384；max-num-seqs=8 |
| Decode | async scheduling；max-num-batched-tokens=256；max-num-seqs=28；FULL_DECODE_ONLY；capture sizes=[6,12,18] |
| 精度基准 / 容差 | 96.74% / ±1 个百分点，允许区间 [95.74%, 97.74%] |

两轮都使用：

```bash
ais_bench --models vllm_api_general_chat_custom \
  --datasets gsm8k_gen_0_shot_cot_chat_prompt \
  --dump-eval-details --debug
```

后处理顺序：`extract_non_reasoning_content` 移除配对的思考段 → `gsm8k_postprocess` 提取最后一个数字 → `Gsm8kEvaluator` 与标准答案比较。文本检查针对后处理前的完整回答。第一轮下载权重期间做过 dummy 连通性预检，其输出完全排除于精度及文本质量统计。

## 5. 测试结果

| 轮次 | 已保存并复算样本 | 正确 / 错误 | 部分样本精度 | 完整性与结论 |
|---|---:|---:|---:|---|
| 第一轮源码安装 | 1314 / 1319 | 1244 / 70 | 94.6728% | 按用户要求提前切换镜像；非最终全量分数。剩余 5 条即使全部正确，全量上界也只有 94.6929%，低于用例下限 |
| 第二轮 v0.23.0-a3 | 552 / 1319 | 536 / 16 | 97.1014% | 按用户要求停止；对已保存输出按相同后处理规则离线复算，非 AISBench 最终全量分数；不能据此认定全量精度恢复 |

第一轮监控曾收到 1,316 条返回，但冻结文件只保存 1,314 条；仅对可复查的已保存回答评分。未保存编号为 187、380、434、901、1005，编号从 0 开始。

第二轮保存的 552 条均 success=true，包含 552 个不同样本 ID。停止时仍有大量样本未完成，已完成子集可能偏向短回答/较容易的题目，不能与 CI 全量得分或第一轮近全量得分直接比较。

### 原始输出质量

| 检查项 | 第一轮 1314 条 | 第二轮 552 条 |
|---|---:|---:|
| 含 U+FFFD 替换字符 `�` | 497 | 218 |
| 重复 16-word 片段至少 8 次 | 11 | 2 |
| `</think>` 至少 3 次 | 18 | 0 |
| 长行重复至少 6 次 | 1 | 0 |
| `<think>` 未闭合 | 7 | 2 |
| 未提取到数字答案 | 6 | 0 |

不同标记可能重叠。重复短语属于启发式标记，需要结合原文判断；正常推理中的复核也可能触发。第二轮的零计数仅针对已保存 552 条，不代表全量没有该现象。

第一轮另检查到：没有整段回答完全重复的组，也没有重复 prompt。497 条含替换字符的回答中有 478 条最终数字仍被判对，说明精度得分不能代替文本质量检查。

明确异常例子：

- 第一轮样本 122：反复生成 `Wait, wait.`，但最终答案 48 正确。
- 第一轮样本 429：英文数学题突兀出现“游戏副本”，重复 7 次 `</think>`，输出 14，标准答案 13。
- 第一轮样本 330：563 次 `</think>`，未提取到答案。
- 第一轮样本 569：1,240 次 `</think>`，大量重复 `nonenonenone…`，结尾 `Mitchell sold �t`，未提取到答案。
- 第二轮样本 32：输入为正常的 `10 dogs`，原始输出开头为 `John takes care of 1�t;� dogs. Wait, no, the question says 10 dogs.`；最终 `answer:35` 正确，仍含两个真实 U+FFFD 字符。

上述替换字符存在于严格 UTF-8 解码后的原始 JSON 字符串中，不是本地终端显示问题。两套服务镜像都观察到异常，目前未确定根因。

## 6. 历史 CI 对照及解释边界

2026-08-01 至 09-19 的历史统计覆盖 28 次实际执行：16 次出分，14 次通过、2 次未达标，12 次未出分；有效得分均值约 95.8870%，范围 92.1152%–96.5883%。未出分不按零分计算。折线图见 [accuracy-trend.png](accuracy-trend.png)。

最近指定的 [CI job 105704758372](https://github.com/vllm-project/vllm-ascend/actions/runs/35376647863/job/105704758372) 全量得分为 **96.28506444275966%（1270 / 1319）**。

与本地相比：

1. CI 直接使用 nightly-ci-main-a3 预装环境，`is_pr_test=false`，经 run.sh / pytest 执行。本地第一轮源码重装并直接调用测试函数；第二轮独立启动服务和客户端。
2. CI vLLM 与第一轮是同一个 v0.28.0 commit。Ascend 为 c8addbb24fe8，第一轮多了三个提交；第二轮则同时换成 v0.23.0 服务栈及多个不同依赖，并非单组件对照。
3. 服务拓扑、主要启动参数、AISBench 采样和评分基本一致。CI 与 e139 之间，相关测试脚本和代理没有代码改动。本地新增 dump 参数以及地址/路径适配，并把 Prefill 的 data-parallel-address 字面值 12321 改成节点 IP。
4. **本地权重符合此次官方清单，不等于已证明与 CI 缓存内容相同。** CI 实际权重、tokenizer、generation_config、数据哈希，以及完整镜像/依赖与各节点驱动尚未逐项对齐。
5. CI 未加 --dump-eval-details；96.xx 仅代表数字答案精度，不能证明当时不存在乱码或重复。

当前结论仅为：第一轮近全量结果低于门限；两轮都观察到文本异常；第二轮全量精度未知。尚不能归因于某个提交、采样参数、算子、权重或通信模块。

## 7. 文件位置与复查入口

服务器共享根目录：

```text
/mnt/share/qwen235-acc-20260919
```

容器内对应 `/workspace`。重要路径：

| 内容 | 容器内路径 |
|---|---|
| 原始 / 本地配置 | /workspace/QWEN3_235B_PD.original.yaml、/workspace/QWEN3_235B_PD.local.yaml |
| 第一轮冻结输出及审计 | /workspace/run01/partial-final/ |
| 第一轮原始 outputs | /workspace/run01/node-0/outputs/ |
| 第一轮服务日志 | /workspace/run01/rank-logs/ |
| 第一轮证据包 | /workspace/run01-partial-evidence.tar.gz |
| 第二轮原始 outputs | /workspace/run02-v023/client/outputs/ |
| 第二轮服务日志 | /workspace/run02-v023/rank-logs/ |
| 第二轮停止记录 | /workspace/run02-v023/STOPPED.json |
| 第二轮离线部分结果汇总 | /workspace/run02-v023/offline-partial-summary.json |
| 第二轮版本清单 | /workspace/run02-v023/run-manifest.json、/workspace/run02-v023/environment/ |
| 第二轮保留服务脚本 | /workspace/services_v023.py、/workspace/services_v023.sh |
| AISBench 启动脚本 | /workspace/benchmark_v023.sh |
| 审计脚本 | /workspace/audit_outputs.py |

仓库已保存的证据：

- [历史结果与折线图](report.md)
- [第一轮结果、异常原文和完整证据包](local-20260919/README.md)
- [CI 与本地测试方式的详细审视](ci-vs-local-review.md)

本次汇总只做只读状态核查和已保存输出的离线统计，没有恢复 AISBench，也没有向模型发送新的测试题目。

## 9. 第三轮：QuaRot 权重（2026-09-20，运行中）

复用第一轮的软件环境和容器，仅修改两个模型路径配置项。三机 68 个文件哈希一致；对比第一轮官方模型，56 个权重分片不同，8 个配置/分词文件相同。

294 条快照局部精度为 **241/294 = 81.972789%**，53 条错误。60 条存在长串重复字符、56 条思考未闭合；没有发现 `�`、完整回答重复或题目重复。样本 12 在正常推理后连续输出 29,004 个感叹号。服务指标记录 64 次 `length` 结束，表明部分请求达到输出长度上限。

这仍是部分结果。前 234 条的 99.15% 随长回答返回已降为 81.97%，不能用早期局部高分代表最终精度。动态 EPLB 更新与长输出现象时间接近，尚未验证因果。

测试继续运行，完整路径、配置、审计和原始样本见 [第三轮报告](local-quarot-20260920/README.md)。
