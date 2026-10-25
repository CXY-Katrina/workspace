# MiniMax M3.1 Flash A5 PD 分离与 DSpark 精度测试报告

正式结果快照：**2026-10-25T12:28:03+08:00（北京时间）**。本次使用两台 Ascend 950DT 服务器进行 PD 分离部署，主模型为 W8A8 / MXFP8 / QuaRot 权重，开启 DSpark，每步草稿长度为 7。

**AIME2026 repeat4：112/120，93.33%；GPQA Diamond 已完成四轮：671/792，84.72%。** 对应加权草稿采信率分别为 **28.59%** 和 **24.34%**。

本报告固定记录已完成的四轮 AIME 与四轮 GPQA。随后按用户要求停止 DSpark 循环：GPQA 第五轮停在 80/198，部分结果不计入正式成绩；另启动关闭 DSpark 的两轮 GPQA 对照测试，其结果独立记录，不混入下表。

四轮 AIME2026 的完整 AISBench 原始 outputs（含全部返回、评分和配置）见 [outputs 归档](aisbench_outputs/aime2026-repeat4/README.md)。逐文件 SHA256 已随归档保存。

## 两台服务器与容器

| 项目 | Prefill | Decode |
| --- | --- | --- |
| 宿主机 | `177.11.26.1` | `177.11.27.1` |
| 计算网络 | `197.13.50.1` / `enp39s0f0` | `197.13.51.1` / `enp39s0f0` |
| 容器 | `cxy-vllm-ascend-0.30.0rc2-20261024` | `cxy-minimax-pd-decode-20261025` |
| 容器 ID | `dd2011d761a0` | `1d97280dcb40` |
| 镜像 | `cxy-vllm-ascend:handoff-current-20261024-235017` | `cxy-minimax-pd:decode-existing-20261025` |
| NPU | 8 × Ascend950DT | 8 × Ascend950DT |
| 拓扑 | DP1 × TP4 × PP2，开启 EP | DP2 × TP4 × PP1，开启 EP |
| 服务端口 | `31050` | `31060` |
| vLLM Ascend 源码 | `/workspace/vllm-ascend` | `/vllm-workspace/vllm-ascend` |
| vLLM 源码 | `/workspace/vllm` | `/vllm-workspace/vllm` |
| 宿主机运行目录 | `/data/cxy/vllm-ascend-0.30.0rc2-20261024/pd-pr19723-20261025` | `/home/x00939195/cxy-pd-pr19723-20261025` |

两端使用 host network / host IPC。PD 代理位于 Prefill 容器，端口 `8009`；请求入口为 `http://177.11.26.1:8009/v1/chat/completions`，模型名为 **`minimax-m3.1-flash`**。实际跨机 KV 传输验证记录了 Decode 端 914 个来自 `external_kv_transfer` 的 prompt token，并验证了客户端取消后两端回到空闲状态，见 [KV 传输证据](data/runtime-kv-transfer-proof.json)。

容器与源码状态在结果快照前完成核验，两端 `/health` 均返回 200；之后为关闭 DSpark 做过停启。完整容器 ID、镜像 ID、只读挂载及版本信息见 [Prefill 审计](data/prefill-runtime-audit.json) 与 [Decode 审计](data/decode-runtime-audit.json)。

## 权重与代码版本

| 项目 | 路径或版本 |
| --- | --- |
| Prefill 主模型，宿主机与容器内相同 | `/data/l00845740/minimax_m3dot1_flash_w8a8_mxfp8_quarot_l32` |
| Decode 主模型，宿主机与容器内相同 | `/mnt/weight/minimax_m3dot1_flash_w8a8_mxfp8_quarot_l32` |
| Decode NFS 源 | `197.13.50.1:/data/l00845740/minimax_m3dot1_flash_w8a8_mxfp8_quarot_l32`，`nfs4`、只读 |
| DSpark 权重，相对主模型目录 | `dspark/model-dspark-00001.safetensors` |
| QuaRot 权重，相对主模型目录 | `optional/quarot.safetensors` |
| vLLM Ascend 分支 | `releases/v0.30.0rc2` |
| vLLM Ascend commit，两端一致 | `accf5d0fc6005fe89dd909e656848b696bbc18f5` |
| vLLM Ascend tree | `1ee636adf349e6cd6cba147713cc2225c521e6b8` |
| vLLM commit，两端一致 | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| 启动脚本文档快照，PR19723 | `bd26c28e34cb925c2b27565b0f4e2ca86d12d685` |
| PD 代理源码快照 | `0058d5512cfb7d26668284fc392fb5a5ca1520e8` |

两端 vLLM 与 vLLM Ascend 源码工作区均为干净状态，Ascend commit 已包含 PR19。Prefill 容器的安装包元数据仍显示 `0.30.0rc2+pr19.dsparkfix`，Decode 显示 `0.30.0rc2`；实际源码以相同的 commit/tree 和工作区核验结果为准。vLLM 包元数据为 `0.30.0+empty`。

主模型目录均以只读方式挂入容器；Decode 使用 Prefill 同一份权重的只读 NFS 挂载。本轮测试未改写权重，草稿权重保持原生 MXFP8；草稿 **KV cache** 使用 BF16，不代表对草稿权重反量化。`quant_model_description.json` 的 `optional.quarot.rotation_map.global_rotation` 为 `optional/quarot.safetensors`，见 [路径核验](data/rotation-file-evidence.json)。

环境：CANN 9.2.0、Python 3.12.13、Torch 2.10.0+cpu、torch-npu 2.10.0.post7、Transformers 5.14.1。A5 使用当前 release 的 CANN FA3 路径，未安装第三方 `flash_attn_npu_3` 包。

服务脚本来源：[PR19723](https://github.com/vllm-project/vllm-ascend/pull/19723)，[固定版本文档](https://github.com/vllm-project/vllm-ascend/blob/bd26c28e34cb925c2b27565b0f4e2ca86d12d685/docs/source/tutorials/models/MiniMax-M3.1-Flash.md)。归档的 [Prefill 脚本](scripts/dspark/run_pd_prefill.sh) 和 [Decode 脚本](scripts/dspark/run_pd_decode.sh) 在替换模型路径、LOCAL_IP、NIC_NAME 后与该文档一致，见两端 [启动核验](data/prefill-start-audit.json) / [启动核验](data/decode-start-audit.json)。

## 服务与测试配置

| 配置 | 值 |
| --- | --- |
| 上下文长度 | `133000` |
| 最大输出长度 | `81920` |
| 并发 | `16` |
| temperature / top_p | `1` / `0.95` |
| 请求 chat_template_kwargs | `reasoning_effort=max`、`thinking_mode=enabled` |
| 请求 seed | AIME：42、43、44、45；GPQA：1042、1043、1044、1045 |
| DSpark | `method=dspark`，`num_speculative_tokens=7`，`draft_sample_method=probabilistic`，草稿 `enforce_eager=true` |
| 主模型 / 草稿 KV cache | `fp8` / `bfloat16` |
| Prefill 执行 | eager，chunked prefill，prefix caching |
| Decode 图模式 | `FULL_DECODE_ONLY` |
| Decode ascend_compilation_config | `fuse_norm_quant=false`，`fuse_qknorm_rope=true` |
| 多流与共享专家 | `multistream_overlap_shared_expert=true`，`enable_shared_expert_dp=true` |
| PD KV 连接 | `MooncakeConnectorV1`，`use_ascend_direct=true` |

使用环境中已有 AISBench，容器为 `j00893003-minimax-a5-dev`，可执行文件为 `/home/j00893003/minimax-dspark/aisbench-venv/bin/ais_bench`。每轮单独推理、再用未修改的官方 evaluator 评分；所有实际返回均为成功请求，没有将异常 HTTP 请求算作精度结果。配置原文见各轮 `data/rounds/*/config.py` 与 `settings.json`。

AIME 使用 30 题完整集，四次独立串行轮次，共 120 个计划请求；不是 pass@4。仅剩 ID15 时按用户要求中止，四次中止均按未作答计错并保留在分母中。GPQA 使用 Diamond 198 题，每轮完整评分；GPQA ID 是 CSV 行号加 1，错题同时保留 Record ID 和该次请求的完整选项，避免与数据集原生 ID 或其他轮次选项混淆。

数据集 SHA256：AIME2026 `52822957957a3f577d1e9706c36a66a8108a3f99b6aff424cfb72dff0094a9ee`；GPQA Diamond `41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305`。

## AIME2026 正式结果

| 轮次 | seed | 正确 / 总数 | 精度 | 草稿采信率 | 平均每步输出，含主模型 token | 错题 ID |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 42 | 27/30 | 90.00% | 27.12% | 2.898229 | [ID4](wrong_answers/aime2026-0001/ID004.md)、[ID14](wrong_answers/aime2026-0001/ID014.md)、[ID15](wrong_answers/aime2026-0001/ID015.md) |
| 2 | 43 | 28/30 | 93.33% | 28.92% | 3.024423 | [ID14](wrong_answers/aime2026-0002/ID014.md)、[ID15](wrong_answers/aime2026-0002/ID015.md) |
| 3 | 44 | 28/30 | 93.33% | 28.74% | 3.011633 | [ID10](wrong_answers/aime2026-0003/ID010.md)、[ID15](wrong_answers/aime2026-0003/ID015.md) |
| 4 | 45 | 29/30 | 96.67% | 29.23% | 3.046417 | [ID15](wrong_answers/aime2026-0004/ID015.md) |
| **合计** | — | **112/120** | **93.33%** | **28.59%** | **3.001367** | ID4 错 1 次、ID10 错 1 次、ID14 错 2 次、ID15 中止 4 次 |

116 个请求正常返回，4 个 ID15 请求按策略中止。除 ID15 外，实际返回的错答共 4 次；正式成绩仍按完整 120 个请求计算。

## GPQA Diamond 正式结果

| 轮次 | seed | 正确 / 总数 | 精度 | 草稿采信率 | 平均每步输出，含主模型 token | 错题数 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1042 | 169/198 | 85.35% | 24.44% | 2.710763 | 29 |
| 2 | 1043 | 170/198 | 85.86% | 24.03% | 2.682107 | 28 |
| 3 | 1044 | 166/198 | 83.84% | 24.43% | 2.710288 | 32 |
| 4 | 1045 | 166/198 | 83.84% | 24.46% | 2.712227 | 32 |
| **合计** | — | **671/792** | **84.72%** | **24.34%** | **2.703863** | **121** |

四轮共有 52 道不同题目至少错一次；每一轮的全部错题及原始回答见 [逐轮错题索引](wrong_answers.md)。四轮均错的题目为：ID33、72、76、77、80、82、91、107、146、161、168、175。

## 采信率计算口径

仅统计 Decode 角色，分别对每轮测试前后的计数器做差，再跨 Decode DP engine 相加；Prefill 的草稿计数为 0，不与 Decode 混合计算。轮次合计用 token 数加权，不能对百分比直接取算术平均。

```text
采信率 = accepted_draft_tokens / proposed_draft_tokens
平均每步采信草稿数 = accepted_draft_tokens / speculation_rounds
平均每步输出 token 数 = 1 + accepted_draft_tokens / speculation_rounds
```

最后一式包含每次主模型采样输出的 1 个 token。AIME 的计数包含 ID15 在被中止前已生成的部分。正式测试前的启动验证请求不计入每轮基线。

| 数据集 | speculative rounds | 草稿 token 总数 | 被采信草稿 token 总数 |
| --- | --- | --- | --- |
| AIME2026 四轮 | 573525 | 4014675 | 1147834 |
| GPQA 四轮 | 3566783 | 24967481 | 6077310 |

每轮指标原文保存于 `data/rounds/<轮次>/before-decode.metrics.txt` 与 `after-decode.metrics.txt`；结构化统计见 [aggregate.json](data/aggregate.json)。

## 输出质量与错题回答

已完成的 116 个 AIME 回答和 792 个 GPQA 回答均未检出 NUL 或异常控制字符。GPQA 第 1 / 第 4 轮分别出现 4 / 1 个 Unicode 替换字符 `U+FFFD`。缺少 `</think>` 结束标记的回答数：AIME 为 0、1、0、0；GPQA 为 10、6、9、6。部分 GPQA 回答包含中文；这些是自动文本标记，不能仅凭标记认定整个回答乱码或重复失控。

原始回答保留全部推理内容，不只保留 evaluator 提取出的选项或最终数字。所有 129 次正式错题记录均有独立条目：125 个已返回的完整回答，以及 4 个按要求中止而没有完整返回的 ID15 条目。未完成的 GPQA 第五轮不加入这些错题数。

入口：[逐轮错题与完整回答](wrong_answers.md)；[结构化错题索引](data/wrong-question-index.json)。每个条目包含题目、实际请求选项、参考答案、官方提取答案，以及未经改写的模型回答。

## 后续无 DSpark GPQA 对照测试

用户要求切换后，先停止 DSpark 测试进程，再停止并重启上述两台容器中的服务。服务参数仅删除完整的 `--speculative-config` 块；源代码、主模型权重、主模型 FP8 KV、PD 拓扑及其余服务参数保持不变。原 DSpark 权重文件保留在只读目录中，未删除或改写。

两端启动日志均明确为 `speculative_config=None`；两端健康检查返回 200，实际 PD 请求返回 `2`，Decode 收到 696 个来自跨机 KV 传输的 prompt token，草稿相关计数为 0。详见 [Prefill 启动证据](data/nonspec-switch/prefill-startup-proof.json)、[Decode 启动证据](data/nonspec-switch/decode-startup-proof.json) 与 [请求验证](data/nonspec-switch/qualification.json)。

无 DSpark GPQA 按原采样配置连续测试两轮，共396个请求。每轮正式成绩、同 seed 的精度对照与全部错题回答见 [无 DSpark GPQA 对照报告](../2026-10-25-a5-pd-no-dspark/README.md)。关闭 DSpark 后采信率不适用。

- [会话配置](data/nonspec-switch/session.json)
- [第一轮 AISBench 配置](data/nonspec-switch/gpqa-round1-config.py)
- [Prefill 参数差异](scripts/nonspec/prefill-script-change.diff) / [Decode 参数差异](scripts/nonspec/decode-script-change.diff)
- [原 DSpark 第五轮停止记录](data/nonspec-switch/dspark-gpqa5-stopped.json)：80/198，仅保留部分输出，不纳入正式汇总。

## 日志与证据位置

- DSpark 测试根目录：`177.11.26.1:/home/j00893003/minimax-dspark/cxy-pd-pr19723-20261025`。
- 逐轮结果：上述目录的 `runs/aime2026-0001` 至 `0004`、`runs/gpqa-0001` 至 `0004`。
- 停止的 GPQA 第五轮：`runs/gpqa-0005/user-stopped-partial-round.json`，保留 80 个已返回结果，未正式评分。
- DSpark 服务停机前脚本和日志：两端运行目录的 `runs/dspark-before-nonspec-20261025`。
- 无 DSpark 两轮 GPQA 测试目录：`177.11.26.1:/home/j00893003/minimax-dspark/cxy-pd-pr19723-nonspec-gpqa2-20261025`。

各轮 `summary.json`、逐题评分和错题索引保留官方成绩；`aisbench-score-summary.json` 保留官方分数与服务器原始结果文件的 SHA256。所有提交文件的校验值见 [manifest.json](manifest.json)。
