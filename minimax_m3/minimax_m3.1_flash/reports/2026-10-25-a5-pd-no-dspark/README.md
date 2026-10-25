# MiniMax M3.1 Flash 无 DSpark GPQA 精度对照测试

结果更新时间：**2026-10-25T14:27:46+08:00（北京时间）**。

两台服务器沿用 [DSpark 测试报告](../2026-10-25-a5-pd-dspark/README.md) 中的容器、只读主模型权重、代码提交和 PD 拓扑。本次仅从两端启动命令中删除 `--speculative-config` 整段，不删除或改写草稿权重文件。

## 正式成绩

| 轮次 | 请求 seed | 完成状态 | 正确 / 总数 | 精度 | DSpark采信率 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1042 | 正式评分完成 | 165/198 | 83.33% | 不适用，已关闭 |
| 2 | 1043 | 按用户要求停止，130/198已返回 | — | — | 不适用，已关闭 |

**已完成轮次合计：165/198，83.33%。** 计划总数为396；未完成轮次不计入此正式分数。

## 停止状态

用户要求停止服务并释放177.11.26.1。两端本次模型服务、PD代理、AISBench测试及自动报告更新进程均已停止；177.11.26.1的31050、8009端口，以及177.11.27.1的31060端口均已关闭。177.11.26.1八张NPU均无运行进程，HBM回落至每卡约4.8GB基础占用。

第二轮在130/198个请求返回时中止，未做完整轮次正式评分，不合并到第一轮165/198的正式成绩。已返回的原始输出和停止前指标保存在服务器原测试目录。见 [中止记录](data/stopped/partial-stop.json)、[Prefill停止核验](data/stopped/prefill-user-stop.json)、[Decode停止核验](data/stopped/decode-user-stop.json) 和 [NPU释放证据](data/stopped/prefill-npu-final-after-stop.txt)。容器、代码与权重均保留，未删除权重。

## 第二轮130条已返回记录

第二轮已返回子集临时核对为 **114/130，87.69%**；16题未匹配，另68题未获得返回。完整原始回答、实际请求题目与选项、Record ID、临时提取答案和参考选项已逐条保存至 [第二轮130条记录](partial_round2/README.md)，并提供原始JSONL压缩包。该子集结果不作为完整轮次正式成绩，不与第一轮合并总分。

## 与 DSpark 相同请求 seed 的对照

| seed | DSpark 正确 / 总数 | 无 DSpark 正确 / 总数 | 精度差值，无 DSpark减DSpark |
| --- | --- | --- | --- |
| 1042 | 169/198 | 165/198 | -2.02个百分点 |
| 1043 | 170/198 | 中止，未正式评分 | — |

此前 DSpark 四轮总分为671/792（84.72%）；上表仅将前两轮相同请求 seed 对齐比较。

## 配置与 DSpark 关闭证据

- Prefill：`177.11.26.1`，容器 `cxy-vllm-ascend-0.30.0rc2-20261024`；DP1 × TP4 × PP2。
- Decode：`177.11.27.1`，容器 `cxy-minimax-pd-decode-20261025`；DP2 × TP4 × PP1。
- Prefill 权重：`/data/l00845740/minimax_m3dot1_flash_w8a8_mxfp8_quarot_l32`。
- Decode 权重：`/mnt/weight/minimax_m3dot1_flash_w8a8_mxfp8_quarot_l32`，只读 NFS 挂载同一份权重。
- vLLM Ascend：`accf5d0fc6005fe89dd909e656848b696bbc18f5`；vLLM：`ced6857afa0ea7b2e3f0846a62e1394e90f15607`。
- 上下文133000，最大输出81920，并发16，temperature=1，top_p=0.95，主模型 KV cache 为FP8。
- `chat_template_kwargs`：`reasoning_effort=max`，`thinking_mode=enabled`。
- 两端运行日志为 `speculative_config=None`，启动验证跨机 KV 传输正常，草稿计数为0；关闭后采信率不适用。

启动参数、日志核验与请求验证见 [切换证据](../2026-10-25-a5-pd-dspark/data/nonspec-switch/qualification.json) 和 [无 DSpark 脚本](../2026-10-25-a5-pd-dspark/scripts/nonspec)。客户端历史 `abbr=minimax-pd-dspark` 仅作为输出目录名沿用，不代表启用了 DSpark。

## 错题回答与原始记录

[逐轮完整错题回答](wrong_answers.md) 包含实际请求题目与选项、参考答案、官方提取答案及全部原始回答。完整轮次的逐题评分、配置与指标位于 `data/rounds`；[结构化汇总](data/aggregate.json)、[错题索引](data/wrong-question-index.json) 与 [本次快照](data/snapshot.json) 可供复核。

服务器结果根目录：`177.11.26.1:/home/j00893003/minimax-dspark/cxy-pd-pr19723-nonspec-gpqa2-20261025`。原计划运行两轮GPQA，现已按用户要求停止第二轮及全部服务，并停止自动更新。
