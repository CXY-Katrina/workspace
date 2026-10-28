# 2026-10-28 main 分支完整 PD 纯 D 性能

P：177.11.21.1，TP4/PP2/DP1；D：177.11.28.1 + 177.11.22.1，DP4/TP4/EP16。全部测量请求经过 proxy → P → D。

Ascend 测试提交：`1b3df109893a98954531ab4bb4dd531f6ff3ad05`；vLLM main：`544f182d9f0c9bfe8bd7645525951d5e3692a542`。集成代码见 [PR #7](https://github.com/CXY-Katrina/vllm-ascend/pull/7)。

DSpark 7，probabilistic；目标 KV FP8，草稿 KV BF16；D FULL_DECODE_ONLY，capture size 1024，HCCL_BUFFSIZE 1024。P/D 均启用 prefix cache，前缀数 1；16K 名义 100% 命中，128K 名义 90% 命中。

每次请求输出 1024 tokens，ignore_eos，RR=0，请求数为并发的 4 倍。`k=1024`。平均 TPOT 判定，允许最多超 1 ms。

以下选择已测有效结果中各档吞吐最高者，并保留重启确认轮。未将 230 并发的 21.2 ms 算作 20 ms 档通过。

| Case | 目标 TPOT | 并发 | TPOT 均值/P90/P99(ms) | 稳态吞吐(token/s) | 原始稳态表 | aisbench.log |
| --- | --- | --- | --- | --- | --- | --- |
| 16K/1K | 20 ms | 224 | 20.4 / 22.1 / 39.8 | 10027.3042 | [table](runs/resume_16k_cc224_target20/steady-aisbench-table.txt) | [log](runs/resume_16k_cc224_target20/aisbench/aisbench.log) |
| 16K/1K | 50 ms | 417 | 33.7 / 36.8 / 66.1 | 11032.0619 | [table](runs/resume_confirm_16k_cc417/steady-aisbench-table.txt) | [log](runs/resume_confirm_16k_cc417/aisbench/aisbench.log) |
| 128K/1K | 30 ms | 55 | 24.2 / 31.3 / 42.7 | 1656.8217 | [table](runs/resume_128k_cc55_capacity/steady-aisbench-table.txt) | [log](runs/resume_128k_cc55_capacity/aisbench/aisbench.log) |
| 128K/1K | 50 ms | 55 | 24.2 / 31.3 / 42.7 | 1656.8217 | [table](runs/resume_128k_cc55_capacity/steady-aisbench-table.txt) | [log](runs/resume_128k_cc55_capacity/aisbench/aisbench.log) |

## 全部测量与确认轮

| Run | 并发 | TPOT均值(ms) | 稳态吞吐(token/s) | 原始稳态表 |
| --- | --- | --- | --- | --- |
| resume_128k_cc55_capacity | 55 | 24.2 | 1656.8217 | [table](runs/resume_128k_cc55_capacity/steady-aisbench-table.txt) |
| resume_16k_cc224_target20 | 224 | 20.4 | 10027.3042 | [table](runs/resume_16k_cc224_target20/steady-aisbench-table.txt) |
| resume_16k_cc230_target20 | 230 | 21.2 | 9644.0627 | [table](runs/resume_16k_cc230_target20/steady-aisbench-table.txt) |
| resume_16k_cc417_capacity | 417 | 33.5 | 11012.6032 | [table](runs/resume_16k_cc417_capacity/steady-aisbench-table.txt) |
| resume_confirm_128k_cc55 | 55 | 24.5 | 1640.1058 | [table](runs/resume_confirm_128k_cc55/steady-aisbench-table.txt) |
| resume_confirm_16k_cc224 | 224 | 20.9 | 9787.9616 | [table](runs/resume_confirm_16k_cc224/steady-aisbench-table.txt) |
| resume_confirm_16k_cc417 | 417 | 33.7 | 11032.0619 | [table](runs/resume_confirm_16k_cc417/steady-aisbench-table.txt) |

## 口径与校验

稳态数据直接来自 AISBench `StablePerfMetricCalculator` 原始打屏，表格没有重算或改写。稳态窗口按客户端达到最大并发的时间区间计算；原始日志注明起止时间。稳态表中的请求数是该区间样本数，与全程 4×并发的请求数不同。每轮完整请求数、成功数、实际长度及 P/D prefix 命中均见对应 `result.json`。

D 总 KV 容量为 7,275,100 tokens，按用户要求 `floor(总KV/(输入+输出))` 得 16K 上限 417、128K 上限 55。128K 的 30/50 ms 两档使用同一容量上限测点，不虚构成两次独立测试。

16K 实测 cache hit 99.21875%，最后一个 128-token block 需重算；128K 实测约 89.94%。128K 原始实际输入存在分词拼接造成的少量偏差（初轮 131071–131074 tokens），逐轮分布见 result.json。输出均严格为 1024 tokens。

首次 16K/417 测量在新启动服务上运行；此后每轮使用独立 cache_salt，预热与测量共享该盐值，预热覆盖四个 D 缓存域。服务没有开放 reset_prefix_cache 接口，日志中的该接口 404 已保留；后续跨轮缓存隔离通过 cache_salt 实现。

原始完整运行目录保留在 D28 `/data/pd-main-20261028/runs/<run>/`。仓库只保存用户所需原始稳态表、AISBench 日志及复现/校验元数据，不包含性能请求全文。

之前 GPQA 为 158/198（79.79798%），采信率 24.36466%；精度未达 >80% 原门槛。用户随后明确要求继续性能测试。本次未重跑或改变精度结果，完整 GPQA outputs 已另行保存。

测试后仅停止本次会话的容器与进程，三台停止证明见 deployment 下的 resume-final-shutdown.json；未停止其他人的服务。实际部署脚本也保存在 deployment 下。
