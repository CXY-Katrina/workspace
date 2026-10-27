# MiniMax M3.1 Flash：A5 PD + DSpark 启动脚本

## 最新双机 Decode 部署

- [2026年10月27日 A5：1P + 双机 D（DP4 TP4 EP16）](deployments/2026-10-27-a5-1p2d/README.md)：P=27.1，D=28.1+21.1；graph size 1024、P/D prefix cache、DSpark 7。含实测启动脚本、proxy、容器准备及网络说明。

从 `MiniMax_A5_PD_DSpark_启动交接说明.html` 第 0 章引用的服务器文件导出，日期：2026-10-23。脚本保留导出时的原始内容，不包含 SSH 登录密码。

## 性能测试报告

- [2026年10月26日纯 D 稳态离线重分析](reports/2026-10-26-decode-steady-state/README.md)：34 组历史数据、最佳完整 AISBench outputs；旧单机直连 D 与新双机完整 PD 分组列出。graph 1024 优化补测进行中。

## 精度测试报告

- [2026年10月26日 release 5666 双机 PD + DSpark GPQA 完整 outputs](reports/2026-10-26-a5-pd-dspark-release5666-gpqa/README.md)：单轮官方成绩 168/198（84.85%），含原始预测、逐题评分、完整错答及评分口径审计。

- [无 DSpark GPQA 两轮对照结果与错题](reports/2026-10-25-a5-pd-no-dspark/README.md)。

- [2026年10月25日 A5 PD 分离与 DSpark 测试](reports/2026-10-25-a5-pd-dspark/README.md)：AIME2026 四轮 112/120（93.33%），GPQA 四轮 671/792（84.72%），包含两台服务器环境、权重路径、采信率及逐题原始错答。

下文为 2026年10月23日导出的历史启动脚本与部署说明；本次测试的容器、权重和拓扑以测试报告为准。

## 文件与部署位置

| 文件 | 容器内位置 | 角色 |
| --- | --- | --- |
| `runtime_env.sh` | `/work/runtime_env.sh` | 两台共用运行环境 |
| `prefill/start.sh` | `/work/prefill/start.sh` | Prefill 启动入口 |
| `prefill/run_dp_template.sh` | `/work/prefill/run_dp_template.sh` | Prefill 服务参数 |
| `decode/start.sh` | `/work/decode/start.sh` | Decode 启动入口 |
| `decode/run_dp_template.sh` | `/work/decode/run_dp_template.sh` | Decode 服务参数 |
| `start_proxy.sh` | `/work/start_proxy.sh` | PD 代理启动入口 |

Prefill 和代理位于 `177.11.26.1`，容器为 `x00939195-m31-pd-prefill-release`，计算网络 IP 为 `197.13.50.1`。Decode 位于 `177.11.27.1`，容器为 `x00939195-m31-pd-decode-release`，计算网络 IP 为 `197.13.51.1`。

两台宿主机的 `/home/x00939195/minimax-pd-release-20261023` 映射为容器内 `/work`。每台使用全部 8 张 NPU，拓扑为 DP2 × TP4 × PP1、EP8。脚本依赖现有容器、虚拟环境、源码和已准备好的权重，并非完整的首次部署安装包。

## 运行前提

- 主模型：`/home/t00932610/minimax_m3dot1_mxfp8`。
- 草稿模型：`/work/pr9-8627ba952/draft-prepared`；DSpark 每轮 7 个草稿 token。
- 配套 vLLM：`ced6857afa0ea7b2e3f0846a62e1394e90f15607`，另应用 PR #9 自带的核心 guard。
- 交接说明记录的 vllm-ascend 基线：`8627ba952`；Decode 另包含 [PR #11](https://github.com/ycl-ai-infra/vllm-ascend-private/pull/11) 修复 `03114943f`。
- 当前脚本保留 `fuse_qknorm_rope=true`。Decode 使用该配置时需要上述融合修复；未部署修复时，可将模板中该开关设为 `false` 绕过对应编译错误。
- 环境：CANN 9.2、Torch 2.10.0、Triton / Triton-Ascend 3.6.0，Python 环境位于 `/work/.venv`。

## 启动已有环境

先检查容器、现有服务和端口，避免重复启动。容器停止时先启动对应容器。

```bash
# 26 宿主机：Prefill
docker exec -d x00939195-m31-pd-prefill-release bash /work/prefill/start.sh
curl --noproxy '*' -fsS http://127.0.0.1:31050/health

# 27 宿主机：Decode
docker exec -d x00939195-m31-pd-decode-release bash /work/decode/start.sh
curl --noproxy '*' -fsS http://127.0.0.1:31060/health

# 两端健康检查成功且 8009 没有现存代理后，在 26 宿主机启动代理
docker exec -d x00939195-m31-pd-prefill-release bash /work/start_proxy.sh
```

模型加载需要时间，启动命令返回不代表服务已经就绪。代理入口为 `http://177.11.26.1:8009/v1/chat/completions`，模型名为 `minimax-m3`。代理没有 `/health`，应通过实际请求检查。

日志位于 `/work/logs/prefill-rank0.log`、`/work/logs/decode-rank0.log` 和 `/work/logs/proxy.log`。

导出验证：6 个脚本均通过 `bash -n`；两台 `runtime_env.sh` 内容一致。本次仅导出文件，未启动或重启服务器上的服务。
