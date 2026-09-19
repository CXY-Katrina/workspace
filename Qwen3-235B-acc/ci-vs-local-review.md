# 本地测试方式与指定 CI 流程的审视

对照任务：[35376647863 / 105704758372](https://github.com/vllm-project/vllm-ascend/actions/runs/35376647863/job/105704758372)。结论：核心服务命令和精度参数已对齐，但两轮本地测试均不是该 CI 的完整环境复刻，也不是只改变一个变量的严格回归实验。

## 流程差异

| 项目 | 指定 CI | 本地第一轮 | 本地第二轮 v0.23.0 |
|---|---|---|---|
| 部署入口 | Kubernetes LeaderWorkerSet，三个 Pod；run.sh → pytest test_external_dp.py | 三台机器 Docker host 网络；直接调用 test_external_dp() | 三台机器 Docker host 网络；独立服务启动脚本 + AISBench 客户端 |
| 软件提供方式 | is_pr_test=false，使用 nightly-ci-main-a3 镜像预装组件，不走 PR 源码重装分支 | 8 月 22 日缓存 nightly 基础镜像，源码编译 vLLM/Ascend | 用户指定 v0.23.0-a3 原始镜像，未重装 vLLM/Ascend |
| vLLM | v0.28.0 / 2cf0a6915ce54 | 与 CI 相同 commit | v0.23.0 / 0fc695fc6d1d |
| Ascend | c8addbb24fe8 | e139b7d573d3，比 CI 多三个提交 | v0.23.0 / 5cb98caaadef |
| 测试脚本和代理 | CI revision 内脚本 | e139 revision；相关脚本/代理相对 CI revision 无改动；AISBench 增加 dump 参数 | 仍使用 e139 的测试脚本和代理，服务包来自 v0.23.0 镜像 |
| AISBench 部署位置 | Leader Pod 内，与服务共用 CI 镜像环境 | 197 测试容器 | 197 原第一轮容器，仅作为客户端；服务在另一个 v0.23.0 容器 |
| pytest 配套行为 | 加载 conftest、执行会话 fixture、正式生成结果、清理和归档 | 直接调用函数，未经过 pytest/conftest 入口 | 独立服务管理器和 run_aisbench_cases，未经过 pytest/conftest 入口 |
| 生命周期 | 评测完成后退出上下文并清理服务 | 中途按用户要求保留服务，后切换镜像时终止 | 按用户要求停止客户端，三台服务保留 |
| 完成度 | 1319 条完整评测，96.28506444275966% | 冻结 1314 条，临时 94.67275494672755%；非完整结果 | 用户停止，保存 552 条；没有最终全量分数 |

run.sh 的安装分支仅在 IS_PR_TEST=true 时执行 checkout_src、install_vllm_ascend、install_aisbench。该 CI 日志明确 is_pr_test=false，随后打印预装包版本。因此“在旧 nightly 上重新安装相同源码”不等价于“复用了该次 CI 镜像”。

本地绕过 pytest 是实际方法差异。conftest 存在 adapt_patch(True/False) 和 session cleanup_model_cache；尚无证据证明它造成服务输出异常。vllm serve 是独立启动的进程，不能直接把父 pytest 进程中的 Python patch 等同于服务进程 patch。若做严格复现，应恢复原入口，而不是未经验证地把这一差异宣布为根因。

## 已对齐的项目

对 CI 日志与第一轮实际服务命令做 shlex 参数解析，三个节点的 rank 0 只有模型标识/本地路径和 data-parallel-address 不同；第二轮使用同一份本地 YAML。

- Prefill TP16/DP1；Decode TP1/DP32/EP32。
- seed=1024、max-model-len=36864、W8A8 ascend 量化、关闭 prefix caching。
- Prefill eager、max-num-batched-tokens=16384、max-num-seqs=8。
- Decode async scheduling、max-num-batched-tokens=256、max-num-seqs=28、FULL_DECODE_ONLY、capture sizes [6,12,18]。
- finegrained TP、EPLB、MLAPO、MooncakeConnectorV1 和 A3 direct KV transfer 配置。
- GSM8K 0-shot CoT；max_out_len=32768、batch_size=64、temperature=0.6、top_p=0.95、thinking=true、ignore_eos=false、stream=false、request_rate=0、retry=2。
- 模型级 extract_non_reasoning_content，数据集级 gsm8k_postprocess/Gsm8kEvaluator。

CI 和本地请求均未显式传 top_k。本地两轮的全部 33 个 API rank 均确认从模型 generation_config 继承 top_k=20；CI 的客户端配置相同，但仅凭客户端日志不能独立证明 CI 缓存中的 generation_config 内容哈希也相同。

本地多出的 --dump-eval-details 用于保存评分原文和逐题结果，不改变采样或答案评分函数。该参数不适合作为乱码根因的直接推断。

## 还没有对齐或证明一致的项目

1. **权重、tokenizer、generation_config、数据集的内容哈希。** CI 使用模型仓库名与缓存；本地使用此次下载并与官方清单校验的文件。校验通过只证明本地文件符合此次官方清单，不能证明与 CI 缓存逐字节一致。CI 在准备 benchmark 时还出现“无法确认缓存是否对应 master revision”的 Modelscope 警告；该警告不能单独证明权重不同，也不能明确归属某个文件。需要 CI 实际文件清单和哈希。
2. **基础镜像及底层依赖。** CI 未提供本次核查足够的完整 pip freeze/镜像不可变摘要，不能把“CANN 都显示 9.1.0”当作所有构建与依赖一致。CI 日志中 CANN innerversion=V100R001C11SPC001B243。两轮本地已确认 torch_npu 从 2.10.0.post4.dev20260715 变为 2.10.0.post4，transformers 从 5.14.1 变为 5.5.4。
3. **网络与驱动。** CI 是 Pod eth0 / 10.0.0.*；本地是 host 网络、data0.173 / 178.27.4.*，Mooncake 日志中还有管理网 192.168.13.* 的传输端点。本地显式补充 LOCAL_IP/HCCL_IF_IP/GLOO_SOCKET_IFNAME/HCCL_SOCKET_IFNAME/TP_SOCKET_IFNAME 等；CI 的 Decode 命令未显式列出这些变量。未列出不代表运行环境一定没有设置。本地 197 驱动 25.5.0，198/159 为 26.0.rc1，CI 各节点驱动尚未完整对齐。
4. **本地配置修改。** CI 的 Prefill --data-parallel-address 字面值为 12321，本地修正为 178.27.4.197；模型仓库名改为 /workspace/model-official；数据集显式本地路径。这些修改均需列入复现记录，不能声称 YAML 完全未改。
5. **动态库加载环境。** 第一轮 LD_LIBRARY_PATH 与 CI 有差异，LD_PRELOAD 中 jemalloc 路径重复一次；尚无证据证明重复路径导致乱码。镜像隐式环境变量和实际加载库未做全量对等证明。
6. **请求调度与随机性。** 服务端 seed 相同，不能保证并发 64 的跨节点请求路由和随机采样逐题一致；不能要求两次输出逐字相同，也不能仅靠随机性解释已经观察到的异常字符。

## 当前可以与不可以得出的结论

- 第一轮全量得分即使将未保存的 5 条全部算对，上界也只有 94.6929%，低于 CI 的 96.2851%；但这是中止后基于已保存数据计算的上界，不是实际全量最终得分。
- v0.23.0 服务栈也观察到 U+FFFD，所以不能只归因于第一轮相对 CI 新增的三个 Ascend 提交；仍不能排除跨版本共有的软件问题。
- 第二轮同时更换 vLLM、Ascend、torch_npu、transformers 等组件，并继续使用当前测试代理和原 AISBench 客户端。因此它是“整套服务镜像对照”，不是单组件二分，也不是完全采用 v0.23.0 当时的测试流程。
- **CI 96.2851% 不代表其输出没有乱码或重复。** CI 命令未加 --dump-eval-details；本地已经看到含乱码但数字答案正确的样本。该 job 的得分日志不足以比较文本质量。

如后续继续定位，优先顺序应为：获取 CI 实际权重/tokenizer/数据哈希与镜像摘要 → 固定 CI Ascend commit c8addbb24fe8、镜像和原 pytest 入口 → 只增加 --dump-eval-details → 对比异常样本。当前仅完成只读审视，未恢复已停止的测试。
