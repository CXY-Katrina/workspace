# MiniMax M3.1 Flash：vLLM 融合算子、代码路径与输入输出契约

核查日期：2026-10-26（Asia/Shanghai）。依据 vLLM 社区源码的两个固定快照；以下结论不随链接中的 main 浮动。

| 来源 | 固定版本 / 状态 |
|---|---|
| [vLLM PR #63232](https://github.com/vllm-project/vllm/pull/63232) | head `32ad976c451304fd66b5f0d65897ded406bde97c`；核查时 Open、尚未合并 |
| [vLLM main](https://github.com/vllm-project/vllm/tree/6674eedd28d785bed1daf6309615ba337e791aa6) | `6674eedd28d785bed1daf6309615ba337e791aa6` |
| [官方 MiniMaxAI/MiniMax-M3.1-Flash 配置](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/262ae6830934f3512211081b8a3f16425b93a239/config.json) | HF revision `262ae6830934f3512211081b8a3f16425b93a239` |
| PR 安装的 [MSA 源码](https://github.com/vllm-project/MSA/tree/25c144700f727b638067e98066a60507081b63ad) | `25c144700f727b638067e98066a60507081b63ad`；由 vLLM CMake 安装到 `vllm/third_party/fmha_sm100/` |
| main 安装的 MSA | `968061ab16cb14991a5c7d4c0e8eabf531be0dcf`，与 PR 依赖版本不同 |

**汇总口径。** 本文覆盖这两个快照中 MiniMax M3.1 Flash 主模型、可选推测解码及视觉模型可达的**显式融合类别**，包括横向投影合并、CUDA/HIP/Triton/CuTe-DSL/FlyDSL 数值融合、融合通信以及直接调用的外部 fused API。相同算法的 tile/autotune 变体不重复计数；仅存在于仓库但当前 checkpoint/配置无法触发的实现另行标注。输入输出以模型调用契约和kernel校验为准，区分 tensor 存储 dtype、量化编码与计算精度。

**“所有”有配置边界。** 未指定 GPU、TP/EP、MoE backend、KV/indexer dtype、speculative 配置和编译选项时，不存在一个固定、同时启用的 kernel 清单。本文给出源码级全集和分支条件；不声称一次推理会执行所有条目，也不把函数名中的 fused 等同于一个 kernel launch。没有运行GPU、做数值/性能测试或导出实际编译图；外部 FlashInfer/AITER/DeepGEMM 只确认 vLLM 调用可见契约，内部未证实的精度或launch数量不作确定断言。

**实现语言列的口径。** 本文将“CUDA C”记为 **CUDA C++**（通常为`.cu/.cuh`）；**Triton、CuTe-DSL、FlyDSL、Triton Gluon、HIP C++**分别列出。CuTe-DSL虽用Python写kernel且运行于CUDA GPU，也不等于Triton或手写CUDA C++。Python wrapper、横向GEMM合并、编译器图改写本身不决定底层kernel语言。标为“外部库、内部未固定核实”的条目仅确认vLLM调用与后端身份，不依据函数名推断内部语言；同一编号的多种技术栈按分支选择，不能据此重复计数。

## 阅读索引与版本差异

| 类别 | 实现语言 / 技术栈 | 固定 main | PR #63232 快照 | 详细位置 |
|---| --- |---|---|---|
| QKV + index-Q/index-K 横向 GEMM | Python封装；GEMM依量化后端选择 | 有，后接A02 | 常规分支使用；命中A03时不单独执行 | A01 |
| Gemma QK norm + partial RoPE + cache write | CUDA C++；ROCm共享实现编译为HIP C++ | 有，主KV可NVFP4；index cache BF16/FP8 | 常规分支接A01；命中A03时不单独执行；扩展NVFP4 index-K等 | A02、A2 |
| MXFP8 QKV GEMM + norm/RoPE/NVFP4 cache epilogue | CuTe-DSL | 没有该模块 | **条件替代A01+A02**，SM100、T≥512等条件；不是第三个串行算子 | A03 |
| Indexer score/max、Top-K/强制local页 | Triton / CuTe-DSL / CUDA C++（按分支） | 有BF16/FP8 Triton/MSA路径 | 新增NVFP4 decode与NVFP4→FP8 prefill、可选TP4 decode CP | A04–A13，含选择树 |
| 稀疏QK + softmax + PV | Triton / CuTe-DSL / CUDA C++ CUTLASS（按分支） | Triton/MSA；已有NVFP4主KV路径 | 专用Q8KV4 prefill、decode计划/内存等改动 | A15–A20；A14为metadata旁路 |
| SwiGLU-OAI clamp/activation/mul | CUDA C++；GEMM1融合版为CuTe-DSL | 有 | 有；新增MXFP8 GEMM1+activation+quant epilogue | B1–B2 |
| Sigmoid+bias+Top4+renorm、普通FusedMoE | 路由CUDA C++；MoE为外部FlashInfer TRTLLM或CuTe-DSL | 有通用融合后端 | 明确M3.1 per-token NVFP4 QAT合同，接入相应分支 | B3、B6 |
| DeepGEMM NVFP4 MegaMoE | prepare为Triton；MegaMoE为外部DeepGEMM | 通用模块已有，M3未接入 | M3接入，可融合MXFP8 shared experts | B4–B5 |
| All-reduce + residual + Gemma norm | B7为CuTe-DSL；B8/B9调用外部FlashInfer | 有 | 有；增加MoE finalize+共享专家+AR+norm+MXFP8跨层融合 | B7–B9 |
| ROCm AITER/Triton常规路径 | Triton / Triton Gluon / HIP C++ / 外部AITER | 有，含main独有CP扩展 | 有对应路径；增加FlyDSL NVFP4系列 | C |
| ROCm persistent mono | FlyDSL | main是FFN半层；官方checkpoint当前不满足检查 | PR是受严格条件限制的整层融合 | C“Persistent mono” |
| MiniMax DSpark | CUDA C++ / Triton / 外部Attention库 / 后端GEMM | 没有MiniMax专有DSpark类 | QK/RoPE/cache、多层context KV writer、Markov sampler等 | D1–D7 |
| 视觉MRoPE与Flash attention | MRoPE为Triton；attention依所选外部后端 | 有 | 同类融合；patch embedding改FP32不计新融合 | D视觉部分 |

以上“新增/已有”表示**两个固定快照的对比**。PR与当前main并非线性包含关系，不能据此把所有差异认定为这个PR原创或已经合入main。每个细项的链接定位实际定义/调用。

## 模型尺寸与 dtype 速查

官方配置：`hidden_size=6144`、60层、64个Q头、4个KV头、head_dim=128、partial rotary_dim=64、index-query头数4、index-key为单个共享向量、稀疏块128 tokens、Top16块（15历史+1 local）；前3层dense MLP intermediate=12288，后57层MoE，128 routed experts、每token Top4、expert intermediate=3072、1个shared expert intermediate=3072。SwiGLU-OAI `alpha=1.702,limit=7`，routed scaling factor=2。60层attention全部稀疏。模型基础激活BF16；主量化linear为MXFP8，routed experts为NVFP4，具体忽略量化的模块以checkpoint配置为准。[官方配置](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/262ae6830934f3512211081b8a3f16425b93a239/config.json)

TP=4、无额外indexer CP时，可用以下尺寸对照表格：`Q[T,16,128]`、`K/V[T,1,128]`、`index-Q[T,1,128]`、`index-K[T,1,128]`；五路packed投影 `[T,2560]`；MoE router logits `[T,128]` FP32，路由结果 `[T,4]`；最终层间hidden `[T,6144]` BF16。更多TP/CP/EP组合必须按各实现的分片合同计算。

| 表中写法 | 真正含义 |
|---|---|
| BF16 / FP16 / FP32 | `torch.bfloat16 / float16 / float32`；未特指时BF16是官方模型常见激活精度 |
| FP8 E4M3 | 通常 `torch.float8_e4m3fn`；缓存也可能先用UINT8 backing再view成FP8 |
| MXFP8 | E4M3数据 + 每32元素一个E8M0/UE8M0幂次scale；scale可能UINT8、float8_e8m0fnu或INT32 packed view，需看实际kernel |
| NVFP4 | E2M1两值/byte + 每16值一个E4M3 block scale +（适用时）FP32 global scale；数据容器可UINT8或INT8，但不是INT8整数量化 |
| MXFP4（ROCm部分专家） | E2M1数据 + E8M0 block scale；不能与NVFP4的E4M3 scale混用 |
| INT32 | 主要用于页号、长度、expert id；也可能是4个scale byte的packed存储，必须结合参数名 |
| INT64 | 主要用于token position、slot mapping、指针/offset；CP候选也可把score和page id打包进INT64 |
| FP32 accumulation | 表示归约/GEMM累加/softmax统计等内部计算精度；不代表输入输出都是FP32 |

Gemma norm与SwiGLU-OAI是两处重要语义差别：
- Gemma norm：`x * rsqrt(mean(x*x)+eps) * (1+weight)`。
- SwiGLU-OAI：`g=min(gate,limit)`，`u=clamp(up,-limit,limit)`，`g*sigmoid(alpha*g)*(u+beta)`，本模型beta=1。
- MoE Top4按 `sigmoid(logits)+correction_bias` 选专家，混合权重取**不带bias**的sigmoid值、归一化后乘2；Indexer Top16按一页内QK点积的max选页，两者完全不同。

来源：[norm](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/model.py#L276)、[SwiGLU-OAI](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/model_executor/layers/activation.py#L207)、[router](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py#L163)。

---

## A. NVIDIA Attention / Indexer 融合目录

本节的 `T,H,Hq,Hkv,Hi,D,B,P,Kb` 分别为本 rank token 数、hidden size、Q 头数、KV 头数、index-query 头数、head dim、cache block size、物理 cache 页数、所选稀疏块数。标准配置 `H=6144,D=128,B=128,Kb=16`；TP=4 时 `Hq=16,Hkv=Hi=1`。index-K 是所有 index-query 头共享的单头，不是 `Hi` 个 K 头。

### A1. QKV 与 norm / RoPE / cache

**A01+A02 与 A03 是同一处理阶段的两条互斥执行路径。** 对一次正常的 sparse-attention forward，命中A03时不调用A01的常规投影forward，也不再调用A02；未命中时执行A01→A02。表中三个编号描述实现选择，不能相加为该次forward连续执行的三个算子。

| 本次forward条件 | 实际执行 | 中间QKV | 计数说明 |
|---|---|---|---|
| 满足A03所有硬件/量化/shape条件，`T≥512`，且非FlashInfer autotuning | 必要的输入MXFP8量化 → **A03** | 不物化BF16 packed QKV | A03替代A01+A02；前置量化是否另有kernel取决于输入是否已量化 |
| `T<512`，或任一A03支持条件不满足，或正在autotuning | **A01 → A02** | A01生成packed QKV，A02读取 | 两个处理阶段；A01内部可能还包含后端输入量化，不能据此断言总共只有两次launch |
| 固定main快照 | **A01 → A02** | 物化packed QKV | 此版本没有A03模块 |

这里的`T`是本次forward的token行数，不是单个请求的上下文长度。A03按条件和token数量分支，代码没有把它限定为prefill；小batch decode通常因`T<512`落到A01+A02。

```text
                       ┌─ 满足A03条件 ─→ [必要的输入量化] → A03 ─┐
hidden_states ──────────┤                                      ├─→ Indexer score / Top-K → Attention
                       └─ 否则 ─────────────────→ A01 → A02 ──┘
```

A03仅融合投影与其norm/RoPE/cache epilogue，**不包含后续Indexer打分、Top-K或Attention**。它仍读取`qkv_proj.weight/weight_scale`，因此替代的是A01的执行调用，不是删除投影模块或其权重；A01/A02还必须服务不满足条件的配置和batch。[分支与调用证据](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/model.py#L937)

| 编号 / 算子 | 实现语言 / 技术栈 | 融合内容、代码路径 | 输入：shape / dtype / 含义 | 输出：shape / dtype / 含义 | main / PR 与条件 |
|---| --- |---|---|---|---|
| A01 `MinimaxM3QKVParallelLinearWithIndexer` | Python模块；底层GEMM按量化后端派发，MXFP8可走FlashInfer CUTLASS或CuTe-DSL | 把 Q、K、V、index-Q、index-K 五个线性投影横向合并为一次 GEMM；[调用及构造](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/model.py#L792)，实现 `vllm/model_executor/layers/linear.py` | hidden `[T,H]` BF16；量化后端可接收 `QuantizedActivation`：FP8 E4M3 数据 + E8M0 scales；权重按 checkpoint 为 MXFP8（忽略量化列表中的层除外） | packed `qkv[T,(Hq+2Hkv+Hi+1)D]`，常规输出 BF16；段序为 Q/K/V/index-Q/index-K | 两版都有。**常规分支第一阶段，后接A02；命中A03时不单独执行。** 是横向GEMM合并，不包含attention计算。 |
| A02 `fused_minimax_m3_qknorm_rope_kv_insert` | **CUDA C++**（`.cu`；同源ROCm分支为HIP C++） | 每头 Gemma RMSNorm + 部分 NeoX RoPE；同时处理 main Q/K 和 index Q/K；Q 去交错写连续输出；K/V/index-K 量化并 scatter 写页缓存。[Python 契约](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/_custom_ops.py#L2711)，[CUDA/HIP 实现及 dtype 检查](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/csrc/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu#L2199)，[模型调用](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/model.py#L1055) | packed QKV 同 A01，BF16 或 FP16；四个 norm weight `[D]` 与 QKV 同 dtype；cos/sin `[max_pos,rotary_dim]` 同 dtype；positions、两个 slot_mapping `[T]` INT64；eps 标量；NVFP4 主缓存 global K/V scale 为 CUDA FP32 tensor | Q `[T,HqD]` 为输入 dtype 或 FP8 E4M3；index-Q `[T,HiD]` 为 BF16/模型 dtype 或 FP8 E4M3；原位写 K/V 与 index cache，详见 A2；可原位更新 packed QKV。norm、RoPE 以 FP32 运算后舍入 | 两版都有，main 已支持主 KV NVFP4；PR 扩展 **index-K NVFP4**、persistent row-group kernel 和 `write_qkv=False` 路径。**常规分支接A01；命中A03时不单独执行。** FP8-only Q 可省去不再使用的 K/index-K 回写。 |
| A03 `minimax_m3_qkv_index_fused_epilogue` | **CuTe-DSL**（Python编写GPU kernel） | 将 A01 GEMM 与 A02 的 norm/RoPE/cache epilogue 合入 CuTe-DSL GEMM，避免物化中间 QKV。[接口完整文档](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/ops/qkv_index_fused_epilogue.py#L958)，[选择条件](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/model.py#L161)，[forward 门槛](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/model.py#L1022) | `xq[T,H]`、`wq[N,H]` FP8 E4M3；`xsf,wsf` E8M0、F8_128x4 swizzled block scales；`N=(19Hkv+1)128`；四个 norm weight BF16 `[128]`；cos/sin BF16 `[max_pos,64]`；positions/slot mappings INT64 `[T]` | Q BF16 和/或 FP8 E4M3 `[T,16Hkv*128]`；index-Q FP8 E4M3 `[T,Hkv*128]`；主 KV 与 index-K NVFP4 uint8 cache 原位写。GEMM 累加/norm 运算 FP32，按融合语义舍入/量化 | **PR 新增；条件替代A01+A02，不与二者串行执行。** SM100 family；FlashInfer MXFP8 CuTe-DSL/CUTLASS weight layout；两种 cache 均 NVFP4；`Hkv∈{1,2,4},Hq=16Hkv,Hi=Hkv,D=128,rotary_dim=64,H%128=0,T≥512`；autotuning 时退回常规路径。输入 BF16→MXFP8 量化可能仍是独立 kernel，除非上游已给量化激活。 |

Gemma RMSNorm 的语义为 `y=x·rsqrt(mean(x²)+eps)·(1+w)`，与直接乘 `w` 的普通 RMSNorm 不同。RoPE 只作用于前 64 维，剩余 64 维保留；V 不做 QK norm/RoPE。A02 的实现硬编码 head dim=128；不要从 Python 通用参数推断支持任意 head dim。见 [CUDA 定义与计算](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/csrc/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu#L118)。

A03 直接对 FP32 GEMM accumulator 做 norm/RoPE，没有常规路径的 GEMM→BF16→norm 往返舍入，因此不能声称与 A01+A02 逐位一致。B1 的 GEMM1 epilogue也省去独立BF16激活中间量；这是融合后数值舍入位置的区别。[A03实现说明](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/ops/qkv_index_fused_epilogue.py#L3)

### A2. 缓存 dtype 与 shape 的精确解释

| 对象 | Tensor dtype / 逻辑 shape | 数值含义 |
|---|---|---|
| BF16 主 KV | BF16，`[P,Hkv,B,2D]`，末轴 K/V | 直接保存归一化旋转后的 K 和原 V。 |
| FP8 主 KV | backing storage 常为 UINT8，attention 读时 view 成所选 FP8；逻辑 `[P,Hkv,B,2D]` | FP8 位模式不是 UINT8 整数值。MSA CUTLASS 路径为 E4M3；E5M2 等支持须按所选 backend 检查，不能套用 CUTLASS 条件。 |
| NVFP4 主 KV | UINT8，`[P,2Hkv,B,72]`，slot=`2*head+side`，side=K/V | 每个 head-page 内先放全部 token 的 E2M1 packed 数据，再放全部 token 的 E4M3 block scales；`72=128/2+128/16` 是每 token 平均存储量。**不能把 tensor 的单行当作「该 token 的64B数据+8B scale」直接切片**。每16个数共享一个 E4M3 scale，另有 FP32 global scale。 |
| BF16 / FP8 index-K | BF16 或 `float8_e4m3fn`，`[P,128,128]` | 单个共享 index-K 头。BF16 index-Q 对 BF16 cache；FP8 index-Q 对 FP8 cache。 |
| NVFP4 index-K | UINT8，`[P,128,72]` | 同样是 page 内数据区+scale区；E2M1 packed + 每16元素一份 E4M3 scale；**无额外 global scale**。对应 index-Q 始终为 FP8 E4M3，不是 FP4。 |

来源：[缓存对象及 query_dtype](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/common/indexer.py#L118)、[A02 cache 契约](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/_custom_ops.py#L2766)、[gather 的物理布局](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/ops/gather_nvfp4_index_k.py#L3)。PR Q8KV4 sparse-attention 实现进一步要求主 K/V global scales 均为 1；A02 接口能接收非 1 scale，不代表完整 attention 路径都支持，见 [forward 检查](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/sparse_attention_msa.py#L537)。

### A3. Indexer：后端选择、执行路径与去重后的步骤

**A04–A13是实现目录，不是串行流水线。** 先在初始化时选择一个Indexer实现，再在每次forward中分别处理decode/prefill token区域。两类token可以同处一个mixed batch，因此“prefill/decode路径互斥”只对同一token成立，不代表同一次forward只能运行其中一个。A14是主Attention的metadata准备，已移到A4表中，保留编号便于对照。

原表混合了完整wrapper和内部步骤，容易重复计数。下面统一记法：**A06a=普通Triton prefill Top-K，A06b=普通Triton decode partial Top-K+merge；A07a=MSA prefill OnlyScore，A07b=MSA BF16/FP8路径共享Top-K；A09/A10仅指各自score步骤，A11单列其wrapper内部Top-K。** 对应输入输出见下方细表。

#### A3.1 第一级：选择哪个Indexer实现（NVIDIA）

选择入口是 `common/indexer.py::select_indexer_impl_cls`，主要依据 **indexer_kv_dtype**（index-K缓存格式），不是主KV缓存格式，也不是A03是否命中。

| 条件 | PR选择结果 | 固定main |
|---|---|---|
| indexer cache = NVFP4，GPU为SM100/103/107，满足NVFP4实现合同 | `MiniMaxM3IndexerNvfp4Impl` | 不支持该indexer格式，抛异常；不自动回退BF16/FP8 |
| indexer cache = BF16/FP8 E4M3，SM100 family，Top-K=16 | `MiniMaxM3IndexerMSAImpl` | 同类选择 |
| indexer cache = BF16/FP8 E4M3，其余受支持情况 | `MiniMaxM3IndexerTritonImpl` | 同类选择 |
| NVFP4设备不支持，或indexer格式本身不支持 | 抛异常；不是“失败后再尝试所有实现” | 对应不支持条件同样拒绝 |

FP8 indexer还要求CUDA平台声明支持FP8。NVFP4实现进一步要求 `topk=16,init_blocks=0,local_blocks=1,block_size=128,head_dim=128,score_type=max`，每rank index头数1/2/4，decode每request查询长度≤8；不满足这些条件不能把上述NVFP4分支当作可用。

来源：[PR selector](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/common/indexer.py#L510)、[NVFP4合同](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/indexer_nvfp4.py#L379)、[main selector](https://github.com/vllm-project/vllm/blob/6674eedd28d785bed1daf6309615ba337e791aa6/vllm/models/minimax_m3/common/indexer.py#L496)。ROCm另有AMD模型入口和AITER/FlyDSL分支，见C节，不能直接套此NVIDIA选择表。

#### A3.2 第二级：选定实现后实际运行哪些步骤

| 已选择实现 / token类型 | 实际执行链（按本报告的步骤口径） | 不应重复计数的部分 |
|---|---|---|
| Triton / prefill | **A04 → A06a** | 不执行A05、A07、A08–A13 |
| Triton / decode | **A05-Triton → A06b** | `minimax_m3_index_decode()`是这条链的wrapper，内部已经调用A05；A06b包含partial+merge两个kernel，wrapper不能再算一个额外算子 |
| MSA BF16/FP8 / prefill | **A07a → A07b** | 不执行A04/A06；不是A07之后再跑A11 |
| MSA BF16/FP8 / decode | **A05-CuTe或A05-Triton → A07b** | 条件为 `Hi*max_decode_query_len≤32` 时用CuTe scorer，否则Triton scorer；这里复用A05但**不接A06b** |
| NVFP4 / prefill | **A08 → A10-score → A11** | `BatchPrefillIndexerQ8KV8Wrapper.run()`已经包含A10-score+A11；按workspace预算分chunk时每chunk执行该链。即使gather输出FP8，也不重新选择A07路径 |
| NVFP4 / decode，CP未启用或不满足条件 | **A09-score → A11** | `BatchDecodeIndexerQ8KV4Wrapper.run()`已经包含两步；A11不能在wrapper后再额外算一次 |
| NVFP4 / decode，CP启用且满足条件 | **Q交换 → A09-score（本rank页分片、所有index头）→ A12 → A13** | 只调用Q8KV4 wrapper的 `_run_scores()`；A12+A13**替代A11**，不是接在A11后；Q交换是Triton通信helper |

MSA BF16/FP8 mixed batch有一个额外的合并关系：decode的A05和prefill的A07a分别写同一个FP32分数buffer的两个token区域，最后**一次A07b**覆盖所有token；不能把“decode一次Top-K+prefill一次Top-K”机械相加。

```text
Indexer backend（初始化时选择）
├─ Triton
│  ├─ prefill：A04 → A06a
│  └─ decode ：A05-Triton → A06b（partial → merge）
├─ MSA BF16/FP8
│  ├─ prefill：A07a ────────────────┐
│  └─ decode ：A05-CuTe或Triton ────┴→ A07b（统一Top-K）
└─ NVFP4
   ├─ prefill：A08 → A10-score → A11（每chunk）
   └─ decode
      ├─ stock：A09-score → A11
      └─ CP   ：Q交换 → A09-score → A12 → A13

A14：主Attention decode metadata准备，属于旁路；不参与Indexer后端选择。
```

源码：[Triton forward](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/common/indexer.py#L418)、[Triton decode wrapper内含score](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/common/ops/index_topk.py#L895)、[MSA统一score/Top-K](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/indexer_msa.py#L259)、[NVFP4 decode/prefill分流](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/indexer_nvfp4.py#L420)、[Q8 decode run](https://github.com/vllm-project/MSA/blob/25c144700f727b638067e98066a60507081b63ad/python/fmha_sm100/cute/q8_indexer_interface.py#L554)、[Q8 prefill run](https://github.com/vllm-project/MSA/blob/25c144700f727b638067e98066a60507081b63ad/python/fmha_sm100/cute/q8_indexer_interface.py#L921)、[CP forward](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/indexer_nvfp4_decode_cp.py#L403)。

#### A3.3 CP、dtype与上游A03的关系

- **CP不是第四套scorer。** PR的NVFP4 decode CP复用Q8KV4 score，只改变读哪些页、计算哪些index头，以及如何跨rank选择结果。要求显式 `indexer_decode_cp=true`、TP=4、全局index头4/本rank头1、decode query长度≤8、没有另开DCP>1、TP rank之间支持所需对称内存。CP条件不满足时退回stock NVFP4 decode；prefill仍是A08→A10→A11。[CP guard](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/indexer_nvfp4_decode_cp.py#L470)
- **A07b与A11是同类任务的不同实现入口。** 前者是BF16/FP8 MSA路径的 `sparse_topk_select`，后者是Q8 NVFP4-indexer路径的 `indexer_topk_select`；不串行叠加。CP用A12/A13另行完成选择；其score比较/平局处理与stock实现可能不同，不应声称逐位一致。[CP说明](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/indexer_nvfp4_decode_cp.py#L3)
- **输入输出层次统一。** A04/A05/A07a/A09/A10都只生成FP32页分数；A06a/A06b/A07b/A11/A13生成INT32逻辑页号。A08输出FP8 K pages；A12输出INT64打包候选；A14输出INT32长度/offset而不是Top-K。
- **A03没有消除Indexer。** A03只提供index-Q并写index-K cache，之后还要执行上述链。A03要求NVFP4 indexer cache，因此它命中时后面使用NVFP4分支；但A03因 `T<512` 退回A01+A02时，如果缓存配置仍为NVFP4，后面仍是同一NVFP4分支。不能把“A01+A02”与“Triton indexer”绑定，也不能把“A03”只与prefill绑定。
- **A14应按metadata build计数。** 它由 `MiniMaxM3SparseMSAMetadataBuilder.build` 在主Attention CUTLASS decode计划满足条件时触发，服务A18；不属于 `MiniMaxM3Indexer.forward`。其启用由主KV格式、attention decode backend、head geometry及batch/query limits决定，与是否选A11/A12无直接替代关系；不能当作每层Indexer都额外执行的步骤。[builder](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/sparse_attention_msa.py#L202)、[计划条件](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/msa_cutlass_sparse_decode.py#L333)

#### A3.4 各步骤的I/O与代码路径

| 编号 / 算子 | 实现语言 / 技术栈 | 融合内容、代码路径 | 输入：shape / dtype / 含义 | 输出：shape / dtype / 含义 | 版本 / 条件 / 融合边界 |
|---| --- |---|---|---|---|
| A04 `minimax_m3_index_score` / `_index_block_score_kernel` | **Triton** | paged index-K gather + QK 点积 + causal mask + 页内 max。[common/ops/index_topk.py](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/common/ops/index_topk.py#L674) | index-Q `[T,Hi,128]`，index-K `[P,128,128]`，BF16 或 FP8 E4M3；block_table、cu_seqlens、seq_lens 等 INT32 | scores `[Hi,T,ceil(Lmax/128) padded]` FP32；每一项为该 query 对该页内有效 key 的最大点积分数 | 两版都有，Triton prefill fallback；混合输入 dtype 时实现会转 FP16，dot 累加 FP32。Top-K 是后续独立调用。 |
| A05 `minimax_m3_index_decode_score` / `minimax_m3_index_decode_score_cutedsl` | **Triton / CuTe-DSL**：MSA decode在`Hi*max_decode_query_len≤32`时选CuTe-DSL，否则Triton；普通Triton后端使用Triton | paged gather + QK + causal mask + 页内 max；Triton 版本还把固定 init/local 块的优先级编码到 score。[Triton](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/common/ops/index_topk.py#L797)、[CuTe-DSL](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/ops/index_decode_score.py#L453) | index-Q `[T,Hi,128]`、index-K cache，BF16 或 E4M3；INT32 block_table/seq_lens；decode_query_len 标量 | FP32 scores `[Hi,T,max_pages]`，可写到共享 `[T,Hi,max_pages]` 的转置 view | 两版都有。Triton backend decode固定使用Triton scorer，后接A06b；MSA BF16/FP8 backend decode在`Hi*max_decode_query_len≤32`时选CuTe scorer，否则Triton，后接A07b。两种scorer互斥；FP8 CuTe operand可在MMA前转FP16 fragment，累加FP32。 |
| A06a `minimax_m3_index_topk`（prefill选择） | **Triton** | 排序/选择 + 强制init/local页 + 无效页处理。[prefill top-k](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/common/ops/index_topk.py#L744) | FP32 页分数、INT32 有效长度/上下文信息 | INT32 `[Hi,T,Kb]`，0-based逻辑页号，-1为无效项。 | 两版都有；仅Triton backend prefill链，接A04。与decode的A06b为不同token区域路径。 |
| A06b `_topk_index_partial_kernel` + `_topk_index_merge_kernel`（decode选择） | **Triton**（partial与merge两个kernel） | 分片Top-K，再merge；读取A05已经编码的init/local优先级及有效长度。[decode wrapper](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/common/ops/index_topk.py#L895) | A05-Triton生成的FP32页分数、INT32 seq_lens | partial scores FP32、partial ids INT32；最终INT32 `[Hi,T,Kb]`逻辑页号/-1 | 两版都有；Triton backend decode执行A05→A06b。`minimax_m3_index_decode()`包住这整条链；A06b本身是partial+merge两个kernel，wrapper不另计一次。 |
| A07a MSA `_fmha_sm100(...output_o=False,output_maxscore=True)`（prefill打分） | **CUDA C++ / CUTLASS**（MSA OnlyScore模板） | OnlyScore融合QK与页内max，不读V占位参数，不计算softmax-PV。[MSA forward](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/indexer_msa.py#L259) | index-Q BF16/E4M3 `[T,Hi,128]`；index-K 同 dtype `[P,128,128]`；paged metadata | 仅FP32 scores，写共享 `[T,Hi,max_pages]` buffer的prefill区域；不输出Top-K页号。 | 两版都有；MSA BF16/FP8 backend的prefill scorer，后接A07b。不是A04后面的额外打分。 |
| A07b MSA `sparse_topk_select`（统一选择） | **CUDA C++**（MSA `sparse_topk_select.cu`） | 页分数Top-K + 有效页过滤 + 强制init/local页。[MSA forward](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/indexer_msa.py#L347) | 共享FP32 scores `[T,Hi,max_pages]`，INT32 num_valid_pages；decode区域由A05写、prefill区域由A07a写 | INT32 `[T,Hi,16]`逻辑页号/-1，写共享topk buffer | 两版都有；一次调用覆盖本次forward的decode及prefill区域。不是A06或A11；这三个Top-K实现属于不同backend链，不串行执行。 |
| A08 `minimax_m3_gather_nvfp4_index_k_fp8` | **Triton**，可含QMUL4 inline PTX | 按 request gather 页 + 解包 E2M1 + 乘 E4M3 scale + 饱和/舍入为 E4M3。[ops/gather_nvfp4_index_k.py](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/ops/gather_nvfp4_index_k.py#L131) | index cache UINT8 `[P,128,72]`；block_table `[R,max_pages]`、num_pages/page_offsets `[R]`，均 INT32 | 连续 E4M3 `[sum(num_pages),128,128]` | **PR 新增**；NVFP4 indexer prefill 没有直接 NVFP4 score kernel，因此先使用该融合转换。QMUL4 指令或等价 FP32乘法后一次E4M3舍入。 |
| A09 Q8KV4 `_run_scores()` / score kernel（仅打分） | **Hi=1：CUDA C++ / CUTLASS；Hi=2或4：CuTe-DSL**。这里Hi是传入wrapper的头数；CP交换后用4头，走CuTe-DSL | NVFP4解包/scale + QK + causal/页内max；反量化精度随硬件/工具链变化：SM100/103在CUDA≥13.4使用QMUL4，旧工具链采用精确FP16反量化；SM107使用精确FP16乘积而不舍入至E4M3，因此分数可能不同。[vLLM indexer](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/indexer_nvfp4.py#L379)、[MSA接口](https://github.com/vllm-project/MSA/blob/25c144700f727b638067e98066a60507081b63ad/python/fmha_sm100/cute/q8_indexer_interface.py#L590)，底层 `vllm/third_party/fmha_sm100/csrc/include/q8kv4_indexer/` 及 SM107 变体 | index-Q E4M3 `[R*q_len,Hi,128]`，index-K UINT8 `[P,128,72]`；block_table/seq_lens INT32 | 仅FP32页分数，内部为request/query-tile/head/page布局；INT32最终Top-K由A11或A13生成，不属于A09输出。 | PR新增indexer路径；SM100/103/107，Hi=1/2/4，q_len≤8。stock执行A09→A11；CP复用A09但后接A12→A13。`BatchDecodeIndexerQ8KV4Wrapper.run()`=A09+A11，不能再把wrapper与内部步骤重复计数。 |
| A10 Q8KV8 prefill `_run_scores()` / score kernel（仅打分） | **CuTe-DSL**（`Q8KV8PrefillIndexerSm100`） | FP8 QK + causal mask + 页内 max；后接同一 forced-tail Top-K。[vLLM prefill调用](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/indexer_nvfp4.py#L460)、[MSA接口](https://github.com/vllm-project/MSA/blob/25c144700f727b638067e98066a60507081b63ad/python/fmha_sm100/cute/q8_indexer_interface.py#L724) | E4M3 Q `[T,Hi,128]`；A08生成的 E4M3 compact K pages；INT32计划元数据 | 仅FP32页分数，按token/head展平行并按wrapper计划padding；INT32最终Top-K由A11生成。 | PR NVFP4 indexer prefill每chunk执行A08→A10→A11。`BatchPrefillIndexerQ8KV8Wrapper.run()`=A10+A11；不能把run算一次后又把A11重复加一次。 |
| A11 `indexer_topk_select` | **CUDA C++**（MSA `.cu` / `.cuh`） | 页分数排序选择 + forced local tail + 无效页填充。[MSA C++ binding](https://github.com/vllm-project/MSA/blob/25c144700f727b638067e98066a60507081b63ad/python/fmha_sm100/csrc/indexer_topk_select.cu#L20)、kernel `csrc/include/indexer_topk/m3_topk.cuh` | scores FP32 `[rows,cols]` 或分组 strided view；lengths INT32 `[rows]` | INT32 `[rows,16]`；长序列选15个历史页+自身local页（最后一个有效项）；短序列保留全部有效页，其余-1 | PR NVFP4 stock decode与prefill复用：分别接A09与A10。**CP decode不用A11，改用A12→A13。** 此函数与A07b不同；仅selection内部融合，没有把score融合进来。 |
| A12 `_local_topk_push_kernel` | **Triton**（选择+对端写入） | 每 rank 页分数精确 Top16 + 分数/页号编码 + 直接写对端候选缓冲。[indexer_nvfp4_decode_cp.py](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/indexer_nvfp4_decode_cp.py#L177) | FP32局部分数 `[R,8*Hi_global,width]`；seq_lens INT32；对端地址数组 INT64 | 对端候选 `[CP,rows_max,16]` INT64，打包可排序score与逻辑页号；不是INT64浮点分数 | PR可选NVFP4 decode CP；接Q交换和A09分片score，再接A13。TP4、4个全局index头/每rank1头、query长度≤8、对称内存等条件需满足。A12+A13替代A11，不是追加在A11后。 |
| A13 `_merge_kernel` | **Triton** | 轮询候选 + 跨rank Top16 merge + local页强制插入 + 清理候选和Q tile sentinel。[同文件](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/indexer_nvfp4_decode_cp.py#L235) | INT64候选、INT32 seq_lens/Q tile存储view | INT32 `[T,1,16]` topk；原位重置通信buffer | PR A12之后。前置 `_q_exchange_kernel`（同文件L133）把本rank E4M3 Q以INT32位模式复制到所有rank并等待，是通信/同步helper，不是attention数值计算融合。 |

注意区分两种 Top-K：MoE Top4 选**专家**；Indexer Top16 选**128-token历史页**。Indexer score 为页内 token 点积分数的 max，不是对专家logits做sigmoid，也不是softmax概率。算法证据见 [common indexer说明](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/common/ops/index_topk.py#L3) 和 [MSA Q8 indexer契约](https://github.com/vllm-project/MSA/blob/25c144700f727b638067e98066a60507081b63ad/python/fmha_sm100/cute/q8_indexer_interface.py#L4)。

### A4. 稀疏 Attention 与辅助构建

| 编号 / 算子 | 实现语言 / 技术栈 | 融合内容、代码路径 | 输入：shape / dtype / 含义 | 输出：shape / dtype / 含义 | 版本 / 触发条件 |
|---| --- |---|---|---|---|
| A15 `minimax_m3_sparse_attn` / `_gqa_sparse_fwd_kernel` | **Triton** | paged gather/FP8反量化 + GQA QK + causal mask + online softmax + PV。[common/ops/sparse_attn.py](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/common/ops/sparse_attn.py#L529) | Q `[T,Hq,128]` BF16/FP16；KV cache模型dtype或FP8；topk `[Hkv,T,16]` INT32；block_table和长度INT32；可选K/V scales FP32 | O `[T,Hq,128]` 与output/model dtype一致。QK/softmax/O accumulator FP32 | 两版，通用Triton prefill；不是NVFP4 fallback。所选topk是逻辑页号，经block_table取物理页。 |
| A16 `minimax_m3_sparse_attn_decode` + `_merge_topk_attn_out_kernel` | **Triton**（attention与merge） | split-K的每段融合 gather/dequant/QK/softmax/PV；merge融合LSE权重归一化及分段O求和。[decode入口](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/common/ops/sparse_attn.py#L616)、[merge](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/common/ops/sparse_attn.py#L434) | 同A15，decode请求/长度；中间partial O `[splits,T,Hq,D]` query dtype，partial LSE `[splits,T,Hq]` FP32 | 最终O为model/output dtype；merge计算FP32 | 两版；**split attention + merge至少两个kernel**。KV scales既可scalar也可实现所支持的per-token/head形式，M3主路径通常scalar。 |
| A17 MSA `sparse_atten_func` | **CuTe-DSL**主attention kernel；Python调度wrapper | 稀疏paged GQA attention，将QK、mask、online softmax、PV融合；可能另有split归约。[vLLM调用](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/sparse_attention_msa.py#L415)；安装后 `vllm/third_party/fmha_sm100/cute/interface.py` | Q `[T,Hq,D]` BF16或E4M3；K/V paged cache BF16或E4M3；INT32 CSR、schedule、页表、长度 | O `[T,Hq,D]` BF16（模型dtype）；内部按kernel使用FP32累加/softmax统计 | 两版SM100 MSA BF16/FP8 cache prefill。选定硬件/量化geometry决定实际kernel；不是任意混合dtype都可运行。 |
| A18 `msa_cutlass_sparse_decode` → `fmha_sm100` | **CUDA C++ / CUTLASS**（MSA；Python封装） | FP8/NVFP4 cache读取与反量化 + 稀疏QK/softmax/PV；split时另有reduce。[vLLM wrapper](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/msa_cutlass_sparse_decode.py#L405)；底层安装目录 `vllm/third_party/fmha_sm100/decode_q8kv4/` 和 `csrc/` | Q E4M3 `[T,Hq,128]`；KV FP8 E4M3或NVFP4 UINT8 head-slot pages；topk `[T,Hkv,16]` INT32；plan/page_table；scale为float或FP32 tensor | O `[T,Hq,128]` BF16/model dtype | **main已有主KV NVFP4 decode**；PR优化计划/内存/PDL及大batch等路径。需要SM100系和支持的head geometry、B128/Kb16。 |
| A19 NVFP4 prefill：main `sparse_atten_nvfp4_kv_func`；PR `prefill_q8kv4.run_prefill` | **main：CuTe-DSL；PR：CUDA C++ / CUTLASS** | packed NVFP4解包/scale + sparse attention QK/softmax/PV；可能另有split reduce。[main调用](https://github.com/vllm-project/vllm/blob/6674eedd28d785bed1daf6309615ba337e791aa6/vllm/models/minimax_m3/nvidia/sparse_attention_msa.py#L367)、[PR调用](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/sparse_attention_msa.py#L581) | Q `[T,Hq,128]`，PR强制E4M3；NVFP4 K/V data(UINT8)与E4M3 block-scale视图；FP32 global scale；CSR/schedule/length INT32 | O `[T,Hq,128]` BF16/model dtype | main已支持NVFP4主cache，但走CuTe sparse函数；PR专用CUTLASS Q8KV4 prefill，要求global scales=1；没有BF16-query kernel或Triton NVFP4 fallback。 |
| A20 `build_k2q_csr(...return_schedule=True)` | **CUDA C++**（`cute/src/sm100/build_k2q_csr/build_k2q_csr.cu`；目录名cute不代表CuTe-DSL） | 把query→selected key pages反转为key→query CSR并构建sparse attention调度；其中schedule构建与部分CSR步骤融合。[vLLM调用](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/sparse_attention_msa.py#L280)、[MSA接口](https://github.com/vllm-project/MSA/blob/25c144700f727b638067e98066a60507081b63ad/python/fmha_sm100/cute/sparse_index_utils.py#L331) | q2k `[Hkv,T,16]` INT32；cu_seqlens_q/k `[R+1]` INT32；shape上界参数 | `k2q_row_ptr[Hkv,total_rows+1]`、`k2q_q_indices[Hkv,nnz_capacity]` INT32；schedule元数据/计数INT32 | 两版MSA prefill；**这是多kernel构建流程**（row map/hist/prefix/scatter及调度），不能把“fused CSR builder”理解为整个调用只有一个kernel。 |
| A14 `_update_runtime_metadata_kernel`（主Attention旁路） | **Triton** | 一次生成KV段长度、query offset、可选request seq_lens。[msa_cutlass_sparse_decode.py](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/msa_cutlass_sparse_decode.py#L98) | seq_lens `[R]` INT32，固定decode_query_len | INT32 `[R*q_len]` 段长度和offset；可选 `[R]` request lengths | main有基础版本，PR扩展输出；由主Attention metadata builder准备CUTLASS decode计划时触发，服务A18。**不属于Indexer forward，不与A04–A13相加为Indexer执行链。** 按metadata build计数，不按层数计数。 |

A17–A20 的外部实现不直接作为普通源码文件存放在 vLLM tree：CMake 从 MSA 安装到 `vllm/third_party/fmha_sm100/`。PR 固定 MSA commit `25c144700f727b638067e98066a60507081b63ad`；main 固定 `968061ab16cb14991a5c7d4c0e8eabf531be0dcf`。追踪源码时应先看对应版本的 [PR CMake](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/cmake/external_projects/fmha_sm100.cmake#L17) / [main CMake](https://github.com/vllm-project/vllm/blob/6674eedd28d785bed1daf6309615ba337e791aa6/cmake/external_projects/fmha_sm100.cmake#L17)，不要直接搜索一个当前checkout不存在的 `third_party/fmha_sm100` 文件就判定实现缺失。

**默认配置范围说明。** 官方 M3.1 Flash 的60层均为sparse attention。模型类里的 dense `MiniMaxM3Attention` 分支有 QKV横向合并与 QK-norm/RoPE融合，但不是该checkpoint默认target计算路径；与之不同，前三层的**dense MLP**确实会执行，不能混为一谈。MTP可复用target decoder融合；DSpark使用独立draft结构，见后文。

---

## B. NVIDIA MLP / MoE / 残差通信融合

核对点：PR #63232 head `32ad976c451304fd66b5f0d65897ded406bde97c`（下称 P），main `6674eedd28d785bed1daf6309615ba337e791aa6`（下称 M）。仅静态读源码，未运行 GPU；外部 FlashInfer/DeepGEMM 的内部 kernel 数量不能由 vLLM wrapper 名称推断。本节符号：T=本 rank token 数，H=hidden_size，I=本 rank/本专家 intermediate size，E=总 routed experts，El=本 rank local experts，K=每 token 选中的专家数，S=shared expert 总 intermediate size，Prows=路由重排后的 padded 行数。

### 结论与可达性

1. PR 新增 `dense_gemm1_mxfp8.py`、`finalize_allreduce_norm.py`，并为 M3 接入 NVFP4 MegaMoE。通用 `deep_gemm_mega_moe.py` 在固定 main 已存在，不能将其全文件当 PR 新增。`prepare_megamoe.py` 已有对应功能，PR 改为 `(token, split)` 网格优化。两者的模型调用链变化见 [P model.py](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/model.py#L209)。
2. M3 的 `_use_mega_moe` 只允许 `moe_quant_algo == NVFP4`、显式 `moe_backend=deep_gemm_mega_moe`、开启 EP；调用 `MEGA_MOE_NVFP4`。通用模块存在 `MEGA_MOE_MXFP8/MXFP4/BF16` 不代表 M3 会用这些 routed 路径。M3 NVFP4 MegaMoE 可以在同一 launch 融合 MXFP8 shared experts；shared 不满足未分片/格式/已安装 DeepGEMM API 条件时退回独立 MLP。[P model 209–229、473–530](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/model.py#L473)、[MegaMoE shared 判定](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/model_executor/layers/fused_moe/deep_gemm_mega_moe.py#L337)。
3. `csrc/minimax_reduce_rms_kernel.cu` / `minimax_allreduce_rms_qk` **不属于该 M3 forward 可达算子**：它由 `MiniMaxText01RMSNormTP`（Text01 系列 TP QK norm）使用，M3 使用逐 head Gemma norm。不要按名字含 MiniMax 就算入 M3.1 融合清单。[Text01 入口](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py#L332)、[M3 norm 类](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/model.py#L276)。

### 实现语言速查

下表对应后续B1–B9及附录的完整I/O说明；外部依赖未审计的底层实现与已确认的本地kernel分开标注。

| 编号 / 算子 | 实现语言 / 技术栈 | 实现边界 |
|---|---|---|
| B1 `fused_gemm1` / `Sm100DenseGemm1Mxfp8` | **CuTe-DSL** | GEMM1+OAI+输出MXFP8量化；输入量化由外部FlashInfer独立调用 |
| B2 `silu_and_mul_with_clamp` | **CUDA C++** | `csrc/activation_kernels.cu` |
| B3 `topk_sigmoid` | **CUDA C++** | `csrc/moe/topk_softmax_kernels.cu`；Python router封装 |
| B4 NVFP4 MegaMoE prepare | **Triton** | `prepare_megamoe.py`中的量化/重排kernel |
| B5 `nvfp4_mega_moe` | **外部DeepGEMM** | vLLM为Python调用；本报告未按固定依赖源码核实内部语言/模板 |
| B6 TRTLLM modular / monolithic MoE | **外部FlashInfer TRTLLM CUDA后端** | vLLM封装可见；内部源码语言未固定核实，不能按包装文件`.py`判定为Triton |
| B6 FlashInfer CuteDSL MoE备用分支 | **CuTe-DSL（外部FlashInfer）** | 与TRTLLM为后端选择关系；量化/调度等辅助步骤不保证同一语言 |
| B7 finalize+allreduce+Gemma norm+MXFP8 | **CuTe-DSL** | publish与consumer两个PDL串联kernel |
| B8 allreduce+residual+Gemma norm | **外部FlashInfer TRTLLM CUDA后端** | Python helper调用；底层语言未按固定依赖源码核实 |
| B9 residual+Gemma RMSNorm；无residual的Gemma RMSNorm | **外部FlashInfer norm** | NVIDIA模型直接调用FlashInfer；具体kernel语言依依赖版本/后端，未固定核实 |
| 可选Marlin `moe_wna16_marlin_gemm` | **CUDA C++** | Marlin GEMM kernel；activation/求和为pipeline中的其他步骤 |
| `MergedColumnParallelLinear` gate/up横向投影合并 | **Python模块；GEMM依量化后端** | 不是独立固定语言kernel；命中B1时走CuTe-DSL融合分支 |

### B1. Dense/shared GEMM1 + clamped SwiGLU-OAI + MXFP8 量化

- 入口：`vllm/models/minimax_m3/nvidia/ops/dense_gemm1_mxfp8.py::fused_gemm1` L590、`run` L633；实现 `Sm100DenseGemm1Mxfp8.kernel` L136；调用 `MiniMaxM3MLP.forward` model.py L361–377。
- wrapper 输入 `x[T,H] BF16`。wrapper 先单独调用 `mxfp8_e4m3_quantize`，再启动融合 kernel；不能说输入量化也在同一个 GEMM kernel 内。
- kernel 输入：`xq[T,H] float8_e4m3fn`、`weight[2I,H] float8_e4m3fn`（gate/up，Cutedsl linear 保存 `[H,2I]` 列主 view，wrapper 转置）；`x_scale`、`weight_scale` 为 E8M0 scale 的 F8_128x4 swizzled flat 存储（编译签名 float8_e8m0fnu；可 uint8 view）。每 32 个 K 元素共享一个 scale；`limit/alpha/beta` 为 FP32 标量。
- kernel 内 FP32 accumulator，令 g=min(XWgᵀ,limit)，u=clamp(XWuᵀ,−limit,+limit)，A=g·sigmoid(alpha·g)·(u+beta)，直接量化输出，无 BF16 intermediate。
- 输出 `QuantizedActivation.data[T,I] float8_e4m3fn`；`.scale[ceil(T/128)*128*(I/32)] uint8`，内容是 UE8M0（每 32 值一 scale，128×4 swizzle）；`orig_dtype=BF16`，`orig_shape=[T,I]`，量化 key `kMxfp8Dynamic`。scale 是反量化乘数，重构约为 q*2^(byte−127)，与 NVFP4 E4M3 block scale 不同。
- 触发：CUDA SM100 family；gate_up 必须 FlashInfer Cutedsl/Cutlass MXFP8 layout；down 输入 quant key 为 kMxfp8Dynamic；I%256=0，H%128=0；T **严格大于** max(1024, `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD`)，且非 FlashInfer autotuning。
- 版本：P 新增；M 的 dense/shared MLP 为 merged gate-up GEMM → activation → down GEMM。
- 证据：[kernel 输入/输出](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/ops/dense_gemm1_mxfp8.py#L590)、[epilogue 数学/scale](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/ops/dense_gemm1_mxfp8.py#L406)、[模型阈值](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/model.py#L348)。

### B2. SwiGLU-OAI 激活本身（dense/shared fallback 与部分 MoE backend）

- `vllm/model_executor/layers/activation.py::SiluAndMulWithClamp.forward_cuda` L246 → `torch.ops._C.silu_and_mul_with_clamp`；C++ host 函数实际名 `csrc/activation_kernels.cu::silu_and_mul_clamp` L306，kernel `act_and_mul_kernel` L110。
- 输入 `[T,2I]`（也支持任意前缀 `[...,2I]`）BF16/FP16/FP32，前 I 是 gate、后 I 是 up；输出 `[...,I]`、dtype 同输入。融合 gate 上界 clamp、up 双侧 clamp、sigmoid(alpha*g)、乘 gate、给 up 加 beta、相乘。数学与上一条一致，输入为 GEMM 后 tensor，所以会存在 GEMM intermediate。
- 路由专家配置使用 `swigluoai_uninterleave`，不能误写成 GPT-OSS 的逐元素 gate/up 交错 layout；`fused_moe/activation.py` L382 将该枚举映射到相同 clamp custom op，masked 路径 `csrc/activation_kernels.cu` L1133 也支持它。
- P/M 上述 activation 实现相同；P 增加更大范围的 GEMM1 fusion，不会让小 batch fallback 消失。
- [Activation contract](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/model_executor/layers/activation.py#L208)、[CUDA 实现](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/csrc/activation_kernels.cu#L306)、[MoE 映射](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/model_executor/layers/fused_moe/activation.py#L382)。

### B3. Sigmoid + correction bias + TopK + renorm + routed scaling

- `vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py::fused_topk_bias` L163 → `vllm_topk_sigmoid` L61 → `ops.topk_sigmoid`。M3 MegaMoE 在 model.py L622 显式调用；普通模块化 MoE 使用 FusedTopKBiasRouter；TRTLLM monolithic 则由外部 MoE backend 自己路由。
- 输入 `router_logits[T,E] FP32`（GateLinear 用 FP32 weights、将 BF16 hidden upcast 后算 logits）；`e_score_correction_bias[E] FP32`；hidden_states `[T,H] BF16` 只用于 token 维/分配；可选 `is_padding[T] bool`。
- 输出 `topk_weights[T,K] FP32`、`topk_ids[T,K] int32`（可通过 indices_type 改 dtype）、`token_expert_indices[T,K] int32` 临时索引。
- 选择用 `sigmoid(logits)+bias`，最终权重从 **未加 bias** 的 sigmoid 分数 gather；对选中 K 项归一化，再乘 routed_scaling_factor。避免把 bias 当成最终 mixing 权重组成部分。
- P/M router 文件相同。源码 [fused_topk_bias](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py#L163)、[模型 FP32 gate](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/model.py#L408)。

### B4. NVFP4 dispatch preparation：量化 + 路由重排 + shared MXFP8 staging + L2 alpha

- `vllm/model_executor/layers/fused_moe/prepare_megamoe.py::prepare_megamoe_nvfp4_inputs` L574，Triton `_prepare_megamoe_nvfp4_inputs_kernel` L268；由 DeepGemmMegaMoEExperts.forward 调用。
- 输入 hidden `[T,H] BF16`；topk_weights `[T,K] FP32`；topk_ids `[T,K] int32`（逻辑/physical expert ids，EPLB 可预先映射）；is_padding `[T] bool`；w2_global_scale `[El] FP32`；l2_activation_scale FP32 scalar。
- 输出 x `[T,H/2] int8`（实际 bit 内容是两个 FP4 E2M1 值/byte）；x_sf `[T,H/64] int32`（4 个 E4M3 block scales 打包在一个 int32；每 16 激活一个 scale）；x_scales `[T] FP32`，为 `max(row_amax,1e-12)/2688` 的逐 token **反量化全局 scale**；topk_idx_out `[T,K] int64`、topk_weights_out `[T,K] FP32`，padding token expert id 设 -1。
- 可选 shared_x `[T,H] float8_e4m3fn`，shared_x_sf 为 `int32` packed UE8M0 `[padded_T,H/128]`，4 个 scale/word，每 32 值一 scale，按 shared runtime BLOCK_M / UTCCP 排列；若 BF16 shared，则 shared_x 是 BF16 copy 且无 scales。l2_alpha `[El] FP32 = w2_global_scale * l2_activation_scale`。
- 这是一条明确的 fusion kernel；不是把所有上述步骤当 5 个独立 launch。P/M 均有；P 把网格改为 `(token, split)`，在 split=0 做 topk 和 token-scale，拆分 K 工作量。
- [Triton](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/model_executor/layers/fused_moe/prepare_megamoe.py#L268)、[contract](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/model_executor/layers/fused_moe/prepare_megamoe.py#L574)、[精确 dtype 测试证据](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/tests/kernels/moe/test_deep_gemm_mega_moe.py#L944)。

### B5. DeepGEMM NVFP4 MegaMoE（可融合 MXFP8/BF16 shared experts）

- `vllm/model_executor/layers/fused_moe/deep_gemm_mega_moe.py::DeepGemmMegaMoEExperts.forward` L722，末尾 `deep_gemm.nvfp4_mega_moe(y,l1_weights,l2_weights,symm_buffer,...)`。融合 EP dispatch、expert GEMM1、clamped SwiGLU epilogue、GEMM2、routing-weight combine；满足条件时 shared expert 也进入同一个 MegaMoE launch。独立 topk 和 prepare 不在此 launch 内。
- public wrapper 输入 hidden `[T,H] BF16`、topk weights `[T,K] FP32`、ids `[T,K] int32`；内部输入是上一条 symmetric buffer。
- routed W13 `[El,2I,H/2] int8 packed E2M1`，W2 `[El,H,I/2] int8 packed E2M1`；W13 block scale `[El,2I,H/64] int32`、W2 scale `[El,H,I/64] int32`，每 word 4 个 FP8 E4M3 scale（每 16 权重一个）；scale 为 MN-major / UTCCP tensor view，不能把 int32 当数值 scale。
- `l1_alpha[El] FP32` = W13 global dequant scale；`l2_alpha[El] FP32` = W2 global dequant scale × L2 activation dequant scale；逐 token x_scales 配合 `use_x_scales=True`。M3.1 checkpoint `weight_global_scale` 在 loader 取倒数；FC1 placeholder input_global_scale 丢弃；FC2 固定 dequant scale **1/16**（原 QAT quant multiplier=16），不能错说 placeholder 1.0 就是真 scale。
- shared MXFP8 W13 `[2S,H] E4M3`、W2 `[H,S] E4M3`，block scale 各 `[2S,H/128]`、`[H,S/128] int32`，每 word 4 UE8M0（每 32 值一 scale）。可选 BF16 shared 没有这些 scale。
- 输出 `y[T,H] BF16`，已经 routed + shared（若共享不能 fused 则 wrapper 另算 shared MLP 并加到 y）。SM100、H/I 均 128 整除、E 可被 EP size 整除；TP>1 时按 token 做 sequence-parallel shard，再 gather；若下层 norm 能融合通信则以零填充其他 rank 行、延后 TP allreduce。
- M 已有通用 MegaMoE 类但 M3 model 未接入；P 在 model 加入以上 NVFP4 配置和 scale 转换。
- [weights/layout](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/model_executor/layers/fused_moe/deep_gemm_mega_moe.py#L410)、[forward](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/model_executor/layers/fused_moe/deep_gemm_mega_moe.py#L722)、[M3 scale 转换](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/model.py#L232)。

### B6. 普通 NVFP4 FusedMoE（非 MegaMoE；外部库 fused pipeline）

- `vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py::TrtLlmNvFp4ExpertsMonolithic.apply` L502 → `flashinfer.fused_moe.trtllm_fp4_block_scale_moe` L552：接收 router logits `[T,E] FP32`、bias `[E] FP32`，内含 routing；模块化 `TrtLlmNvFp4ExpertsModular._invoke_kernel` L324 → `trtllm_fp4_block_scale_routed_moe` L353：输入已经路由的 ids `[T,K] int32` 和 weights `[T,K] FP32`。
- 两者 public inputs hidden `[T,H] BF16`、packed NVFP4 W13/W2 + E4M3 block scales + FP32 global scales/activation参数；M3 PR 把 `qat_activation_key=kNvfp4DynamicToken` 明确传给 FusedMoEFactory。`quantize_nvfp4_per_token_input` 为外部 nvfp4_quantize 单独调用：packed activation `[T,H/2]` byte、block scale `[T,H/16] E4M3`、per_token_scale `[T] FP32`；该前置量化不能宣称已和 routing/GEMM 同 kernel 融合。
- 整体功能：routing/dispatch（monolithic）或已路由 dispatch（modular）→ NVFP4 GEMM1 + clamped SwiGLU-OAI + NVFP4 GEMM2 → topk 加权/归并。vLLM wrapper 证明 fused API 调用，不证明“一次 CUDA launch”。正常输出 `[T,H] BF16`；defer 时输出 `gemm2_permuted[Prows,H] BF16`、expert_weights `[T,K] BF16`（monolithic 内部 routing）或 FP32（modular router）、expanded_idx_to_permuted_idx `[T,K] int32`（-1 代表专家不在本 rank）。后者流向下一条跨层融合。
- 可达备用：`experts/flashinfer_cutedsl_moe.py::FlashInferCuteDSLExperts.apply` L152 → `flashinfer_cute_dsl_fused_moe_nvfp4` L224，接受相同类型 packed x、x_sf `[T,H/16,1] E4M3`，topk ids int32/weights FP32，输出 `[T,H] BF16`。GEMM1 activation参数从配置传入。它支持 per-token activation，但不支持这里的 deferred finalize。
- P/M 上述普通 backend 均已有；P 新增 M3 的 per-token QAT 指定、deferred finalize 接线。P oracle 显式拒绝不支持 per-token activation 的普通 backend；白名单只有 TRTLLM 和 Cutedsl（另有 Marlin W4A16 特例）。不能把 oracle 枚举所有 NVFP4 backend 都列为 M3.1 正常 A4 路径。
- [TRTLLM modular](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py#L324)、[monolithic](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py#L502)、[output contract](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/model_executor/layers/fused_moe/moe_output.py#L11)、[CuteDSL](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/model_executor/layers/fused_moe/experts/flashinfer_cutedsl_moe.py#L152)、[backend guard](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py#L635)。

### B7. MoE finalize + shared add + TP allreduce + residual add + Gemma RMSNorm + MXFP8

- `vllm/models/minimax_m3/nvidia/ops/finalize_allreduce_norm.py::finalize_allreduce_gemma_rms_norm_mxfp8` L772 → `FinalizeAllReduceGemmaRMSNormMxfp8.__call__` L644；publish 复用 `vllm/models/deepseek_v41/nvidia/ops/cute_dsl/all_reduce_mhc.py::_QuadFinalizePublishDeviceKernel`，consumer `_LamportGemmaRMSNormMxfp8Kernel` L205。**明确是两个 PDL chained kernel launches，不是一个**。
- 输入 routed `gemm2_permuted[Prows,H] BF16`、`expert_weights[T,K] BF16/FP32`（见上一条）、`expanded_idx_to_permuted_idx[T,K] int32`；shared_output `[T,H] BF16`（各 rank TP partial）；residual `[T,H] BF16`；norm_weight `[H] BF16`；eps float。
- publish 做本 rank expert 加权求和 + shared，并写入多播 Lamport mailbox（内部 `[3,TP,capacity,H] BF16`，stage_state `[2] int32`）；consumer 等所有 rank、FP32 累加各 rank fragment、加 residual 后 round BF16 并写 `residual_output[T,H] BF16`；计算 Gemma norm `(1+w)*r/sqrt(mean(r²)+eps)`（FP32 统计/运算，norm result BF16 round），再 MXFP8 quant。
- 输出 `QuantizedActivation.data[T,H] E4M3`、`.scale[ceil(T/128)*128*ceil((H/32)/4)*4] uint8`（UE8M0、32-value block、128x4 swizzle），和 `residual_output[T,H] BF16`。这是下一层 qkv projection 的输入，不是最终 lm_head 输入。
- 触发：TP>1、**不开 EP**、CUDA SM100 family、模型 BF16、所有 rank NVLink multicast 支持；普通 TRTLLM NVFP4 monolithic/modular experts 能 defer；1<=T<=32（也被 backend chunk size 限制）；下一层 qkv 输入 quant key 是 MXFP8 dynamic；同 pipeline stage 内有后继层。最后一层不 defer（final norm/lm_head 需要 BF16）；MegaMoE 不采用此路径。M 不存在本模块，P 新增。
- [class contract](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/ops/finalize_allreduce_norm.py#L528)、[input/output](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/ops/finalize_allreduce_norm.py#L644)、[门限/拓扑](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/ops/finalize_allreduce_norm.py#L726)、[模型接线](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/model.py#L1337)。

### B8. 常规 TP allreduce + residual add + Gemma RMSNorm

- `vllm/model_executor/layers/fused_allreduce_gemma_rms_norm.py::fused_allreduce_gemma_rms_norm` L152 → `flashinfer_trtllm_fused_allreduce_norm`（定义 `vllm/compilation/passes/fusion/allreduce_rms_fusion.py`），指定 pattern `kARResidualRMSNorm`、weight_bias=1.0、fp32_acc=True、PDL。
- 输入 hidden_states `[T,H] BF16/FP16`（各 rank partial）；residual `[T,H]` 同 dtype；norm.weight `[H]` 模型 dtype；eps scalar。
- 输出 norm_out `[T,H]` 同 dtype 和 new_residual `[T,H]` 同 dtype；新 residual 写回 hidden_states buffer，传入的 residual 保持不变。语义 r=allreduce(x)+residual；y=r*rsqrt(mean(r²)+eps)*(1+w)。无量化。
- 用于 attention o_proj→post_attention_layernorm，也用于上一层 FFN→下一层 input norm、最终 norm（代码顶部“attention only”是历史注释，不应覆盖实际 model 调用）。TP=1 直接 norm；CUDA contiguous 2D、dtype 支持、workspace/NVSwitch 可用且 token 数和 buffer capacity 合适才走 FlashInfer；失败则显式 allreduce + norm。
- P/M helper 相同，M 已有跨层 FFN allreduce deferral；P 增加 MegaMoE 和更大的 finalize fusion 路径。
- [helper](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/model_executor/layers/fused_allreduce_gemma_rms_norm.py#L139)、[输入约束](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/model_executor/layers/fused_allreduce_gemma_rms_norm.py#L72)、[model 使用](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/model.py#L1202)。

### B9. residual add + Gemma RMSNorm（无通信/通信 fallback 后）

- `vllm/models/minimax_m3/nvidia/model.py::MiniMAXGemmaRMSNorm.forward` L290 → `flashinfer.norm.gemma_fused_add_rmsnorm` L303。
- 输入 x/residual `[T,H]`，通常 BF16（泛化 FlashInfer 路径支持 FP16，模型实际 dtype 以 model config 为准）；weight `[H]` 同模型 dtype；输出原地覆盖 x 为 normalized output、residual 为 x_old+residual_old。和上一条不同，residual 在这里被原地更新。
- 无 residual 时调用普通 `gemma_rmsnorm`，不计 add+norm fusion。P/M 同样存在。
- [函数](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/model.py#L276)。

### 可选 W4A16 附录：Marlin

P 的 oracle 为 `MARLIN && use_nvfp4_w4a16` 显式关闭 per-token A4 metadata。它不是上述 NVFP4 A4 推理分支。`experts/marlin_moe.py::_fused_marlin_moe` L60 的 `ops.moe_wna16_marlin_gemm` 负责 packed 权重反量化 + GEMM；第二个 GEMM 把 topk routing weight 乘入输出（M3 apply_router_weight_on_input=False）；activation 仍是第2条独立 custom kernel，最后 `ops.moe_sum` 做 K 路相加，不能把函数名 fused_marlin_moe 理解为单 kernel 完整 MLP。输入 BF16 `[T,H]`、Marlin re-packed quant weights/scales、路由表；GEMM1 intermediate `[T*K,2I] BF16`，activation `[T*K,I] BF16`，GEMM2 `[T*K,H] BF16`，最终 `[T,H] BF16`。权重物理布局由 Marlin packer 决定，不套用 MegaMoE `[El,2I,H/2] int8` kernel layout。[Marlin 调用](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py#L131)、[第二 GEMM](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py#L195)。

### 横向投影与量化格式补充

- merged gate/up projection 是权重横向合并：`MergedColumnParallelLinear` 的 `[T,H]→[T,2I]`；该横向投影合并在本报告覆盖范围内；它与第1条epilogue fusion属于不同层次。
- GateLinear FP32 是 router GEMM，不等于 sigmoid/topk fusion。各 backend 的 topk 计算位置要区分。
- 不把 `MXFP8 scale uint8`、`NVFP4 E4M3 scale int32-packed`、`FP32 global/dequant scale` 混成同一个 dtype；报告应同时标注存储 dtype 与数值编码。
- main 固定 commit 的原文件可用同一路径替换 SHA 阅读，例如 [main model.py](https://github.com/vllm-project/vllm/blob/6674eedd28d785bed1daf6309615ba337e791aa6/vllm/models/minimax_m3/nvidia/model.py)、[main MegaMoE](https://github.com/vllm-project/vllm/blob/6674eedd28d785bed1daf6309615ba337e791aa6/vllm/model_executor/layers/fused_moe/deep_gemm_mega_moe.py)。PR 与 main 不同 ancestry 造成的通用 backend 差异（例如 post_gemm2_hook）不能一概称为该 PR 引入。


### 官方 checkpoint 参数代入（HF revision 262ae6830934f3512211081b8a3f16425b93a239）

来源为前文列出的固定版本官方模型配置。H=6144、总 E=128、K=4、routed I=3072、shared S=3072；前3层 dense MLP I_dense=12288、后57层 MoE；全部60层为 sparse attention，所以 dense MLP 可达不代表普通 dense-attention 实现也默认可达。BF16 activation；主 linear MXFP8、routed experts NVFP4；alpha=1.702、clamp=7、routed factor=2。TP=t 的普通 dense/shared MLP GEMM1 输出宽度分别 2*12288/t 与 2*3072/t；fusion 激活输出 `[T,12288/t]` 与 `[T,3072/t]`。路由 logits `[T,128] FP32`，topk `[T,4]`；finalize 输出 E4M3 `[T,6144]`，scales `ceil(T/128)*128*192 uint8`，residual BF16 `[T,6144]`。

MegaMoE 每 local expert W13 logical `[6144,6144]`，packed `[6144,3072] int8`；W2 logical `[6144,3072]`，packed `[6144,1536] int8`；W13 scales `[6144,96] int32`（4×E4M3 packed），W2 scales `[6144,48] int32`；前面再加 local expert 维 `El=128/EP`。shared MXFP8 W13 `[6144,6144] E4M3` 和 scale `[6144,48] int32`；W2 `[6144,3072] E4M3` 和 scale `[6144,24] int32`。MegaMoE input prepare 输出 x `[T,3072] int8`、x_sf `[T,96] int32`、x_scales `[T] FP32`。

TopK CUDA 精确路径补充：`csrc/moe/topk_softmax_kernels.cu::topk_sigmoid` L862 → `topkGatingKernelLauncher` L657 → `topkGating` L281。该官方 E=128 命中 fused launcher 的 case128（L701 附近），所以可以确认此配置下 sigmoid+bias+topk+renorm/scaling 在 fused topk kernel，而不是仅 wrapper 的概括。[CUDA source](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/csrc/moe/topk_softmax_kernels.cu#L862)。

普通 TRTLLM/CuteDSL routed 权重的存储补充：compressed_tensors loader 明确分配 W13 `[El,2I,H/2] uint8`、W2 `[El,H,I/2] uint8`；block scales W13 `[El,2I,H/16] float8_e4m3fn`、W2 `[El,H,I/16] float8_e4m3fn`；global scales FP32。TRTLLM 后处理再做 gated-row reorder / shuffle，输出 packed weight uint8、scale 的 byte buffers 会在调用点 `view(float8_e4m3fn)`；不要把它们错误套用为 MegaMoE int8 weight / int32 packed-scale 布局。[loader](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_nvfp4.py#L89)、[TRTLLM pack](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py#L240)。FlashInfer `quantize_nvfp4_per_token_input` 输出 packed activation 是 API 的 8-bit byte tensor，wrapper 本身未 assert dtype；需要精确到 installed FlashInfer 返回的 torch dtype 时应进一步按该依赖版本核对，不能只凭“FP4”推断 torch.float4。

---

## C. ROCm 融合与分支条件

审计版本：PR63232 head `32ad976c451304fd66b5f0d65897ded406bde97c`（下文 P）；main `6674eedd28d785bed1daf6309615ba337e791aa6`（M）。目录实际为 `vllm/models/minimax_m3/amd`，不是 `rocm`。此报告只覆盖 AMD 专有实现及其直接公共依赖；NVIDIA与DSpark/视觉见其他章节。静态源码审读，未运行 GPU。

记号：T=token rows，H=hidden size，D=attention head dimension，Q/K/I=每 TP rank 的 query/KV/index-query head 数，E=专家数，J=每 rank FFN intermediate，k=MoE top-k，B=逻辑稀疏 block 数，P=cache pages。典型专用 decode 固定 H=6144、D=128、Q=16、K=1、I=1（CP 可投影4个 index heads）、J=768、E=128、k=4、block=128、稀疏 topk=16。不要把这些专用核常量当作任意 M3 配置的通用 ABI。

### 首先区分两个版本

- P 包含 `amd/mono/{dispatch,runner,kernels/pre_attn,kernels/post_attn,kernels/index_score}.py`：从层输入 norm 一直融合到 attention、router、MoE、TP reduce 的整层 persistent kernel。M 已改为 `amd/specialized_decode/mono/ffn.py`：仅融合 attention 后半层，输入 norm 和 attention 走普通模块。二者不是重命名后的同一个融合范围。[P runner](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/amd/mono/runner.py#L3)；[M README](https://github.com/vllm-project/vllm/blob/6674eedd28d785bed1daf6309615ba337e791aa6/vllm/models/minimax_m3/amd/specialized_decode/README.md#L3)
- P 的 NVFP4 FlyDSL cache/indexer/attention 系列在上述 M 中不存在。M 不可直接引用这些 P 路径为 main 已支持功能。[P NVFP4 路径入口](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/amd/model.py#L1162)；[M attention 入口](https://github.com/vllm-project/vllm/blob/6674eedd28d785bed1daf6309615ba337e791aa6/vllm/models/minimax_m3/amd/model.py#L980)
- P `indexer_aiter.py` 的 AITER MSA 路径，在 M 变为 `indexer_msa.py` 并增加 AITER CP selector；M 还增加 BF16 Triton CP 的 score/local-topk/gather/merge。[M MSA](https://github.com/vllm-project/vllm/blob/6674eedd28d785bed1daf6309615ba337e791aa6/vllm/models/minimax_m3/amd/indexer_msa.py#L411)；[M BF16 CP](https://github.com/vllm-project/vllm/blob/6674eedd28d785bed1daf6309615ba337e791aa6/vllm/models/minimax_m3/amd/indexer_context_parallel.py#L3)

### 两版本共有或对应实现

|融合内容 / 实际函数| 实现语言 / 技术栈 |输入 → 输出；dtype / 计算含义|触发/边界与源码|
|---| --- |---|---|
|Gemma RMSNorm；`gemma_rmsnorm` → `_gemma_rmsnorm_kernel`| **Triton** |x `[...,N]`、weight `[N]` → out同shape同x dtype。load 后 FP32：`x / sqrt(mean(x²)+eps) * (1+w)`；最终 cast 回 x dtype。weight 模型默认 BF16，但 wrapper 本身不硬断言 BF16。|输入/最终 norm、普通路径每 head QK norm。N 可以 H 或 D。单 Triton pass；[M amd/ops/gemma_rmsnorm.py:28,105](https://github.com/vllm-project/vllm/blob/6674eedd28d785bed1daf6309615ba337e791aa6/vllm/models/minimax_m3/amd/ops/gemma_rmsnorm.py#L28)，P 同路径同函数。|
|Residual-add + Gemma RMSNorm；`gemma_fused_add_rmsnorm`| **Triton** |x,res `[...,N]`、weight `[N]` → `(normed,res_out)`，两者同shape、x dtype；FP32 s=x+res，res_out=cast(s)，norm使用未舍入 FP32 s。|存在 residual 时，单 Triton pass。[M :56,126](https://github.com/vllm-project/vllm/blob/6674eedd28d785bed1daf6309615ba337e791aa6/vllm/models/minimax_m3/amd/ops/gemma_rmsnorm.py#L56)。|
|SwiGLU-OAI clamp+activation+mul；`swiglu_oai_split`| **Triton** |gate_up `[...,2J]` split layout `[gate / up]` → `[...,J]`，默认输入dtype，可指定out_dtype。FP32 `g=min(g,L); u=clip(u,-L,L); y=g*sigmoid(alpha*g)*(u+beta)`。|dense/shared MLP；若 limit=None 无 clamp。是真单核，不包括上下游 GEMM。[M swiglu_oai.py:31,187](https://github.com/vllm-project/vllm/blob/6674eedd28d785bed1daf6309615ba337e791aa6/vllm/models/minimax_m3/amd/ops/swiglu_oai.py#L31)。|
|SwiGLU-OAI + block MXFP8 quant；`swiglu_oai_quantize_mxfp8`| **Triton** |gate_up `[M,2J]`（MoE M=T*k）→ act_q `[M,J]` `float8_e4m3fn`，act_scale `[M,J/32]` `uint8` E8M0。FP32 SwiGLU、每32值amax、pow2 scale，直接FP8 store，无中间BF16 activation。|native MXFP8 MoE 两个 GEMM 之间；J%32=0。scale byte 表示2^(byte−127)，不是普通整数scale。[M swiglu_oai.py:69,130](https://github.com/vllm-project/vllm/blob/6674eedd28d785bed1daf6309615ba337e791aa6/vllm/models/minimax_m3/amd/ops/swiglu_oai.py#L69)；[P调用 native_moe:303](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/model_executor/layers/fused_moe/experts/mxfp8_native_moe.py#L303)。|
|attention TP all-reduce+residual-add+Gemma RMSNorm；`fused_allreduce_gemma_rms_norm` → AITER fused AR op| **外部AITER**；vLLM为Python封装，具体HIP/汇编实现未固定核实 |rank-local partial `[T,H]`、residual `[T,H]`、weight `[H]` → normed/new-residual `[T,H]`，支持FP16/BF16；`norm(sum_r partial_r + residual)`。|TP>1、AITER custom AR enabled、contiguous、消息大小/拓扑/1stage fused AR eligible。条件失败=独立 all_reduce+norm，不是真通信融合。权重bias=1。[M helper:115,139,177](https://github.com/vllm-project/vllm/blob/6674eedd28d785bed1daf6309615ba337e791aa6/vllm/model_executor/layers/fused_allreduce_gemma_rms_norm.py#L115)。外部AITER核的计算dtype不能只凭该wrapper完整证明。|
|AITER consolidated QK/index QK norm + partial NeoX RoPE + cache insert；`rocm_aiter_ops.fused_qknorm_idxrqknorm`| **外部AITER**；Python binding可见，底层实现语言待依赖源码核实 |qkv `[T,(Q+2K+I+1)D]`、4个norm weight `[D]`、cos_sin `[max_pos,rotary_dim]`、positions/slot maps `[T]` int64 → q `[T,QD]`（qkv dtype），index_q `[T,ID]`（BF16或E4M3、与index cache匹配），原位写K/V paged cache以及index-K cache。输入qkv通常BF16；cache auto/BF16或E4M3，FP8 K/V另带FP32 scale。|AITER enabled+`use_aiter_sparse_pa`；auto，或fp8/fp8_e4m3且k/v scale非空。wrapper强制AITER block_size=16、asm_layout=True；index-K仍按稀疏128粒度管理。`skip_index_branch` 可跳过index。[M _aiter_ops.py:2475,2496](https://github.com/vllm-project/vllm/blob/6674eedd28d785bed1daf6309615ba337e791aa6/vllm/_aiter_ops.py#L2475)；[P model:1178](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/amd/model.py#L1178)。AITER实际kernel不在此repo，勿声称已审读其内部计算dtype。|
|公共 HIP `fused_minimax_m3_qknorm_rope_kv_insert`| **HIP C++**（共享`.cu`源码，经ROCm编译） |同上norm/rope；直接通用cache为 `[P,K,128,2D]`，index cache `[P,128,D]`；q output独立，某些分支k/index-k回写qkv供单独insert。支持qkv FP16/BF16；index_q可qkv dtype或E4M3。|AITER consolidated 不适用时；若仍用AITER PA，HIP norm/rope后另调用cache insert，因此这一分支不能写成norm+cache单核。P gfx942专门 native bypass，见下面排除项。[P model:1217,1307](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/amd/model.py#L1307)；[P csrc:2376](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/csrc/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu#L2376)。|
|Triton decode adaptive top-k +可选page-table生成；`_decode_topk_fused_kernel`（入口`minimax_m3_index_decode`）| **Triton** |先独立score：idx_q `[T,I,D]`、index cache `[P,128,D]`（BF16/E4M3），FP32 score `[I,T,B]`；selector输出 int32 topk `[K,T,topk]`，逻辑block id，-1 padding；可额外输出 int32 sparse table `[T*K,topk*8]`、context lengths `[T*K]`（128→16 pages）。partial keys int64，completion int32。|score与topk是两个launch；“fused”指selector内多chunk合并与table生成。score FP32累积，BF16/FP8 dot依分支。topk num index heads需等于KV heads。table参数必须成组提供。[P index_topk.py:1298,1603](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/amd/ops/index_topk.py#L1298)，M同路径有对应实现。|
|AITER MSA top-k + sparse page table；`pa_sparse_block_topk`| **外部AITER编译算子**；底层HIP/汇编实现未固定核实 |FP32 score `[I,T,rounded_B]`、int32 seq lens/page table → int32 topk `[I,T,16]`、page16 table/context lengths；归并score、固定init/local、映射物理页在selector内。|P gfx950、index heads1/2、D128、block128、topk16、max score blocks8192，Q/index cache E4M3；M检查见 `msa_indexer_unsupported_reason`，不应照搬P所有限制。score decode/prefill 与topk是独立kernels。[P indexer_aiter.py:3](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/amd/indexer_aiter.py#L3)；[M indexer_msa.py:515](https://github.com/vllm-project/vllm/blob/6674eedd28d785bed1daf6309615ba337e791aa6/vllm/models/minimax_m3/amd/indexer_msa.py#L515)。|
|Block-sparse attention（QK+causal mask+online softmax+PV）AMD Triton prefill；`_gqa_sparse_fwd_kernel`| **Triton** |q `[T,Q,D]`、cache `[P,K,128,2D]`（query dtype 或 FP8）、topk `[K,T,topk]` int32、metadata int32 → out `[T,Q,D]` query dtype；FP8 cache可scalar/per-token scale，load dequant到query dtype，score/softmax/acc FP32。|gfx942/gfx950 subtiles分别32/64；decode复用common split-K attention，不能算AMD新增decode核。[P sparse_attn.py:3,80](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/amd/ops/sparse_attn.py#L80)。|
|AITER sparse PA（本身attention融合，wrapper不是单核）`minimax_m3_sparse_attn_{decode,prefill}_aiter` → `pa_decode_gluon`| **Triton Gluon**（外部AITER `ops.triton.gluon`） |q/out `[T,Q,D]`通常BF16；ASM paged K/V BF16或FP8；page16 sparse table和context lengths int32。exp_sums/max_logits FP32；split tmp_out为q dtype。FP8 cache时compute_type=AITER fp8，否则q dtype。|prefill/decode都可映射到gluon decode；另有split reduce，因此不能把Pythonwrapper说成单launch。[M sparse_pa.py:432,479,523](https://github.com/vllm-project/vllm/blob/6674eedd28d785bed1daf6309615ba337e791aa6/vllm/models/minimax_m3/amd/ops/sparse_pa.py#L523)。|
|AITER routing sigmoid+bias+top-k+shared slot追加（FSE）| **外部AITER** routing；shared-slot追加可能另含PyTorch操作，不能归成一个固定语言kernel |router logits `[T,E]` FP32、bias `[E]` FP32 → routed ids/weights `[T,k+n_shared]`（ids整型、weights FP32）；shared expert作为固定slot同批专家执行。|FSE opt-in `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS`、非EP；gfx950+AITER MoE才能shared追加融合入grouped-topk，设num_expert_group=topk_group=1。不是整个MoE单核；其它路径可仅torch concat。[P model:331,425,461](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/amd/model.py#L331)；[M model:327](https://github.com/vllm-project/vllm/blob/6674eedd28d785bed1daf6309615ba337e791aa6/vllm/models/minimax_m3/amd/model.py#L327)。|
|Native MXFP8 grouped GEMM2+route-weight multiply；`_mxfp8_grouped_gemm_kernel`| **Triton** |act FP8 `[T*k,J]`、E8M0 scales `[T*k,J/32]`、w2 FP8 `[E,H,J]`、scales uint8 `[E,H,J/32]`、routing weights FP32 → g2 `[T*k,H]` FP32，内核dot_scaled FP32累加后乘route weight。|完整`fused_moe_mxfp8_native`还执行sort、量化、GEMM1、SwiGLU+quant、GEMM2、sum，故多kernel；最终sum→`[T,H]`原hidden dtype（BF16）。[P native_moe:57,244,306](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/model_executor/layers/fused_moe/experts/mxfp8_native_moe.py#L244)。|

### P 独有 NVFP4 路径

这些kernel的cache不是普通FP8。D128每token平均占64字节packed E2M1值 + 8字节E4M3 per-16 scale，共72 bytes；物理上head-page内先存所有token的数据区、再存scale区（见A2）；cache tensor dtype=uint8。main KV cache `[P,2K,128,72]`，index cache `[P,128,72]`；还有k_scale/v_scale FP32 global scale。FlyDSL gfx950，block128、D128、topk16、GQA group16为主要固定合同。

|算子| 实现语言 / 技术栈 |I/O和计算dtype|条件、融合范围与源码|
|---| --- |---|---|
|`minimax_m3_norm_rope_nvfp4_insert`| **FlyDSL** |qkv `[T,(Q+2K+I+1)*128]` BF16 contiguous，4 norm weights BF16，cos_sin BF16，positions/main slot/index slot int64，k/v global scales FP32 → q `[T,Q*128]` BF16；index_q `[T,I*128]` E4M3；原位写上述uint8 caches。norm/rope FP32中间。|Q/K/index-Q/index-K Gemma norm+RoPE+NVFP4量化+两套cache写入，真正一launch；K和index-K不回写qkv。[host:27](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/amd/ops/norm_rope_nvfp4_insert/host.py#L27)；[kernel:3](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/amd/ops/norm_rope_nvfp4_insert/kernel.py#L3)。|
|`minimax_m3_nvfp4_index_score_decode`| **FlyDSL** |index q `[R*query_len,I,128]` E4M3，uint8 cache、block_table `[R,B]` int32、seq_lens `[R]` int32 → score `[I,R*query_len,>=B]` FP32。|NVFP4在寄存器解量化为round-sat E4M3，FP8 MFMA、FP32 score，fuse causal max reduction/init/local sentinel；不是topk融合。[host:81](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/amd/ops/nvfp4_index_decode/host.py#L81)；[kernel:3](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/amd/ops/nvfp4_index_decode/kernel.py#L3)。|
|`minimax_m3_nvfp4_index_score_prefill`| **FlyDSL** |q `[T,I,128]` E4M3，uint8 index cache、block_table `[R,B]`、cu_seqlens `[R+1]`/seq_lens `[R]` int32 → `[I,T,>=B]` FP32。|I=1/2/4；同decode的dequant+dot+max+causal/sentinel语义，调度选择shared/nonshared FlyDSL，topk另launch。[host:94](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/amd/ops/nvfp4_index_prefill/host.py#L94)。|
|`minimax_m3_nvfp4_index_score_push_cp`| **FlyDSL** |本rank q `[R*4,128]` E4M3、cache uint8、metadata int32、peer指针/flags → 各head所属rank FP32 score `[rows,row_stride]`。|TP4+配置indexer_decode_cp、query_len=4等runtime条件；融合Q交换、NVFP4解量化score及向head owner push、等待完成。topk仍另launch。[host:165](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/amd/ops/nvfp4_index_decode/host.py#L165)；[dispatch:69,128](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/amd/indexer_nvfp4.py#L69)。|
|`minimax_m3_index_topk_{prefill,decode}_indices`| **FlyDSL** |FP32 scores `[I,T,S]`、有效pages或seq_lens int32 → int32 `[I,T,16]` logical ids。|gfx950、block128、topk16、<=8192 blocks；独立topk核，不与NVFP4 score融合，也不生成PA page table。[host:33,50,83](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/amd/ops/index_topk_decode/host.py#L33)。应列辅助算子，不能虚增为“score+topk fusion”。|
|`minimax_m3_nvfp4_sparse_decode_fly`| **FlyDSL** |q/out `[T,Q,128]` BF16，uint8 KV `[P,2K,128,72]`，topk `[K,T,k]` int32，metadata int32，global scales FP32。|融合NVFP4→E4M3舍入→BF16解量化、QK、causal mask、FP32 online softmax、BF16 P、PV；多个split时partial FP32、最后到达split在同kernel归并。GQA Q/K=16；可支持uniform speculative query_len。[host:112](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/amd/ops/nvfp4_sparse_decode/host.py#L112)；[kernel:3](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/amd/ops/nvfp4_sparse_decode/kernel.py#L3)。|
|`minimax_m3_nvfp4_sparse_prefill`| **FlyDSL** |q/out `[T,Q,128]` BF16，uint8 KV，topk `[K,T,16]` int32，query_req_id/query_abs_pos `[T]` int32。|每WG4query tokens，走所选页union；K/V为E2M1×E4M3的FP16精确product（注意不同于decode E4M3舍入），Q/P FP16，score/softmax/acc FP32，输出BF16。[host:40](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/amd/ops/nvfp4_prefill/host.py#L40)；[kernel:3](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/amd/ops/nvfp4_prefill/kernel.py#L3)。|

### Persistent与CP实现语言补充

| 算子 / 路径 | 实现语言 / 技术栈 | 对应说明 |
|---|---|---|
| PR整层persistent mono | **FlyDSL** | pre-attention、index score、attention、post-attention/FFN等整层融合，条件见下文 |
| main FFN半层mono | **FlyDSL** | `amd/specialized_decode/mono/ffn.py`；融合范围不同于PR整层 |
| main `pa_sparse_block_topk_cp` | **外部AITER编译算子** | vLLM为Python调用，内部HIP/汇编实现未固定核实 |
| main `indexer_context_scores` / `_context_score` | **Triton** | BF16 CP局部score |
| main `local_topk_keys` / `merge_topk_keys` | **Triton** | 两侧选择kernel；中间collective all-gather是独立通信步骤 |

### Persistent mono：融合范围必须按版本分开

#### P 整层

入口 `MonoDecodeRunner.run_sparse_layer`、`build_post_attn_kernel`，K1 `emit_k1` 被内联进 post-attn kernel。层输入 `(ar,res)` 均 `[S,6144]` BF16 → `(ar_out,h_mid)` 同shape BF16；主缓存FP8、index cache E4M3；positions/slot int64，block table/seq_lens int32。`ar`为上层已reduce MoE输出，`res`为残差；h_mid为本层attention后residual，ar_out为本层已reduce MoE输出。

融合链：FP32 residual add→Gemma norm→per-token FP8 activation→ptpc-FP8 QKV/index GEMV→per-head norm/RoPE/cache写入→index score/max/topk/page table→FP8 sparse QK/PV和softmax→FP8量化+O GEMV→TP allreduce+residual+post norm→router→sigmoid+bias top4→MXFP4 up/gate→SwiGLU→down+route-weight→TP allreduce。BF16残差/激活边界、FP32norm/score/softmax/累积；小batch MoE A16W4，大batch阈值分支改A8W4并使用MXFP8中间；shared expert可单独MXFP8或者融合到MXFP4专家表。

支持约束：显式 `VLLM_ROCM_MINIMAX_M3_MONO`、TP4/DP1/PP1、非EP、FP8 KV/index cache、block128、context<=1M、1..16 decode token rows、无prefill/microbatch、metadata合法、所有256 CTA驻留（MI355X）、缓存slot共享及weight几何严格校验。embedding、leading dense layers、final norm仍普通模块，不能称整个model一个kernel。

源码：[runner:204,230](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/amd/mono/runner.py#L204)、[pre_attn:6](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/amd/mono/kernels/pre_attn.py#L6)、[post_attn:6](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/amd/mono/kernels/post_attn.py#L6)、[dispatch:59](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/amd/mono/dispatch.py#L59)。

#### M FFN半层

`FfnRunner.__call__`→`build_ffn`/`kffn`。输入unreduced o_proj `part [S,6144]` BF16、residual同shape BF16、slot `[>=S]` int64、postnorm weight BF16、router weight `[128,6144]` FP32/bias FP32、AITER shuffle的专家weight `float4_e2m1fn_x2`、scale uint8 E8M0。输出 `(out,res_out)` `[S,6144]` BF16，out为reduced MoE输出，res_out为attention后residual。

单launch融合：attention TP allreduce（rank-ordered FP32 sum→BF16）+FP32 residual add→Gemma norm→BF16 X→FP32 router GEMM/logits→sigmoid+bias top4、renorm×2→shared+selected experts A16W4 MFMA→SwiGLU（alpha1.702,beta1,limit7）→BF16 mid→weighted down→MoE allreduce。H6144,J768,E128,k4，shared专家index128；FP32 norm/router/reductions，BF16专家activation。没有QKV/RoPE/index score/attention融合。

约束：`--monokernel`、gfx950/256 CUs、TP4、Model Runner V2、`--moe-backend aiter`、AITER_MXFP4_MXFP4 post-load weights、FP8 KV/index cache/block128、1..16 plain decode或FULL graph；DP/PP/EP、DCP/PCP、ubatching、LoRA、KV connector/offload、weight updates、cross-layer index reuse等被拒绝。前3层普通、sparse层3到last由配置控制。slot<0 padding不参加expert、MoE output为0，residual output未指定。

源码：[ffn.py:3,853](https://github.com/vllm-project/vllm/blob/6674eedd28d785bed1daf6309615ba337e791aa6/vllm/models/minimax_m3/amd/specialized_decode/mono/ffn.py#L3)、[runner:126](https://github.com/vllm-project/vllm/blob/6674eedd28d785bed1daf6309615ba337e791aa6/vllm/models/minimax_m3/amd/specialized_decode/mono/runner.py#L126)、[weights:165](https://github.com/vllm-project/vllm/blob/6674eedd28d785bed1daf6309615ba337e791aa6/vllm/models/minimax_m3/amd/specialized_decode/runtime/weights.py#L165)、[spec:23](https://github.com/vllm-project/vllm/blob/6674eedd28d785bed1daf6309615ba337e791aa6/vllm/models/minimax_m3/amd/specialized_decode/runtime/spec.py#L23)。

### M 新增 CP 相关融合

- **AITER `pa_sparse_block_topk_cp`**：输入每rank分片FP32score、IPC candidate buffers/epoch counters、int32 seq_lens/page table；输出owned-head int32 topk、sparse page16 table/context lengths。在一个selector launch中融合本地候选topk、IPC exchange、全局merge及page table生成；其前面的 `pa_sparse_block_score_decode` 仍独立launch。调用方明确“两kernels”，不能写成score+select一体。`index_query [T,total_I,128]` E4M3，cache E4M3，score local-shard FP32。[indexer_msa.py:411](https://github.com/vllm-project/vllm/blob/6674eedd28d785bed1daf6309615ba337e791aa6/vllm/models/minimax_m3/amd/indexer_msa.py#L411)。
- **BF16 CP score `indexer_context_scores`/`_context_score`**：q `[T,total_I,128]` BF16、index cache `[P,128,128]` BF16、int32 tables/lens → score `[total_I,T,ceil(B/world)]` FP32；fuse paged read+BF16 QK dot+scale+causal block max，round-robin shard。main的 `minimax_m3_indexer_cp=true`、ROCm TP>1 BF16 index cache。[ops/indexer_context_parallel.py:45,160](https://github.com/vllm-project/vllm/blob/6674eedd28d785bed1daf6309615ba337e791aa6/vllm/models/minimax_m3/amd/ops/indexer_context_parallel.py#L45)。
- **`local_topk_keys`**：FP32 local score `[heads,T,local_B]`、int32 seq_lens→int64 `[heads,T,k]` packed `(score,global block id)`；fuse forced init/local、selection、key packing。**`merge_topk_keys`**：int64 gathered `[world,heads,T,k]`→int32 `[owned_heads,T,k]`，可融合sparse page table/context lengths。中间 `all_gather` 单独collective，不是单核communication fusion。[ops/indexer_cp_exchange.py:196,282](https://github.com/vllm-project/vllm/blob/6674eedd28d785bed1daf6309615ba337e791aa6/vllm/models/minimax_m3/amd/ops/indexer_cp_exchange.py#L196)。

### 排除项 / 不确定边界

- `qkv_proj` 把Q/K/V/index-Q/index-K合并一个GEMM，`gate_up_proj`把gate/up合并一个GEMM；这是横向projection合并，常被称fused projection，但不是额外norm/activation融合kernel。[P model:709,1128](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/amd/model.py#L709)。
- P `gfx942` sparse norm/rope/cache走 `_native_gfx942_qknorm_rope_cache`，避免HIP fused op崩溃；这里有独立norm/rope/insert，不能把gfx950融合能力覆盖到所有ROCm。[P model:1011,1153](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/amd/model.py#L1011)。
- P MiniMax-M3.1的ModelOpt `moe_quant_algo: NVFP4` checkpoint在ROCm没有原生NVFP4 MoE backend；需online requant为MXFP8或MXFP4。**NVFP4 KV支持不等于NVFP4专家支持**。[P model:372](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/amd/model.py#L372)。
- MTP使用同AMD decoder/norm，enorm、hnorm分别执行后torch.cat+eh_proj，没有在本文件发现特殊“两个norm+concat+projection”融合。[P amd/mtp.py:100](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/amd/mtp.py#L100)。
- 外部AITER op只有vLLM调用合同在本次审计范围；未固定并审读AITER自身commit，故其内部计算dtype、launch数量（除明确描述）应标外部依赖待核。通用AITER MoE backend分支由设备/量化/shape动态选择，不能从`FusedMoEFactory`类名推断全层单核。

### 官方 M3.1 Flash checkpoint 的关键可达性补充（优先于泛化条目）

前文固定版本官方配置显示：H6144、Q64/K4/index4、D128、60层全部sparse attention；仅MoE频率前三层0、后57层1，即前三层是 **sparse attention + dense MLP**。main specialized_decode `runtime/spec.py` 明确要求 sparse layer列表恰为 `[3,...,last]`，所以原始官方配置会在启用 `--monokernel` 时拒绝；即使绕过该条件，`runtime/weights.py` 又要求QKV和O projection的post-load weight dtype=BF16，而不能直接把官方MXFP8加载权重视作符合合同。因此 M 的FFN persistent kernel应放“代码存在但该checkpoint默认不满足”的条件附录，不计作官方M3.1 Flash已启用融合。此处只做源码条件判断，未做实际加载验证。[spec:94](https://github.com/vllm-project/vllm/blob/6674eedd28d785bed1daf6309615ba337e791aa6/vllm/models/minimax_m3/amd/specialized_decode/runtime/spec.py#L94)；[weights:213](https://github.com/vllm-project/vllm/blob/6674eedd28d785bed1daf6309615ba337e791aa6/vllm/models/minimax_m3/amd/specialized_decode/runtime/weights.py#L213)。

P `dense_prefix` 按 `is_moe_layer` 判断前缀，不要求前三层dense attention，因此该layer-mix条件允许官方配置。P mono把attention MXFP8权重在初始化时转BF16再转per-channel FP8，并把router FP32 gate转BF16；其router GEMV使用BF16 gate/activation、累加FP32，和 M mono保留FP32 router权重的合同不同。不要写P mono router weight仍是FP32。[P runner:80](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/amd/mono/runner.py#L80)；[P weights:3](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/amd/mono/weights.py#L3)。

---

## D. DSpark、视觉与编译融合边界

固定版本：PR #63232 head `32ad976c451304fd66b5f0d65897ded406bde97c`；main `6674eedd28d785bed1daf6309615ba337e791aa6`。本节是只读源码分析，没有运行 GPU 或验证特定部署的编译图。`T` 为当前 draft/query token 数，`C` 为 context token 数，`L` 为 draft 层数，`H` 为 hidden_size，`hq/hkv` 为本 TP rank 的 query/KV head 数，`D=128`，`R` 为 rotary_dim，`B` 为请求数，`V` 为词表大小。

### 1. 版本与可达性

PR 新增 `vllm/models/minimax_m3/common/dspark.py`，main 此 SHA 没有该文件。PR 的 `MiniMaxM3DSparkForCausalLM` 继承 Qwen3 DSpark，替换为 MiniMax Gemma RMSNorm、partial RoPE 和 dense SwiGLU-OAI FFN；draft 不自带 embedding/LM head 时共享 target。配置入口将 `DSparkMiniMaxDraftModel` 解析为 `minimax_m3_dspark`，`dflash_config.causal` 默认 false，因此不能拿 target 稀疏 attention 的融合清单直接套到 DSpark。源码：[DSpark 定义](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/common/dspark.py#L1)、[配置适配](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/config/speculative.py#L1060)。

### 算子实现语言速查

下表为D1–D7、V0–V3新增实现列；后续tensor契约表继续按输入/输出组织，不将某个输入tensor误标成一种kernel语言。

| 编号 / 算子 | 实现语言 / 技术栈 | 实现边界 |
|---|---|---|
| D1 draft QK norm+RoPE+KV insert | **CUDA C++**；共享源码含HIP分支 | Python custom-op调用`.cu` kernel；完整DSpark后端可达性仍以正文限制为准 |
| D2 多层context KV writer | **CUDA C++** | CUDA专用context norm/RoPE/cache kernel |
| D3 多层context K/V横向GEMM | **Python / PyTorch封装；GEMM依后端** | `F.linear`或量化linear后端；不是专有Triton融合kernel |
| D4 residual+Gemma RMSNorm | **NVIDIA：外部FlashInfer；AMD对应norm：Triton** | 同B9/C的复用类别；不表示完整DSpark在所有平台均可运行 |
| D5 SwiGLU-OAI | **NVIDIA：CUDA C++；ROCm所述通用fallback：PyTorch组合** | 原生组合表达式不自动计成一个融合kernel |
| D6 Markov bias+temperature+Gumbel+block argmax | **Triton** | sampler融合kernel；后续全局选取等步骤按正文契约 |
| D7 draft dense paged attention | **外部FlashInfer TS / FlashAttention，按后端选择** | 两个入口见D7表；vLLM调用源码不足以统一断言底层语言 |
| V0 视觉QKV横向GEMM | **Python模块；GEMM依量化后端** | `QKVParallelLinear`，不与V1/V2/attention自动合成一个kernel |
| V1 `vit_mrope_setup` | **Triton** | position/frequency/cos/sin准备 |
| V2 `triton_mrope` | **Triton** | Q/K partial MRoPE |
| V3 `MMEncoderAttention` | **外部FlashAttention / FlashInfer / cuDNN / ROCm后端** | 后端身份可确认；底层语言随安装版本与选择变化，未统一固定核实 |

### 2. DSpark 确认可达的融合

#### D1. 每个 draft 层的 Q/K Gemma RMSNorm + partial NeoX RoPE + K/V cache 写入 + 可选 FP8 Q

调用路径：`MiniMaxM3DSparkAttention.forward` → `torch.ops.vllm.minimax_m3_dspark_qknorm_rope_kv_insert` → `dspark_qknorm_rope_kv_insert` → `ops.fused_minimax_m3_qknorm_rope_kv_insert` → CUDA `fused_minimax_m3_qknorm_rope_kv_insert`。

| 输入/输出 | shape / dtype | 含义 |
|---|---|---|
| 输入 `qkv` | `[T,(hq+2*hkv)*128]`，FP16/BF16；contiguous | 一次 QKV GEMM 的 packed 结果；这里无 indexer 分支 |
| 输入 `q_norm_weight`,`k_norm_weight` | 各 `[128]`，与 qkv 同 dtype | Gemma 权重，使用 `1+w`，不是直接乘 `w` |
| 输入 `cos_sin_cache` | `[max_pos,R]`，与 qkv 同 dtype | partial RoPE cos/sin；`R` 为 8 的正倍数且不大于 128 |
| 输入 `positions` | `[T]`，INT64 | token 的绝对位置 |
| 输入 `slot_mapping` | `[T]`，INT64，可无 | cache 物理槽位；在 custom op 运行时从 attention context 读取 |
| 输出/原地 `qkv` | 同输入 shape/dtype | K 归一化/旋转；Q 按输出模式写入，V 不作 norm/RoPE |
| 可选输出 `q_fp8_out` | `[T,hq*128]`，`float8_e4m3fn`（CUDA） | 直接输出归一化/旋转后、按 `q_fp8_scale` 缩放的 FP8 Q，省一次后续 Q quantization |
| 输出/原地 `kv_cache` | `[num_blocks,hkv,block_size,256]`；BF16/FP16（auto）或 UINT8（存储 E4M3 bits） | 最后一维前 128 是 K，后 128 是 V；只在有 slot_mapping 时插入 |

公式：`q' = RoPE(q * rsqrt(mean(q²)+eps) * (1+wq))`，K 同理。FP8 Q 模式下 caller 直接使用 `q_fp8_out`，不要把 qkv 中 Q 切片称为有效的最终 Q。该 wrapper 传 `q_fp8_scale=layer._q_scale_float`，但没有向通用 kernel 传 KV scale 参数；不能写成“DSpark draft token K/V 按任意每层 scale 缩放”。context writer D2 才明确传每层 K/V scales。

触发：PR MiniMax DSpark attention 每次 forward 都调用；是否写 cache 由实时 slot mapping 决定。`query_pre_quantized` 在 draft cache dtype 字符串以 `fp8` 开头时启用。后续 attention output 明确回到 hidden dtype；若 FP8 Q + `FLASHINFER_TS` + MXFP8 `o_proj`，attention output 可用 FP16，GEMM 再写模型 dtype。源码：[caller/FP8 output](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/common/dspark.py#L178)、[运行时 context wrapper](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/common/dspark.py#L82)、[CUDA contract](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/csrc/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu#L2199)。

版本：通用 target kernel 与 `q_fp8_out` 在 main 已存在，main wrapper 从 [L960](https://github.com/vllm-project/vllm/blob/6674eedd28d785bed1daf6309615ba337e791aa6/csrc/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu#L960) 起；PR 新增 MiniMax DSpark 调用路径，不能称整个 QK-norm/RoPE kernel 全为 PR 首创。

#### D2. 所有 draft 层 context K norm + RoPE + cache 写入/量化，一次 kernel

调用路径：`MiniMaxM3DSparkModel.precompute_and_store_context_kv` → `DSparkContextKVWriter.__call__` → `ops.fused_minimax_m3_dspark_context_kv` → CUDA `fused_minimax_m3_dspark_context_kv` / `launchFusedMiniMaxM3DSparkContextKV`。

| 输入/输出 | shape / dtype | 含义 |
|---|---|---|
| 输入 `all_kv` | `[C,L,2,hkv,128]` BF16 | D3 生成的 token-major、多层 K/V；第三维 0=K，1=V |
| 输入 `k_norm_weight` | `[L,128]` BF16 contiguous | 各 draft 层 K 的 Gemma norm 权重 |
| 输入 `cos_sin_cache` | `[max_pos,R]` BF16 contiguous | 所有 draft 层共用 RoPE 表；`R∈{16,32,64}` |
| 输入 `positions`,`slot_mapping` | 各 `[C]` INT64 contiguous | context 绝对位置和物理槽位；slot<0 不写入 |
| 输入 `cache_offsets` | `[L]` INT64 | 各层独立 cache allocation 相对第 0 层的元素地址偏移 |
| 输入 `k_scale`,`scale_offsets` | `k_scale` FP32 scalar；offsets `[2,L]` INT64 | 第 0 层 K scale 地址基准；各层 K、V scale 的 float 地址偏移 |
| 输出/原地各层 cache | 各 `[num_blocks,hkv,block_size,256]` BF16，或 UINT8 保存 E4M3 | 归一化并旋转 K、直接 V，按对应 K/V scale 编码 FP8（`x/scale`） |

融合覆盖原本的 permute copy、每层 K norm + stack、positions.repeat、RoPE、每层 reshape/cache write。K norm 在 FP32 算，再按 BF16 materialization 语义做后续 RoPE；无独立返回 tensor。

触发条件很严格：CUDA；BF16 context；head_dim=128；NeoX；R=16/32/64；draft模型未配置额外 `v_scale` 变换（不是指cache反量化scale）；所有 draft 层共用 slot mapping（同 data_ptr），cache 已绑定、shape/stride/dtype 一致且连续、block size 为 2 的幂、cache 相对地址以 16 元素对齐；cache dtype 允许 `auto/bfloat16/fp8/fp8_e4m3`。ROCm、非 BF16 或不支持的 layout 走基类路径，不应把 D2 标为全平台默认。源码：[调用与 fallback](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/common/dspark.py#L339)、[layout 检查/语义](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/common/ops/dspark_context_kv.py#L3)、[CUDA 输入校验](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/csrc/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu#L2492)。PR 新增，main 无此函数。

#### D3. 多 draft 层 context K/V 的水平 GEMM 合并

`DFlashQwen3Model._build_context_kv_buffers` 把所有层 K/V 权重拼接，`_project_context_kv_token_major` 仅做一次 `F.linear` 或一个 `ReplicatedLinear`。输入为先独立 Gemma norm 后的 `[C,H]`，输出 `[C,L,2,hkv,128]`；BF16 未量化权重 `[L*2*hkv*128,H]` 与 activation 同 dtype。MXFP8 变体把各层权重和每行 `weight_scale` 原样拼接，内层 linear 仍按量化后端执行，输出恢复模型 dtype。这里是多 GEMM 的水平合并，**没有把 hidden norm、GEMM、D2 融成一个 kernel**。

只有全部投影为 unquantized，或全部 MXFP8 且加载后尚未预处理重排权重时建合并权重；其余量化走逐层 projection。源码：[权重合并](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/model_executor/models/qwen3_dflash.py#L539)、[MXFP8 合并](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/model_executor/models/qwen3_dflash.py#L580)、[实际投影](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/model_executor/models/qwen3_dflash.py#L661)。PR 为 MiniMax 复用通用基类，不能按文件存在就称 main 已支持 MiniMax DSpark。

#### D4. Gemma residual add + RMSNorm；D5. SwiGLU-OAI clamp + sigmoid + multiply

这些复用 B 节 target 通用算子；这里说明 DSpark 的调用位置，不作为额外kernel类别重复计数。

- D4：`MiniMAXGemmaRMSNorm.forward(x,residual)` → FlashInfer `gemma_fused_add_rmsnorm`；`x,residual:[T,H]`，weight `[H]`，模型激活 dtype（典型 BF16）；原地输出 `residual'=x+residual` 与 `x'=residual' * rsqrt(mean(residual'²)+eps)*(1+w)`。无 residual 时为 `gemma_rmsnorm`。DSpark 的 input/post-attention/hidden/final norm 均用此 MiniMax norm，不是 Qwen 普通 RMSNorm。[DSpark 配置](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/common/dspark.py#L306)、[CUDA wrapper](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/nvidia/model.py#L276)。ROCm 使用 AMD `MiniMAXGemmaRMSNorm`，由 AMD 部分展开。
- D5：`MiniMaxM3DSparkMLP.forward` → `SiluAndMulWithClamp.forward_cuda` → `_C.silu_and_mul_with_clamp`；packed `gate_up:[T,2I]` → `[T,I]`，输入输出同 dtype。`g=min(gate,limit)`，`u=clamp(up,-limit,limit)`，`y=g*sigmoid(alpha*g)*(u+beta)`。gate/up 投影为水平 merged GEMM，但不是这个激活 kernel 的一部分。ROCm 非默认 alpha/beta 走 `forward_native`，不能算作相同 C++ fused op。[DSpark MLP](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/common/dspark.py#L235)、[activation contract](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/model_executor/layers/activation.py#L207)。

#### D6. Markov bias add + temperature + Gumbel draw + logits-cache 写入 + block argmax

`DSparkSpeculator._sample_markov_step` → `gumbel_sample` → `_gumbel_sample_kernel` + `_gumbel_combine_kernel`。第一 kernel 融 bias、可选 logits-cache store、temperature、Gumbel 与 block 内 argmax；第二 kernel 合并各词表块 winner，**不是一次 launch 完成整个 sampler**。Markov embedding 和 Markov projection GEMM 仍在外部。

输入 `logits:[B,V]` 与 `bias:[B_bias,V]` 同浮点 dtype（通常 BF16，wrapper 并未限定只支持 BF16），`bias_rows:[B]` INT64 指定每行 bias；`expanded_idx_mapping:[B]` 整数请求状态索引，`temperature:[max_reqs]` FP32，`seed:[max_reqs]` 和 `pos:[B]` INT64；可选 `logits_cache:[max_reqs,N,V]` 与 logits 同 dtype，col 为标量或 `[B]` 整数。输出 sampled IDs `[B]` INT64；临时 block maxima `[B,ceil(V/1024)]` FP32（`use_fp64=True` 时 FP64）和同形 INT64 IDs。bias addition FP32 算后 round 回 logits dtype，随后 FP32 采样；cache 保存温度缩放前的 biased logits，供 rejection sampler 使用。

融合 bias 仅在 `draft_logits` buffer 存在、无 draft-to-target scatter、无 watermark、无 acceptance estimator 时启用；否则先显式加 bias，再普通采样。PR 新增 in-kernel bias add 以及相应调用；main 有 Gumbel sampler，但无此 bias 参数融合。源码：[caller/guards](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py#L198)、[kernel](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/v1/worker/gpu/sample/gumbel.py#L289)、[wrapper/output dtype](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/v1/worker/gpu/sample/gumbel.py#L393)。

#### D7. Draft dense paged attention：QKᵀ + scale/mask + softmax + PV

与 target 稀疏 attention 分开计：`MiniMaxM3DSparkAttention.forward` 在 D1 写 cache 后调用 `self.attn(q,None,None,output_dtype=...)`，从已写好的 context + draft token KV cache 做 attention。融合计算为 `O=softmax(scale*QKᵀ+mask)V`；D1/D2 写 cache 与 attention 本体仍是不同 kernel。输入 Q 在模型边界为 `[T,hq*128]`，`Attention.forward` reshape 成 `[T,hq,128]`；输出分配 `[T,hq*128]`，backend 使用 `[T,hq,128]` view。`query_pre_quantized=True` 明确跳过 Attention 的第二次 QuantFP8。[调用](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/common/dspark.py#L224)、[跳过重复 Q quantization 与输出分配](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/model_executor/layers/attention/attention.py#L467)。

| backend/入口 | 实现语言 / 技术栈 | 实际输入 | 输出与触发条件 |
|---| --- |---|---|
| PR 新增 `FlashInferTSImpl.forward` → `flashinfer.attention.prims_ts.BatchDecodePagedTSWrapper.run` | **外部FlashInfer `prims_ts`**；vLLM为Python调用，底层语言未按依赖版本固定核实 | Q `[B,Sq,hq,128]`，每请求相同 query-block 长度 `Sq`，`T=B*Sq`；K/V 为 `[pages,hkv,page_size,128]` strided views，来自 packed `[pages,hkv,page_size,256]`；Q/K/V 同 dtype：未量化为 FP16/BF16，FP8 cache 为 UINT8 storage 转 E4M3 view，Q 为 D1 直接写的 E4M3；`seq_lens:[B]` INT32，`block_tables:[B,max_pages]` INT32；`bmm1_scale=attn_scale*q_scale*k_scale`，`bmm2_scale=v_scale` 为 float | kernel O `[B,Sq,hq,128]`。FP8 Q 的原生 O 为 FP16（此 wrapper 不选择 FP8 O）；若 caller 请求 BF16，则先 FP16 temporary 再 copy/cast 到 BF16。**MiniMax DSpark 在 FP8 Q + FLASHINFER_TS + MXFP8 o_proj 时主动请求 FP16 O**，这样避免 FP16→BF16 中转，后续 o_proj quantizes FP16 并输出模型 dtype。未量化 Q 直接写 caller dtype。拒绝 output_scale/output_block_scale，因此没有本路径的 fused output quantization。 |
| `FlashAttentionImpl.forward` → `_FA4_DENSE_ATTENTION_KERNEL` → `flash_attn_varlen_func`（按选择 FA 版本派发） | **外部FlashAttention，随FA版本/后端变化**；不能把Python custom-op包装统一视为Triton或CUDA C++ | Q `[T,hq,128]`；K/V view `[pages,page_size,hkv,128]`，FP8 storage 也 view 为 platform FP8；`query_start_loc:[B+1]` INT32、`seq_lens:[B]` INT32、`block_table:[B,max_pages]` INT32；FP8 时 q/k/v descale 为 `[B,hkv]` FP32 expand views，FA4 固定单位 scale 可传 None | 模型 caller 明确请求 `hidden_states.dtype`（典型 BF16）；FP8 Q + FA4 在 DSpark 注释中说明使用 BF16 output。非 FP8 Q 的 FP16/BF16按模型 dtype。backend 不支持 output_scale/output_block_scale。FA3/FA4具体设备与dtype组合受 backend选择限制，不能声称所有GPU支持FP8Q。 |

FLASHINFER_TS 的特定限制：SM100/103/107、page_size=16/32/64/128、head dim=64/128/256（MiniMax 使用128），无 sink/ALiBi/softcap/KV-sharing，dense non-MLA，FlashInfer >=0.7.1；非因果 draft 使用 `mask_type="dense"`。非因果 sliding window 在 wrapper 通过 page table 截取近似处理，页边界会多见最多 `page_size-1` 旧 token，同一 query block 共享 window start，相比逐 query 精确窗口还有最多 `Sq-1` drift。因此不要把 TS non-causal sliding-window 与精确 token-level mask 写成数学上完全等价。源码：[backend完整契约](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/v1/attention/backends/flashinfer_ts.py#L3)、[plan dtype/mask](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/v1/attention/backends/flashinfer_ts.py#L194)、[run与cast](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/v1/attention/backends/flashinfer_ts.py#L397)、[caller FP16条件](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/common/dspark.py#L181)。main此SHA没有 `flashinfer_ts.py`。

FLASH_ATTN 源码：[forward契约/cache view/descales](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/v1/attention/backends/flash_attn.py#L1196)、[实际dense调用](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/v1/attention/backends/flash_attn.py#L1417)。这是 main 已有共享 attention backend；PR新增 MiniMax DSpark caller。FA4 SM90 的 FP8-KV 路径明确期望 native FP16/BF16 Q、在 kernel 内 dequant K/V，而 MiniMax DSpark 当前根据 cache dtype 直接 emit FP8 Q；该组合存在静态代码契约不一致风险，未作GPU测试，应列为未确认组合，不能泛化“所有FA4 FP8-KV都兼容此draft”。[SM90区别](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/v1/attention/backends/flash_attn.py#L1150)。

其他 DSpark 融合复查：未发现漏掉的 MiniMax 专有数值 fused kernel。QKV 的 merged projection、gate/up merged projection属于水平 GEMM 合并（已在 D1/D5注明）；confidence head 是 cat/cast + linear + sigmoid，Markov embedding/project在D6之外，不能称为已确认一个 kernel。基类 context fallback 的普通 RMS/RoPE/cache函数不应与D2同时按默认执行路径计数。

#### 明确排除：NVFP4 gathered Markov bias

`qwen3_dspark.py` 有 `nvfp4_gathered_bias`，能融合 candidate weight gather、NVFP4 dequant、dot、base logit add 和 scatter；但 PR 配置 [speculative.py:L1566](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/config/speculative.py#L1566) 只允许 `Qwen3DSparkModel/Qwen3OmniDSparkModel` 设置 `dspark_draft_topk`，MiniMax `DSparkMiniMaxDraftModel` 会抛 ValueError。因此虽继承该 method，正常配置不可达，不应列为 MiniMax 已启用的融合。kernel 对比 main 未新增；PR 对 `nvfp4_emulation_utils.py` 的变化主要是权重离线反量化分块，不是此 sampler 融合。

### 3. 视觉侧确认可达的融合

#### V0. Q/K/V 横向投影合并

`MiniMaxVLAttention.qkv_proj` 使用 `QKVParallelLinear`，一次投影同时生成Q/K/V：输入 `x[N,embed_dim]`（通常BF16）→ packed `[N,3*hv*Dv]`（模型输出dtype）；权重与bias经该linear的量化后端处理，`bias=True`。与target的五路投影不同，这里是视觉三路QKV，没有indexer分支；它也没有与后续MRoPE/attention合成一个kernel。两版都有。[构造与调用](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/common/vision_tower.py#L103)

#### V1. 3D positions + per-axis frequency multiply + cos/sin setup

PR `vit_mrope_setup` → `_vit_mrope_setup_kernel`；main `VitMropeSetup.forward` → 同名 Triton kernel。输入三个 FP32 inv_freq，shape `[t_dim/2]`、`[h_dim/2]`、`[w_dim/2]`；host `grid_thw:[G,3]` 和 spatial_merge_size，内部一次 H2D 打包 INT32 `cu_seqlens:[G+1]` 和 `grids:[G,3]`。输出 cos、sin 各 `[3,N,Rv/2]` FP32，`N=sum(t*h*w)`、`Rv=t_dim+h_dim+w_dim`。按 spatial-merge patch 顺序算 t/h/w 坐标并乘 inverse frequency，再求 cos/sin。**只有各轴自己的 section 有效，其余 columns 未初始化，消费者用 mask 只读取有效部分。**

每次 vision metadata prepare（eager 或 graph replay 准备）可达。PR 与 main 的数值 kernel 相同，变化是 main 的 pluggable class 改为函数调用，不应计为 PR 新增融合。[PR kernel/wrapper](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/model_executor/layers/rotary_embedding/mrope_vit_setup.py#L17)、[main class](https://github.com/vllm-project/vllm/blob/6674eedd28d785bed1daf6309615ba337e791aa6/vllm/model_executor/layers/rotary_embedding/mrope_vit_setup.py#L95)、[模型调用](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/common/vision_tower.py#L458)。

#### V2. Q/K 同 kernel 的 partial 3D NeoX MRoPE

`MiniMaxVLAttention.forward` → `triton_mrope` → `_triton_mrope_forward`。输入 q、k 均为 packed qkv 的 strided view `[N,hv*Dv]`，模型浮点 dtype（通常 BF16）；cos/sin 为 V1 的 FP32 `[3,N,Rv/2]`；section lengths 标记 t/h/w 范围。输出原地 q、k，同 shape、同输入 dtype；旋转算术 FP32，未旋转 tail 保持原样。无 repack 和独立 upcast tensor，Q/K 同一次 launch。main 和 PR 都有。[模型调用](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/common/vision_tower.py#L134)、[Triton wrapper](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/model_executor/layers/rotary_embedding/mrope.py#L163)。

V3（共享 attention 类别）：同一个 `MiniMaxVLAttention.forward` 的 `MMEncoderAttention` 对 q/k/v `[1,N,hv,Dv]` 做 fused/Flash attention，输出同形 context，再 reshape/project。具体 CUDA FlashAttention、FlashInfer/cuDNN、ROCm 后端按选择变化，并不是 MiniMax 专有 kernel；这是与语言模型共用的attention融合类别，在此标记视觉调用点。`cu_seqlens`、max-seqlen 等元数据为 INT32，具体含义随 backend 有 token offsets 或 element offsets 转换，不能用一个固定 layout 描述所有后端。[caller](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/common/vision_tower.py#L169)、[backend metadata](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/models/minimax_m3/common/vision_tower.py#L433)。

视觉层的 residual add + LayerNorm 没有发现显式 MiniMax 专有 fused kernel；源码是 `nn.LayerNorm` 和显式 residual add。PR 把 patch Conv3d 权重固定 FP32、再 cast 回 pixel dtype，这是精度变更，不能称为新的融合。CUDA graph capture 是减少 launch 调度开销，也不等同把整座 vision tower 融成单 kernel。

### 4. 通用 compiler fusion 不等于本模型算子清单

[pass_manager.py:L163](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/compilation/passes/pass_manager.py#L163) 根据开关注册 allreduce-rms、norm-quant、act-quant、QK norm/RoPE/cache 等 pass；注册仅代表允许匹配，不证明该模型 graph 中存在匹配。

| 通用 pass | 实现语言 / 技术栈 | 本模型判断 |
|---| --- |---|
| RMSNormQuant / allreduce-RMS | Python图改写；目标kernel依匹配后端 | NVIDIA MiniMAXGemmaRMSNorm 已直接调用 FlashInfer Gemma kernel，而通用 NormQuant 匹配 `vllm_ir.rms_norm/fused_add_rms_norm`；不能拿通用普通 RMS fusion 加入确认列表。target 自有 allreduce+Gemma norm 在B节单独说明。 |
| ActivationQuant | Python图改写；目标kernel依匹配后端 | 当前 matcher 使用 plain `silu_and_mul`，不匹配 MiniMax 的 `SiluAndMulWithClamp`（alpha/beta/clamp）。不存在凭 `fuse_act_quant=True` 就自动生成 MiniMax OAI+MXFP8 的证据。target 的自有 GEMM1/activation/quant fusion 应另列。 |
| QK norm+RoPE / RoPE+cache | Python图改写；目标kernel依匹配后端 | target/DSpark 显式调用专有 custom op，内部 norm/RoPE 已封装；不能把通用 pass 名称再次算作额外执行 kernel。 |
| AttnQuant | Python图改写；目标kernel依匹配后端 | 当前注册 static FP8 与 NVFP4 dynamic output 模式，而且需 backend `fused_output_quant_supported`。没有 MXFP8 output pattern，因此 released MXFP8 o_proj 不能据此声称 attention→MXFP8 输出融合。其他量化配置是否匹配属于条件可达，需要运行图确认，不作默认事实。 |
| MLA 相关 | Python图改写；本模型不据此认定可达 | MiniMax 此路径用 MHA/GQA sparse/dense attention，不是 MLA；不能因 repo 有 MLA fusion 就列入。 |
| TorchInductor 自动 elementwise fusion | TorchInductor代码生成；生成语言依设备与后端，未导出实际编译图 | 可能随编译版本/shape 生成，源码静态研究不能枚举固定 kernel 名、数量或稳定 I/O ABI；应报告为编译依赖范围，不冒充确定融合。 |

证据：[NormQuant 的 IR target](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/compilation/passes/fusion/rms_quant_fusion.py#L43)、[ActQuant matcher](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/compilation/passes/fusion/act_quant_fusion.py#L33)、[AttnQuant registration](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/vllm/compilation/passes/fusion/attn_quant_fusion.py#L405)。

### 5. 范围与验证限制

- 这里的“所有”按源码可达、显式数值融合类别列举；不把 serving 输入准备、token bookkeeping、通信缓冲搬运、通用 rejection sampler 所有内部 Triton 子函数逐一称为“MiniMax 模型融合算子”。runtime sampler属于模型以外共享路径；D6因PR专门融合MiniMax DSpark的Markov bias而单列。
- dtype 区分 logical FP8 与物理 UINT8；E4M3 cache 的 UINT8 不是 INT8 quantization。D2 的 offset tables 是地址偏移，不是 norm 参数。
- 查看了测试 `tests/models/minimax_m3/test_dspark_draft_compile.py`：其意图检查 compiled draft attention 实际写 KV；没有在本环境执行该测试，不能写“测试通过”。[测试](https://github.com/vllm-project/vllm/blob/32ad976c451304fd66b5f0d65897ded406bde97c/tests/models/minimax_m3/test_dspark_draft_compile.py#L40)。
