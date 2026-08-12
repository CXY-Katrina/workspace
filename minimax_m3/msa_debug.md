结论：#13987 合入后的乱码更像 attention 输出存在未初始化值或跨核 combine 读取了未完成的 partial workspace，不是 tokenizer 问题。当前代码中已经能确认一个缺陷，并发现一个高度可疑的 A3 FD 竞态。
1. 已确认：validTopK == 0 时输出没有初始化
#13987 将 MiniMax M3 decode 从 Triton 切换到 npu_sparse_attention_score。PR #13987
Arch22 kernel 中：
非 FD 路径遇到 validTopK == 0 直接 continue；
FD 路径只有 isFdPartial == true 时才写 neutral partial；
FD 未分片任务、普通路径、部分 kvSLoopNum == 0 情况都可能不写 gO。
位置见 [sparse_attention_score_kernel_arch22.h (line 239)](/C:/Users/Katrina/Desktop/cxy/vllm_project/vllm-ascend/csrc/attention/sparse_attention_score/op_kernel/arch22/sparse_attention_score_kernel_arch22.h:239)。
测试里甚至已经把这个问题标记为 known limit，并用 xfail 跳过：
select_num_idx=0 does not initialize all normal/FD output or partial-result storage

见 [test_flash_decoding_generalized.py (line 342)](/C:/Users/Katrina/Desktop/cxy/vllm_project/vllm-ascend/csrc/attention/sparse_attention_score/tests/test_flash_decoding_generalized.py:342)。
这在 FULL_DECODE_ONLY 下尤其危险：图内 padding 行的 select_num_idx 通常为 0，而 NPU 算子输出由运行时分配，不保证为 0。垃圾 attention output 继续进入 residual、MoE 和后续层后，logits 会被污染，最终表现就是重复 token、异常 Unicode 或乱码。
修复要求是：
validTopK == 0 时必须把对应 Q token、KV head group 的 gO 写 0；
如果输出 LSE，也应写 -inf；
FD split task 写 neutral partial；
FD unsplit task直接写零到最终输出；
将现有 xfail 改成强制通过。
2. 高度可疑：A3 neutral partial 写入和 combine 之间缺少明确完成事件
MiniMax 的短序列通常只有 1 个有效 block，但 Host tiling 读取不到 device 上的 select_num_idx，会退化为按 topK=16 估算：
[sparse_attention_score_tiling.cpp (line 250)](/C:/Users/Katrina/Desktop/cxy/vllm_project/vllm-ascend/csrc/attention/sparse_attention_score/op_host/sparse_attention_score_tiling.cpp:250)
因此单请求 A3 decode 会出现：
runtime 实际有效 block：1；
host 按 16 个 block 将任务拆到多个 AIC；
一个 core 写真实 partial；
其余 core 写 neutral partial；
combine 随后跨核读取这些 partial。
但 WriteNeutralPartial() 在两次 GM DataCopy 后，没有建立明确的 MTE3 completion event：
[block_epilogue_rescale_o.hpp (line 530)](/C:/Users/Katrina/Desktop/cxy/vllm_project/vllm-ascend/csrc/attention/sparse_attention_score/op_kernel/attn_infra/epilogue/block/block_epilogue_rescale_o.hpp:530)
随后 kernel 直接进入 PipeBarrier + SyncAll + combine。如果 SyncAll 不能保证之前的 GM MTE3 写入对其他 AIV 已完全可见，combine 就可能读到上一次 workspace 内容。这个现象会是偶现的，并且非常符合“有时正常、有时乱码”。
建议在 neutral partial 写完后增加受支持的 MTE3 completion/fence，并在进入 combine 前保证：
所有 partial O/LSE GM 写完成
        ↓
跨核同步
        ↓
combine 读取 partial workspace
不要仅依赖当前的 PipeBarrier<PIPE_ALL>()。
3. A3 FD 的验证覆盖不足
当前 functional FD 测试主要按照 Arch35/Ascend 950 的 28 AIC 环境设计；Arch22 只有 host tiling-key 单测，缺少真正的 kernel accuracy 测试。同时文档还写着 A2/A3 不支持 inner_precise=4，但 Python wrapper 固定传入 4，并且 Arch22 FD gate 也要求 4：
[msa_m3_npu.py (line 8)](/C:/Users/Katrina/Desktop/cxy/vllm_project/vllm-ascend/vllm_ascend/models/minimax_m3/ops/msa_m3_npu.py:8)
[sparse_attention_score_tiling.cpp (line 279)](/C:/Users/Katrina/Desktop/cxy/vllm_project/vllm-ascend/csrc/attention/sparse_attention_score/op_host/sparse_attention_score_tiling.cpp:279)
这至少说明 A3 FD 的契约和测试还没有闭环。
最短定位方式
建议依次做三组隔离：
保留 NPU decode，但临时禁止 enableFd_。
乱码消失：问题在 FD partial/combine。
仍乱码：重点查零有效行和普通 NPU decode。

关闭 FULL_DECODE_ONLY，运行 eager。
eager 稳定、graph 乱码：基本确认是 padding 行未初始化。
eager 也乱码：优先检查 valid=1 时 neutral partial 的写入可见性。

加两个精确 micro case，循环至少 1000 次：
q_tokens=1, topK=16, select_num=1：检查真实 partial + 多个 neutral partial。
q_tokens=8，第一行 select_num=1、其余行 select_num=0：模拟 graph padding，并断言后七行严格为 0。

当前判断优先级是：
select_num_idx=0 输出未初始化——已确认；
A3 neutral partial 写入与 combine 的 GM 可见性——高度可疑；
Arch22 inner_precise=4/FD 未做完整功能验证；
PR review 提到的 Arch35 UB overlap 在当前最新提交里看起来已经调整过，不是 A3 乱码的首要原因。
