# Qwen3-235B PD 本地精度与输出质量复测（2026-09-19）

## 环境与复现范围

使用三台服务器的 48 个 NPU：197 为 Prefill TP16/DP1，198 和 159 合计 Decode TP1/DP32/EP32。各节点使用独立测试容器 `qwen235-acc-20260919`，SSH 均使用私钥。

- vLLM Ascend：`e139b7d573d3769fd1407d5027d7d4831f5469f0`，启动任务时获取的 main 版本。
- vLLM：对应 release tag `v0.28.0`，commit `2cf0a6915ce544dc493a0990f2ea38d81601128a`。
- `.github/vllm-main-verified.commit` 的 `84030bbe3d74d99bad477a3d2e37a973ccd8865c` 仅记录备查；本次按照“对应 release tag”的要求安装 v0.28.0。
- AISBench：与用例一致的 `v3.1-20260609-master`，commit `0da56eadb2ac85c31c2540f4f5b69af3ec5717a5`。
- 基础 nightly 镜像、依赖清单和设备信息见 [run-manifest.json](run-manifest.json) 与 [environment](environment)。Python 3.12.13、torch 2.10.0+cpu、torch_npu 2.10.0.post4.dev20260715、CANN 9.1.0。

模型使用官方 `vllm-ascend/Qwen3-235B-A22B-W8A8`。仓库 revision 为 `a19b5fffb5a09ee570acfa0473146a76a1910a05`；66 个文件（含 56 个权重分片）全部通过 SHA-256 比对，见 [official-model-integrity.json](official-model-integrity.json)。原机器上同名近似的 QuaRot 权重经检查 56 个分片均不同，因此未用于精度测试。

GSM8K test 为 1,319 条，数据文件 SHA-256 为 `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`。保留原用例参数：temperature=0.6、top_p=0.95、thinking=true、batch_size=64、max_out_len=32768、服务 seed=1024。基准 96.74%，容差 ±1 个百分点，允许区间 [95.74%, 97.74%]。

本地适配仅包括服务器地址、模型/数据本地路径、Prefill 的 `--data-parallel-address` 从原配置字面值 `12321` 改为节点 IP、setup.py 的两处 ROOT_DIR 改用 abspath，以及 AISBench 增加 `--dump-eval-details`。配置见 [原始 YAML](QWEN3_235B_PD.original.yaml)、[本地 YAML](QWEN3_235B_PD.local.yaml)。vLLM 先构建 Rust frontend，再按用户要求进行 editable 安装；Ascend 在三台机器均完成源码编译安装。

实际评测命令：

```bash
ais_bench --models vllm_api_general_chat_custom \
  --datasets gsm8k_gen_0_shot_cot_chat_prompt \
  --dump-eval-details --debug
```

在真实权重下载期间做过 dummy 权重的服务连通性预检；dummy 输出没有纳入本报告的任何精度或文本质量统计。

## 检查方法与边界

对 eval details 中每题的完整 `origin_prediction` 做严格 UTF-8 解码、U+FFFD/异常控制字符扫描、重复短语/长行/结束标签检查、空输出与答案提取失败检查，并比较规范化后的整段输出和 prompt 是否重复。数字答案相同不视为重复输出。逐题 `correct` 用于独立复算准确率。

重复短语启发式阈值为同一 16-word 片段至少 8 次；长行至少 30 字符且重复至少 6 次；`</think>` 至少 3 次会标记。合理的推理复核也可能触发重复短语，因此该类标记需人工阅读确认，不能全部直接称为模型退化。乱码计数只把实际 U+FFFD 当作明确替换字符，不把正常中文或其他语言文字一概视作乱码。

保存文本重新分词所得 token 长度不等于原始生成 token ID 序列，也不能替代 finish_reason。超长样本仅据此标记接近上限；根因定位需要另外的对照实验，本次不将现象归因于某个具体提交、驱动或通信模块。
