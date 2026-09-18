证据链已完全闭合。所有关键事实确认完毕,现在输出最终报告。

---

# CI 失败分析报告:Nightly-A2 / qwen3-30b-a3b-bf16-a2-performance

## 观察到的失败

失败步骤 **Step #15 `Run Pytest (YAML-driven)`**,exit code 1。

测试 `tests/e2e/nightly/single_node/models/scripts/test_single_node.py::test_single_node[Qwen3-30B-A3B-BF16-A2-TP4]` 失败,根断言点:

```
tools/aisbench.py:349: AssertionError
AssertionError: some aisbench cases failed, info were shown above.
```

唯一失败的 aisbench case 为性能用例(`case_type=performance`,GSM8K,180 prompts,batch_size=45)。直接失败原因:

```
Performance verification failed. The current Output Token Throughput is 757.0391 token/s,
which is not greater than or equal to 0.97 * baseline 802.4382.
```

即当前吞吐 **757.04 token/s** < 阈值 **0.97 × 802.4382 = 778.37 token/s**,未达标。

## 已确认事实

### 1. good/bad 运行环境与配置完全一致(排除环境/硬件/镜像/配置差异)

| 维度 | 成功 (good `bdab8bd`) | 失败 (bad `c7ca0b6`) |
|---|---|---|
| Image | `vllm-ascend:nightly-ci-main-a2` | 同 |
| CANN | `9.1.0_20260731131545` | 同 |
| Runner label | `linux-aarch64-a2b3-4`(A2 ×4) | 同 |
| vLLM 版本 | v0.28.0 | 同 |
| server_cmd | `--async-scheduling --tensor-parallel-size 4 ... --additional-config {"enable_cpu_binding":true} --compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY",...}` | 完全一致 |
| baseline / threshold | 802.4382 / 0.97 | 同 |
| 配置文件 `Qwen3-30B-A3B-BF16-A2.yaml` | — | **good..bad 区间内未改动,两版本 IDENTICAL** |

配置文件注释明确:**baseline 是 V1 路径下三次稳定 A2 ×4 测量的均值(804.6923 / 802.1003 / 800.5221 tok/s),保留 3% 容差,基线变更需团队批准**。good 运行实测 791.08 tok/s(> 778.37 阈值)真实通过,证明基线合理。

### 2. good/bad 运行路径发生根本性分歧(根因机制)

| 运行时指标 | 成功 (good) | 失败 (bad) |
|---|---|---|
| Model Runner V2 | **未启用**(日志 0 处) | **已启用**(`mrv2_utils.py:200] Model Runner V2 is enabled for Qwen3MoeForCausalLM`,共 6 处) |
| Worker 警告 | 无 | `npu model runner v2 is in developing, some features doesn't work for now.`(×4 workers) |
| 加载模型模块 | `model_runner_v1.py:4015` | `model_runner.py:358`(V2 路径) |
| init engine 编译耗时 | 10.71s | **57.97s**(约 5.4×) |
| init engine 总耗时 | 57.50s | 126.09s |
| `ascend_moe_forward_complete` 编译失败重编 | 无 | **4 个 rank 均触发**(`'_OpNamespace' 'vllm' object has no attribute 'ascend_moe_forward_complete'`) |
| TPOT | 18.9 ms | 19.7 ms(↑ 单 token 时延) |
| Output Token Throughput | 791.08 tok/s(过阈) | 757.04 tok/s(未过阈) |

bad 运行切到了尚未优化完成的 V2 model runner 路径,且 V2 路径下 MoE 的 `ascend_moe_forward_complete` 算子缺失导致编译失败重编,直接拉高编译开销并推高 TPOT,最终拉低 Output Token Throughput。

## 回归边界

- **good_commit**(被测,成功):`bdab8bde86aad5eba3b362f923a3ff03b2c0e772`(分支 main,已验证是 bad 的祖先)
- **bad_commit**(被测,失败):`c7ca0b676b9668535f467b78cff1274a4ddb63b2`(分支 main)
- 区间共 185 个提交。区间内唯一把 `Qwen3MoeForCausalLM` 纳入默认 V2 白名单的提交,即 **bad_commit 本身**。

## 主要嫌疑(高置信)

**PR #16626** `c7ca0b676b9668535f467b78cff1274a4ddb63b2` —— `[Feature][MRV2] expand default MRv2 architecture whitelist and add dspark`(作者 yjyang62)。

### 支持证据(因果链闭合)

1. **失败日志事实**:吞吐 757.04 < 778.37 阈值 → `aisbench.py:349` 断言失败。
2. **实际运行入口/配置**:`test_single_node → _run_benchmarks → run_aisbench_cases → AisbenchRunner`(日志行 1232-1270 完整调用栈)。本 Job 模型 `Qwen/Qwen3-30B-A3B` 的架构为 `Qwen3MoeForCausalLM`(bad 日志 `mrv2_utils.py:200` 直接打印 `Model Runner V2 is enabled for Qwen3MoeForCausalLM`)。
3. **受影响源码路径**:该 commit **新建** `vllm_ascend/mrv2_utils.py`(diff 为 `--- /dev/null`),其中定义:
   ```python
   DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES = frozenset({
       "Qwen3ForCausalLM",
       "Qwen3MoeForCausalLM",   # ← 本 Job 模型
       "MiniMaxM2ForCausalLM",
       "DeepseekV3ForCausalLM",
       "DeepseekV32ForCausalLM",
       "GlmMoeDsaForCausalLM",
       "DeepseekV4ForCausalLM",
       "Qwen3_5MoeForCausalLM",
   })
   ```
   `is_default_v2_model_runner_model()` 据此白名单决定是否默认启用 V2。
4. **good..bad 边界**:
   - good commit `bdab8bd` 中 `vllm_ascend/mrv2_utils.py` **不存在**(`git cat-file` 报 `does not exist in 'bdab8bd'`),即 good 运行根本无 V2 默认机制 → 走 V1。
   - 该文件由 c7ca0b6(#16626)新增(`--diff-filter=A` 命中,区间内唯一)。
   - 基础白名单 PR #16203(仅默认 `Qwen3ForCausalLM`)不在 good..bad 区间内,不构成混淆。
5. **运行路径日志强佐证**:bad 日志启用 V2(`mrv2_utils.py:200` ×6),good 日志未启用(0 处);bad 出现 V2 专属的 `npu model runner v2 is in developing` 警告与 `ascend_moe_forward_complete` 编译失败重编,good 均无。
6. **性能劣化的代码级解释**:V2 runner 尚在开发中(日志明示),其 MoE forward 完成算子 `ascend_moe_forward_complete` 未注册导致 4 rank 编译失败重编,编译耗时 10.71s→57.97s,TPOT 18.9→19.7ms,吞吐 791.08→757.04。

## 反证与排除

- **排除 Runner/硬件波动**:同 image、同 CANN、同 runner label、同 server_cmd;且日志存在明确的代码路径切换(V1→V2)与算子级编译失败,非随机噪声。
- **排除基线需修改**:baseline 三次稳定测量均值,good 运行 791.08 仅比阈值高 ~1.6% 仍真实通过,证明阈值合理;bad 是真实代码回归(路径切换)导致,非基线漂移。按精度性能基线保护规则,**不建议放宽阈值**。
- **排除配置变更**:配置文件 good/bad IDENTICAL,区间内未改动。
- **排除 #16203 干扰**:它不在 good..bad 区间,good 运行本无 V2 默认机制。

## 证据缺口

- 无法登录 runner 做同配置复跑(规则:不能因无法复跑大模型性能测试而把 otherwise 一致的离线证据链降级为 insufficient)。
- 未逐项验证 V2 runner 在 Qwen3MoE 上的优化路线图;但日志 `is in developing` 警告 + `ascend_moe_forward_complete` 算子缺失已是充分的路径级反证,指向 V2 路径未就绪。

**结论等级:likely**(离线取证充分,待运行复现)。

## 建议的后续动作

1. **首选**:在 `Qwen3-30B-A3B-BF16-A2.yaml` 性能 Job 的 `envs` 中 pin `VLLM_USE_V2_MODEL_RUNNER: "0"`,将其锁定回 V1 runner,直至 MRV2 对 Qwen3MoE 的性能优化与 `ascend_moe_forward_complete` 算子适配完成。这与 #16626 自身在多个测试文件(RLHF conftest、SFA V1 precision、attention V1 precision)里因 MRV2 未就绪而 pin 回 V1 的既有做法一致。
2. 推进 MRV2 在 `Qwen3MoeForCausalLM` 上的性能对齐与 `ascend_moe_forward_complete` 算子注册/适配,完成后即可移除该 pin。
3. 复跑本 Job 验证 V1 路径吞吐恢复至 ~791 tok/s 量级。

## 关联 PR

**PR #16626**(`c7ca0b676b9668535f467b78cff1274a4ddb63b2`)。因果链闭合:失败日志事实 → 运行入口(test_single_node/_run_benchmarks/run_aisbench_cases)→ 受影响源码(`vllm_ascend/mrv2_utils.py` 的 `DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES` 白名单新增 `Qwen3MoeForCausalLM`)→ good..bad 区间内的具体改动(c7ca0b6 新建该文件)。

```json
{
  "problem_category": "开发代码",
  "root_cause_summary": "PR #16626(c7ca0b6)新建 vllm_ascend/mrv2_utils.py 并将 Qwen3MoeForCausalLM 加入默认 MRv2 白名单,导致本 Job 的 Qwen3-30B-A3B 性能用例从 V1 切换到尚未优化完成的 V2 runner;V2 路径下 ascend_moe_forward_complete 算子缺失引发编译失败重编、编译耗时增至 5 倍、TPOT 上升,Output Token Throughput 从 791.08 降至 757.04 token/s,低于 0.97×802.4382=778.37 阈值触发断言。离线取证充分,待运行复现。",
  "improvement_measures_summary": "在该性能 Job 配置 pin VLLM_USE_V2_MODEL_RUNNER=0 锁定回 V1,直至 MRV2 对 Qwen3MoE 的性能优化与 ascend_moe_forward_complete 算子适配完成;基线保持不变,不建议放宽阈值。"
}
```
