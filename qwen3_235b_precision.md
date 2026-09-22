【qwen3_235b_w8a8 gpqa】基线0.7212
80.5.9.128 root/Huawei@12，容器：B070_dyb

在B060版本精度OK，在B061版本精度不OK。
差异只有vllm-ascend
B060 vllm_ascend_commitid:660c4582aa580ce98edc9b681bb6ff6d03153575
B061 vllm_ascend_commitid:c267db731d03e18042047ea1594e033673e46a6f
B070 vllm_ascend_commitid:993782efc842308f646dcf80a562d45288f265d3
vllm_commitid:2cf0a6915ce544dc493a0990f2ea38d81601128a

0、脚本
export HCCL_BUFFSIZE=1024
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

vllm serve /mnt/weight/Qwen3-235B-A22B-w8a8-rot \
    --served-model-name 'qwen' \
    --host 0.0.0.0 \
    --port 8004 \
    --async-scheduling \
    --tensor-parallel-size 4 \
    --data-parallel-size 4 \
    --data-parallel-size-local 4 \
    --data-parallel-start-rank 0 \
    --data-parallel-address 80.5.17.118 \
    --data-parallel-rpc-port 2345 \
    --max-num-seqs 16 \
    --max-model-len 40960 \
    --max-num-batched-tokens 13864 \
    --gpu-memory-utilization 0.9 \
    --enable-expert-parallel \
    --quantization "ascend" \
    --trust-remote-code \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'

1、qwen3-235b的这个case是在测gpqa这个数据集198题，容器内/usr/local/python3.11.10/lib/python3.11/site-packages/ais_bench/datasets目录下是有这个数据集的
2、aisbench精度配置文件内容是：
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="vllm-api-general-chat",
        path="/mnt/weight/Qwen3-235B-A22B-w8a8-rot",
        model="qwen",
        stream=False,
        request_rate=0,
        use_timestamp=False,
        retry=2,
        api_key="",
        host_ip="80.5.17.118",
        host_port = 8004,
        url="",
        max_out_len = 32768,
        batch_size = 64,
        trust_remote_code=False,
        generation_kwargs=dict(
            top_p = 0.95,
            top_k = 20,
            temperature = 0.6,
            ignore_eos=False,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
3、精度测试指令：ais_bench --models vllm_api_general_chat --datasets gpqa_gen_0_shot_cot_chat_prompt --debug --dump-eval-details
4、B070精度输出：/tmp/b070_235b_gpqa
