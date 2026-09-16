# Qwen3-8B-W8A8 单卡部署与 AISBench 入门

面向第一次使用 Ascend、Docker 和 AISBench 的同学。按顺序复制命令，即可完成：启动服务容器 → 启动模型 → 单次 curl → 在另一个容器安装 AISBench → GSM8K lite 精度测试 → 2k 输入、2k 输出、并发 16 性能测试。

## 1. 先看清楚环境和执行位置

| 项目 | 本教程使用的值 |
| --- | --- |
| 服务器 | `192.168.9.160`，SSH 用户 `root` |
| 硬件 | Ascend 910B3，单卡 64 GB；示例使用 0 号卡 |
| 服务镜像 | `quay.io/ascend/vllm-ascend:v0.23.0`，服务器已有 |
| 模型权重 | `/mnt/weight/Qwen3-8B-W8A8` |
| 教程工作目录 | `/mnt/share/c00814587/tutorial` |
| 服务容器 | `qwen3-8b-server` |
| 测试容器 | `qwen3-aisbench` |
| 服务端口 / 模型名 | `8000` / `qwen3-8b` |
| 精度数据 | 服务器现有 GSM8K lite，测试集 32 条 |
| 性能规格 | 输入 2048 tokens、输出 2048 tokens、客户端并发 16、正式请求 128 条 |

**准备两个终端窗口 A、B。** 两个窗口都先通过 SSH 登录服务器。之后 A 用于查看服务，B 用于 curl 和 AISBench。

- **本地终端**：你自己电脑上的 PowerShell、Git Bash 或终端。
- **宿主机**：SSH 登录后的服务器，通常看到 `root@worker-160`。
- **服务容器**：通过 `docker exec` 进入，运行 `vllm serve`。
- **测试容器**：通过 `docker exec` 进入，运行 `ais_bench`。
- 后文 Linux 命令在服务器或容器的 **Bash** 中运行，不要直接粘贴到本地 PowerShell。
- 遇到 `cat ... <<'PY'`，请把整个代码块一起复制，包括最后单独一行的 `PY`；它是在生成配置文件，不需要手动打开编辑器。

**本次核对范围**：已登录服务器核对硬件、镜像、权重和数据；已在不映射 NPU 的临时容器中验证 2k 输入的 tokenizer 长度。本文未实际启动模型或跑完精度、性能测试，因此不提供虚构的精度分数或吞吐量。

服务器登录提示说明它是 MindIE 冒烟环境，使用显存前应联系 `00948935` 确认卡的使用安排。下面按已分配 0 号卡编写；若分配其他卡，只修改 Docker 命令中的 `/dev/davinci0`。不要停止他人的容器或进程。

## 2. 登录并准备目录

### 2.1 在本地终端 A 和 B 分别登录

```bash
ssh root@192.168.9.160
```

按提示输入管理员提供的密码，输入时屏幕不显示字符是正常现象。首次连接时核对服务器指纹并接受。连接不上时，先确认已接入能访问该服务器的内网。

### 2.2 在终端 A：宿主机检查环境

```bash
npu-smi info
docker images quay.io/ascend/vllm-ascend
ls -lh /mnt/weight/Qwen3-8B-W8A8
ss -ltnp 'sport = :8000'
docker ps -a --format 'table {{.Names}}\t{{.Status}}'
```

检查要点：

- 目标卡健康状态为 `OK`，且没有其他业务占用。
- 镜像列表包含 `v0.23.0`。镜像已存在时不需要 `docker pull`。
- 权重目录包含 `config.json`、`tokenizer.json`、`tokenizer_config.json`、`quant_model_description.json` 和 `quant_model_weight_w8a8.safetensors`。
- 8000 端口应没有监听进程。若已占用，先确认现有服务归属，再统一调整本文所有端口配置。
- 首次执行时，本文的两个容器名应尚不存在；后续重复进入见第 9 节。

### 2.3 在终端 A：宿主机准备数据和 AISBench 源码

核对时 `tutorial` 是空目录，但同级目录已有 GSM8K lite。将数据复制到本教程自己的目录：

```bash
mkdir -p /mnt/share/c00814587/tutorial/gsm8k-lite
mkdir -p /mnt/share/c00814587/tutorial/results

cp -n /mnt/share/c00814587/vllm-ascend_gsm8k-lite/train.jsonl \
  /mnt/share/c00814587/tutorial/gsm8k-lite/train.jsonl
cp -n /mnt/share/c00814587/vllm-ascend_gsm8k-lite/test.jsonl \
  /mnt/share/c00814587/tutorial/gsm8k-lite/test.jsonl

wc -l /mnt/share/c00814587/tutorial/gsm8k-lite/*.jsonl
sha256sum /mnt/share/c00814587/tutorial/gsm8k-lite/test.jsonl
```

应看到 `train.jsonl` 和 `test.jsonl` 各 32 行。测试文件 SHA256 应为：

```text
6b1b25e3c772d87e36c01839e3b91e910793caae412785024700e2c270176bb9
```

这里使用现有的 **32 条 lite 数据**，不是完整 GSM8K，也不是另行截取的 100 条数据。保留 `train.jsonl` 是因为 AISBench 的 GSM8K 加载器会读取两个文件；后面的零样本配置不使用训练样例作为提示。

服务器已有 AISBench 源码，首次执行下面两条命令，克隆一份独立副本并固定版本。这样无需通过外网下载源码，也不会带入现有工作目录里的未提交配置：

```bash
git clone --no-hardlinks /mnt/share/c00814587/benchmark \
  /mnt/share/c00814587/tutorial/benchmark
git -C /mnt/share/c00814587/tutorial/benchmark checkout \
  3fd27b4a5fd022fcb5484fb084307f49955491ba
```

出现 `detached HEAD` 是固定到某个提交后的正常提示。如果目标目录已经是上次创建的仓库，不要再次 clone，使用 `git -C /mnt/share/c00814587/tutorial/benchmark rev-parse HEAD` 核对提交即可。

## 3. 启动单卡服务容器

### 3.1 在终端 A：宿主机创建容器

整个代码块是一条 Docker 命令，反斜杠表示续行，其后不要添加空格。

```bash
docker run -dit \
  --name qwen3-8b-server \
  --network host \
  --shm-size 8g \
  --device /dev/davinci0 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /mnt/weight/Qwen3-8B-W8A8:/model:ro \
  -v /mnt/share/c00814587/tutorial:/workspace \
  --entrypoint /bin/bash \
  quay.io/ascend/vllm-ascend:v0.23.0
```

看到一串容器 ID 即表示创建成功。

- 只映射 `/dev/davinci0` 一张计算卡，其余三个 `--device` 是运行时需要的管理设备。
- `/model` 是容器内的权重路径；`ro` 表示只读。
- `/workspace` 对应宿主机 `tutorial`，配置、日志和结果会保留在宿主机。
- `--network host` 让容器直接使用服务器网络。两个容器都可以用 `127.0.0.1:8000` 访问服务。

### 3.2 在终端 A：进入服务容器

```bash
docker exec -it qwen3-8b-server bash
```

以下命令在**服务容器内**执行：

```bash
cd /workspace
python -m pip show vllm vllm-ascend
ls /model
```

### 3.3 在终端 A：服务容器内启动模型

```bash
set -o pipefail
vllm serve /model \
  --served-model-name qwen3-8b \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --quantization ascend \
  --max-model-len 8192 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.8 \
  --no-enable-prefix-caching \
  2>&1 | tee /workspace/server.log
```

**保持终端 A 开着，暂时不要按 Ctrl+C。** 首次加载权重、编译和图捕获可能需要数分钟。出现 `Application startup complete` 后，继续下一节检查接口。

| 参数 | 为什么这样设置 |
| --- | --- |
| `--tensor-parallel-size 1` | 模型只使用一张卡 |
| `--quantization ascend` | 使用这份 Ascend W8A8 量化权重 |
| `--max-model-len 8192` | 单次请求输入与输出总长度上限；覆盖 2048 + 2048，兼顾 lite 精度测试 |
| `--max-num-seqs 16` | 服务端同时调度序列数上限为 16 |
| `--gpu-memory-utilization 0.8` | vLLM 沿用这个参数名，在本环境中控制 NPU 显存预算 |
| `--no-enable-prefix-caching` | 性能测试使用重复的合成输入，关闭前缀缓存，避免复用输入计算 |

不要加 `--enforce-eager` 来跑本文的正式性能数据；它会改变执行方式。如果启动失败，应先根据 `server.log` 解决错误。

## 4. 用单次 curl 验证服务

在**终端 B：宿主机**执行，不要打断终端 A。

先确认服务已就绪：

```bash
curl -f http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
```

`/health` 成功时可能没有正文；`/v1/models` 应返回 JSON，包含模型名 `qwen3-8b`。

发一条聊天请求：

```bash
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-8b",
    "messages": [{"role": "user", "content": "你好，请用一句话介绍自己。"}],
    "max_tokens": 128,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

应能在 `choices[0].message.content` 看到回答。`enable_thinking: false` 关闭 Qwen3 的思考模式，后面的精度配置使用相同设置。

## 5. 在另一个容器安装 AISBench

### 5.1 在终端 B：宿主机创建测试容器

复用本机已有镜像，但使用独立容器。它不映射 NPU，测试通过 HTTP 请求服务；权重挂载只用于读取 tokenizer。

```bash
docker run -dit \
  --name qwen3-aisbench \
  --network host \
  --shm-size 2g \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -v /mnt/weight/Qwen3-8B-W8A8:/model:ro \
  -v /mnt/share/c00814587/tutorial:/workspace \
  -w /workspace \
  --entrypoint /bin/bash \
  quay.io/ascend/vllm-ascend:v0.23.0

docker exec -it qwen3-aisbench bash
```

`TORCH_DEVICE_BACKEND_AUTOLOAD=0` 禁止 PyTorch 自动加载 NPU 后端。由于测试容器没有映射驱动和计算卡，这个设置可避免导入 tokenizer 时出现 `libascend_hal.so` 缺失错误；不要把它加到服务容器。

### 5.2 在终端 B：测试容器内安装

```bash
cd /workspace/benchmark
python --version
python -m pip install -e . --use-pep517
python -m pip install -r requirements/api.txt -r requirements/extra.txt
ais_bench -h
```

Python 应是 3.12；`ais_bench -h` 能打印帮助后再继续。安装依赖需要容器能访问 PyPI 或公司的 Python 镜像源；遇到下载失败，先解决网络/镜像源问题，不要忽略安装错误。

记录版本并再次确认能访问服务：

```bash
git rev-parse HEAD
python -m pip freeze > /workspace/results/aisbench-packages.txt
curl -fsS http://127.0.0.1:8000/v1/models
```

后续第 6～8 节都在这个**测试容器**中执行。

## 6. 跑 GSM8K lite 精度

### 6.1 创建精度配置

下面复用 AISBench 自带的零样本数学问答模板和答案判分器，只修改模型、数据路径等必要配置。

```bash
cd /workspace
cat > accuracy.py <<'PY'
from mmengine.config import read_base

with read_base():
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_chat import models
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_chat_prompt import gsm8k_datasets

models[0].update(
    abbr="qwen3-8b-accuracy",
    path="/model",
    model="qwen3-8b",
    host_ip="127.0.0.1",
    host_port=8000,
    batch_size=16,
    request_rate=0,
    retry=0,
    max_out_len=4096,
    generation_kwargs=dict(
        temperature=0,
        seed=42,
        ignore_eos=False,
        chat_template_kwargs=dict(enable_thinking=False),
    ),
)
gsm8k_datasets[0].update(
    abbr="gsm8k-lite",
    path="/workspace/gsm8k-lite",
)
datasets = gsm8k_datasets
PY
```

精度测试允许自然结束，不强制每题输出 2k。`max_out_len=4096` 是答案长度上限；如果输出被截断，应先检查长度限制，再解释精度变化。

### 6.2 执行精度测试

```bash
ais_bench /workspace/accuracy.py \
  --mode all \
  --dump-eval-details \
  --work-dir /workspace/results/accuracy
```

`--mode all` 包含生成回答与判分。等待命令正常结束，然后寻找摘要：

```bash
find /workspace/results/accuracy -type f -path '*/summary/*'
```

用 `cat` 查看上述命令列出的 `.txt` 或 `.csv` 文件。检查：

1. 数据集名为 `gsm8k-lite`，测试条数为 **32**。
2. 请求全部成功；失败请求不能当成正常低精度结果接受。
3. `accuracy` 是百分比；例如 30/32 对应 93.75%，**这只是计算示例，不是本教程实测结果**。
4. 有错误时，结合输出目录中的 `predictions`、`results` 和 `logs` 查看回答、抽取答案及判分详情。

这是小样本入门验证结果，不代表完整 GSM8K 分数；比较两次结果时应保持数据、提示词、思考模式、生成参数和软件版本一致。

## 7. 跑 2k 输入、2k 输出、bs=16 性能

### 7.1 明确测量口径

- `2k` 在本文中就是 **2048 tokens**，不是 2000 个字或字符。
- `bs=16` 指 AISBench **最多 16 个在途请求**。vLLM 动态调度，起始与结束阶段不保证每一步恰好有 16 条序列。
- 正式测试 128 条请求，总请求数和并发数是两个不同参数。
- 使用流式 `/v1/completions`，便于统计首 token 和后续 token 延迟，也避免聊天模板给输入增加额外 tokens。
- 输入是 2048 个用空格分隔的 `A`，用于固定长度合成测试。它不代表真实业务文本分布。
- 关闭前缀缓存，设置 `ignore_eos=True`、输出下限/上限均为 2048，以固定输出长度。

### 7.2 验证 tokenizer 的输入长度

```bash
python - <<'PY'
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/model")
prompt = " ".join(["A"] * 2048)
length = len(tokenizer.encode(prompt, add_special_tokens=False))
print("input tokens:", length)
assert length == 2048, f"输入长度不符合要求：{length}"
PY
```

应输出 `input tokens: 2048`。本次已用服务器上的 tokenizer 验证这一点；如果以后换权重，必须重新检查。

### 7.3 创建性能配置

```bash
cat > /workspace/performance.py <<'PY'
from mmengine.config import read_base

with read_base():
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_stream import models
    from ais_bench.benchmark.configs.datasets.synthetic.synthetic_gen import synthetic_datasets

models[0].update(
    abbr="qwen3-8b-2k2k-bs16",
    path="/model",
    model="qwen3-8b",
    host_ip="127.0.0.1",
    host_port=8000,
    stream=True,
    batch_size=16,
    request_rate=0,
    retry=0,
    max_out_len=2048,
    generation_kwargs=dict(
        temperature=0,
        seed=42,
        ignore_eos=True,
        min_tokens=2048,
        add_special_tokens=False,
    ),
)
synthetic_datasets[0].update(
    abbr="synthetic-2k2k",
    config={
        "Type": "string",
        "RequestCount": 128,
        "StringConfig": {
            "Input": {
                "Method": "uniform",
                "Params": {"MinValue": 2048, "MaxValue": 2048},
            },
            "Output": {
                "Method": "uniform",
                "Params": {"MinValue": 2048, "MaxValue": 2048},
            },
        },
    },
)
datasets = synthetic_datasets
PY
```

`request_rate=0` 表示不额外限制发送速率，由并发数控制压力；`retry=0` 避免自动重试掩盖失败。AISBench 的 string 合成器按上述配置生成 `A A A ...`，其长度已在上一步校验。

### 7.4 先预热 16 条

仅将请求总数 128 改为 16，生成一份预热配置：

```bash
sed 's/"RequestCount": 128/"RequestCount": 16/' \
  /workspace/performance.py > /workspace/warmup.py

ais_bench /workspace/warmup.py \
  --mode perf \
  --work-dir /workspace/results/warmup
```

等待预热完成且请求无失败。预热结果单独保存，不作为正式性能数据。

### 7.5 再测正式的 128 条

```bash
ais_bench /workspace/performance.py \
  --mode perf \
  --work-dir /workspace/results/performance
```

精度和性能测试依次执行，不要同时跑，也不要让其他客户端在这段时间请求该服务。

### 7.6 查看结果并验收规格

```bash
find /workspace/results/performance -type f -path '*/performances/*'
```

本版本性能结果位于时间戳子目录下的 `performances/qwen3-8b-2k2k-bs16/`。重点查看 `synthetic-2k2k.json`（总体指标）、`synthetic-2k2k.csv`（延迟等统计）和 `synthetic-2k2k_details.jsonl`（请求明细）。用 `cat` 加上上述命令列出的实际文件路径查看；重复运行时选择本次时间戳对应的文件。

从终端报告及结果文件中记录以下信息，单位以实际报告为准：

| 项目 | 要检查或记录什么 |
| --- | --- |
| 成功 / 失败请求 | 正式请求应为 128 条成功、0 条失败 |
| 输入 / 输出 token 长度 | 检查最小值、平均值、最大值，均应为 2048；必要时查看逐请求记录 |
| 输入 / 输出 token 总数 | 正式阶段各应为 128 × 2048 = 262144，不含预热 |
| Request Throughput | 每秒完成请求数，通常为 requests/s |
| Output Token Throughput | 整个服务每秒输出 token 数，通常为 tokens/s |
| TTFT | 从发起请求到收到首个 token 的时间；记录平均值、P50、P99 |
| TPOT | 首 token 之后，平均每个输出 token 的耗时；记录平均值、P50、P99 |
| E2E Latency | 一条请求从发送到完整结束的耗时 |

如果报告只展示均值，不足以证明每条都是 2k/2k，应进一步查看该次运行保存的请求明细。出现长度不符、失败或提前结束时，先检查请求记录和 `server.log`，不要将结果标记为“2k/2k bs16 已通过”。

不要把 `Total Token Throughput`（可能包括输入）当成输出吞吐量；也不要直接用 `1000 / TPOT(ms)` 代替整个服务的输出吞吐量。AISBench 报告的输出吞吐量才是本次需要记录的服务指标。

## 8. 结果保存在哪里

容器内 `/workspace` 就是宿主机 `/mnt/share/c00814587/tutorial`，不需要额外 `docker cp`。

```text
/mnt/share/c00814587/tutorial/
├── gsm8k-lite/               # train.jsonl 和 test.jsonl
├── benchmark/                # 固定版本的 AISBench 源码
├── accuracy.py               # 精度配置
├── performance.py            # 正式性能配置
├── warmup.py                 # 预热配置
├── server.log                # 服务日志
└── results/
    ├── aisbench-packages.txt # 测试容器依赖版本
    ├── accuracy/             # 各次精度运行，含时间戳子目录
    ├── warmup/               # 预热结果
    └── performance/          # 正式性能结果
```

记录结果时至少附上：镜像版本、AISBench 提交、数据文件校验值、启动命令、思考模式、32 条精度分数，以及性能测试的成功数、实际输入/输出长度、输出吞吐量、TTFT、TPOT。

要将结果下载到自己的电脑，请在**新的本地终端**执行：

```bash
scp -r root@192.168.9.160:/mnt/share/c00814587/tutorial/results ./qwen3-results
```

## 9. 测试结束与下次继续

### 9.1 本次结束

1. 等终端 B 的 AISBench 命令结束，输入 `exit` 退出测试容器，返回宿主机。
2. 在终端 A 按 `Ctrl+C` 停止模型服务，等待服务退出，再输入 `exit` 返回宿主机。
3. 在任一宿主机终端停止本文创建的两个容器：

```bash
docker stop qwen3-8b-server qwen3-aisbench
npu-smi info
```

确认模型进程已退出并释放显存。容器和 `tutorial` 内的文件仍保留。

### 9.2 下次继续

宿主机执行：

```bash
docker start qwen3-8b-server qwen3-aisbench
```

- 终端 A：`docker exec -it qwen3-8b-server bash`，重新执行第 3.3 节的启动命令。
- 终端 B：`docker exec -it qwen3-aisbench bash`，等待服务就绪后，重新执行评测命令。原容器还在时不需要重新安装 AISBench。
- `docker start` 只恢复容器的 Bash，不会自动启动 `vllm serve`。
- `tee /workspace/server.log` 会覆盖旧服务日志；需要比较两次运行时，请提前给旧日志改名保存。

## 10. 常见问题速查

| 现象 | 处理方式 |
| --- | --- |
| `Conflict. The container name ... is already in use` | 先 `docker ps -a` 确认；自己上次创建的容器按第 9.2 节恢复 |
| `Connection refused` | 检查终端 A 是否仍在加载，或 `server.log` 是否报错；先通过 `/health` |
| 返回模型不存在 / 404 | 请求和 AISBench 的模型名必须为 `qwen3-8b`，与 `--served-model-name` 一致 |
| 测试容器导入时缺 `libascend_hal.so` | 检查创建测试容器时是否设置了 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` |
| 权重或数据不存在 | 区分宿主机路径与容器路径；容器权重在 `/model`，数据在 `/workspace/gsm8k-lite` |
| `ais_bench: command not found` | 确认在测试容器内，且两条 pip 安装命令都成功 |
| 找不到 `train.jsonl` | GSM8K 加载器需要 train/test 两个文件，即使使用零样本提示也要保留两者 |
| NPU 显存不足 | 先检查目标卡占用和服务日志，确认使用单卡 W8A8；不要直接杀其他进程 |
| 输出不是 2048 tokens | 确认运行的是 performance.py，检查 `min_tokens`、`ignore_eos`、最大输出及失败请求 |
| 精度分数与他人不同 | 核对是否同一份 32 条 lite、相同提示词、思考模式、输出上限和软件版本 |

## 11. 参考与版本依据

- [vLLM Ascend Qwen3 Dense 教程](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-Dense.html)：模型和容器部署背景；本文具体使用服务器现有的 v0.23.0 镜像。
- [AISBench 安装说明](https://github.com/AISBench/benchmark#readme)：源码安装及 API 依赖安装方式。
- [AISBench GSM8K 零样本配置](https://github.com/AISBench/benchmark/blob/3fd27b4a5fd022fcb5484fb084307f49955491ba/ais_bench/benchmark/configs/datasets/gsm8k/gsm8k_gen_0_shot_cot_chat_prompt.py)：提示模板和判分配置。
- [AISBench 合成数据实现](https://github.com/AISBench/benchmark/blob/3fd27b4a5fd022fcb5484fb084307f49955491ba/ais_bench/benchmark/datasets/synthetic.py)：固定 string 输入及输出长度的生成方式。
- [AISBench 流式 completion 配置](https://github.com/AISBench/benchmark/blob/3fd27b4a5fd022fcb5484fb084307f49955491ba/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_stream.py)：流式 API 配置依据。

上游文档会变化。复现本文时，优先使用第 2 节固定的 AISBench 提交，不要在安装时随手升级到 master。
